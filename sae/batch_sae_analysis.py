#!/usr/bin/env python3

import os
import json
import torch
import argparse
import multiprocessing
from datetime import datetime
from datasets import load_dataset
from sparsify.sparsify import Sae
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from typing import Dict, List, Set, Any, Tuple
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Batch process prompts through Llama 3 with SAE hooks')
    parser.add_argument('--prompts', type=str, nargs='+', default=[], 
                      help='Prompts to process (can specify multiple)')
    parser.add_argument('--prompts-file', type=str, default=None,
                      help='Path to file containing prompts (one per line)')
    parser.add_argument('--output-dir', type=str, default='sae_results',
                      help='Directory to save results')
    parser.add_argument('--max-iter', type=int, default=100,
                      help='Maximum iterations for constant hook')
    parser.add_argument('--layer', type=int, default=25, 
                      help='Layer to apply SAE to (1-indexed)')
    parser.add_argument('--hook-types', type=str, nargs='+', 
                      default=['none', 'single', 'double', 'constant'],
                      help='Hook types to run (none, single, double, constant)')
    parser.add_argument('--num-gpus', type=int, default=8,
                      help='Number of GPUs to use for parallel processing')
    return parser.parse_args()

class SaeAnalyzer:
    def __init__(self, layer_idx=24, max_iterations=100, gpu_id=0):
        """Initialize the analyzer with models and parameters.
        
        Args:
            layer_idx: 0-indexed layer number (25 in 1-indexing would be 24)
            max_iterations: Maximum iterations for constant hook
            gpu_id: GPU ID to use for this analyzer instance
        """
        self.max_iterations = max_iterations
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        print(f"Initializing analyzer on {self.device}")
        
        # Load models
        print(f"Loading SAE model on {self.device}...")
        self.sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x-v2", hookpoint=f"layers.{layer_idx}")
        self.sae = self.sae.to(self.device)
        
        print(f"Loading Llama-3 model on {self.device}...")
        self.model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B", device=self.device)
        
        # Set up hook point
        self.layer_idx = layer_idx  # 0-indexed layer number
        self.hook_name = get_act_name("resid_post", layer_idx)
    
    def sae_hook_single(self, act: torch.Tensor, hook: Dict) -> torch.Tensor:
        """Single SAE hook that processes the last token once."""
        batch_size, seq_len, hidden_size = act.shape
        with torch.no_grad():
            # Only process the last token position
            last_token_pos = seq_len - 1
            last_token_act = act[0, last_token_pos].unsqueeze(0)  # Shape: [1, hidden_size]
            
            encoded = self.sae.encode(last_token_act)
            top_acts = encoded.top_acts
            top_indices = encoded.top_indices
            
            # Store metrics for later analysis
            metrics = {
                'active_features': len(set(top_indices.cpu().numpy().flatten().tolist())),
                'feature_indices': top_indices.cpu().numpy().flatten().tolist()
            }
            self.single_hook_metrics = metrics
            
            decoded = self.sae.decode(top_acts, top_indices)
            
            # Keep all other positions unchanged
            result = act.clone()
            result[0, last_token_pos] = decoded.squeeze(0)
            
        return result

    def sae_hook_double(self, act: torch.Tensor, hook: Dict) -> torch.Tensor:
        """Double SAE hook that processes the last token twice."""
        batch_size, seq_len, hidden_size = act.shape
        with torch.no_grad():
            # Only process the last token position
            last_token_pos = seq_len - 1
            last_token_act = act[0, last_token_pos].unsqueeze(0)  # Shape: [1, hidden_size]
            
            # First pass through SAE
            encoded1 = self.sae.encode(last_token_act)
            top_acts1 = encoded1.top_acts
            top_indices1 = encoded1.top_indices
            decoded1 = self.sae.decode(top_acts1, top_indices1)
            
            # Second pass through SAE
            encoded2 = self.sae.encode(decoded1)
            top_acts2 = encoded2.top_acts
            top_indices2 = encoded2.top_indices
            
            # Convert to sets of unique indices for comparison
            indices1_set = set(top_indices1.cpu().numpy().flatten().tolist())
            indices2_set = set(top_indices2.cpu().numpy().flatten().tolist())
            
            # Calculate feature overlap statistics
            shared_indices = indices1_set.intersection(indices2_set)
            reactivation_ratio = len(shared_indices) / len(indices1_set) if len(indices1_set) > 0 else 0
            
            union_size = len(indices1_set.union(indices2_set))
            jaccard = len(shared_indices) / union_size if union_size > 0 else 0
            
            # Store metrics for later analysis
            metrics = {
                'first_pass_features': len(indices1_set),
                'second_pass_features': len(indices2_set),
                'shared_features': len(shared_indices),
                'reactivation_ratio': reactivation_ratio,
                'jaccard_similarity': jaccard,
                'first_pass_indices': list(indices1_set),
                'second_pass_indices': list(indices2_set)
            }
            self.double_hook_metrics = metrics
            
            # Apply the final result
            result = act.clone()
            result[0, last_token_pos] = self.sae.decode(top_acts2, top_indices2).squeeze(0)
        
        return result

    def sae_hook_constant(self, act: torch.Tensor, hook: Dict) -> torch.Tensor:
        """Constant hook that iterates until convergence or cycle detection."""
        batch_size, seq_len, hidden_size = act.shape
        
        with torch.no_grad():
            # Only focus on the last token position
            last_token_pos = seq_len - 1
            token_act = act[0, last_token_pos].unsqueeze(0)  # Shape: [1, hidden_size]
            current_act = token_act.clone()
            
            # Track feature indices across iterations
            prev_indices_set = None
            iteration = 0
            converged = False
            
            # Store iteration data for analysis
            iteration_indices_sets = []
            indices_history = []  # For cycle detection
            jaccard_history = []
            
            while not converged and iteration < self.max_iterations:
                # Encode and decode
                encoded = self.sae.encode(current_act)
                top_acts = encoded.top_acts
                top_indices = encoded.top_indices
                
                # Convert to frozenset for hashable comparison
                current_indices_set = frozenset(top_indices.cpu().numpy().flatten().tolist())
                
                # Check for cycle
                if current_indices_set in indices_history:
                    cycle_start = indices_history.index(current_indices_set)
                    cycle_length = iteration - cycle_start
                    converged = True
                    convergence_type = 'cycle'
                    break
                
                # Store current indices
                indices_history.append(current_indices_set)
                
                # Save regular set version for analysis
                current_indices_regular_set = set(current_indices_set)
                iteration_indices_sets.append(current_indices_regular_set)
                
                # Check for convergence with previous iteration
                if prev_indices_set is not None:
                    # Calculate Jaccard similarity
                    shared_indices = current_indices_regular_set.intersection(prev_indices_set)
                    union_size = len(current_indices_regular_set.union(prev_indices_set))
                    jaccard = len(shared_indices) / union_size if union_size > 0 else 1.0
                    jaccard_history.append(jaccard)
                    
                    # Check if converged based on threshold (perfect match)
                    if jaccard >= 1.0:
                        converged = True
                        convergence_type = 'fixed_point'
                        break
                
                # Decode for next iteration
                current_act = self.sae.decode(top_acts, top_indices)
                
                # Update for next iteration
                prev_indices_set = current_indices_regular_set
                iteration += 1
            
            # Calculate feature stability metrics
            if len(iteration_indices_sets) >= 2:
                first_set = iteration_indices_sets[0]
                final_set = iteration_indices_sets[-1]
                shared = first_set.intersection(final_set)
                first_final_jaccard = len(shared) / len(first_set.union(final_set)) if len(first_set.union(final_set)) > 0 else 1.0
            else:
                first_final_jaccard = 1.0
                
            # Store metrics for later analysis
            if not converged:
                convergence_type = 'max_iter'
                
            metrics = {
                'iterations': iteration + 1,
                'convergence_type': convergence_type,
                'first_final_jaccard': first_final_jaccard,
                'feature_counts': [len(s) for s in iteration_indices_sets],
                'jaccard_history': jaccard_history,
            }
            
            if convergence_type == 'cycle':
                metrics['cycle_start'] = cycle_start + 1  # 1-indexed for human readability
                metrics['cycle_length'] = cycle_length
            
            self.constant_hook_metrics = metrics
            
            # Create the result tensor
            result = act.clone()
            result[0, last_token_pos] = current_act.squeeze(0)
        
        return result

    def run_model(self, prompt, hook_type='none'):
        """Run the model with specified hook type and return results."""
        input_ids = self.model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        
        if hook_type == 'none':
            # Run without hooks
            output = self.model(input_ids)
        else:
            # Select hook function based on type
            if hook_type == 'double':
                hook_fn = self.sae_hook_double
            elif hook_type == 'constant':
                hook_fn = self.sae_hook_constant
            else:
                hook_fn = self.sae_hook_single
                
            fwd_hooks = [(self.hook_name, hook_fn)]
            
            # Run with the selected hook
            output = self.model.run_with_hooks(
                input_ids,
                fwd_hooks=fwd_hooks,
                return_type="logits"
            )
        
        # Get top token predictions
        next_token_id = torch.argmax(output[0, -1]).item()
        top5_logits, top5_indices = torch.topk(output[0, -1], 5)
        top5_probs = torch.softmax(top5_logits, dim=0)
        
        # Create results dict
        results = {
            'next_token': {
                'id': next_token_id,
                'text': self.model.tokenizer.decode([next_token_id])
            },
            'top5': {
                'indices': top5_indices.cpu().numpy().tolist(),
                'tokens': [self.model.tokenizer.decode([idx.item()]) for idx in top5_indices],
                'probs': top5_probs.cpu().numpy().tolist()
            },
            'full_logits': output[0, -1].cpu().numpy().tolist()
        }
        
        # Add hook-specific metrics if applicable
        if hook_type == 'single' and hasattr(self, 'single_hook_metrics'):
            results['hook_metrics'] = self.single_hook_metrics
        elif hook_type == 'double' and hasattr(self, 'double_hook_metrics'):
            results['hook_metrics'] = self.double_hook_metrics
        elif hook_type == 'constant' and hasattr(self, 'constant_hook_metrics'):
            results['hook_metrics'] = self.constant_hook_metrics
        
        return results

    def analyze_prompt(self, prompt, hook_types=None):
        """Run all hooks on a prompt and return comparative results."""
        if hook_types is None:
            hook_types = ['none', 'single', 'double', 'constant']
            
        results = {}
        
        # Run each hook type
        for hook_type in hook_types:
            print(f"Running {hook_type} hook for prompt: {prompt[:30]}...")
            results[hook_type] = self.run_model(prompt, hook_type)
        
        # Calculate KL divergences between distributions
        kl_divergences = {}
        for i, hook1 in enumerate(hook_types):
            for j, hook2 in enumerate(hook_types):
                if i < j:  # Only calculate each pair once
                    # Get the logits
                    logits1 = results[hook1]['full_logits']
                    logits2 = results[hook2]['full_logits']
                    
                    # Convert to PyTorch tensors
                    logits1_tensor = torch.tensor(logits1)
                    logits2_tensor = torch.tensor(logits2)
                    
                    # Calculate KL divergence
                    p_log = torch.log_softmax(logits1_tensor, dim=0)
                    q_log = torch.log_softmax(logits2_tensor, dim=0)
                    kl_div = torch.sum(torch.exp(p_log) * (p_log - q_log)).item()
                    
                    kl_divergences[f"{hook1}_vs_{hook2}"] = kl_div
        
        # Add KL divergences to results
        results['kl_divergences'] = kl_divergences
        
        # Check for prediction changes
        prediction_changes = {}
        for i, hook1 in enumerate(hook_types):
            for j, hook2 in enumerate(hook_types):
                if i < j:  # Only calculate each pair once
                    token1 = results[hook1]['next_token']['id']
                    token2 = results[hook2]['next_token']['id']
                    
                    prediction_changes[f"{hook1}_vs_{hook2}"] = token1 != token2
        
        # Add prediction changes to results
        results['prediction_changes'] = prediction_changes
        
        return results

def process_prompt(args_tuple):
    """Process a single prompt on a specific GPU.
    
    Args:
        args_tuple: Tuple containing (prompt_index, prompt, args, gpu_id)
        
    Returns:
        Dict containing the results for the prompt
    """
    prompt_index, prompt, args, gpu_id = args_tuple
    
    print(f"GPU {gpu_id}: Processing prompt {prompt_index}: {prompt[:30]}...")
    
    # Create analyzer with specific GPU
    analyzer = SaeAnalyzer(
        layer_idx=args.layer-1,  # Convert 1-indexed to 0-indexed (default 25->24)
        max_iterations=args.max_iter,
        gpu_id=gpu_id
    )
    
    # Analyze prompt with specified hook types
    results = analyzer.analyze_prompt(prompt, hook_types=args.hook_types)
    
    # Save individual prompt result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prompt_filename = f"prompt_{prompt_index}_{timestamp}_gpu{gpu_id}.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, prompt_filename), 'w') as f:
        json.dump({
            'prompt': prompt,
            'results': results,
            'gpu_id': gpu_id
        }, f, indent=2)
    
    return prompt, results

def process_prompts(prompts, args):
    """Process a list of prompts in parallel using multiple GPUs."""
    # Check GPU availability
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus < 1:
        print("No GPUs available. Using CPU.")
        num_gpus = 1
    
    print(f"Processing prompts using {num_gpus} GPUs")
    
    # Create task arguments for parallel processing
    tasks = []
    for i, prompt in enumerate(prompts):
        # Assign each prompt to a GPU in round-robin fashion
        gpu_id = i % num_gpus
        tasks.append((i+1, prompt, args, gpu_id))
    
    # Process prompts in parallel
    all_results = {}
    
    if num_gpus > 1:
        # Use multiprocessing to distribute across GPUs
        with multiprocessing.Pool(processes=num_gpus) as pool:
            results = pool.map(process_prompt, tasks)
            
            # Collect results
            for prompt, result in results:
                all_results[prompt] = result
    else:
        # Process sequentially if only one GPU/CPU
        for task in tasks:
            prompt, result = process_prompt(task)
            all_results[prompt] = result
    
    # Save combined results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    combined_filename = f"all_prompts_{timestamp}.json"
    with open(os.path.join(args.output_dir, combined_filename), 'w') as f:
        json.dump({
            'prompts': prompts,
            'layer': args.layer,
            'max_iter': args.max_iter,
            'hook_types': args.hook_types,
            'num_gpus': num_gpus,
            'results': all_results
        }, f, indent=2)
    
    return all_results

def main():
    # Parse arguments
    args = parse_args()
    
    # Get prompts from command line or file
    prompts = args.prompts
    if args.prompts_file:
        try:
            with open(args.prompts_file, 'r') as f:
                file_prompts = [line.strip() for line in f if line.strip()]
                prompts.extend(file_prompts)
        except Exception as e:
            print(f"Error reading prompts file: {e}")
            sys.exit(1)
    
    # Check if we have prompts
    if not prompts:
        print("No prompts provided. Please specify prompts with --prompts or --prompts-file")
        parser.print_help()
        sys.exit(1)
    
    # Process prompts
    print(f"Processing {len(prompts)} prompts...")
    results = process_prompts(prompts, args)
    
    print(f"Results saved to {args.output_dir}")

if __name__ == "__main__":
    # Required for multi-GPU processing to work properly
    multiprocessing.set_start_method('spawn', force=True)
    main()