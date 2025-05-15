from datasets import load_dataset
from sparsify.sparsify import Sae
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from typing import Dict, Tuple, List, Set
import argparse
import os
import sys
import time
import csv

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Llama 3 with SAE hooks')
parser.add_argument('--prompt', type=str, default=None, help='Prompt to test with')
parser.add_argument('--prompts-file', type=str, default=None, help='File with prompts to process (one per line)')
parser.add_argument('--answers-file', type=str, default=None, help='File with correct next tokens (one per line)')
parser.add_argument('--output-dir', type=str, default='sae_results', help='Directory to save results')
parser.add_argument('--gpu-id', type=int, default=0, help='GPU ID to use (default: 0)')
args = parser.parse_args()

# Create output directory if needed
os.makedirs(args.output_dir, exist_ok=True)

# Set CUDA device if specified
if args.gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    print(f"Using GPU {args.gpu_id}")

# Get prompts - either from file or single prompt
prompts = []
correct_answers = []

if args.prompts_file:
    print(f"Reading prompts from file: {args.prompts_file}")
    try:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip().replace("Respond with only one token. ", "") for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts")
        
        # Load answers if provided
        if args.answers_file:
            print(f"Reading answers from file: {args.answers_file}")
            with open(args.answers_file, 'r') as f:
                correct_answers = [line.strip() for line in f if line.strip()]
            if len(correct_answers) != len(prompts):
                print(f"Warning: Number of answers ({len(correct_answers)}) doesn't match number of prompts ({len(prompts)})")
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)
elif args.prompt:
    prompts = [args.prompt]
else:
    prompts = ["The best way to predict the future is to"]

# Load dataset
print("Loading dataset...")
ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", streaming=True, trust_remote_code=True)

# Load SAE
print("Loading SAE...")
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x-v2", hookpoint="layers.24").to('cuda')

# Load model
print("Loading model...")
model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B").to('cuda')

# Global dictionary to store hook metrics
hook_metrics = {}

# Single hook (original approach)
def sae_hook_single(
    act: torch.Tensor, 
    hook: Dict
) -> torch.Tensor:
    print("Processing with single SAE hook...")
    print(f"Input shape: {act.shape}")
    batch_size, seq_len, hidden_size = act.shape
    with torch.no_grad():
        # Only process the last token position
        last_token_pos = seq_len - 1
        last_token_act = act[0, last_token_pos].unsqueeze(0)  # Shape: [1, hidden_size]
        
        encoded = sae.encode(last_token_act)
        top_acts = encoded.top_acts
        top_indices = encoded.top_indices
        
        # Store metrics in global dict instead of hook
        global hook_metrics
        hook_metrics['single'] = {
            'active_features': len(set(top_indices.cpu().numpy().flatten().tolist())),
            'feature_indices': top_indices.cpu().numpy().flatten().tolist()
        }
        
        print(f"Last token activated features: {top_indices.shape}")
        
        decoded = sae.decode(top_acts, top_indices)
        
        # Keep all other positions unchanged
        result = act.clone()
        result[0, last_token_pos] = decoded.squeeze(0)
        
    print(f"Output shape: {result.shape}")
    return result

# Double hook (2-hook approach)
def sae_hook_double(
    act: torch.Tensor, 
    hook: Dict
) -> torch.Tensor:
    print("Processing with double SAE hook (2-hook)...")
    print(f"Input shape: {act.shape}")
    batch_size, seq_len, hidden_size = act.shape
    with torch.no_grad():
        try:
            # Only process the last token position
            last_token_pos = seq_len - 1
            last_token_act = act[0, last_token_pos].unsqueeze(0)  # Shape: [1, hidden_size]
            
            # First pass through SAE for last token
            encoded1 = sae.encode(last_token_act)
            top_acts1 = encoded1.top_acts
            top_indices1 = encoded1.top_indices
            decoded1 = sae.decode(top_acts1, top_indices1)
            
            # Second pass through SAE for last token
            encoded2 = sae.encode(decoded1)
            top_acts2 = encoded2.top_acts
            top_indices2 = encoded2.top_indices
            
            # Convert to sets of unique indices for proper comparison
            indices1_set = set(top_indices1.cpu().numpy().flatten().tolist())
            indices2_set = set(top_indices2.cpu().numpy().flatten().tolist())
            
            # Calculate feature overlap statistics
            shared_indices = indices1_set.intersection(indices2_set)
            reactivation_ratio = len(shared_indices) / len(indices1_set) if len(indices1_set) > 0 else 0
            
            union_size = len(indices1_set.union(indices2_set))
            jaccard = len(shared_indices) / union_size if union_size > 0 else 0
            
            # Store metrics in global dict instead of hook
            global hook_metrics
            hook_metrics['double'] = {
                'first_pass_features': len(indices1_set),
                'second_pass_features': len(indices2_set),
                'shared_features': len(shared_indices),
                'reactivation_ratio': reactivation_ratio,
                'jaccard_similarity': jaccard,
                'first_pass_indices': list(indices1_set),
                'second_pass_indices': list(indices2_set)
            }
            
            print(f"\nLast token feature statistics:")
            print(f"Features in first pass: {len(indices1_set)}")
            print(f"Features in second pass: {len(indices2_set)}")
            print(f"Shared features: {len(shared_indices)}")
            print(f"Reactivation ratio: {reactivation_ratio:.4f} ({reactivation_ratio*100:.2f}%)")
            print(f"Jaccard similarity: {jaccard:.4f} ({jaccard*100:.2f}%)")
            
            # Apply the final result
            result = act.clone()
            result[0, last_token_pos] = sae.decode(top_acts2, top_indices2).squeeze(0)
        except Exception as e:
            print(f"Error in double SAE hook: {e}")
            import traceback
            traceback.print_exc()
            # Return original activations in case of error
            return act
    
    print(f"Output shape: {result.shape}")
    return result

# Constant hook (iterative until convergence)
def sae_hook_constant(
    act: torch.Tensor, 
    hook: Dict
) -> torch.Tensor:
    print("Processing with constant SAE hook (converging features)...")
    print(f"Input shape: {act.shape}")
    
    MAX_ITERATIONS = 100  # Maximum number of iterations to prevent infinite loops
    CONVERGENCE_THRESHOLD = 1.0  # 100% match required for convergence
    
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
        convergence_type = 'max_iter'
        
        # Store iteration data for analysis
        iteration_indices_sets = []
        iteration_indices = []
        indices_history = []  # Keep track of all previous sets of indices to detect cycles
        jaccard_history = []
        
        print(f"\nLast token position: Starting convergence loop")
        
        while not converged and iteration < MAX_ITERATIONS:
            # Encode and decode
            encoded = sae.encode(current_act)
            top_acts = encoded.top_acts
            top_indices = encoded.top_indices
            
            # Convert to set for comparison
            current_indices_set = frozenset(top_indices.cpu().numpy().flatten().tolist())
            
            # Check if we've seen these indices before (cycle detection)
            if current_indices_set in indices_history:
                cycle_start = indices_history.index(current_indices_set)
                cycle_length = iteration - cycle_start
                print(f"Detected cycle! Indices at iteration {iteration+1} match those from iteration {cycle_start+1}")
                print(f"Cycle length: {cycle_length}")
                converged = True
                convergence_type = 'cycle'
                break
            
            # Store current indices for cycle detection
            indices_history.append(current_indices_set)
            iteration_indices.append(list(current_indices_set))
            
            # Save regular set version for normal analysis
            current_indices_regular_set = set(current_indices_set)
            iteration_indices_sets.append(current_indices_regular_set)
            
            # Check for convergence with previous iteration
            if prev_indices_set is not None:
                # Calculate Jaccard similarity
                shared_indices = current_indices_regular_set.intersection(prev_indices_set)
                union_size = len(current_indices_regular_set.union(prev_indices_set))
                jaccard = len(shared_indices) / union_size if union_size > 0 else 1.0
                jaccard_history.append(jaccard)
                
                # Check if converged based on threshold
                if jaccard >= CONVERGENCE_THRESHOLD:
                    converged = True
                    convergence_type = 'fixed_point'
                    print(f"Converged after {iteration+1} iterations (perfect match with previous iteration)")
                else:
                    print(f"Iteration {iteration+1}, Jaccard similarity: {jaccard:.4f}")
                    print(f"  Active features: {len(current_indices_regular_set)}")
            else:
                print(f"Iteration 1, Active features: {len(current_indices_regular_set)}")
            
            # Decode for next iteration
            current_act = sae.decode(top_acts, top_indices)
            
            # Update for next iteration
            prev_indices_set = current_indices_regular_set
            iteration += 1
        
        # Calculate feature stability metrics
        first_final_jaccard = 0
        shared_features = 0
        if len(iteration_indices_sets) >= 2:
            # Compare first and final sets
            first_set = iteration_indices_sets[0]
            final_set = iteration_indices_sets[-1]
            shared = first_set.intersection(final_set)
            union = first_set.union(final_set)
            first_final_jaccard = len(shared) / len(union) if len(union) > 0 else 1.0
            shared_features = len(shared)
            
            # Analyze feature stability across iterations
            print(f"First→Final Jaccard similarity: {first_final_jaccard:.4f}")
            print(f"First iteration: {len(first_set)} features, Final iteration: {len(final_set)} features")
            print(f"Shared features between first and final: {shared_features}")
        
        # Store metrics in global dict instead of hook
        global hook_metrics
        hook_metrics['constant'] = {
            'iterations': iteration,
            'convergence_type': convergence_type,
            'first_final_jaccard': first_final_jaccard,
            'shared_features': shared_features,
            'feature_counts': [len(s) for s in iteration_indices_sets],
            'jaccard_history': jaccard_history,
            'iteration_indices': iteration_indices
        }
        
        if convergence_type == 'cycle':
            hook_metrics['constant'].update({
                'cycle_start': cycle_start + 1,  # 1-indexed for human readability
                'cycle_length': cycle_length
            })
        
        # Create the result tensor (leave all other positions unchanged)
        result = act.clone()
        result[0, last_token_pos] = current_act.squeeze(0)
    
    return result

# The hook point
hook_name = get_act_name("resid_post", 24)  # Layer 25 (1-indexed)

def run_model(prompt, hook_type='none'):
    """Run the model with specified hook type and return results."""
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
    
    global hook_metrics
    if hook_type == 'none':
        # Run without hooks
        output = model(input_ids)
        hook_metrics['none'] = {}
    else:
        # Select hook function based on type
        if hook_type == 'double':
            hook_fn = sae_hook_double
        elif hook_type == 'constant':
            hook_fn = sae_hook_constant
        else:
            hook_fn = sae_hook_single
        
        # Run with the selected hook
        output = model.run_with_hooks(
            input_ids,
            fwd_hooks=[(hook_name, hook_fn)],
            return_type="logits"
        )
    
    return {
        'logits': output,
        'next_token': {
            'id': torch.argmax(output[0, -1]).item(),
            'text': model.tokenizer.decode([torch.argmax(output[0, -1]).item()])
        },
        'top5': {
            'logits': torch.topk(output[0, -1], 5)[0],
            'indices': torch.topk(output[0, -1], 5)[1],
            'tokens': [model.tokenizer.decode([idx.item()]) for idx in torch.topk(output[0, -1], 5)[1]],
            'probs': torch.softmax(torch.topk(output[0, -1], 5)[0], dim=0)
        },
        'distribution': torch.softmax(output[0, -1], dim=0)
    }

def calculate_kl_divergence(p, q):
    """Calculate KL divergence between two probability distributions."""
    p = torch.softmax(p, dim=0)
    q = torch.softmax(q, dim=0)
    return torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10)))

def process_prompt(prompt, idx, correct_answer=None):
    """Process a single prompt and save results to files"""
    print(f"Processing prompt {idx}: {prompt[:50]}...")
    
    # Clear previous hook metrics
    global hook_metrics
    hook_metrics = {}
    
    # Create output files
    output_file = os.path.join(args.output_dir, f"prompt_{idx}_gpu{args.gpu_id}.txt")
    csv_file = os.path.join(args.output_dir, f"prompt_{idx}_metrics.csv")
    
    # Redirect stdout to capture all output
    original_stdout = sys.stdout
    with open(output_file, 'w') as f:
        sys.stdout = f
        
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        
        try:
            # Run all versions
            results = {}
            for hook_type in ['none', 'single', 'double', 'constant']:
                results[hook_type] = run_model(prompt, hook_type)
            
            # Calculate metrics for CSV
            csv_data = []
            base_logits = results['none']['logits'][0, -1]
            
            # Add base version
            csv_data.append({
                'version': 'base',
                'top_token': results['none']['next_token']['text'],
                'top_token_is_right': str(results['none']['next_token']['text'].strip().lower() == correct_answer.strip().lower() if correct_answer else 'NA'),
                'top_5_right': str(any(t.strip().lower() == correct_answer.strip().lower() for t in results['none']['top5']['tokens']) if correct_answer else 'NA'),
                'jaccard': 'NA',
                'kl_div': 'NA'
            })
            
            # Add other versions
            for hook_type in ['single', 'double', 'constant']:
                # Get KL divergence with base distribution
                kl_div = calculate_kl_divergence(
                    base_logits,
                    results[hook_type]['logits'][0, -1]
                ).item()
                
                # Get Jaccard similarity
                jaccard = 'NA'
                if hook_type == 'double' and 'jaccard_similarity' in hook_metrics.get('double', {}):
                    jaccard = hook_metrics['double']['jaccard_similarity']
                elif hook_type == 'constant' and 'first_final_jaccard' in hook_metrics.get('constant', {}):
                    jaccard = hook_metrics['constant']['first_final_jaccard']
                
                csv_data.append({
                    'version': hook_type,
                    'top_token': results[hook_type]['next_token']['text'],
                    'top_token_is_right': str(results[hook_type]['next_token']['text'].strip().lower() == correct_answer.strip().lower() if correct_answer else 'NA'),
                    'top_5_right': str(any(t.strip().lower() == correct_answer.strip().lower() for t in results[hook_type]['top5']['tokens']) if correct_answer else 'NA'),
                    'jaccard': str(jaccard),
                    'kl_div': str(kl_div)
                })
            
            # Write CSV file
            with open(csv_file, 'w', newline='') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=['version', 'top_token', 'top_token_is_right', 'top_5_right', 'jaccard', 'kl_div'])
                writer.writeheader()
                writer.writerows(csv_data)
            
            # Print comparison results
            for hook_type, result in results.items():
                print(f"\n{hook_type.upper()} HOOK:")
                print(f"Next token: '{result['next_token']['text']}'")
                print("Top 5 tokens:")
                for i in range(5):
                    print(f"  {i+1}. '{result['top5']['tokens'][i]}' (prob: {result['top5']['probs'][i].item():.4f})")
                
                if hook_type in hook_metrics:
                    if hook_type == 'constant':
                        metrics = hook_metrics[hook_type]
                        print(f"\nConvergence metrics:")
                        print(f"Iterations: {metrics['iterations']}")
                        print(f"Convergence type: {metrics['convergence_type']}")
                        print(f"First→Final Jaccard: {metrics['first_final_jaccard']:.4f}")
                        print(f"Shared features: {metrics['shared_features']}")
                    elif hook_type == 'double':
                        metrics = hook_metrics[hook_type]
                        print(f"\nFeature overlap metrics:")
                        print(f"Jaccard similarity: {metrics['jaccard_similarity']:.4f}")
                        print(f"Shared features: {metrics['shared_features']}")
            
            # Print KL divergences
            print("\nDISTRIBUTION DIFFERENCES (KL DIVERGENCE):")
            hook_types = list(results.keys())
            for i in range(len(hook_types)):
                for j in range(i + 1, len(hook_types)):
                    type1, type2 = hook_types[i], hook_types[j]
                    kl_div = calculate_kl_divergence(
                        results[type1]['logits'][0, -1],
                        results[type2]['logits'][0, -1]
                    ).item()
                    print(f"{type1.capitalize()} vs {type2.capitalize()} hook: {kl_div:.6f}")
            
        except Exception as e:
            print(f"ERROR processing prompt: {e}")
            import traceback
            traceback.print_exc()
        
    # Restore stdout
    sys.stdout = original_stdout
    print(f"Finished prompt {idx}")
    print(f"Results saved to {output_file}")
    print(f"Metrics saved to {csv_file}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()

def main():
    """Main function to process all prompts and generate summary"""
    # Initialize statistics tracking
    stats = {
        'base': {'correct': 0, 'total': 0, 'top5_correct': 0},
        'single': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'top5_correct': 0},
        'double': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'jaccard_sum': 0.0, 'top5_correct': 0},
        'constant': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'jaccard_sum': 0.0, 'top5_correct': 0}
    }

    start_time = time.time()
    
    # Process each prompt
    for idx, prompt in enumerate(prompts):
        # Pass the correct answer if available
        correct_answer = correct_answers[idx] if idx < len(correct_answers) else None
        process_prompt(prompt, idx, correct_answer)
        
        # Update statistics based on results
        for version in ['base', 'single', 'double', 'constant']:
            metrics_file = os.path.join(args.output_dir, f"prompt_{idx}_metrics.csv")
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['version'] == version:
                            if row['top_token_is_right'].lower() == 'true':
                                stats[version]['correct'] += 1
                            if row['top_5_right'].lower() == 'true':
                                stats[version]['top5_correct'] += 1
                            stats[version]['total'] += 1
                            if version != 'base' and row['kl_div'] != 'NA':
                                try:
                                    stats[version]['kl_sum'] += float(row['kl_div'])
                                except ValueError:
                                    pass
                            if version in ['double', 'constant'] and row['jaccard'] != 'NA':
                                try:
                                    stats[version]['jaccard_sum'] += float(row['jaccard'])
                                except ValueError:
                                    pass
    
    elapsed_time = time.time() - start_time
    
    # Calculate summary statistics
    summary = {
        'total_prompts': len(prompts),
        'processing_time': elapsed_time,
        'accuracy': {},
        'top5_accuracy': {},
        'avg_jaccard': {},
        'avg_kl_div': {}
    }
    
    # Calculate percentages and averages
    for version in stats:
        # Calculate accuracy
        total = stats[version]['total']
        if total > 0:
            summary['accuracy'][version] = (stats[version]['correct'] / total) * 100
            summary['top5_accuracy'][version] = (stats[version]['top5_correct'] / total) * 100
        else:
            summary['accuracy'][version] = 0
            summary['top5_accuracy'][version] = 0
        
        # Calculate average KL divergence for non-base versions
        if version != 'base' and total > 0:
            summary['avg_kl_div'][version] = stats[version]['kl_sum'] / total
        
        # Calculate average Jaccard for double and constant
        if version in ['double', 'constant'] and total > 0:
            summary['avg_jaccard'][version] = stats[version]['jaccard_sum'] / total
    
    # Write summary to file
    summary_file = os.path.join(args.output_dir, "summary_stats.txt")
    with open(summary_file, 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("=================\n\n")
        f.write(f"Total prompts processed: {summary['total_prompts']}\n")
        f.write(f"Total processing time: {summary['processing_time']:.1f} seconds\n\n")
        
        f.write("Top-1 Accuracy (% correct predictions)\n")
        f.write("---------------------------------\n")
        for version in ['base', 'single', 'double', 'constant']:
            f.write(f"{version:8}: {summary['accuracy'][version]:6.2f}%\n")
        f.write("\n")
        
        f.write("Top-5 Accuracy (% correct in top 5)\n")
        f.write("---------------------------------\n")
        for version in ['base', 'single', 'double', 'constant']:
            f.write(f"{version:8}: {summary['top5_accuracy'][version]:6.2f}%\n")
        f.write("\n")
        
        f.write("Average Jaccard Similarity\n")
        f.write("-------------------------\n")
        for version in ['double', 'constant']:
            f.write(f"{version:8}: {summary['avg_jaccard'].get(version, 0):6.4f}\n")
        f.write("\n")
        
        f.write("Average KL Divergence\n")
        f.write("--------------------\n")
        for version in ['single', 'double', 'constant']:
            f.write(f"{version:8}: {summary['avg_kl_div'].get(version, 0):6.4f}\n")
    
    print(f"All prompts processed in {elapsed_time:.1f} seconds")
    print(f"Results saved in {args.output_dir}")
    print(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()