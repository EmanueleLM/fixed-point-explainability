from datasets import load_dataset
from sparsify.sparsify import Sae
import torch
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from typing import Dict, Tuple, List, Set

# Fixed prompt
PROMPT = "The best way to predict the future is to"

# Load dataset
print("Loading dataset...")
ds = load_dataset("togethercomputer/RedPajama-Data-V2", name="sample", streaming=True, trust_remote_code=True)

# Load SAE
print("Loading SAE...")
sae = Sae.load_from_hub("EleutherAI/sae-llama-3-8b-32x-v2", hookpoint="layers.24").to('cuda')

# Load model
print("Loading model...")
model = HookedTransformer.from_pretrained("meta-llama/Meta-Llama-3-8B").to('cuda')

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
        
        print(f"\nLast token feature statistics:")
        print(f"Features in first pass: {len(indices1_set)}")
        print(f"Features in second pass: {len(indices2_set)}")
        print(f"Shared features: {len(shared_indices)}")
        print(f"Reactivation ratio: {reactivation_ratio:.4f} ({reactivation_ratio*100:.2f}%)")
        print(f"Jaccard similarity: {jaccard:.4f} ({jaccard*100:.2f}%)")
        
        # Apply the final result
        result = act.clone()
        result[0, last_token_pos] = sae.decode(top_acts2, top_indices2).squeeze(0)
    
    print(f"Output shape: {result.shape}")
    return result

# Constant hook (iterative until convergence)
def sae_hook_constant(
    act: torch.Tensor, 
    hook: Dict
) -> torch.Tensor:
    print("Processing with constant SAE hook (converging features)...")
    print(f"Input shape: {act.shape}")
    
    MAX_ITERATIONS = 1000  # Maximum number of iterations to prevent infinite loops
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
        
        # Store iteration data for analysis
        iteration_indices_sets = []
        iteration_acts = []
        
        print(f"\nLast token position: Starting convergence loop")
        
        while not converged and iteration < MAX_ITERATIONS:
            # Encode and decode
            encoded = sae.encode(current_act)
            top_acts = encoded.top_acts
            top_indices = encoded.top_indices
            
            # Store for later analysis
            iteration_acts.append(top_acts.clone())
            
            # Convert to set for comparison
            current_indices_set = set(top_indices.cpu().numpy().flatten().tolist())
            iteration_indices_sets.append(current_indices_set)
            
            # Check for convergence
            if prev_indices_set is not None:
                # Calculate Jaccard similarity
                shared_indices = current_indices_set.intersection(prev_indices_set)
                union_size = len(current_indices_set.union(prev_indices_set))
                jaccard = len(shared_indices) / union_size if union_size > 0 else 1.0
                
                # Check if converged based on threshold
                if jaccard >= CONVERGENCE_THRESHOLD:
                    converged = True
                    print(f"Converged after {iteration+1} iterations")
                else:
                    print(f"Iteration {iteration+1}, Jaccard similarity: {jaccard:.4f}")
                    print(f"  Active features: {len(current_indices_set)}")
            else:
                print(f"Iteration 1, Active features: {len(current_indices_set)}")
            
            # Decode for next iteration
            current_act = sae.decode(top_acts, top_indices)
            
            # Update for next iteration
            prev_indices_set = current_indices_set
            iteration += 1
        
        # Calculate feature stability metrics
        if len(iteration_indices_sets) >= 2:
            # Compare first and final sets
            first_set = iteration_indices_sets[0]
            final_set = iteration_indices_sets[-1]
            shared = first_set.intersection(final_set)
            union = first_set.union(final_set)
            final_jaccard = len(shared) / len(union) if len(union) > 0 else 1.0
            
            # Analyze feature stability across iterations
            print(f"First→Final Jaccard similarity: {final_jaccard:.4f}")
            print(f"First iteration: {len(first_set)} features, Final iteration: {len(final_set)} features")
            print(f"Shared features between first and final: {len(shared)}")
        
        # Create the result tensor (leave all other positions unchanged)
        result = act.clone()
        result[0, last_token_pos] = current_act.squeeze(0)
    
    return result
 
# The hook point
hook_name = get_act_name("resid_post", 24)  # Layer 25 (1-indexed)

def run_model(prompt, hook_type='none'):
    """Run the model with specified hook type and return results."""
    input_ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"].to('cuda')
    
    if hook_type == 'none':
        # Run without hooks
        output = model(input_ids)
    else:
        # Select hook function based on type
        if hook_type == 'double':
            hook_fn = sae_hook_double
        elif hook_type == 'constant':
            hook_fn = sae_hook_constant
        else:
            hook_fn = sae_hook_single
            
        fwd_hooks = [(hook_name, hook_fn)]
        
        # Run with the selected hook
        output = model.run_with_hooks(
            input_ids,
            fwd_hooks=fwd_hooks,
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
    p_log = torch.log_softmax(p, dim=0)
    q_log = torch.log_softmax(q, dim=0)
    return torch.sum(torch.exp(p_log) * (p_log - q_log))

def compare_results(results):
    """Compare and print results from different hook types."""
    print(f"\n{'='*60}")
    print(f"PROMPT: {PROMPT}")
    print(f"{'='*60}")
    
    # Print top predictions for each method
    for hook_type, result in results.items():
        print(f"\n{hook_type.upper()} HOOK:")
        print(f"Next token: '{result['next_token']['text']}'")
        print("Top 5 tokens:")
        for i in range(5):
            print(f"  {i+1}. '{result['top5']['tokens'][i]}' (prob: {result['top5']['probs'][i].item():.4f})")
    
    # Compare top predictions across all methods
    hook_types = list(results.keys())
    for i in range(len(hook_types)):
        for j in range(i + 1, len(hook_types)):
            type1, type2 = hook_types[i], hook_types[j]
            if results[type1]['next_token']['id'] != results[type2]['next_token']['id']:
                print(f"\n⚠️ {type1.capitalize()} hook and {type2.capitalize()} hook predict different tokens!")
    
    # Calculate KL divergences between all distributions
    print("\nDISTRIBUTION DIFFERENCES (KL DIVERGENCE):")
    for i in range(len(hook_types)):
        for j in range(i + 1, len(hook_types)):
            type1, type2 = hook_types[i], hook_types[j]
            kl_div = calculate_kl_divergence(
                results[type1]['logits'][0, -1],
                results[type2]['logits'][0, -1]
            ).item()
            print(f"{type1.capitalize()} vs {type2.capitalize()} hook: {kl_div:.6f}")

def main():
    """Main function to run the comparisons."""
    # Always run all hook types and compare
    print("Running model with all hook types...")
    results = {
        'none': run_model(PROMPT, 'none'),
        'single': run_model(PROMPT, 'single'),
        'double': run_model(PROMPT, 'double'),
        'constant': run_model(PROMPT, 'constant')
    }
    compare_results(results)

if __name__ == "__main__":
    main()