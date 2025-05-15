#!/bin/bash

# Directory for results
RESULTS_DIR="sae_results"
mkdir -p $RESULTS_DIR

# Check if a prompts file is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <prompts-file>"
    echo "  <prompts-file>: Path to file containing prompts (one per line)"
    exit 1
fi

PROMPTS_FILE=$1

# Count total lines in the prompts file
TOTAL_LINES=$(wc -l < $PROMPTS_FILE)
TOTAL_GPUS=8
echo "Total prompts: $TOTAL_LINES"

# Create temp directory for splitting files
TEMP_DIR="${RESULTS_DIR}/temp"
mkdir -p $TEMP_DIR

# Split the prompts file into chunks for each GPU
# Using prefix 'x' to ensure consistent naming across systems
split -n l/$TOTAL_GPUS $PROMPTS_FILE "$TEMP_DIR/x"

# List the created chunk files
echo "Created chunk files:"
ls -l $TEMP_DIR/

# Function to process a single prompt on a specific GPU
process_prompt() {
    local prompt="$1"
    local gpu_id="$2"
    local prompt_idx="$3"
    local output_file="${RESULTS_DIR}/prompt_${prompt_idx}_gpu${gpu_id}.txt"
    
    echo "GPU $gpu_id: Processing prompt: $prompt"
    
    # Use CUDA_VISIBLE_DEVICES to restrict to a specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id python recursive_sae.py --prompt "$prompt" > "$output_file" 2>&1
    
    echo "GPU $gpu_id: Finished prompt $prompt_idx"
}

# Launch processes for each GPU
gpu_id=0
for chunk_file in $TEMP_DIR/x*; do
    if [ ! -f "$chunk_file" ]; then
        echo "No chunk file for GPU $gpu_id, skipping"
        continue
    fi
    
    echo "Processing chunk for GPU $gpu_id: $chunk_file"
    
    # Process each prompt in the chunk file
    prompt_count=0
    while IFS= read -r prompt; do
        # Skip empty lines
        if [ -z "$prompt" ]; then
            continue
        fi
        
        prompt_count=$((prompt_count+1))
        global_idx=$((gpu_id*1000 + prompt_count))
        
        # Process this prompt on the GPU
        process_prompt "$prompt" $gpu_id $global_idx &
        
        # Wait a bit to avoid overwhelming the system
        sleep 0.5
    done < "$chunk_file"
    
    echo "Launched $prompt_count prompts on GPU $gpu_id"
    gpu_id=$((gpu_id+1))
    
    # Break if we've used all available GPUs
    if [ $gpu_id -ge $TOTAL_GPUS ]; then
        break
    fi
done

# Wait for all background processes to complete
echo "Waiting for all processes to complete..."
wait

echo "All prompts processed! Results saved to $RESULTS_DIR/"

# Clean up temporary files
rm -rf $TEMP_DIR

# Create a summary file
echo "Creating summary file..."
{
    echo "Processing completed at $(date)"
    echo "Total prompts processed: $TOTAL_LINES"
    echo "Results stored in: $RESULTS_DIR"
} > "${RESULTS_DIR}/summary.txt"

echo "Done!"