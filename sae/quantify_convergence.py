import os
import re
import csv
import numpy as np
from collections import defaultdict

# Define the three categories
categories = ['8b_choice_result', '8b_number_result', '8b_token_result']
results = {}
distributions = {cat: defaultdict(int) for cat in categories}
all_iterations = {cat: [] for cat in categories}  # To store all iteration values for statistical calculations

# Process each category
for category in categories:
    path = os.path.join('/home/lse/recursive_explainations', category)
    total_files = 0
    files_under_99 = 0
    
    # Process each prompt file
    for filename in os.listdir(path):
        if not filename.startswith('prompt_') or not filename.endswith('_gpu0.txt'):
            continue
        
        total_files += 1
        file_path = os.path.join(path, filename)
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Find the iterations value using regex
            match = re.search(r'Convergence metrics:\s*\nIterations:\s*(\d+)', content)
            if match:
                iterations = int(match.group(1))
                distributions[category][iterations] += 1
                all_iterations[category].append(iterations)  # Add to list for statistical calculations
                
                if iterations < 99:
                    files_under_99 += 1
        except Exception as e:
            print(f'Error processing {file_path}: {e}')
    
    # Calculate percentage
    percentage = (files_under_99 / total_files) * 100 if total_files > 0 else 0
    results[category] = {
        'total_files': total_files,
        'files_under_99': files_under_99,
        'percentage': percentage
    }

# Calculate statistics
statistics = {}
for category in categories:
    if all_iterations[category]:
        mean = np.mean(all_iterations[category])
        std_dev = np.std(all_iterations[category])
        statistics[category] = {
            'mean': mean,
            'std_dev': std_dev
        }
    else:
        statistics[category] = {
            'mean': 0,
            'std_dev': 0
        }

# Print summary
print('SUMMARY OF ITERATIONS < 99:')
for category, data in results.items():
    print(f'{category}: {data["files_under_99"]} out of {data["total_files"]} files ({data["percentage"]:.2f}%)')

# Print statistics
print('\nSTATISTICS:')
for category, data in statistics.items():
    print(f'{category}: Mean = {data["mean"]:.2f}, Standard Deviation = {data["std_dev"]:.2f}')

# Write distributions to CSV
with open('iteration_distributions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    
    # Find all possible iteration values across all categories
    all_iterations_set = set()
    for cat_dist in distributions.values():
        all_iterations_set.update(cat_dist.keys())
    all_iterations_set = sorted(all_iterations_set)
    
    # Write header
    writer.writerow(['Iterations'] + categories)
    
    # Write data rows
    for iteration in all_iterations_set:
        writer.writerow([
            iteration,
            distributions['8b_choice_result'][iteration],
            distributions['8b_number_result'][iteration],
            distributions['8b_token_result'][iteration]
        ])

print('\nDistribution saved to iteration_distributions.csv')
