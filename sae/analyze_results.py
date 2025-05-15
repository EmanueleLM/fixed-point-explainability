import os
import glob
import csv

def analyze_metrics_files(results_dir):
    """Analyze all metrics CSV files in the given directory and generate summary statistics"""
    # Statistics tracking
    stats = {
        'base': {'correct': 0, 'total': 0, 'top5_correct': 0},
        'single': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'top5_correct': 0},
        'double': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'jaccard_sum': 0.0, 'top5_correct': 0},
        'constant': {'correct': 0, 'total': 0, 'kl_sum': 0.0, 'jaccard_sum': 0.0, 'top5_correct': 0}
    }
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(results_dir, "prompt_*_metrics.csv"))
    total_files = len(metrics_files)
    print(f"Found {total_files} metrics files")
    
    # Process each file
    for csv_file in metrics_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                version = row['version']
                # Update correctness counts
                if row['top_token_is_right'].lower() == 'true':
                    stats[version]['correct'] += 1
                if row['top_5_right'].lower() == 'true':
                    stats[version]['top5_correct'] += 1
                stats[version]['total'] += 1
                
                # Update KL divergence sums for non-base versions
                if version != 'base' and row['kl_div'] != 'NA':
                    try:
                        stats[version]['kl_sum'] += float(row['kl_div'])
                    except ValueError:
                        pass
                
                # Update Jaccard sums for double and constant
                if version in ['double', 'constant'] and row['jaccard'] != 'NA':
                    try:
                        stats[version]['jaccard_sum'] += float(row['jaccard'])
                    except ValueError:
                        pass
    
    # Calculate summary statistics
    summary = {
        'total_prompts': total_files,
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
    summary_file = os.path.join(results_dir, "summary_stats.txt")
    with open(summary_file, 'w') as f:
        f.write("SUMMARY STATISTICS\n")
        f.write("=================\n\n")
        f.write(f"Total prompts analyzed: {summary['total_prompts']}\n\n")
        
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
    
    print(f"Summary statistics saved to {summary_file}")
    return summary