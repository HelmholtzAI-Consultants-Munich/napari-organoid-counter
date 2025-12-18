#!/usr/bin/env python3
"""
Timing Analysis Script for Organoid Counter Experiments

Analyzes JSON timing logs and computes mean ± std for different configurations.
Groups results by model name and (window_sizes, downsampling) pairs.

Usage:
    python analyze_timings.py <directory_path>
    python analyze_timings.py /path/to/timing_logs --output results.txt
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import numpy as np


def load_timing_files(directory: Path) -> List[Dict]:
    """
    Load all JSON timing files from the specified directory.
    
    Args:
        directory: Path to directory containing JSON timing logs
        
    Returns:
        List of dictionaries containing timing data
    """
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{directory}'.", file=sys.stderr)
        return []
    
    timing_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                timing_data.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse '{json_file.name}'. Skipping.", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error reading '{json_file.name}': {e}. Skipping.", file=sys.stderr)
    
    print(f"Loaded {len(timing_data)} timing files from {len(json_files)} JSON files.\n")
    return timing_data


def create_config_key(window_sizes: List[int], downsampling: List[int]) -> str:
    """
    Create a standardized configuration key from window sizes and downsampling values.
    
    Args:
        window_sizes: List of window sizes
        downsampling: List of downsampling values
        
    Returns:
        Formatted string like "[1024, 512]ws, [2, 8]ds"
    """
    ws_str = str(window_sizes).replace(" ", "")
    ds_str = str(downsampling).replace(" ", "")
    return f"{ws_str}ws, {ds_str}ds"


def group_timings(timing_data: List[Dict]) -> Dict[Tuple[str, str], List[float]]:
    """
    Group timing data by (model_name, config) and collect elapsed times.
    
    Args:
        timing_data: List of timing dictionaries
        
    Returns:
        Dictionary mapping (model_name, config_key) to list of elapsed times
    """
    grouped = defaultdict(list)
    
    for entry in timing_data:
        try:
            model_name = entry.get("model_name", "unknown")
            window_sizes = entry.get("window_sizes", [])
            downsampling = entry.get("downsampling", [])
            elapsed_time = entry.get("elapsed_time_seconds")
            
            if elapsed_time is None:
                print(f"Warning: Missing elapsed_time_seconds in entry. Skipping.", file=sys.stderr)
                continue
            
            config_key = create_config_key(window_sizes, downsampling)
            grouped[(model_name, config_key)].append(elapsed_time)
            
        except Exception as e:
            print(f"Warning: Error processing entry: {e}. Skipping.", file=sys.stderr)
    
    return grouped


def compute_statistics(times: List[float]) -> Tuple[float, float, int]:
    """
    Compute mean and standard deviation of timing data.
    
    Args:
        times: List of elapsed times in seconds
        
    Returns:
        Tuple of (mean, std, count)
    """
    times_array = np.array(times)
    mean = np.mean(times_array)
    std = np.std(times_array, ddof=1) if len(times) > 1 else 0.0
    return mean, std, len(times)


def format_output(grouped_data: Dict[Tuple[str, str], List[float]]) -> str:
    """
    Format the grouped timing data into a readable output.
    Groups by configuration first to allow easy model comparison.
    Within each configuration, models are sorted by speed (fastest first).
    
    Args:
        grouped_data: Dictionary mapping (model, config) to list of times
        
    Returns:
        Formatted string with statistics
    """
    if not grouped_data:
        return "No timing data to display."
    
    # Reorganize data by config first, then by model
    config_to_models = defaultdict(dict)
    for (model_name, config_key), times in grouped_data.items():
        mean, std, count = compute_statistics(times)
        config_to_models[config_key][model_name] = {
            'mean': mean,
            'std': std,
            'count': count
        }
    
    # Sort configurations (you can customize this order if needed)
    sorted_configs = sorted(config_to_models.keys())
    
    lines = []
    lines.append("=" * 80)
    lines.append("TIMING ANALYSIS RESULTS")
    lines.append("=" * 80)
    lines.append("")
    
    for config_key in sorted_configs:
        models_dict = config_to_models[config_key]
        
        # Sort models by mean time (fastest first)
        sorted_models = sorted(models_dict.items(), key=lambda x: x[1]['mean'])
        
        # Get the fastest time for percentage calculation
        fastest_time = sorted_models[0][1]['mean'] if sorted_models else 0
        
        for idx, (model_name, stats) in enumerate(sorted_models):
            mean = stats['mean']
            std = stats['std']
            count = stats['count']
            
            # Calculate percentage increase from fastest
            if idx == 0:
                # First (fastest) model - baseline
                percent_str = "(baseline)"
            else:
                percent_increase = ((mean - fastest_time) / fastest_time) * 100
                percent_str = f"(+{percent_increase:.1f}%)"
            
            lines.append(f"{model_name} with {config_key}: "
                        f"{mean:.2f} ± {std:.2f} sec (n={count}) {percent_str}")
        
        # Add empty line between configuration blocks
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def main():
    """Main entry point for the timing analysis script."""
    parser = argparse.ArgumentParser(
        description="Analyze timing logs from organoid counter experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/timing_logs
  %(prog)s ./timing_logs --output results.txt
        """
    )
    
    parser.add_argument(
        "directory",
        type=Path,
        help="Path to directory containing JSON timing log files"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Optional output file path (default: print to stdout)"
    )
    
    args = parser.parse_args()
    
    # Load timing files
    timing_data = load_timing_files(args.directory)
    
    if not timing_data:
        print("No valid timing data found. Exiting.")
        sys.exit(1)
    
    # Group by model and configuration
    grouped_data = group_timings(timing_data)
    
    # Format output
    output_text = format_output(grouped_data)
    
    # Write or print output
    if args.output:
        try:
            with open(args.output, 'w') as f:
                f.write(output_text)
            print(f"Results saved to: {args.output}")
        except Exception as e:
            print(f"Error writing to '{args.output}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_text)


if __name__ == "__main__":
    main()