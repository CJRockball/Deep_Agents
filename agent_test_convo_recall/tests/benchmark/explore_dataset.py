#!/usr/bin/env python3
"""
Explore LONGMEMEVAL dataset structure
Shows question type distribution and allows stratified sampling
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import random

def explore_dataset(json_file: str):
    """Analyze dataset composition"""
    
    print(f"Loading {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    print(f"Total tests: {len(data)}")
    
    # Extract question types
    question_types = [test.get('question_type', 'unknown') for test in data]
    type_counts = Counter(question_types)
    
    print(f"\n{'='*60}")
    print(f"QUESTION TYPES")
    print(f"{'='*60}")
    
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        percentage = (count / len(data)) * 100
        print(f"{qtype:30s}: {count:4d} ({percentage:5.1f}%)")
    
    # Show sample questions from each type
    print(f"\n{'='*60}")
    print(f"SAMPLE QUESTIONS BY TYPE")
    print(f"{'='*60}")
    
    for qtype in sorted(type_counts.keys()):
        samples = [t for t in data if t.get('question_type') == qtype][:2]
        print(f"\n{qtype}:")
        for i, sample in enumerate(samples, 1):
            # Safe string conversion
            q = str(sample.get('question', 'N/A'))
            a = str(sample.get('answer', 'N/A'))
            
            # Truncate safely
            q_display = q[:60] + "..." if len(q) > 60 else q
            a_display = a[:60] + "..." if len(a) > 60 else a
            
            print(f"  {i}. Q: {q_display}")
            print(f"     A: {a_display}")
    
    return data, type_counts

def create_stratified_sample(
    data: list,
    type_counts: Counter,
    n_samples: int = 50,
    output_file: str = 'data/longmemeval_stratified_50.json'
):
    """Create stratified sample with proportional representation"""
    
    print(f"\n{'='*60}")
    print(f"CREATING STRATIFIED SAMPLE")
    print(f"{'='*60}")
    
    total = sum(type_counts.values())
    samples_per_type = {}
    
    for qtype, count in type_counts.items():
        proportion = count / total
        n_type = max(1, int(n_samples * proportion))
        samples_per_type[qtype] = n_type
    
    print(f"\nTarget samples: {n_samples}")
    print(f"Samples per type:")
    for qtype, n in sorted(samples_per_type.items()):
        print(f"  {qtype:30s}: {n:4d}")
    
    stratified_data = []
    
    for qtype, n_needed in samples_per_type.items():
        type_tests = [t for t in data if t.get('question_type') == qtype]
        
        if len(type_tests) < n_needed:
            print(f"  Warning: Only {len(type_tests)} available for {qtype}")
            sampled = type_tests
        else:
            sampled = random.sample(type_tests, n_needed)
        
        stratified_data.extend(sampled)
    
    random.shuffle(stratified_data)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stratified_data, f, indent=2)
    
    print(f"\n✓ Saved {len(stratified_data)} stratified samples to {output_file}")
    return stratified_data

def create_balanced_sample(
    data: list,
    type_counts: Counter,
    n_per_type: int = 10,
    output_file: str = 'data/longmemeval_balanced_50.json'
):
    """Create balanced sample with equal representation"""
    
    print(f"\n{'='*60}")
    print(f"CREATING BALANCED SAMPLE")
    print(f"{'='*60}")
    
    print(f"\nTarget: {n_per_type} samples per type")
    
    balanced_data = []
    
    for qtype in sorted(type_counts.keys()):
        type_tests = [t for t in data if t.get('question_type') == qtype]
        
        if len(type_tests) < n_per_type:
            print(f"  Warning: Only {len(type_tests)} for {qtype}")
            sampled = type_tests
        else:
            sampled = random.sample(type_tests, n_per_type)
        
        balanced_data.extend(sampled)
        print(f"  {qtype:30s}: {len(sampled):4d} samples")
    
    random.shuffle(balanced_data)
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(balanced_data, f, indent=2)
    
    print(f"\n✓ Saved {len(balanced_data)} balanced samples to {output_file}")
    return balanced_data

def main():
    parser = argparse.ArgumentParser(
        description='Explore LONGMEMEVAL dataset and create samples'
    )
    parser.add_argument(
        '--json-file',
        default='data/longmemeval_s_cleaned.json',
        help='Path to LONGMEMEVAL JSON file'
    )
    parser.add_argument(
        '--create-stratified',
        type=int,
        default=None,
        help='Create stratified sample with N tests'
    )
    parser.add_argument(
        '--create-balanced',
        type=int,
        default=None,
        help='Create balanced sample with N tests per type'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Explore dataset
    data, type_counts = explore_dataset(args.json_file)
    
    # Create samples if requested
    if args.create_stratified:
        create_stratified_sample(
            data, type_counts,
            n_samples=args.create_stratified,
            output_file=f'data/longmemeval_stratified_{args.create_stratified}.json'
        )
    
    if args.create_balanced:
        create_balanced_sample(
            data, type_counts,
            n_per_type=args.create_balanced,
            output_file=f'data/longmemeval_balanced_{args.create_balanced * len(type_counts)}.json'
        )

if __name__ == '__main__':
    main()
