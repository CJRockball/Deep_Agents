#!/usr/bin/env python3
"""
Analyze LONGMEMEVAL benchmark results

This script:
1. Loads benchmark results from JSONL file
2. Compares agent answers to ground truth
3. Uses LLM-as-judge to score answer quality
4. Generates comprehensive metrics and visualizations

Usage:
    python analyze_results.py
    python analyze_results.py --results results/benchmark_results.jsonl
    python analyze_results.py --use-llm-judge  # More accurate but uses API
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import re

# Add agent source to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
AGENT_SRC = PROJECT_ROOT / 'services' / 'agent-core' / 'src'
sys.path.insert(0, str(AGENT_SRC))

def load_results(results_file: str) -> pd.DataFrame:
    """Load benchmark results from JSONL file"""
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    
    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} test results from {results_file}")
    return df

def simple_answer_match(ground_truth: str, agent_answer: str) -> float:
    """
    Simple keyword-based matching
    Returns score 0.0 to 1.0
    """
    # Normalize
    gt_lower = ground_truth.lower().strip()
    ans_lower = agent_answer.lower().strip()
    
    # Exact match
    if gt_lower == ans_lower:
        return 1.0
    
    # Contains ground truth
    if gt_lower in ans_lower:
        return 0.8
    
    # Check for key words
    gt_words = set(re.findall(r'\w+', gt_lower))
    ans_words = set(re.findall(r'\w+', ans_lower))
    
    if len(gt_words) == 0:
        return 0.0
    
    # Word overlap
    overlap = len(gt_words & ans_words) / len(gt_words)
    
    # Penalize if agent says "I don't know" or similar
    refusal_phrases = [
        "don't have access",
        "cannot provide",
        "don't know",
        "no information",
        "unable to answer"
    ]
    
    for phrase in refusal_phrases:
        if phrase in ans_lower:
            return 0.0
    
    return overlap

def llm_judge_score(ground_truth: str, agent_answer: str, question: str) -> dict:
    """
    Use LLM as judge to score answer quality
    Returns dict with score and reasoning
    """
    try:
        import boot
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
        prompt = f"""You are evaluating a conversational AI agent's answer quality.

Question: {question}
Ground Truth Answer: {ground_truth}
Agent's Answer: {agent_answer}

Rate the agent's answer on a scale of 0-5:
- 5: Perfect match, semantically equivalent
- 4: Correct with minor differences
- 3: Partially correct
- 2: Contains some relevant info but mostly wrong
- 1: Completely wrong or refuses to answer
- 0: No answer or error

Respond in JSON format:
{{"score": <0-5>, "reasoning": "<brief explanation>"}}"""

        response = llm.invoke([{"role": "user", "content": prompt}])
        
        # Parse JSON from response
        content = response.content
        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        result = json.loads(content)
        return result
        
    except Exception as e:
        print(f"LLM judge error: {e}")
        return {"score": 0, "reasoning": f"Error: {e}"}

def calculate_metrics(df: pd.DataFrame, use_llm_judge: bool = False) -> dict:
    """Calculate evaluation metrics"""
    
    print("\nCalculating metrics...")
    
    # Simple keyword matching for all
    df['simple_score'] = df.apply(
        lambda row: simple_answer_match(
            row['ground_truth_answer'],
            row['agent_answer']
        ),
        axis=1
    )
    
    # LLM judge scoring (optional, slower)
    if use_llm_judge:
        print("Using LLM-as-judge (this may take a while)...")
        
        llm_results = []
        for idx, row in df.iterrows():
            print(f"  Judging {idx+1}/{len(df)}...", end='\r')
            result = llm_judge_score(
                row['ground_truth_answer'],
                row['agent_answer'],
                row['question']
            )
            llm_results.append(result)
        
        df['llm_score'] = [r['score'] / 5.0 for r in llm_results]  # Normalize to 0-1
        df['llm_reasoning'] = [r['reasoning'] for r in llm_results]
        print("\n✓ LLM judging complete")
    
    # Calculate metrics
    metrics = {
        'total_tests': len(df),
        'successful_runs': len(df[df['error'].isna()]),
        'failed_runs': len(df[df['error'].notna()]),
        'avg_latency_ms': df['latency_ms'].mean(),
        'median_latency_ms': df['latency_ms'].median(),
        'simple_score_mean': df['simple_score'].mean(),
        'simple_score_median': df['simple_score'].median(),
    }
    
    if use_llm_judge:
        metrics['llm_score_mean'] = df['llm_score'].mean()
        metrics['llm_score_median'] = df['llm_score'].median()
    
    # Score distribution
    df['score_bin'] = pd.cut(df['simple_score'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                              labels=['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'],
                              include_lowest=True)
    
    metrics['score_distribution'] = df['score_bin'].value_counts().to_dict()
    
    # By question type
    if 'question_type' in df.columns:
        metrics['by_type'] = {}
        for qtype in df['question_type'].unique():
            type_df = df[df['question_type'] == qtype]
            metrics['by_type'][qtype] = {
                'count': len(type_df),
                'avg_score': type_df['simple_score'].mean()
            }
    
    return metrics

def print_metrics_report(metrics: dict):
    """Print formatted metrics report"""
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Overall Performance:")
    print(f"  Total tests: {metrics['total_tests']}")
    print(f"  Successful: {metrics['successful_runs']}")
    print(f"  Failed: {metrics['failed_runs']}")
    
    print(f"\n⏱️  Latency:")
    print(f"  Average: {metrics['avg_latency_ms']:.2f}ms")
    print(f"  Median: {metrics['median_latency_ms']:.2f}ms")
    
    print(f"\n🎯 Answer Quality (Simple Matching):")
    print(f"  Mean score: {metrics['simple_score_mean']:.3f}")
    print(f"  Median score: {metrics['simple_score_median']:.3f}")
    
    if 'llm_score_mean' in metrics:
        print(f"\n🤖 Answer Quality (LLM Judge):")
        print(f"  Mean score: {metrics['llm_score_mean']:.3f}")
        print(f"  Median score: {metrics['llm_score_median']:.3f}")
    
    print(f"\n📈 Score Distribution:")
    for bin_label, count in sorted(metrics['score_distribution'].items()):
        print(f"  {bin_label}: {count} tests")
    
    if 'by_type' in metrics:
        print(f"\n🏷️  Performance by Question Type:")
        for qtype, stats in metrics['by_type'].items():
            print(f"  {qtype}: {stats['avg_score']:.3f} ({stats['count']} tests)")
    
    print("\n" + "="*60)

def export_detailed_results(df: pd.DataFrame, output_file: str):
    """Export detailed results to CSV"""
    
    # Select columns for export
    columns = [
        'test_index', 'question_id', 'question_type', 'question',
        'ground_truth_answer', 'agent_answer', 'simple_score',
        'latency_ms', 'error', 'timestamp'
    ]
    
    if 'llm_score' in df.columns:
        columns.extend(['llm_score', 'llm_reasoning'])
    
    export_df = df[columns]
    export_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results exported to: {output_file}")

def show_examples(df: pd.DataFrame, n_best: int = 3, n_worst: int = 3):
    """Show best and worst examples"""
    
    print("\n" + "="*60)
    print(f"TOP {n_best} BEST ANSWERS")
    print("="*60)
    
    best = df.nlargest(n_best, 'simple_score')
    for idx, row in best.iterrows():
        print(f"\n[Score: {row['simple_score']:.2f}] {row['question_id']}")
        print(f"Q: {row['question']}")
        print(f"Expected: {row['ground_truth_answer']}")
        print(f"Got: {row['agent_answer'][:200]}...")
    
    print("\n" + "="*60)
    print(f"TOP {n_worst} WORST ANSWERS")
    print("="*60)
    
    worst = df.nsmallest(n_worst, 'simple_score')
    for idx, row in worst.iterrows():
        print(f"\n[Score: {row['simple_score']:.2f}] {row['question_id']}")
        print(f"Q: {row['question']}")
        print(f"Expected: {row['ground_truth_answer']}")
        print(f"Got: {row['agent_answer'][:200]}...")
    
    print("\n" + "="*60)

def main():
    """Main analysis workflow"""
    
    parser = argparse.ArgumentParser(
        description='Analyze LONGMEMEVAL benchmark results'
    )
    parser.add_argument(
        '--results',
        default='results/benchmark_results.jsonl',
        help='Path to results JSONL file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Analyze only first N results (useful for quick checks)'
    )
    parser.add_argument(
        '--use-llm-judge',
        action='store_true',
        help='Use LLM-as-judge for more accurate scoring (slower, uses API)'
    )
    parser.add_argument(
        '--export-csv',
        default='results/detailed_results.csv',
        help='Path to export detailed CSV'
    )
    parser.add_argument(
        '--show-examples',
        type=int,
        default=3,
        help='Number of best/worst examples to show'
    )
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results).exists():
        print(f"Error: Results file not found: {args.results}")
        print("Run benchmark first: python run_benchmark.py --max-tests 5")
        return 1
    
    # Load results
    df = load_results(args.results)
    
    if len(df) == 0:
        print("No results found in file")
        return 1
    
    # Apply limit if specified
    if args.limit:
        original_count = len(df)
        df = df.head(args.limit)
        print(f"\n⚠️  Analyzing first {len(df)} of {original_count} results (--limit {args.limit})")
    
    # Calculate metrics
    metrics = calculate_metrics(df, use_llm_judge=args.use_llm_judge)
    
    # Rest of analysis...

    # Print report
    print_metrics_report(metrics)
    
    # Show examples
    if args.show_examples > 0:
        show_examples(df, n_best=args.show_examples, n_worst=args.show_examples)
    
    # Export detailed results
    export_detailed_results(df, args.export_csv)
    
    # Save metrics to JSON
    metrics_file = Path(args.results).parent / 'metrics_summary.json'
    with open(metrics_file, 'w') as f:
        # Convert non-serializable types
        metrics_clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                metrics_clean[k] = v
            elif isinstance(v, dict):
                metrics_clean[k] = {str(k2): float(v2) if isinstance(v2, (int, float)) else v2 
                                   for k2, v2 in v.items()}
            else:
                metrics_clean[k] = str(v)
        
        json.dump(metrics_clean, f, indent=2)
    
    print(f"\n✓ Metrics summary saved to: {metrics_file}")
    
    # Overall assessment
    print("\n" + "="*60)
    print("ASSESSMENT")
    print("="*60)
    
    avg_score = metrics['simple_score_mean']
    
    if avg_score >= 0.7:
        print("✓ GOOD: Your agent's memory system is working well!")
    elif avg_score >= 0.4:
        print("⚠ MODERATE: Memory works but needs improvement")
    else:
        print("✗ POOR: Memory system needs significant work")
    
    print(f"\nAverage score: {avg_score:.2%}")
    print("="*60)
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
