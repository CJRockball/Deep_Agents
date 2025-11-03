# analyze_results.py

import pandas as pd
import matplotlib.pyplot as plt

class BenchmarkAnalyzer:
    """
    Analyze benchmark results by question type, memory span, etc.
    """
    def __init__(self, results_file):
        # Load results
        self.df = pd.read_json(results_file, lines=True)
        
    def aggregate_by_question_type(self):
        """
        Compute metrics by question type
        """
        grouped = self.df.groupby('question_type').agg({
            'evaluation.score': ['mean', 'std'],
            'evaluation.correctness': 'mean',
            'latency_ms': ['mean', 'median', lambda x: x.quantile(0.95)],
            'retrieval_metrics.recall@k': 'mean',
            'retrieval_metrics.precision@k': 'mean'
        })
        
        print("\n=== Performance by Question Type ===")
        print(grouped)
        
        return grouped
        
    def identify_failure_modes(self, threshold=50):
        """
        Find questions where agent scored below threshold
        """
        failures = self.df[self.df['evaluation.score'] < threshold]
        
        print(f"\n=== Failure Analysis ({len(failures)} failures) ===")
        print(failures[['question_id', 'question_type', 'evaluation.score', 
                       'evaluation.explanation']].to_string())
        
        return failures
        
    def plot_performance_breakdown(self):
        """
        Visualize performance across dimensions
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Score by question type
        self.df.groupby('question_type')['evaluation.score'].mean().plot(
            kind='bar', ax=axes[0, 0], title='Avg Score by Question Type'
        )
        
        # Plot 2: Latency distribution
        self.df['latency_ms'].hist(bins=50, ax=axes[0, 1])
        axes[0, 1].set_title('Response Latency Distribution')
        
        # Plot 3: Recall vs Precision
        axes[1, 0].scatter(
            self.df['retrieval_metrics.recall@k'],
            self.df['retrieval_metrics.precision@k'],
            alpha=0.5
        )
        axes[1, 0].set_xlabel('Recall@K')
        axes[1, 0].set_ylabel('Precision@K')
        axes[1, 0].set_title('Retrieval Quality')
        
        # Plot 4: Correctness components
        components = ['correctness', 'completeness', 'clarity']
        means = [self.df[f'evaluation.{c}'].mean() for c in components]
        axes[1, 1].bar(components, means)
        axes[1, 1].set_title('Answer Quality Components')
        
        plt.tight_layout()
        plt.savefig('benchmark_analysis.png', dpi=300)
        print("✓ Plots saved to benchmark_analysis.png")

# Run analysis
analyzer = BenchmarkAnalyzer('results/longmemeval_results.jsonl')
analyzer.aggregate_by_question_type()
analyzer.identify_failure_modes()
analyzer.plot_performance_breakdown()
