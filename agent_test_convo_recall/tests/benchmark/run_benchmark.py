#!/usr/bin/env python3
"""
Run LONGMEMEVAL benchmark on IntelliFinQ agent

This script:
1. Loads test data from database
2. Asks your agent each test question
3. Compares agent answers to ground truth
4. Saves results for analysis

Usage:
    python run_benchmark.py --max-tests 10  # Run first 10 tests
    python run_benchmark.py                 # Run all 500 tests
"""

import json
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
import logging

# Add your agent source to Python path
AGENT_SRC = Path(__file__).parent.parent.parent / 'services' / 'agent-core' / 'src'
sys.path.insert(0, str(AGENT_SRC))

# Import your agent
import boot
from agents.react_agent import FinancialReActAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_single_test(
    agent: FinancialReActAgent,
    test_instance: dict,
    test_index: int,
    user_id_prefix: str = 'test_user'
) -> dict:
    """
    Run a single test: ask the agent a question and record response
    
    Args:
        agent: Your IntelliFinQ agent instance
        test_instance: Test data from LONGMEMEVAL
        test_index: Index of this test
        user_id_prefix: Prefix for test user IDs
        
    Returns:
        Dict with test results
    """
    
    # Get test details
    user_id = f"{user_id_prefix}_{test_index:04d}"
    question = test_instance.get('question')
    ground_truth = test_instance.get('answer')
    question_id = test_instance.get('question_id')
    question_type = test_instance.get('question_type')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Test {test_index}: {question_id}")
    logger.info(f"Type: {question_type}")
    logger.info(f"Question: {question}")
    logger.info(f"Expected Answer: {ground_truth}")
    logger.info(f"{'='*60}")
    
    # Ask the agent
    start_time = time.time()
    
    try:
        agent_response = agent.handle_query(
            user_id=user_id,
            query=question
        )
        latency_ms = (time.time() - start_time) * 1000
        error = None
        
    except Exception as e:
        agent_response = ""
        latency_ms = 0
        error = str(e)
        logger.error(f"Agent error: {e}")
    
    logger.info(f"\nAgent Answer: {agent_response}")
    logger.info(f"Latency: {latency_ms:.2f}ms")
    
    # Collect result
    result = {
        'test_index': test_index,
        'question_id': question_id,
        'question_type': question_type,
        'question': question,
        'ground_truth_answer': ground_truth,
        'agent_answer': agent_response,
        'latency_ms': latency_ms,
        'error': error,
        'timestamp': datetime.now().isoformat()
    }
    
    return result

def run_benchmark(
    json_file: str = 'data/longmemeval_s_cleaned.json',
    max_tests: int = None,
    user_id_prefix: str = 'test_user',
    output_file: str = 'results/benchmark_results.jsonl'
):
    """
    Run full benchmark on all tests
    
    Args:
        json_file: Path to LONGMEMEVAL JSON
        max_tests: Limit number of tests
        user_id_prefix: Prefix for test user IDs
        output_file: Where to save results
    """
    
    # Load test data
    logger.info(f"Loading test data from {json_file}")
    with open(json_file, 'r') as f:
        test_data = json.load(f)
    
    if max_tests:
        test_data = test_data[:max_tests]
        logger.info(f"Limited to first {max_tests} tests")
    
    logger.info(f"Total tests to run: {len(test_data)}")
    
    # Initialize agent
    logger.info("Initializing IntelliFinQ agent...")
    agent = FinancialReActAgent()
    
    # Create output directory
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Run all tests
    results = []
    
    for idx, test_instance in enumerate(test_data):
        try:
            result = run_single_test(
                agent=agent,
                test_instance=test_instance,
                test_index=idx,
                user_id_prefix=user_id_prefix
            )
            
            results.append(result)
            
            # Save incrementally
            with open(output_file, 'w') as f:
                for r in results:
                    f.write(json.dumps(r) + '\n')
            
            logger.info(f"✓ Completed {idx+1}/{len(test_data)}")
            
        except Exception as e:
            logger.error(f"Failed test {idx}: {e}")
            continue
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Benchmark complete!")
    logger.info(f"Tests run: {len(results)}/{len(test_data)}")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*60}\n")
    
    return results

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run LONGMEMEVAL benchmark on IntelliFinQ agent'
    )
    parser.add_argument(
        '--json-file',
        default='data/longmemeval_s_cleaned.json',
        help='Path to LONGMEMEVAL JSON file'
    )
    parser.add_argument(
        '--max-tests',
        type=int,
        default=None,
        help='Limit number of tests to run'
    )
    parser.add_argument(
        '--user-prefix',
        default='test_user',
        help='Prefix for test user IDs (must match load_test_data.py)'
    )
    parser.add_argument(
        '--output',
        default='results/benchmark_results.jsonl',
        help='Output file for results'
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    run_benchmark(
        json_file=args.json_file,
        max_tests=args.max_tests,
        user_id_prefix=args.user_prefix,
        output_file=args.output
    )

if __name__ == '__main__':
    main()
