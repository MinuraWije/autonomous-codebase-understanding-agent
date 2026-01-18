"""Evaluation runner for the agent."""
import json
from pathlib import Path
from typing import List, Dict
import statistics
from agent.graph import create_agent_graph
from eval.metrics import calculate_metrics


def load_test_questions(dataset_path: Path = None) -> List[Dict]:
    """Load test questions from JSON file."""
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "datasets" / "test_questions.json"
    
    with open(dataset_path, 'r') as f:
        return json.load(f)


def run_evaluation(repo_id: str, test_questions: List[Dict] = None) -> Dict:
    """
    Run evaluation on a repository.
    
    Args:
        repo_id: Repository ID to evaluate
        test_questions: List of test questions (optional)
    
    Returns:
        Evaluation results
    """
    if test_questions is None:
        test_questions = load_test_questions()
    
    agent = create_agent_graph()
    results = []
    
    print(f"Running evaluation on {len(test_questions)} questions...")
    
    for i, test_q in enumerate(test_questions, 1):
        print(f"\n[{i}/{len(test_questions)}] {test_q['question']}")
        
        try:
            # Run agent
            result = agent.invoke({
                'question': test_q['question'],
                'repo_id': repo_id,
                'retrieval_iteration': 0,
                'reasoning_trace': []
            })
            
            # Calculate metrics
            metrics = calculate_metrics(result, test_q.get('expected_files'))
            
            results.append({
                'question': test_q['question'],
                'type': test_q.get('type', 'unknown'),
                'answer': result.get('final_answer', ''),
                'metrics': metrics,
                'success': True
            })
            
            print(f"  ✓ Groundedness: {metrics['groundedness']:.2%}")
            print(f"  ✓ Citations: {metrics['citation_count']}")
            print(f"  ✓ Hallucination: {metrics['hallucination_rate']:.2%}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'question': test_q['question'],
                'type': test_q.get('type', 'unknown'),
                'error': str(e),
                'success': False
            })
    
    # Aggregate metrics
    successful = [r for r in results if r['success']]
    
    if successful:
        aggregate = {
            'total_questions': len(test_questions),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'avg_groundedness': statistics.mean([r['metrics']['groundedness'] for r in successful]),
            'avg_hallucination_rate': statistics.mean([r['metrics']['hallucination_rate'] for r in successful]),
            'avg_citations': statistics.mean([r['metrics']['citation_count'] for r in successful]),
            'avg_chunks_retrieved': statistics.mean([r['metrics']['chunks_retrieved'] for r in successful]),
        }
        
        # Calculate retrieval hit rate if available
        with_hit_rate = [r for r in successful if 'retrieval_hit_rate' in r['metrics']]
        if with_hit_rate:
            aggregate['avg_retrieval_hit_rate'] = statistics.mean([
                r['metrics']['retrieval_hit_rate'] for r in with_hit_rate
            ])
    else:
        aggregate = {
            'total_questions': len(test_questions),
            'successful': 0,
            'failed': len(results)
        }
    
    return {
        'results': results,
        'aggregate': aggregate
    }


def print_summary_table(eval_results: Dict):
    """Print a summary table of evaluation results."""
    aggregate = eval_results['aggregate']
    
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal Questions: {aggregate['total_questions']}")
    print(f"Successful: {aggregate['successful']}")
    print(f"Failed: {aggregate['failed']}")
    
    if aggregate['successful'] > 0:
        print("\n" + "-"*60)
        print(f"{'Metric':<30} {'Score':>10}")
        print("-"*60)
        print(f"{'Groundedness':<30} {aggregate['avg_groundedness']:>9.1%}")
        print(f"{'Hallucination Rate':<30} {aggregate['avg_hallucination_rate']:>9.1%}")
        print(f"{'Avg Citations per Answer':<30} {aggregate['avg_citations']:>10.1f}")
        print(f"{'Avg Chunks Retrieved':<30} {aggregate['avg_chunks_retrieved']:>10.1f}")
        
        if 'avg_retrieval_hit_rate' in aggregate:
            print(f"{'Retrieval Hit Rate':<30} {aggregate['avg_retrieval_hit_rate']:>9.1%}")
        
        print("-"*60)


def save_results(eval_results: Dict, output_path: Path):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval_runner.py <repo_id>")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    
    # Run evaluation
    results = run_evaluation(repo_id)
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    save_results(results, output_dir / f"eval_{repo_id}.json")
