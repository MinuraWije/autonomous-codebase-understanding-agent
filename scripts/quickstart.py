"""Quick start script to test the agent on a sample repository."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from indexing.pipeline import index_repository
from agent.graph import create_agent_graph


def quickstart_demo():
    """Run a quick demo of the agent."""
    print("=" * 60)
    print("AUTONOMOUS CODEBASE UNDERSTANDING AGENT - QUICKSTART")
    print("=" * 60)
    
    # Example: Index a small public repo
    print("\n1. Indexing a sample repository...")
    print("   This may take a few minutes...\n")
    
    # You can use a small public repo for testing
    # Example: https://github.com/pallets/click (small, well-structured)
    repo_url = input("Enter a GitHub repo URL (or press Enter for default): ").strip()
    
    if not repo_url:
        repo_url = "https://github.com/pallets/click"
        print(f"   Using default: {repo_url}")
    
    try:
        repo_metadata = index_repository(github_url=repo_url)
        print(f"\n✓ Repository indexed successfully!")
        print(f"   Repo ID: {repo_metadata.repo_id}")
        print(f"   Total files: {repo_metadata.stats['total_files']}")
        print(f"   Languages: {repo_metadata.stats['by_language']}")
        
        # Ask a sample question
        print("\n2. Asking a question...\n")
        
        question = input("Enter a question (or press Enter for default): ").strip()
        if not question:
            question = "Where is command line parsing implemented?"
            print(f"   Using default: {question}")
        
        print("\n   Processing... (this may take 30-60 seconds)")
        
        agent = create_agent_graph()
        result = agent.invoke({
            'question': question,
            'repo_id': repo_metadata.repo_id,
            'retrieval_iteration': 0,
            'reasoning_trace': []
        })
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(result.get('final_answer', 'No answer generated'))
        
        print("\n" + "=" * 60)
        print("REASONING TRACE:")
        print("=" * 60)
        for step in result.get('reasoning_trace', []):
            print(f"  • {step}")
        
        print("\n✓ Demo complete!")
        print("\nNext steps:")
        print("  1. Start the API: uvicorn app.main:app --reload")
        print("  2. Start the UI: streamlit run ui/streamlit_app.py")
        print("  3. Visit http://localhost:8501 to use the full interface")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure you have set OPENAI_API_KEY in .env")
        print("  2. Check your internet connection")
        print("  3. Verify the repository URL is correct and public")


if __name__ == "__main__":
    quickstart_demo()
