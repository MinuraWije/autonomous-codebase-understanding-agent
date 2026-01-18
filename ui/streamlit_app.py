"""Streamlit UI for the Codebase Understanding Agent."""
import streamlit as st
import requests
from typing import List, Dict
import time


# API Configuration
API_URL = "http://localhost:8000/api/v1"


def get_repos() -> List[Dict]:
    """Get list of indexed repositories."""
    try:
        response = requests.get(f"{API_URL}/repos")
        if response.status_code == 200:
            return response.json()
        return []
    except Exception:
        return []


def index_repository(github_url: str = None, local_path: str = None, branch: str = "main"):
    """Start indexing a repository."""
    try:
        payload = {"branch": branch}
        if github_url:
            payload["github_url"] = github_url
        elif local_path:
            payload["local_path"] = local_path
        else:
            return None
        
        response = requests.post(f"{API_URL}/repos/index", json=payload)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def get_repo_status(repo_id: str) -> Dict:
    """Get repository status."""
    try:
        response = requests.get(f"{API_URL}/repos/{repo_id}/status")
        if response.status_code == 200:
            return response.json()
        return {"status": "error"}
    except Exception:
        return {"status": "error"}


def ask_question(repo_id: str, question: str, use_verification: bool = True) -> Dict:
    """Ask a question about the repository."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "repo_id": repo_id,
                "question": question,
                "use_verification": use_verification
            }
        )
        if response.status_code == 200:
            return response.json()
        return {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def get_architecture_summary(repo_id: str) -> Dict:
    """Get architecture summary for repository."""
    try:
        response = requests.get(f"{API_URL}/repos/{repo_id}/summary")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


# Page config
st.set_page_config(
    page_title="Codebase Understanding Agent",
    page_icon="🔍",
    layout="wide"
)

# Title
st.title("🔍 Autonomous Codebase Understanding Agent")
st.markdown("*Ask questions about your codebase and get answers with citations*")

# Sidebar
with st.sidebar:
    st.header("📚 Repository Management")
    
    # Index new repository
    with st.expander("Index New Repository", expanded=False):
        repo_type = st.radio("Repository Type", ["GitHub URL", "Local Path"])
        
        if repo_type == "GitHub URL":
            github_url = st.text_input("GitHub URL", placeholder="https://github.com/user/repo")
            branch = st.text_input("Branch", value="main")
            
            if st.button("🚀 Start Indexing"):
                if github_url:
                    with st.spinner("Starting indexing..."):
                        result = index_repository(github_url=github_url, branch=branch)
                        if result:
                            st.success(f"✅ {result['message']}")
                            st.info(f"Repository ID: `{result['repo_id']}`")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Failed to start indexing")
                else:
                    st.warning("Please enter a GitHub URL")
        else:
            local_path = st.text_input("Local Path", placeholder="/path/to/repo")
            
            if st.button("🚀 Start Indexing"):
                if local_path:
                    with st.spinner("Starting indexing..."):
                        result = index_repository(local_path=local_path)
                        if result:
                            st.success(f"✅ {result['message']}")
                            st.info(f"Repository ID: `{result['repo_id']}`")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error("Failed to start indexing")
                else:
                    st.warning("Please enter a local path")
    
    st.divider()
    
    # List indexed repositories
    st.subheader("Indexed Repositories")
    repos = get_repos()
    
    if repos:
        for repo in repos:
            with st.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    repo_name = repo.get('url', repo.get('local_path', 'Unknown'))
                    if len(repo_name) > 40:
                        repo_name = "..." + repo_name[-37:]
                    st.markdown(f"**{repo_name}**")
                    st.caption(f"ID: `{repo['repo_id']}`")
                    st.caption(f"Files: {repo['stats'].get('total_files', 0)}")
                
                with col2:
                    status = get_repo_status(repo['repo_id'])
                    if status['status'] == 'completed':
                        st.success("✓")
                    else:
                        st.warning("...")
                
                st.divider()
    else:
        st.info("No repositories indexed yet")

# Main content
if repos:
    # Select repository
    repo_options = {
        f"{r.get('url', r.get('local_path', 'Unknown'))} ({r['repo_id'][:8]})": r['repo_id']
        for r in repos
    }
    
    selected_repo_label = st.selectbox(
        "Select Repository",
        options=list(repo_options.keys())
    )
    selected_repo_id = repo_options[selected_repo_label]
    
    # Tabs for non-chat features
    tab1, tab2 = st.tabs(["📋 Summary", "⚙️ Settings"])
    
    # Chat interface (outside tabs)
    st.subheader("💬 Chat")
    
    # Example questions
    with st.expander("📝 Example Questions"):
        st.markdown("""
        - Where is authentication handled?
        - How does the request lifecycle work?
        - Where is input validation implemented?
        - What database is used and how is it configured?
        - How are errors handled in the API?
        """)
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "citations" in message:
                with st.expander("📎 View Citations"):
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f"**{i}. `{citation['file_path']}` "
                                  f"(lines {citation['start_line']}-{citation['end_line']})**")
                        if citation.get('text_snippet'):
                            st.code(citation['text_snippet'], language='python')
            
            if message["role"] == "assistant" and "reasoning" in message:
                with st.expander("🔍 View Reasoning Trace"):
                    for step in message["reasoning"]:
                        st.text(f"• {step}")
    
    # Chat input (must be outside tabs)
    question = st.chat_input("Ask a question about the codebase...")
    
    if question:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                use_verification = st.session_state.get('use_verification', True)
                result = ask_question(selected_repo_id, question, use_verification)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    answer = result.get('answer', 'No answer generated')
                    st.markdown(answer)
                    
                    # Store message with metadata
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "citations": result.get('citations', []),
                        "reasoning": result.get('reasoning_trace', [])
                    })
                    
                    # Show citations
                    citations = result.get('citations', [])
                    if citations:
                        with st.expander("📎 View Citations"):
                            for i, citation in enumerate(citations, 1):
                                st.markdown(f"**{i}. `{citation['file_path']}` "
                                          f"(lines {citation['start_line']}-{citation['end_line']})**")
                                if citation.get('text_snippet'):
                                    st.code(citation['text_snippet'], language='python')
                    
                    # Show reasoning trace
                    reasoning = result.get('reasoning_trace', [])
                    if reasoning:
                        with st.expander("🔍 View Reasoning Trace"):
                            for step in reasoning:
                                st.text(f"• {step}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Summary Tab
    with tab1:
        st.subheader("Architecture Summary")
        
        if st.button("🔄 Generate Summary"):
            with st.spinner("Generating architecture summary..."):
                summary_data = get_architecture_summary(selected_repo_id)
                
                if summary_data:
                    st.markdown("### Overview")
                    st.markdown(summary_data['summary'])
                    
                    st.markdown("### Key Files")
                    for file in summary_data.get('key_files', []):
                        st.markdown(f"- `{file}`")
                    
                    st.markdown("### Directory Structure")
                    
                    def display_tree(tree, indent=0):
                        for key, value in sorted(tree.items()):
                            if value is None:
                                st.text("  " * indent + f"📄 {key}")
                            else:
                                st.text("  " * indent + f"📁 {key}/")
                                if isinstance(value, dict):
                                    display_tree(value, indent + 1)
                    
                    display_tree(summary_data.get('file_structure', {}))
                else:
                    st.error("Failed to generate summary")
    
    # Settings Tab
    with tab2:
        st.subheader("Agent Settings")
        
        use_verification = st.checkbox(
            "Use Verification Loop",
            value=st.session_state.get('use_verification', True),
            help="Enable the verifier to check if answers are grounded in code"
        )
        st.session_state['use_verification'] = use_verification
        
        st.info("""
        **Verification Loop**: When enabled, the agent will verify that answers are 
        grounded in the retrieved code and retrieve additional context if needed. 
        This reduces hallucinations but may be slower.
        """)

else:
    # No repositories indexed yet
    st.info("👈 Start by indexing a repository from the sidebar")
    
    st.markdown("""
    ## How it works
    
    1. **Index a Repository**: Enter a GitHub URL or local path to index
    2. **Ask Questions**: Once indexed, ask questions about the codebase
    3. **Get Cited Answers**: Receive answers with specific file references
    
    ## Features
    
    - 🔍 **Hybrid Search**: Combines semantic and keyword search
    - 🤖 **Agentic Workflow**: Plans, retrieves, synthesizes, and verifies
    - 📝 **Citation-Backed**: Every answer includes file references
    - 🔄 **Verification Loop**: Ensures answers are grounded in actual code
    """)
