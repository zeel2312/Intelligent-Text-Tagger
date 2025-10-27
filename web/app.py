"""
Intelligent Text Tagger - Web Interface
Main Streamlit application
"""

import streamlit as st
import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import session management
from web.utils.session_manager import init_session_state, load_state_from_files, get_state_summary, reload_from_files_if_newer

# Import our existing modules
from src.generate_tags import generate_tags, load_documents
from src.collect_feedback import simulate_feedback, load_tags, load_documents as load_docs_for_feedback
from src.learn_from_feedback import load_feedback, compute_tag_stats, derive_tag_weights, save_weights


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="Intelligent Text Tagger",
        page_icon="🏷️",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    # Auto-load state from files on first run
    if 'state_loaded' not in st.session_state:
        load_state_from_files()
        st.session_state['state_loaded'] = True
    else:
        # Reload if files have been updated externally
        reload_from_files_if_newer()
    
    st.title("🏷️ Intelligent Text Tagger")
    st.markdown("Generate and learn from document tags using AI")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Upload Documents", "Generate Tags", "View Feedback", "Learn from Feedback", "Run Pipeline"]
    )
    
    # Route to different pages
    if page == "Home":
        show_home_page()
    elif page == "Upload Documents":
        show_upload_page()
    elif page == "Generate Tags":
        show_generate_tags_page()
    elif page == "View Feedback":
        show_feedback_page()
    elif page == "Learn from Feedback":
        show_learning_page()
    elif page == "Run Pipeline":
        show_pipeline_page()


def show_home_page():
    """Home page with overview"""
    st.header("Welcome to Intelligent Text Tagger")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 What it does")
        st.markdown("""
        - **Generates tags** for your documents using TF-IDF
        - **Learns from feedback** to improve tag quality
        - **Adapts over time** to your specific document types
        """)
    
    with col2:
        st.subheader("🚀 How to use")
        st.markdown("""
        1. **Upload Documents** - Add your text files
        2. **Generate Tags** - Create tags for your documents
        3. **View Feedback** - See which tags were approved/rejected
        4. **Learn from Feedback** - Process feedback and learn optimal weights
        5. **Run Pipeline** - Execute the complete learning cycle
        """)
    
    st.subheader("📊 System Status")
    if os.path.exists("outputs"):
        st.success("✅ Output directory exists")
    else:
        st.warning("⚠️ Output directory not found")
   
        
def show_upload_page():
    """Upload documents page"""
    st.header("📁 Upload Documents")
    
    from web.components.file_uploader import file_uploader, show_document_list
    
    uploaded_files = file_uploader()
    
    st.divider()
    
    show_document_list()
 
def show_generate_tags_page():
    """Generate tags page"""
    
    from web.components.tag_display import generate_tags_interface, display_tags
    
    generate_tags_interface()
       
def show_feedback_page():
    """Feedback page"""
    
    from web.components.feedback_display import collect_feedback_interface, display_feedback

    collect_feedback_interface()
           
def show_learning_page():
    """Learning from feedback page"""
    
    from web.components.learning_display import learn_from_feedback_interface
    
    learn_from_feedback_interface()

def show_pipeline_page():
    """Pipeline page"""
    
    from web.components.pipeline_runner import run_pipeline_interface

    run_pipeline_interface()

if __name__ == "__main__":
    main()