"""
Feedback display and visualization component
"""

import streamlit as st
import pandas as pd
import json
import os
import time

from src.collect_feedback import simulate_feedback, load_tags, load_documents as load_docs_for_feedback, save_feedback

def display_feedback(feedback_data):
    """Display feedback results"""
    if not feedback_data:
        st.warning("No feedback to display")
        return
    
    st.subheader("📊 Feedback Results")
    
    # Create detailed feedback table
    all_feedback = []
    for doc in feedback_data:
        filename = doc["filename"]
        for feedback in doc["feedback"]:
            all_feedback.append({
                "Document": filename,
                "Tag": feedback["tag"],
                "Status": feedback["status"],
                "Relevance Score": f"{feedback['relevance_score']:.4f}"
            })
    
    if all_feedback:
        df = pd.DataFrame(all_feedback)
        
        # Color code the status
        def color_status(val):
            if val == "approved":
                return "background-color: #d4edda"
            else:
                return "background-color: #f8d7da"
        
        styled_df = df.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_df, width="stretch")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_tags = len(all_feedback)
        approved_tags = len([f for f in all_feedback if f["Status"] == "approved"])
        approval_rate = (approved_tags / total_tags * 100) if total_tags > 0 else 0
        
        with col1:
            st.metric("Total Tags", total_tags)
        with col2:
            st.metric("Approved", approved_tags)
        with col3:
            st.metric("Rejected", total_tags - approved_tags)
        with col4:
            st.metric("Approval Rate", f"{approval_rate:.1f}%")
        
        # Approval rate chart
        st.subheader("📈 Approval Rate by Document")
        doc_stats = []
        for doc in feedback_data:
            total = len(doc["feedback"])
            approved = len([f for f in doc["feedback"] if f["status"] == "approved"])
            rate = (approved / total * 100) if total > 0 else 0
            doc_stats.append({
                "Document": doc["filename"],
                "Approval Rate": rate
            })
        
        if doc_stats:
            doc_df = pd.DataFrame(doc_stats)
            st.bar_chart(doc_df.set_index("Document"))

def collect_feedback_interface():
    """Interface for collecting feedback"""
    st.subheader("🔄 Collect Feedback")
    
    # Check if tags exist
    if not os.path.exists("outputs/tags.json"):
        st.error("❌ No tags found. Please generate tags first.")
        return
    
    # Parameters
    documents_folder = st.text_input("Documents folder", "documents")
    
    # Collect feedback button
    if st.button("🔄 Collect Feedback", type="primary"):
        if not os.path.exists(documents_folder):
            st.error(f"❌ Folder '{documents_folder}' not found")
            return
        
        # Show progress
        with st.spinner("Collecting feedback..."):
            try:
                # Load tags and documents
                tags_data = load_tags("outputs/tags.json")
                docs_texts = load_docs_for_feedback(documents_folder)
                
                if not tags_data or not docs_texts:
                    st.error("❌ No tags or documents found")
                    return
                
                # Simulate feedback
                feedback_results = simulate_feedback(tags_data, docs_texts)
                
                # Store in session state
                st.session_state['collected_feedback'] = feedback_results
                st.session_state['feedback_metadata'] = {
                    'timestamp': time.time(),
                    'documents_folder': documents_folder
                }
                
                # Save feedback
                save_feedback(feedback_results, "outputs/feedback.json")
                
                st.success("✅ Feedback collected successfully")
                
                # Display feedback
                display_feedback(feedback_results)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    st.divider()
    
    # Display existing feedback if available
    if 'collected_feedback' in st.session_state and st.session_state['collected_feedback'] is not None:
        st.info("📂 Loaded previously collected feedback")
        display_feedback(st.session_state['collected_feedback'])