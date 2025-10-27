"""
Pipeline execution component
"""

import streamlit as st
import os
import time
import pandas as pd
from src.generate_tags import generate_tags, load_documents, save_tags_to_json
from src.collect_feedback import simulate_feedback, load_tags, load_documents as load_docs_for_feedback, save_feedback
from src.learn_from_feedback import load_feedback, compute_tag_stats, derive_tag_weights, save_weights

def run_pipeline_interface():
    """Interface for running the complete pipeline"""
    st.subheader("🚀 Run Complete Pipeline")
    
    # Show quick summary if complete pipeline results exist
    if all(key in st.session_state and st.session_state[key] is not None 
           for key in ['generated_tags', 'collected_feedback', 'learned_weights']):
        st.info("📂 Last Pipeline Run Results:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", len(st.session_state['generated_tags']))
        with col2:
            total_tags = sum(len(doc['tags']) for doc in st.session_state['generated_tags'])
            st.metric("Tags Generated", total_tags)
        with col3:
            # Calculate approval rate
            feedback = st.session_state['collected_feedback']
            total = sum(len(doc['feedback']) for doc in feedback)
            approved = sum(len([f for f in doc['feedback'] if f['status'] == 'approved']) for doc in feedback)
            rate = (approved / total * 100) if total > 0 else 0
            st.metric("Approval Rate", f"{rate:.1f}%")
        with col4:
            st.metric("Weights Learned", len(st.session_state['learned_weights']))
        st.divider()
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        documents_folder = st.text_input("Documents folder", "documents")
    with col2:
        top_k = st.slider("Tags per document", 1, 20, 5)
    
    # Run pipeline button
    if st.button("🚀 Run Complete Pipeline", type="primary"):
        if not os.path.exists(documents_folder):
            st.error(f"❌ Folder '{documents_folder}' not found")
            return
        
        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Generate Tags
            status_text.text("Step 1/3: Generating tags...")
            progress_bar.progress(0.33)
            
            documents = load_documents(documents_folder)
            if not documents:
                st.error("❌ No documents found")
                return
            
            tags = generate_tags(documents, top_k=top_k)
            
            # Store in session state
            st.session_state['generated_tags'] = tags
            st.session_state['tags_metadata'] = {
                'timestamp': time.time(),
                'top_k': top_k,
                'documents_folder': documents_folder,
                'num_documents': len(documents)
            }
            save_tags_to_json(tags, "outputs/tags.json")
            
            # Step 2: Collect Feedback
            status_text.text("Step 2/3: Collecting feedback...")
            progress_bar.progress(0.66)
            
            tags_data = load_tags("outputs/tags.json")
            docs_texts = load_docs_for_feedback(documents_folder)
            feedback_results = simulate_feedback(tags_data, docs_texts)
            
            st.session_state['collected_feedback'] = feedback_results
            st.session_state['feedback_metadata'] = {
                'timestamp': time.time(),
                'documents_folder': documents_folder
            }
            save_feedback(feedback_results, "outputs/feedback.json")
            
            # Step 3: Learn from Feedback
            status_text.text("Step 3/3: Learning from feedback...")
            progress_bar.progress(1.0)
            
            feedback_data = load_feedback("outputs/feedback.json")
            tag_rates = compute_tag_stats(feedback_data)
            tag_weights = derive_tag_weights(tag_rates)
            
            st.session_state['learned_weights'] = tag_weights
            st.session_state['tag_rates'] = tag_rates
            save_weights(tag_weights, "outputs/tag_weights.json")
            
            status_text.text("✅ Pipeline completed successfully!")
            
            # Show results summary
            show_pipeline_results(tags, feedback_results, tag_weights)
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

def show_pipeline_results(tags, feedback, weights):
    """Show complete pipeline results summary"""
    st.subheader("📊 Pipeline Results Summary")
    
    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documents Processed", len(tags))
    with col2:
        total_tags = sum(len(doc['tags']) for doc in tags)
        st.metric("Tags Generated", total_tags)
    with col3:
        total_feedback = sum(len(doc['feedback']) for doc in feedback)
        approved_feedback = sum(len([f for f in doc['feedback'] if f['status'] == 'approved']) for doc in feedback)
        approval_rate = (approved_feedback / total_feedback * 100) if total_feedback > 0 else 0
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    with col4:
        st.metric("Weights Learned", len(weights))
    
    st.divider()
    
    # Step 1: Tags Summary
    st.subheader("🏷️ Step 1: Generated Tags")
    tags_summary = []
    for doc in tags:
        tag_names = [tag['tag'] for tag in doc['tags']]
        tags_summary.append({
            "Document": doc['filename'],
            "Tags": ", ".join(tag_names),
            "Count": len(doc['tags'])
        })
    tags_df = pd.DataFrame(tags_summary)
    st.dataframe(tags_df, width="stretch", hide_index=True)
    
    st.divider()
    
    # Step 2: Feedback Summary
    st.subheader("🔄 Step 2: Feedback Analysis")
    
    feedback_col1, feedback_col2 = st.columns(2)
    
    with feedback_col1:
        # Approval breakdown
        approved = sum(len([f for f in doc['feedback'] if f['status'] == 'approved']) for doc in feedback)
        rejected = total_feedback - approved
        st.metric("Approved", approved)
        st.metric("Rejected", rejected)
    
    with feedback_col2:
        # Top approved tags
        tag_approvals = {}
        for doc in feedback:
            for f in doc['feedback']:
                tag = f['tag']
                if tag not in tag_approvals:
                    tag_approvals[tag] = {'approved': 0, 'rejected': 0}
                if f['status'] == 'approved':
                    tag_approvals[tag]['approved'] += 1
                else:
                    tag_approvals[tag]['rejected'] += 1
        
        # Top approved tags
        approved_tags = [(tag, data['approved']) for tag, data in tag_approvals.items() if data['approved'] > 0]
        approved_tags.sort(key=lambda x: x[1], reverse=True)
        st.write("**Top Approved Tags:**")
        for tag, count in approved_tags[:5]:
            st.write(f"✓ {tag}: {count} approval(s)")
    
    # Per-document feedback
    st.write("**Per-Document Approval:**")
    doc_feedback = []
    for doc in feedback:
        doc_approved = len([f for f in doc['feedback'] if f['status'] == 'approved'])
        doc_total = len(doc['feedback'])
        doc_rate = (doc_approved / doc_total * 100) if doc_total > 0 else 0
        doc_feedback.append({
            "Document": doc['filename'],
            "Approved": doc_approved,
            "Rejected": doc_total - doc_approved,
            "Rate": f"{doc_rate:.1f}%"
        })
    doc_feedback_df = pd.DataFrame(doc_feedback)
    st.dataframe(doc_feedback_df, width="stretch", hide_index=True)
    
    st.divider()
    
    # Step 3: Learning Summary
    st.subheader("🧠 Step 3: Learning Results")
    
    # Learning statistics
    learning_col1, learning_col2, learning_col3, learning_col4 = st.columns(4)
    
    with learning_col1:
        boosted_tags = sum(1 for w in weights.values() if w > 1.0)
        st.metric("Boosted", boosted_tags)
    with learning_col2:
        penalized_tags = sum(1 for w in weights.values() if w < 1.0)
        st.metric("Penalized", penalized_tags)
    with learning_col3:
        neutral_tags = sum(1 for w in weights.values() if w == 1.0)
        st.metric("Neutral", neutral_tags)
    with learning_col4:
        total_weighted = len(weights)
        st.metric("Total Tags", total_weighted)
    
    # Show top weights
    if weights:
        weights_df = pd.DataFrame(list(weights.items()), columns=['Tag', 'Weight'])
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Boosted Tags (≥1.1):**")
            boosted_df = weights_df[weights_df['Weight'] >= 1.1].head(10)
            if len(boosted_df) > 0:
                st.dataframe(boosted_df, width="stretch", hide_index=True)
            else:
                st.info("No boosted tags")
        
        with col2:
            st.write("**Top Penalized Tags (≤0.9):**")
            penalized_df = weights_df[weights_df['Weight'] <= 0.9].head(10)
            if len(penalized_df) > 0:
                st.dataframe(penalized_df, width="stretch", hide_index=True)
            else:
                st.info("No penalized tags")
        
        st.divider()
        
        st.subheader("📈 All Learned Weights")
        st.dataframe(weights_df, width="stretch", hide_index=True, height=300)