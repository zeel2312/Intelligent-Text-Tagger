"""
Tag display and visualization component
"""

import streamlit as st
import json
import pandas as pd
import os
import time

from src.generate_tags import generate_tags, load_documents, save_tags_to_json

def display_tags(tags_data=None):
    """Display tags in a nice format"""
    if not tags_data:
        st.warning("No tags to display")
        return
    
    st.subheader("🔧 Generated Tags")
    
    # Create a DataFrame for better display
    all_tags = []
    for doc in tags_data:
        filename = doc["filename"]
        for tag in doc["tags"]:
            all_tags.append({
                "Document": filename,
                "Tag": tag["tag"],
                "TF-IDF Score": f"{tag['tfidf_score']:.4f}"
            })
    
    if all_tags:
        df = pd.DataFrame(all_tags)
        st.dataframe(df, width="stretch")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tags", len(all_tags))
        with col2:
            st.metric("Unique Tags", len(set(tag["Tag"] for tag in all_tags)))
        with col3:
            st.metric("Documents", len(tags_data))
        
        # Tag frequency chart
        tag_counts = pd.DataFrame(all_tags)["Tag"].value_counts()
        if len(tag_counts) > 0:
            st.subheader("📊 Tag Frequency")
            st.bar_chart(tag_counts.head(10))

def generate_tags_interface():
    """Interface for generating tags"""    

    st.subheader("🔧 Generate Tags")
    
    # Parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of tags per document", 1, 20, 5)
    with col2:
        documents_folder = st.text_input("Documents folder", "documents")
    
    # Generate button
    if st.button("🚀 Generate Tags", type="primary"):
        if not os.path.exists(documents_folder):
            st.error(f"❌ Folder '{documents_folder}' not found")
            return
        
        # Show progress
        with st.spinner("Generating tags..."):
            try:
                # Load documents
                documents = load_documents(documents_folder)
                if not documents:
                    st.error("❌ No documents found")
                    return
                
                # Generate tags
                tags = generate_tags(documents, top_k=top_k)
                
                # Store in session state
                st.session_state['generated_tags'] = tags
                st.session_state['tags_metadata'] = {
                    'timestamp': time.time(),
                    'top_k': top_k,
                    'documents_folder': documents_folder,
                    'num_documents': len(documents)
                }
                
                # Save tags
                save_tags_to_json(tags, "outputs/tags.json")
                
                st.success(f"✅ Generated {sum(len(doc['tags']) for doc in tags)} tags")
                
                # Display tags
                display_tags(tags)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                
    st.divider()
    
    # Display existing tags if available
    if 'generated_tags' in st.session_state and st.session_state['generated_tags'] is not None:
        st.info("📂 Loaded previously generated tags")
        display_tags(st.session_state['generated_tags'])