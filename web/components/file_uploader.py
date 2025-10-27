"""
File upload component for documents
"""

import streamlit as st
import os
import tempfile
from pathlib import Path

def file_uploader():
    """File upload component"""
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose text files",
        type=['txt', 'md'],
        accept_multiple_files=True,
        help="Upload .txt or .md files"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Show file details
        for file in uploaded_files:
            st.write(f"📄 {file.name} ({file.size} bytes)")
        
        # Save files option
        if st.button("💾 Save Files"):
            save_uploaded_files(uploaded_files)
    
    return uploaded_files

def save_uploaded_files(uploaded_files):
    """Save uploaded files to documents folder"""
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    saved_count = 0
    for file in uploaded_files:
        file_path = documents_dir / file.name
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_count += 1
    
    st.success(f"✅ Saved {saved_count} files to documents/ folder")
    
    # Refresh the page to show updated file list
    st.rerun()

def show_document_list():
    """Show list of existing documents"""
    st.subheader("📋 Current Documents")
    
    documents_dir = Path("documents")
    if documents_dir.exists():
        files = list(documents_dir.glob("*.txt")) + list(documents_dir.glob("*.md"))
        
        if files:
            for file in files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{file.stat().st_size} bytes")
                with col3:
                    if st.button("🗑️", key=f"delete_{file.name}"):
                        file.unlink()
                        st.rerun()
        else:
            st.info("No documents found. Upload some files!")
    else:
        st.info("Documents folder doesn't exist yet.")