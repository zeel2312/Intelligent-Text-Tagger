"""
Session state management utilities
"""
import streamlit as st
import json
import os
from datetime import datetime


def init_session_state():
    """Initialize all session state keys if not present"""
    if 'generated_tags' not in st.session_state:
        st.session_state['generated_tags'] = None
    if 'collected_feedback' not in st.session_state:
        st.session_state['collected_feedback'] = None
    if 'learned_weights' not in st.session_state:
        st.session_state['learned_weights'] = None
    if 'tag_rates' not in st.session_state:
        st.session_state['tag_rates'] = None
    if 'tags_metadata' not in st.session_state:
        st.session_state['tags_metadata'] = None
    if 'feedback_metadata' not in st.session_state:
        st.session_state['feedback_metadata'] = None


def load_state_from_files():
    """Load all available state from output files"""
    # Load tags
    if os.path.exists("outputs/tags.json"):
        try:
            with open("outputs/tags.json", "r") as f:
                st.session_state['generated_tags'] = json.load(f)
        except:
            pass
    
    # Load feedback
    if os.path.exists("outputs/feedback.json"):
        try:
            with open("outputs/feedback.json", "r") as f:
                st.session_state['collected_feedback'] = json.load(f)
        except:
            pass
    
    # Load weights
    if os.path.exists("outputs/tag_weights.json"):
        try:
            with open("outputs/tag_weights.json", "r") as f:
                st.session_state['learned_weights'] = json.load(f)
        except:
            pass


def reload_from_files_if_newer():
    """Reload from files if file modification time is newer than when we loaded"""
    # Check if files are newer and reload if so
    for file_key, filename in [
        ('generated_tags', 'outputs/tags.json'),
        ('collected_feedback', 'outputs/feedback.json'),
        ('learned_weights', 'outputs/tag_weights.json')
    ]:
        if os.path.exists(filename):
            file_time = os.path.getmtime(filename)
            loaded_time = st.session_state.get(f'{file_key}_loaded_at', 0)
            
            if file_time > loaded_time:
                try:
                    with open(filename, "r") as f:
                        st.session_state[file_key] = json.load(f)
                        st.session_state[f'{file_key}_loaded_at'] = file_time
                except:
                    pass
