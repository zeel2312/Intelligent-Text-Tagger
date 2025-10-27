"""
Learning from feedback display and visualization component
"""

import streamlit as st
import pandas as pd
import json
import os

from src.learn_from_feedback import (
    load_feedback, 
    compute_tag_stats, 
    derive_tag_weights, 
    save_weights,
    print_summary
)
from src.generate_tags import load_tag_weights


def check_prerequisites():
    """Check if required files exist and guide user"""
    status = {
        "tags_exists": os.path.exists("outputs/tags.json"),
        "feedback_exists": os.path.exists("outputs/feedback.json"),
        "weights_exists": os.path.exists("outputs/tag_weights.json")
    }
    
    return status


def display_prerequisites_status():
    """Display prerequisite check with visual indicators"""
    status = check_prerequisites()
    
    st.subheader("📋 Prerequisites")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if status["tags_exists"]:
            st.success("✅ Step 1: Tags Generated")
        else:
            st.error("❌ Step 1: Tags Not Generated")
            st.info("💡 Go to 'Generate Tags' page to create tags first")
    
    with col2:
        if status["feedback_exists"]:
            st.success("✅ Step 2: Feedback Collected")
        else:
            st.error("❌ Step 2: Feedback Not Collected")
            st.info("💡 Go to 'View Feedback' page to collect feedback first")
    
    return status["feedback_exists"]


def display_current_weights():
    """Display existing weights if they exist"""
    if not os.path.exists("outputs/tag_weights.json"):
        return None
    
    try:
        weights = load_tag_weights()
        
        # Check if weights is valid
        if weights is None or not weights:
            return None
        
        st.subheader("📊 Current Learned Weights")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tags", len(weights))
        
        with col2:
            boosted = sum(1 for w in weights.values() if w > 1.0)
            st.metric("Boosted", boosted, delta=None, delta_color="normal")
        
        with col3:
            penalized = sum(1 for w in weights.values() if w < 1.0)
            st.metric("Penalized", penalized, delta=None, delta_color="normal")
        
        with col4:
            neutral = sum(1 for w in weights.values() if w == 1.0)
            st.metric("Neutral", neutral, delta=None, delta_color="normal")
        
        # Show top weights
        if len(weights) > 0:
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            
            with st.expander("View Current Weights"):
                weights_df = pd.DataFrame(sorted_weights, columns=['Tag', 'Weight'])
                st.dataframe(weights_df.head(10), width="stretch", hide_index=True)
        
        return weights
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None


def display_learning_results(weights, tag_rates=None):
    """Display learning statistics and weight breakdown"""
    st.subheader("📈 Learning Results")
    
    # Check if weights is None or empty
    if weights is None or not weights:
        st.warning("No weights to display")
        return
    
    # Calculate metrics
    strong_boosted = sum(1 for w in weights.values() if w >= 1.3)
    mild_boosted = sum(1 for w in weights.values() if 1.1 <= w < 1.3)
    mild_penalized = sum(1 for w in weights.values() if 0.9 <= w < 1.1)
    strong_penalized = sum(1 for w in weights.values() if w < 0.9)
    neutral = sum(1 for w in weights.values() if w == 1.0)
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tags", len(weights))
    
    with col2:
        st.metric("Strong Boost (≥1.3)", strong_boosted, delta=None, delta_color="normal")
        st.caption(f"Mild Boost (1.1): {mild_boosted}")
    
    with col3:
        st.metric("Strong Penalty (≤0.9)", strong_penalized, delta=None, delta_color="normal")
        st.caption(f"Mild Penalty (0.9): {mild_penalized}")
    
    with col4:
        st.metric("Boosted Total", strong_boosted + mild_boosted, delta=None, delta_color="normal")
        st.caption(f"Neutral: {neutral}")
    
    # Detailed weights table
    if weights:
        st.subheader("📊 All Learned Weights")
        
        # Create DataFrame with additional info
        weights_list = []
        for tag, weight in weights.items():
            category = ""
            if weight >= 1.3:
                category = "🟢 Strong Boost"
            elif weight >= 1.1:
                category = "🟡 Mild Boost"
            elif weight > 0.9:
                category = "⚪ Neutral"
            elif weight > 0.5:
                category = "🟠 Mild Penalty"
            else:
                category = "🔴 Strong Penalty"
            
            weights_list.append({
                'Tag': tag,
                'Weight': weight,
                'Category': category
            })
        
        weights_df = pd.DataFrame(weights_list)
        weights_df = weights_df.sort_values('Weight', ascending=False)
        
        # Color code the rows
        def color_weight_category(row):
            weight = row['Weight']
            if weight >= 1.3:
                return ['background-color: #d4edda'] * len(row)
            elif weight >= 1.1:
                return ['background-color: #fff3cd'] * len(row)
            elif weight < 0.9:
                return ['background-color: #f8d7da'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = weights_df.style.apply(color_weight_category, axis=1)
        st.dataframe(styled_df, width="stretch", hide_index=True, height=400)
        
        # Top boosted and penalized sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌟 Top Boosted Tags")
            boosted_tags = weights_df[weights_df['Weight'] > 1.0].head(10)
            if len(boosted_tags) > 0:
                st.dataframe(boosted_tags[['Tag', 'Weight', 'Category']], hide_index=True)
            else:
                st.info("No boosted tags")
        
        with col2:
            st.subheader("⚠️ Top Penalized Tags")
            penalized_tags = weights_df[weights_df['Weight'] < 1.0].head(10)
            if len(penalized_tags) > 0:
                st.dataframe(penalized_tags[['Tag', 'Weight', 'Category']], hide_index=True)
            else:
                st.info("No penalized tags")


def learn_from_feedback_interface():
    """Main interface for learning from feedback"""
    st.subheader("🧠 Learn from Feedback")
    st.markdown("Process feedback data to learn optimal tag weights for future tagging.")
    
    # Display prerequisites
    is_ready = display_prerequisites_status()
    
    # Display current weights if they exist
    current_weights = display_current_weights()
    
    st.divider()
    
    # Action button
    if not is_ready:
        st.error("❌ Cannot learn from feedback: Missing required files. Please complete the previous steps.")
        return
    
    if st.button("🧠 Learn from Feedback", type="primary", use_container_width=True):
        try:
            with st.spinner("Processing feedback and learning weights..."):
                # Load feedback
                feedback_data = load_feedback("outputs/feedback.json")
                
                if not feedback_data:
                    st.error("❌ No feedback data found in outputs/feedback.json")
                    return
                
                # Compute statistics
                tag_rates = compute_tag_stats(feedback_data)
                
                # Derive weights
                tag_weights = derive_tag_weights(tag_rates)
                
                # Save weights
                save_weights(tag_weights, "outputs/tag_weights.json")
                
                # Store in session for display
                st.session_state['learned_weights'] = tag_weights
                st.session_state['tag_rates'] = tag_rates
                
                st.success("✅ Successfully learned weights from feedback!")
        
        except Exception as e:
            st.error(f"❌ Error learning from feedback: {str(e)}")
            st.exception(e)
    
    # Display results if learning was just performed
    if 'learned_weights' in st.session_state and st.session_state['learned_weights'] is not None:
        st.divider()
        display_learning_results(st.session_state['learned_weights'], st.session_state.get('tag_rates'))
