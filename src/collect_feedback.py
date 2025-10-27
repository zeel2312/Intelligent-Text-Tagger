"""
collect_feedback.py
--------------------
Step 2 of Intelligent Text Tagger:
Implements a rule-based simulated feedback mechanism.

Workflow:
1. Load generated tags (tags.json)
2. Load corresponding document texts
3. For each tag, calculate the frequency, position and tfidf scores
4. Combine the scores using the weights
5. Determine the approval status based on the combined score
6. Save results to feedback.json and print a summary
"""

import os
import json
import argparse
import re
import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TFIDF_WEIGHT, FREQUENCY_WEIGHT, POSITION_WEIGHT, APPROVAL_THRESHOLD, POSITION_SCORES

# -------------------------------------------------------------------------
#  Load Documents and Tags
# -------------------------------------------------------------------------
def load_documents(folder_path):
    documents = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md")):
            try:
                with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                    documents[filename] = f.read().lower()
            except Exception as e:
                print(f" Could not read {filename}: {e}")
    return documents


def load_tags(tags_file):
    try:
        with open(tags_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f" Error loading {tags_file}: {e}")
        return []


#  Position Scoring Function
def calculate_position_score(tag, raw_text):
    """
    Scores tag based on where it appears in document.
    Returns 0.0 to 1.0
    """
    lines = raw_text.split('\n')
    tag_lower = tag.lower()
    
    # Check first line (title)
    if lines and tag_lower in lines[0].lower():
        return POSITION_SCORES["title"]
    
    # Check headers (lines with #, all caps, or short lines that might be headers)
    for line in lines[:10]:  # Check first 10 lines for headers
        line_lower = line.lower().strip()
        if tag_lower in line_lower:
            # Check if it's a markdown header
            if line.strip().startswith('#'):
                return POSITION_SCORES["header"]
            # Check if it's an all-caps line (likely header)
            if line.strip().isupper() and len(line.strip()) < 50:
                return POSITION_SCORES["header"]
            # Check if it's a short line that might be a header
            if len(line.strip()) < 30 and ':' in line:
                return POSITION_SCORES["header"]
    
    # Check first paragraph (first substantial text block)
    first_paragraph = ""
    for line in lines:
        if line.strip() and not line.strip().startswith('#') and len(line.strip()) > 20:
            first_paragraph = line.lower()
            break
    
    if first_paragraph and tag_lower in first_paragraph:
        return POSITION_SCORES["first_paragraph"]
    
    # Check if tag appears anywhere in body text
    if tag_lower in raw_text.lower():
        return POSITION_SCORES["body"]
    
    return POSITION_SCORES["not_found"]


#  Frequency Scoring Function
def calculate_frequency_score(tag, cleaned_text):
    """
    Normalized frequency score (0.0 to 1.0).
    Uses logarithmic scaling to avoid over-emphasizing high counts.
    """
    count = cleaned_text.count(tag.lower())
    if count == 0:
        return 0.0
    
    # Logarithmic scaling: log(count + 1) / log(10)
    # This gives diminishing returns for higher counts
    score = min(1.0, math.log(count + 1) / math.log(10)) # TODO: think about this what could be done better?
    return score


#  Text Cleaning Function
def clean_text(text: str) -> str:
    """
    Cleans and normalizes input text for keyword extraction.
    - Converts to lowercase
    - Removes punctuation/numbers
    - Tokenizes
    - lemmatizes
    - Removes stopwords
    Returns cleaned string.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # keep only letters & spaces
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(filtered_tokens)


#  Multi-Signal Feedback Simulation
def simulate_feedback(tags_data, docs_texts):
    """
    Evaluates each tag using weighted multi-signal scoring:
    - TF-IDF score: 50% (statistical importance)
    - Frequency score: 20% (occurrence count)
    - Position score: 30% (structural importance)
    
    Threshold: 0.35 (tunable)
    - >= 0.35: approved
    - < 0.35: rejected
    """
    feedback_results = []

    for item in tags_data:
        filename = item["filename"]
        tags = item["tags"]
        raw_text = docs_texts.get(filename, "")
        cleaned_text = clean_text(raw_text)

        feedback_list = []
        for tag_data in tags:
            tag = tag_data["tag"]
            tfidf_score = tag_data["tfidf_score"]
            
            # Calculate multi-signal scores
            freq_score = calculate_frequency_score(tag, cleaned_text)
            pos_score = calculate_position_score(tag, raw_text)
            
            # Weighted combination
            combined_score = (
                TFIDF_WEIGHT * tfidf_score +
                FREQUENCY_WEIGHT * freq_score +
                POSITION_WEIGHT * pos_score
            )
            
            # Determine approval status
            status = "approved" if combined_score >= APPROVAL_THRESHOLD else "rejected"
            
            feedback_list.append({
                "tag": tag,
                "status": status,
                "relevance_score": round(combined_score, 4)
            })

        feedback_results.append({
            "filename": filename,
            "feedback": feedback_list
        })

    return feedback_results


#  Save feedback to JSON
def save_feedback(feedback, output_path="outputs/feedback.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)
    print(f"\n Feedback saved to {output_path}")


#  Display Summary
def print_summary(feedback_results):
    print("\n Feedback Summary")
    print("-" * 40)

    total_approved = 0
    total_rejected = 0
    
    for item in feedback_results:
        approved = [f for f in item["feedback"] if f["status"] == "approved"]
        rejected = [f for f in item["feedback"] if f["status"] == "rejected"]
        
        total_approved += len(approved)
        total_rejected += len(rejected)

        print(f"\n {item['filename']}")
        print(f" {len(approved)} approved |  {len(rejected)} rejected")
        
        if approved:
            approved_tags = [f"{f['tag']} ({f['relevance_score']})" for f in approved]
            print(f"Approved tags: {', '.join(approved_tags)}")
        else:
            print("Approved tags: None")
            
        if rejected:
            rejected_tags = [f"{f['tag']} ({f['relevance_score']})" for f in rejected]
            print(f"Rejected tags: {', '.join(rejected_tags)}")
        else:
            print("Rejected tags: None")
    
    # Overall summary
    total_tags = total_approved + total_rejected
    approval_rate = (total_approved / total_tags * 100) if total_tags > 0 else 0
    print(f"\n Overall: {total_approved}/{total_tags} approved ({approval_rate:.1f}%)")


#  Main CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate user feedback on tags (rule-based).")
    parser.add_argument("--tags", type=str, default="outputs/tags.json", help="Path to tags.json file")
    parser.add_argument("--docs", type=str, default="documents", help="Path to document folder")
    parser.add_argument("--save", action="store_true", help="Save feedback to feedback.json")
    args = parser.parse_args()

    # Load inputs
    tags_data = load_tags(args.tags)
    docs_texts = load_documents(args.docs)

    if not tags_data or not docs_texts:
        print(" Missing tags or documents. Please check your inputs.")
        exit()

    # Simulate feedback
    feedback_results = simulate_feedback(tags_data, docs_texts)

    # Print summary
    print_summary(feedback_results)

    # Save if requested
    if args.save:
        save_feedback(feedback_results)
