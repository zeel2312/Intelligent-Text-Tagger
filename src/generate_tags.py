"""
generate_tags.py
----------------
Step 1 of Intelligent Text Tagger:
Generates automatic tags for each ingested document using TF-IDF vectorization.

Workflow:
1. Load documents (from folder or pre-loaded list)
2. Clean and normalize text
3. Compute TF-IDF matrix
4. Extract top-k keywords per document
5. Print and optionally save results to JSON
"""

import os
import re
import json
import argparse
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer

# First-time setup (downloads tokenizer + stopword list)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

#  Load learned tag weights (if available)
def load_tag_weights(weights_path="outputs/tag_weights.json"):
    if os.path.exists(weights_path):
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                weights = json.load(f)
                print(f" Loaded {len(weights)} learned tag weights.")
                return weights
        except Exception as e:
            print(f" Could not load {weights_path}: {e}")
    return {}


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


#  Document Loader
def load_documents(folder_path: str) -> List[Dict[str, str]]:
    docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".txt", ".md")):
            try:
                with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                    docs.append({"filename": filename, "content": f.read()})
            except Exception as e:
                print(f"⚠️ Could not read {filename}: {e}")
    return docs


#  TF-IDF Tag Generator
def generate_tags(docs: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, List[Dict[str, float]]]]:
    """
    Takes a list of documents (with 'content' key).
    Returns list of dicts with filename + generated tags with TF-IDF scores.
    """
    # Step 1: Clean each document's text
    cleaned_texts = [clean_text(d['content']) for d in docs]

    # Step 2: Fit TF-IDF on all docs
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()

    # Step 3: Extract top terms per doc (with learned weights)
    tag_weights = load_tag_weights()

    tags_output = []
    for idx, doc in enumerate(docs):
        tfidf_scores = tfidf_matrix[idx].toarray().flatten()
        adjusted_scores = []

        for i, term in enumerate(feature_names):
            original_tfidf_score = tfidf_scores[i]
            adjusted_score = original_tfidf_score * tag_weights.get(term.lower(), 1.0)
            adjusted_scores.append((term, adjusted_score, original_tfidf_score))

        # Sort by adjusted score and pick top-k
        top_terms_with_scores = []
        for term, adjusted_score, original_tfidf_score in sorted(adjusted_scores, key=lambda x: x[1], reverse=True)[:top_k]:
            if adjusted_score > 0:
                top_terms_with_scores.append({
                    "tag": term,
                    "tfidf_score": float(original_tfidf_score),
                    "adjusted_tfidf_score": float(adjusted_score)
                })

        tags_output.append({
            "filename": doc["filename"],
            "tags": top_terms_with_scores
        })
    return tags_output



#  Save tags to JSON
def save_tags_to_json(tags: List[Dict[str, List[str]]], output_path: str = "outputs/tags.json"):
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tags, f, indent=2)
    print(f"\n Tags saved to {output_path}")


#  CLI Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TF-IDF tags for documents.")
    parser.add_argument("--folder", type=str, default="documents", help="Path to document folder")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top tags per document")
    parser.add_argument("--save", action="store_true", help="Save results to tags.json")
    args = parser.parse_args()

    # Load & process
    documents = load_documents(args.folder)
    if not documents:
        print(f" No documents found in {args.folder}")
        exit()

    tags = generate_tags(documents, top_k=args.top_k)

    # Print results nicely
    print("\n Generated Tags:\n")
    for t in tags:
        tag_names = [tag["tag"] for tag in t['tags']]
        print(f"\n --> {t['filename']}: {tag_names}")

    if args.save:
        save_tags_to_json(tags)
