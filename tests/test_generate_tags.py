"""
test_generate_tags.py
--------------------
Basic unit tests for generate_tags.py functionality.

Run with: python -m unittest tests.test_generate_tags -v
"""

import unittest
import os
import sys
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.generate_tags import (
    load_tag_weights, clean_text, load_documents, generate_tags, save_tags_to_json
)

class TestGenerateTags(unittest.TestCase):
    """Test generate_tags.py functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample test documents
        self.sample_docs = [
            {"filename": "test1.txt", "content": "API error timeout browser bug"},
            {"filename": "test2.txt", "content": "Design architecture authentication backend"},
            {"filename": "test3.txt", "content": "Meeting agenda evaluation automation"}
        ]
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    # Test 1: Text Cleaning Function
    def test_clean_text_basic(self):
        """
        Test that clean_text removes punctuation and converts to lowercase.
        
        What this tests: The core text preprocessing function
        Why it's important: If text cleaning fails, TF-IDF won't work properly
        """
        text = "API Error! Timeout 123 Browser Bug."
        cleaned = clean_text(text)
        
        # Check that punctuation and numbers are removed
        self.assertNotIn("!", cleaned)
        self.assertNotIn("123", cleaned)
        
        # Check that content words remain (lowercase)
        self.assertIn("api", cleaned)
        self.assertIn("error", cleaned)
        self.assertIn("timeout", cleaned)
    
    def test_clean_text_stopwords_removal(self):
        """
        Test that clean_text removes common English stopwords.
        
        What this tests: Stopword filtering
        Why it's important: Stopwords like "the", "and" shouldn't become tags
        """
        text = "the api and error timeout browser bug"
        cleaned = clean_text(text)
        
        # Stopwords should be removed
        self.assertNotIn("the", cleaned)
        self.assertNotIn("and", cleaned)
        
        # Content words should remain
        self.assertIn("api", cleaned)
        self.assertIn("error", cleaned)
    
    # Test 2: Document Loading Function
    def test_load_documents_success(self):
        """
        Test loading documents from a folder.
        
        What this tests: File I/O and document parsing
        Why it's important: If we can't load documents, nothing else works
        """
        # Create test files
        for doc in self.sample_docs:
            file_path = os.path.join(self.temp_dir, doc["filename"])
            with open(file_path, 'w') as f:
                f.write(doc["content"])
        
        # Load documents
        documents = load_documents(self.temp_dir)
        
        # Verify we got the right number of documents
        self.assertEqual(len(documents), len(self.sample_docs))
        
        # Verify each document has the right structure
        for doc in documents:
            self.assertIn("filename", doc)
            self.assertIn("content", doc)
            self.assertTrue(doc["filename"].endswith((".txt", ".md")))
    
    def test_load_documents_empty_folder(self):
        """
        Test loading from an empty folder.
        
        What this tests: Edge case handling
        Why it's important: System should handle empty folders gracefully
        """
        documents = load_documents(self.temp_dir)
        self.assertEqual(len(documents), 0)
    
    # Test 3: Tag Weights Loading
    def test_load_tag_weights_existing_file(self):
        """
        Test loading tag weights from an existing file.
        
        What this tests: JSON file loading and error handling
        Why it's important: Learned weights improve tag quality
        """
        # Create a test weights file
        weights_data = {"api": 1.3, "error": 0.5, "design": 1.1}
        weights_file = os.path.join(self.temp_dir, "weights.json")
        
        with open(weights_file, 'w') as f:
            json.dump(weights_data, f)
        
        # Load weights
        weights = load_tag_weights(weights_file)
        
        # Verify we got the right weights
        self.assertEqual(weights, weights_data)
    
    def test_load_tag_weights_missing_file(self):
        """
        Test loading tag weights when file doesn't exist.
        
        What this tests: Graceful handling of missing files
        Why it's important: First run won't have weights file
        """
        weights = load_tag_weights("nonexistent.json")
        self.assertEqual(weights, {})
    
    # Test 4: Tag Generation (Main Function)
    def test_generate_tags_basic(self):
        """
        Test basic tag generation with sample documents.
        
        What this tests: The core TF-IDF algorithm
        Why it's important: This is the main functionality
        """
        tags = generate_tags(self.sample_docs, top_k=3)
        
        # Verify we got tags for all documents
        self.assertEqual(len(tags), len(self.sample_docs))
        
        # Verify each document has tags
        for doc_tags in tags:
            self.assertIn("filename", doc_tags)
            self.assertIn("tags", doc_tags)
            
            # Should have at most 3 tags (top_k=3)
            self.assertLessEqual(len(doc_tags["tags"]), 3)
            
            # Each tag should have the right structure
            for tag in doc_tags["tags"]:
                self.assertIn("tag", tag)
                self.assertIn("tfidf_score", tag)
                self.assertIsInstance(tag["tfidf_score"], float)
                self.assertGreaterEqual(tag["tfidf_score"], 0.0)
                self.assertIn("adjusted_tfidf_score", tag)
                self.assertIsInstance(tag["adjusted_tfidf_score"], float)
                self.assertGreaterEqual(tag["adjusted_tfidf_score"], 0.0)
    
    def test_generate_tags_top_k_limit(self):
        """
        Test that top_k parameter limits the number of tags.
        
        What this tests: Parameter validation
        Why it's important: Users should get exactly the number of tags they request
        """
        tags = generate_tags(self.sample_docs, top_k=2)
        
        # Each document should have at most 2 tags
        for doc_tags in tags:
            self.assertLessEqual(len(doc_tags["tags"]), 2)
    
    def test_generate_tags_empty_documents(self):
        """
        Test tag generation with empty document list.
        
        What this tests: Edge case handling
        Why it's important: System should handle empty input gracefully
        """
        empty_docs = []
        
        # This should either return empty list or raise ValueError
        try:
            tags = generate_tags(empty_docs, top_k=5)
            self.assertEqual(len(tags), 0)
        except ValueError:
            # It's also acceptable for the function to raise ValueError
            pass
    
    # Test 5: File Saving
    def test_save_tags_to_json(self):
        """
        Test saving tags to JSON file.
        
        What this tests: File output functionality
        Why it's important: Results need to be saved for other steps
        """
        sample_tags = [
            {
                "filename": "test1.txt",
                "tags": [
                    {"tag": "api", "tfidf_score": 0.8},
                    {"tag": "error", "tfidf_score": 0.6}
                ]
            }
        ]
        
        output_file = os.path.join(self.temp_dir, "test_tags.json")
        save_tags_to_json(sample_tags, output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check content is correct
        with open(output_file, 'r') as f:
            loaded_tags = json.load(f)
            self.assertEqual(loaded_tags, sample_tags)

if __name__ == '__main__':
    unittest.main()