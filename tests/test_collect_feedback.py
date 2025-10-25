"""
test_collect_feedback.py
------------------------
Basic unit tests for collect_feedback.py functionality.

Run with: python -m unittest tests.test_collect_feedback -v
"""

import unittest
import os
import sys
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.collect_feedback import (
    load_documents, load_tags, calculate_position_score, calculate_frequency_score,
    clean_text, simulate_feedback, save_feedback
)

class TestCollectFeedback(unittest.TestCase):
    """Test collect_feedback.py functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample document texts (raw text with structure)
        self.sample_texts = {
            "test1.txt": "API Error! Timeout 123 Browser Bug.",
            "test2.txt": "# Design Document\nArchitecture authentication backend",
            "test3.txt": "Meeting Notes\nAgenda: evaluation automation"
        }
        
        # Sample tags data (from generate_tags.py output)
        self.sample_tags = [
            {
                "filename": "test1.txt",
                "tags": [
                    {"tag": "api", "tfidf_score": 0.8},
                    {"tag": "error", "tfidf_score": 0.6},
                    {"tag": "timeout", "tfidf_score": 0.4}
                ]
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    # Test 1: Tags Loading Function
    def test_load_tags_success(self):
        """
        Test loading tags from JSON file.
        
        What this tests: JSON file loading and error handling
        Why it's important: Tags data comes from generate_tags.py output
        """
        tags_file = os.path.join(self.temp_dir, "tags.json")
        with open(tags_file, 'w') as f:
            json.dump(self.sample_tags, f)
        
        loaded_tags = load_tags(tags_file)
        self.assertEqual(loaded_tags, self.sample_tags)
    
    def test_load_tags_missing_file(self):
        """
        Test loading tags from missing file.
        
        What this tests: Graceful handling of missing files
        Why it's important: File might not exist or be corrupted
        """
        loaded_tags = load_tags("nonexistent.json")
        self.assertEqual(loaded_tags, [])
    
    # Test 2: Position Scoring Function
    def test_calculate_position_score_title(self):
        """
        Test position scoring for title (first line).
        
        What this tests: Title detection logic
        Why it's important: Tags in titles are most important (score 1.0)
        """
        text = "API Documentation\nThis is about APIs"
        score = calculate_position_score("api", text)
        self.assertEqual(score, 1.0)  # Should find in title
    
    def test_calculate_position_score_header(self):
        """
        Test position scoring for markdown headers.
        
        What this tests: Header detection logic
        Why it's important: Tags in headers are important (score 0.8)
        """
        text = "Introduction\n# API Documentation\nThis is about APIs"
        score = calculate_position_score("api", text)
        self.assertEqual(score, 0.8)  # Should find in header
    
    def test_calculate_position_score_first_paragraph(self):
        """
        Test position scoring for first paragraph.
        
        What this tests: First paragraph detection
        Why it's important: Tags in first paragraph are moderately important (score 0.6)
        """
        text = "Introduction\n\nAPI error timeout browser bug"
        score = calculate_position_score("api", text)
        self.assertEqual(score, 0.6)  # Should find in first paragraph
    
    def test_calculate_position_score_body(self):
        """
        Test position scoring for body text.
        
        What this tests: Body text detection
        Why it's important: Tags in body are less important (score 0.4)
        """
        text = "Introduction paragraph with enough content\n\nSome other content here\n\nAPI error timeout browser bug"
        score = calculate_position_score("api", text)
        self.assertEqual(score, 0.4)  # Should find in body
    
    def test_calculate_position_score_not_found(self):
        """
        Test position scoring when tag is not found.
        
        What this tests: Not found case
        Why it's important: Tags not in document should get score 0.0
        """
        text = "This is about something else"
        score = calculate_position_score("api", text)
        self.assertEqual(score, 0.0)  # Should not find
    
    # Test 3: Frequency Scoring Function
    def test_calculate_frequency_score_single_occurrence(self):
        """
        Test frequency scoring for single occurrence.
        
        What this tests: Basic frequency calculation
        Why it's important: Frequency affects relevance score
        """
        text = "api error timeout"  # "api" appears once
        score = calculate_frequency_score("api", text)
        
        # Should be > 0 but <= 1.0
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_calculate_frequency_score_multiple_occurrences(self):
        """
        Test frequency scoring for multiple occurrences.
        
        What this tests: Logarithmic scaling
        Why it's important: More occurrences should give higher score
        """
        text_single = "api error timeout"
        text_multiple = "api error timeout api api"  # "api" appears 3 times
        
        score_single = calculate_frequency_score("api", text_single)
        score_multiple = calculate_frequency_score("api", text_multiple)
        
        # Multiple occurrences should have higher score
        self.assertGreater(score_multiple, score_single)
        # But not linearly higher (logarithmic scaling)
        self.assertLess(score_multiple, score_single * 3)
    
    def test_calculate_frequency_score_no_occurrence(self):
        """
        Test frequency scoring when tag is not found.
        
        What this tests: Zero frequency case
        Why it's important: Tags not in text should get score 0.0
        """
        text = "something else"
        score = calculate_frequency_score("api", text)
        self.assertEqual(score, 0.0)
    
    # Test 4: Main Feedback Simulation Function
    def test_simulate_feedback_basic(self):
        """
        Test basic feedback simulation.
        
        What this tests: The core multi-signal scoring algorithm
        Why it's important: This is the main functionality
        """
        docs_texts = {"test1.txt": "API error timeout browser bug"}
        
        feedback = simulate_feedback(self.sample_tags, docs_texts)
        
        # Verify structure
        self.assertEqual(len(feedback), len(self.sample_tags))
        self.assertIn("filename", feedback[0])
        self.assertIn("feedback", feedback[0])
        
        # Verify each feedback item has required fields
        for item in feedback[0]["feedback"]:
            self.assertIn("tag", item)
            self.assertIn("status", item)
            self.assertIn("relevance_score", item)
            self.assertIn(item["status"], ["approved", "rejected"])
            self.assertIsInstance(item["relevance_score"], float)
    
    def test_simulate_feedback_approval_logic(self):
        """
        Test that feedback uses proper approval logic.
        
        What this tests: Multi-signal scoring produces different results
        Why it's important: Different tags should get different approval status
        """
        # Create tags with known TF-IDF scores
        test_tags = [
            {
                "filename": "test1.txt",
                "tags": [
                    {"tag": "api", "tfidf_score": 0.8},  # High TF-IDF
                    {"tag": "xyz", "tfidf_score": 0.1}   # Low TF-IDF
                ]
            }
        ]
        
        docs_texts = {"test1.txt": "API error timeout browser bug"}
        
        feedback = simulate_feedback(test_tags, docs_texts)
        
        # Find the feedback for each tag
        api_feedback = next(f for f in feedback[0]["feedback"] if f["tag"] == "api")
        xyz_feedback = next(f for f in feedback[0]["feedback"] if f["tag"] == "xyz")
        
        # API should have higher relevance score (appears in text + high TF-IDF)
        self.assertGreater(api_feedback["relevance_score"], xyz_feedback["relevance_score"])
    
    def test_simulate_feedback_empty_inputs(self):
        """
        Test feedback simulation with empty inputs.
        
        What this tests: Edge case handling
        Why it's important: System should handle empty data gracefully
        """
        feedback = simulate_feedback([], {})
        self.assertEqual(len(feedback), 0)
    
    def test_simulate_feedback_missing_document(self):
        """
        Test feedback simulation when document is missing.
        
        What this tests: Missing document handling
        Why it's important: Documents might be missing or corrupted
        """
        feedback = simulate_feedback(self.sample_tags, {})
        
        # Should still process but with empty document text
        self.assertEqual(len(feedback), len(self.sample_tags))
        for item in feedback[0]["feedback"]:
            self.assertIn("status", item)
            self.assertIn("relevance_score", item)
    
    # Test 5: File Saving Function
    def test_save_feedback(self):
        """
        Test saving feedback to JSON file.
        
        What this tests: File output functionality
        Why it's important: Results need to be saved for learning step
        """
        sample_feedback = [
            {
                "filename": "test1.txt",
                "feedback": [
                    {"tag": "api", "status": "approved", "relevance_score": 0.8}
                ]
            }
        ]
        
        output_file = os.path.join(self.temp_dir, "feedback.json")
        save_feedback(sample_feedback, output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check content is correct
        with open(output_file, 'r') as f:
            loaded_feedback = json.load(f)
            self.assertEqual(loaded_feedback, sample_feedback)

if __name__ == '__main__':
    unittest.main()