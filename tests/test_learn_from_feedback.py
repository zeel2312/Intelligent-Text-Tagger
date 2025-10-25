"""
test_learn_from_feedback.py
---------------------------
Basic unit tests for learn_from_feedback.py functionality.

Run with: python -m unittest tests.test_learn_from_feedback -v
"""

import unittest
import os
import sys
import tempfile
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.learn_from_feedback import (
    load_feedback, compute_tag_stats, derive_tag_weights, save_weights
)

class TestLearnFromFeedback(unittest.TestCase):
    """Test learn_from_feedback.py functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Sample feedback data with different approval rates
        self.sample_feedback = [
            {
                "filename": "doc1.txt",
                "feedback": [
                    {"tag": "api", "status": "approved", "relevance_score": 0.8},
                    {"tag": "error", "status": "approved", "relevance_score": 0.7},
                    {"tag": "browser", "status": "rejected", "relevance_score": 0.2}
                ]
            },
            {
                "filename": "doc2.txt", 
                "feedback": [
                    {"tag": "api", "status": "approved", "relevance_score": 0.9},
                    {"tag": "error", "status": "rejected", "relevance_score": 0.3},
                    {"tag": "design", "status": "approved", "relevance_score": 0.6}
                ]
            }
        ]
    
    def tearDown(self):
        """Clean up test fixtures after each test."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    # Test 1: Feedback Loading Function
    def test_load_feedback_success(self):
        """
        Test loading feedback from JSON file.
        
        What this tests: JSON file loading and error handling
        Why it's important: Feedback data comes from collect_feedback.py output
        """
        feedback_file = os.path.join(self.temp_dir, "feedback.json")
        with open(feedback_file, 'w') as f:
            json.dump(self.sample_feedback, f)
        
        loaded_feedback = load_feedback(feedback_file)
        self.assertEqual(loaded_feedback, self.sample_feedback)
    
    def test_load_feedback_missing_file(self):
        """
        Test loading feedback from missing file.
        
        What this tests: Graceful handling of missing files
        Why it's important: File might not exist or be corrupted
        """
        loaded_feedback = load_feedback("nonexistent.json")
        self.assertEqual(loaded_feedback, [])
    
    def test_load_feedback_invalid_json(self):
        """
        Test loading feedback with invalid JSON.
        
        What this tests: Error handling for corrupted files
        Why it's important: Files can get corrupted or have syntax errors
        """
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        loaded_feedback = load_feedback(invalid_file)
        self.assertEqual(loaded_feedback, [])
    
    # Test 2: Tag Statistics Computation
    def test_compute_tag_stats_basic(self):
        """
        Test basic tag statistics computation.
        
        What this tests: The core learning algorithm - counting approvals/rejections
        Why it's important: This is how the system learns which tags are good
        """
        tag_rates = compute_tag_stats(self.sample_feedback)
        
        # Check that all tags are included
        expected_tags = {"api", "error", "browser", "design"}
        self.assertEqual(set(tag_rates.keys()), expected_tags)
        
        # Check approval rates
        # api: 2 approved, 0 rejected = 100%
        self.assertEqual(tag_rates["api"], 1.0)
        
        # error: 1 approved, 1 rejected = 50%
        self.assertEqual(tag_rates["error"], 0.5)
        
        # browser: 0 approved, 1 rejected = 0%
        self.assertEqual(tag_rates["browser"], 0.0)
        
        # design: 1 approved, 0 rejected = 100%
        self.assertEqual(tag_rates["design"], 1.0)
    
    def test_compute_tag_stats_empty_feedback(self):
        """
        Test tag statistics with empty feedback.
        
        What this tests: Edge case handling
        Why it's important: System should handle empty input gracefully
        """
        tag_rates = compute_tag_stats([])
        self.assertEqual(len(tag_rates), 0)
    
    def test_compute_tag_stats_case_insensitive(self):
        """
        Test that tag statistics are case insensitive.
        
        What this tests: Case normalization
        Why it's important: "API" and "api" should be treated as the same tag
        """
        feedback_with_case = [
            {
                "filename": "doc1.txt",
                "feedback": [
                    {"tag": "API", "status": "approved", "relevance_score": 0.8},
                    {"tag": "api", "status": "approved", "relevance_score": 0.7}
                ]
            }
        ]
        
        tag_rates = compute_tag_stats(feedback_with_case)
        
        # Should combine both "API" and "api" as "api"
        self.assertIn("api", tag_rates)
        self.assertEqual(tag_rates["api"], 1.0)  # 2 approved, 0 rejected
    
    def test_compute_tag_stats_single_approval(self):
        """
        Test tag statistics with single approval.
        
        What this tests: Edge case - single occurrence
        Why it's important: Tags with only one occurrence should still work
        """
        single_approval = [
            {
                "filename": "doc1.txt",
                "feedback": [
                    {"tag": "test", "status": "approved", "relevance_score": 0.8}
                ]
            }
        ]
        
        tag_rates = compute_tag_stats(single_approval)
        self.assertEqual(tag_rates["test"], 1.0)
    
    def test_compute_tag_stats_single_rejection(self):
        """
        Test tag statistics with single rejection.
        
        What this tests: Edge case - single occurrence
        Why it's important: Tags with only one occurrence should still work
        """
        single_rejection = [
            {
                "filename": "doc1.txt",
                "feedback": [
                    {"tag": "test", "status": "rejected", "relevance_score": 0.2}
                ]
            }
        ]
        
        tag_rates = compute_tag_stats(single_rejection)
        self.assertEqual(tag_rates["test"], 0.0)
    
    # Test 3: Tag Weight Derivation
    def test_derive_tag_weights_basic(self):
        """
        Test basic tag weight derivation.
        
        What this tests: The core learning algorithm - converting approval rates to weights
        Why it's important: This is how the system learns to boost good tags and penalize bad ones
        """
        tag_rates = {"api": 1.0, "error": 0.5, "browser": 0.0}
        weights = derive_tag_weights(tag_rates)
        
        # Check that all tags have weights
        self.assertEqual(set(weights.keys()), set(tag_rates.keys()))
        
        # Check weight assignments based on approval rates
        # api: 100% approval should get strong boost (1.3)
        self.assertEqual(weights["api"], 1.3)
        
        # error: 50% approval should get mild boost (1.1)
        self.assertEqual(weights["error"], 1.1)
        
        # browser: 0% approval should get strong penalty (0.5)
        self.assertEqual(weights["browser"], 0.5)
    
    def test_derive_tag_weights_all_scenarios(self):
        """
        Test tag weight derivation for all approval rate scenarios.
        
        What this tests: All learning rules from config
        Why it's important: System should handle all possible approval rates
        """
        tag_rates = {
            "excellent": 0.9,    # Should get strong boost (1.3)
            "good": 0.6,        # Should get mild boost (1.1)
            "poor": 0.3,        # Should get mild penalty (0.9)
            "terrible": 0.1     # Should get strong penalty (0.5)
        }
        
        weights = derive_tag_weights(tag_rates)
        
        self.assertEqual(weights["excellent"], 1.3)
        self.assertEqual(weights["good"], 1.1)
        self.assertEqual(weights["poor"], 0.9)
        self.assertEqual(weights["terrible"], 0.5)
    
    def test_derive_tag_weights_empty_input(self):
        """
        Test tag weight derivation with empty input.
        
        What this tests: Edge case handling
        Why it's important: System should handle empty input gracefully
        """
        weights = derive_tag_weights({})
        self.assertEqual(len(weights), 0)
    
    def test_derive_tag_weights_boundary_cases(self):
        """
        Test tag weight derivation at boundary values.
        
        What this tests: Edge cases at learning thresholds
        Why it's important: System should handle boundary conditions correctly
        """
        # Test exact boundary values
        tag_rates = {
            "boundary_80": 0.8,    # Exactly at strong boost threshold
            "boundary_50": 0.5,    # Exactly at mild boost threshold
            "boundary_20": 0.2,    # Exactly at mild penalty threshold
        }
        
        weights = derive_tag_weights(tag_rates)
        
        self.assertEqual(weights["boundary_80"], 1.3)
        self.assertEqual(weights["boundary_50"], 1.1)
        self.assertEqual(weights["boundary_20"], 0.9)
    
    # Test 4: File Saving Function
    def test_save_weights(self):
        """
        Test saving weights to JSON file.
        
        What this tests: File output functionality
        Why it's important: Results need to be saved for future tag generation
        """
        sample_weights = {"api": 1.3, "error": 0.5, "design": 1.1}
        
        output_file = os.path.join(self.temp_dir, "weights.json")
        save_weights(sample_weights, output_file)
        
        # Check file was created
        self.assertTrue(os.path.exists(output_file))
        
        # Check content is correct
        with open(output_file, 'r') as f:
            loaded_weights = json.load(f)
            self.assertEqual(loaded_weights, sample_weights)
    
    # Test 5: Complete Learning Pipeline
    def test_learning_pipeline_integration(self):
        """
        Test the complete learning pipeline.
        
        What this tests: End-to-end learning process
        Why it's important: This is how the system actually learns from feedback
        """
        # Create feedback file
        feedback_file = os.path.join(self.temp_dir, "feedback.json")
        with open(feedback_file, 'w') as f:
            json.dump(self.sample_feedback, f)
        
        # Run the complete pipeline
        feedback_data = load_feedback(feedback_file)
        tag_rates = compute_tag_stats(feedback_data)
        tag_weights = derive_tag_weights(tag_rates)
        
        # Save weights
        weights_file = os.path.join(self.temp_dir, "weights.json")
        save_weights(tag_weights, weights_file)
        
        # Verify results
        self.assertGreater(len(tag_rates), 0)
        self.assertGreater(len(tag_weights), 0)
        self.assertTrue(os.path.exists(weights_file))
        
        # Check that weights are reasonable
        for tag, weight in tag_weights.items():
            self.assertIsInstance(weight, (int, float))
            self.assertGreaterEqual(weight, 0.0)
            self.assertLessEqual(weight, 2.0)
    
    def test_learning_pipeline_realistic_scenario(self):
        """
        Test learning pipeline with realistic feedback data.
        
        What this tests: Real-world learning scenario
        Why it's important: System should work with realistic data
        """
        # Realistic feedback with mixed approval rates
        realistic_feedback = [
            {
                "filename": "support_ticket.txt",
                "feedback": [
                    {"tag": "api", "status": "approved", "relevance_score": 0.8},
                    {"tag": "error", "status": "approved", "relevance_score": 0.7},
                    {"tag": "timeout", "status": "approved", "relevance_score": 0.6},
                    {"tag": "browser", "status": "rejected", "relevance_score": 0.2},
                    {"tag": "xyz", "status": "rejected", "relevance_score": 0.1}
                ]
            },
            {
                "filename": "bug_report.txt",
                "feedback": [
                    {"tag": "api", "status": "approved", "relevance_score": 0.9},
                    {"tag": "error", "status": "approved", "relevance_score": 0.8},
                    {"tag": "timeout", "status": "rejected", "relevance_score": 0.3},
                    {"tag": "browser", "status": "approved", "relevance_score": 0.5},
                    {"tag": "xyz", "status": "rejected", "relevance_score": 0.1}
                ]
            }
        ]
        
        # Run learning pipeline
        tag_rates = compute_tag_stats(realistic_feedback)
        tag_weights = derive_tag_weights(tag_rates)
        
        # Verify learning results
        # "api" should be boosted (100% approval)
        self.assertEqual(tag_weights["api"], 1.3)
        
        # "error" should be boosted (100% approval)
        self.assertEqual(tag_weights["error"], 1.3)
        
        # "timeout" should be mildly boosted (50% approval)
        self.assertEqual(tag_weights["timeout"], 1.1)
        
        # "browser" should be mildly boosted (50% approval)
        self.assertEqual(tag_weights["browser"], 1.1)
        
        # "xyz" should be penalized (0% approval)
        self.assertEqual(tag_weights["xyz"], 0.5)

if __name__ == '__main__':
    unittest.main()