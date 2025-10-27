"""
pipeline.py
-----------
Simple pipeline orchestrator for the Intelligent Text Tagger.
Runs the complete pipeline: generate_tags -> collect_feedback -> learn_from_feedback
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Import our existing modules
from src.generate_tags import generate_tags, load_documents, save_tags_to_json
from src.collect_feedback import simulate_feedback, load_tags, load_documents as load_docs_for_feedback, save_feedback
from src.learn_from_feedback import load_feedback, compute_tag_stats, derive_tag_weights, save_weights

def run_pipeline(documents_folder="documents", output_dir="outputs", top_k=5):
    """
    Run the complete Intelligent Text Tagger pipeline.
    
    Args:
        documents_folder: Path to folder containing documents
        output_dir: Path to output directory for results
        top_k: Number of top tags to generate per document
    
    Returns:
        dict: Pipeline results and metrics
    """
    start_time = time.time()
    
    print("Starting Intelligent Text Tagger Pipeline")
    print("=" * 50)
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Step 1: Generate Tags
        print("\n Step 1: Generating tags...")
        
        documents = load_documents(documents_folder)
        if not documents:
            raise ValueError(f"No documents found in {documents_folder}")
        
        tags = generate_tags(documents, top_k=top_k)
        tags_file = os.path.join(output_dir, "tags.json")
        save_tags_to_json(tags, tags_file)
        
        print(f"Generated {sum(len(doc['tags']) for doc in tags)} tags for {len(documents)} documents")
        
        # Step 2: Collect Feedback
        print("\n Step 2: Collecting feedback...")
        
        tags_data = load_tags(tags_file)
        docs_texts = load_docs_for_feedback(documents_folder)
        feedback_results = simulate_feedback(tags_data, docs_texts)
        
        feedback_file = os.path.join(output_dir, "feedback.json")
        save_feedback(feedback_results, feedback_file)
        
        # Calculate approval metrics
        total_tags = sum(len(item["feedback"]) for item in feedback_results)
        approved_tags = sum(len([f for f in item["feedback"] if f["status"] == "approved"]) 
                          for item in feedback_results)
        approval_rate = (approved_tags / total_tags * 100) if total_tags > 0 else 0
        
        print(f"Collected feedback for {total_tags} tags ({approval_rate:.1f}% approved)")
        
        # Step 3: Learn from Feedback
        print("\n Step 3: Learning from feedback...")
        
        feedback_data = load_feedback(feedback_file)
        tag_rates = compute_tag_stats(feedback_data)
        tag_weights = derive_tag_weights(tag_rates)
        
        weights_file = os.path.join(output_dir, "tag_weights.json")
        save_weights(tag_weights, weights_file)
        
        # Calculate learning metrics
        boosted_tags = sum(1 for w in tag_weights.values() if w > 1.0)
        penalized_tags = sum(1 for w in tag_weights.values() if w < 1.0)
        
        print(f"Learned weights for {len(tag_weights)} tags ({boosted_tags} boosted, {penalized_tags} penalized)")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Compile results
        results = {
            "status": "success",
            "total_time": total_time,
            "documents_processed": len(documents),
            "tags_generated": sum(len(doc["tags"]) for doc in tags),
            "approval_rate": approval_rate,
            "tags_learned": len(tag_weights),
            "boosted_tags": boosted_tags,
            "penalized_tags": penalized_tags,
            "output_files": {
                "tags": tags_file,
                "feedback": feedback_file,
                "weights": weights_file
            }
        }
        
        print(f"\nPipeline completed successfully!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Results: {results['documents_processed']} docs → {results['tags_generated']} tags → {approval_rate:.1f}% approved → {results['tags_learned']} learned")
        
        return results
        
    except Exception as e:
        total_time = time.time() - start_time
        error_msg = f"Pipeline failed: {e}"
        
        print(f"\n {error_msg}\n")
        
        return {
            "status": "error",
            "error": error_msg,
            "total_time": total_time
        }

def main():
    """Main CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete Intelligent Text Tagger pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python pipeline.py                           # Run with defaults
            python pipeline.py --documents my_docs      # Custom documents folder
            python pipeline.py --output results         # Custom output directory
            python pipeline.py --top-k 10               # Generate 10 tags per document
        """
    )
    
    parser.add_argument("--documents", type=str, default="documents",
                       help="Path to document folder (default: documents)")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Path to output directory (default: outputs)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of top tags per document (default: 5)")
    
    args = parser.parse_args()
    
    # Run pipeline
    results = run_pipeline(
        documents_folder=args.documents,
        output_dir=args.output,
        top_k=args.top_k
    )
    
    # Return appropriate exit code
    return 0 if results["status"] == "success" else 1

if __name__ == "__main__":
    exit(main())
