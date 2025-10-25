"""
learn_from_feedback.py
-----------------------
Step 3 of Intelligent Text Tagger:
Learns from user feedback to improve future tag suggestions.

Workflow:
1. Load feedback.json (produced in Step 2)
2. Compute approval statistics for each tag across all documents
3. Apply simple learning rules to determine tag weights
4. Save weights to tag_weights.json for use in future TF-IDF runs
"""

import json
import argparse
from collections import defaultdict
from config import LEARNING_WEIGHTS

# -------------------------------------------------------------------------
#  Load feedback file
# -------------------------------------------------------------------------
def load_feedback(feedback_path):
    try:
        with open(feedback_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f" Error loading {feedback_path}: {e}")
        return []


# -------------------------------------------------------------------------
#  Compute global tag approval statistics
# -------------------------------------------------------------------------
def compute_tag_stats(feedback_data):
    stats = defaultdict(lambda: {"approved": 0, "rejected": 0})

    for item in feedback_data:
        for f in item["feedback"]:
            tag = f["tag"].lower()
            if f["status"] == "approved":
                stats[tag]["approved"] += 1
            else:
                stats[tag]["rejected"] += 1

    # Compute approval rates
    tag_rates = {}
    for tag, counts in stats.items():
        total = counts["approved"] + counts["rejected"]
        approval_rate = counts["approved"] / total if total > 0 else 0.0
        tag_rates[tag] = approval_rate

    return tag_rates


# -------------------------------------------------------------------------
#  Apply learning rules -> produce weights
# -------------------------------------------------------------------------
def derive_tag_weights(tag_rates):
    """
    Apply nuanced learning rules with more gradations and avoid zero weights.
    """
    weights = {}
    for tag, rate in tag_rates.items():
        # Apply learning rules from config
        if rate >= LEARNING_WEIGHTS["strong_boost"][0]:
            weights[tag] = LEARNING_WEIGHTS["strong_boost"][1]  # 1.3
        elif rate >= LEARNING_WEIGHTS["mild_boost"][0]:
            weights[tag] = LEARNING_WEIGHTS["mild_boost"][1]    # 1.1
        elif rate >= LEARNING_WEIGHTS["mild_penalty"][0]:
            weights[tag] = LEARNING_WEIGHTS["mild_penalty"][1]  # 0.9
        else:
            weights[tag] = LEARNING_WEIGHTS["strong_penalty"][1]  # 0.5
    
    return weights


# -------------------------------------------------------------------------
#  Save weights to JSON
# -------------------------------------------------------------------------
def save_weights(weights, output_path="tag_weights.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)
    print(f"\n Tag weights saved to {output_path}")


# -------------------------------------------------------------------------
#  Print summary
# -------------------------------------------------------------------------
def print_summary(weights):
    strong_boosted = sum(1 for w in weights.values() if w >= 1.3)
    mild_boosted = sum(1 for w in weights.values() if 1.1 <= w < 1.3)
    mild_penalized = sum(1 for w in weights.values() if 0.9 <= w < 1.1)
    strong_penalized = sum(1 for w in weights.values() if w < 0.9)

    print("\n Feedback Learning Summary")
    print("---------------------------------")
    print(f"Total unique tags: {len(weights)}")
    print(f"Strong Boost (1.3+): {strong_boosted}")
    print(f"Mild Boost (1.1): {mild_boosted}")
    print(f"Mild Penalty (0.9): {mild_penalized}")
    print(f"Strong Penalty (0.5): {strong_penalized}")
    
    # Show some examples
    print("\n Weight Examples:")
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    for tag, weight in sorted_weights[:5]:
        print(f"  {tag}: {weight}")


# -------------------------------------------------------------------------
#  CLI Entrypoint
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn tag weights from feedback.json")
    parser.add_argument("--feedback", type=str, default="feedback.json", help="Path to feedback.json")
    parser.add_argument("--save", action="store_true", help="Save results to tag_weights.json")
    args = parser.parse_args()

    feedback_data = load_feedback(args.feedback)
    if not feedback_data:
        print(" No feedback data found. Please run Step 3 first.")
        exit()

    tag_rates = compute_tag_stats(feedback_data)
    tag_weights = derive_tag_weights(tag_rates)

    print_summary(tag_weights)

    if args.save:
        save_weights(tag_weights)
