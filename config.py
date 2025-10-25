"""
config.py
---------
Configuration module for Intelligent Text Tagger.
Centralizes all tunable parameters for easy experimentation and future domain-specific customization.
"""

# Signal weights for multi-signal scoring
TFIDF_WEIGHT = 0.5
FREQUENCY_WEIGHT = 0.2
POSITION_WEIGHT = 0.3

# Feedback threshold for approval/rejection
APPROVAL_THRESHOLD = 0.35

# Learning weights based on approval rates
LEARNING_WEIGHTS = {
    "strong_boost": (0.80, 1.3),    # >=80% approval: strong boost
    "mild_boost": (0.50, 1.1),     # 50-79% approval: mild boost
    "mild_penalty": (0.20, 0.9),   # 20-49% approval: mild penalty
    "strong_penalty": (0.00, 0.5)  # <20% approval: strong penalty
}

# Position scoring weights
POSITION_SCORES = {
    "title": 1.0,        # First line (title)
    "header": 0.8,       # Headers (lines with #, all caps, etc.)
    "first_paragraph": 0.6,  # First paragraph
    "body": 0.4,         # Body text
    "not_found": 0.0     # Not found
}

# Future: document-type specific configs
DOCUMENT_TYPE_CONFIGS = {
    # Example structure for future implementation:
    # "meeting_notes": {
    #     "tfidf_weight": 0.6,
    #     "frequency_weight": 0.2,
    #     "position_weight": 0.2,
    #     "approval_threshold": 0.3
    # },
    # "support_tickets": {
    #     "tfidf_weight": 0.4,
    #     "frequency_weight": 0.4,
    #     "position_weight": 0.2,
    #     "approval_threshold": 0.4
    # }
}
