# Intelligent Text Tagger

A machine learning-powered document tagging system that learns from user feedback to improve tag suggestions over time.

## Overview

The Intelligent Text Tagger generates descriptive tags for documents, collects feedback on tag quality, and learns from that feedback to improve future tag suggestions. The system uses TF-IDF for tag generation and implements a feedback-based learning loop.

### Features

- **Automatic Tag Generation**: Generate tags using TF-IDF vectorization
- **Feedback Collection**: Collect multi-signal feedback on tag quality
- **Learning System**: Improve tag suggestions based on feedback patterns
- **Web Interface**: Interactive Streamlit-based web application
- **CLI Support**: Run the complete pipeline from command line
- **Test Coverage**: Comprehensive tests for all components

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Intelligent-Text-Tagger
```

### Step 2: Create Virtual Environment

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download NLTK Data

The system requires NLTK data for text processing. Download it with:

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Step 5: Prepare Documents

Place your text documents in the `documents/` folder. The system accepts `.txt` and `.md` files.

**Example:**

```
documents/
├── document1.txt
├── document2.txt
└── document3.md
```

---

## Usage

### Option 1: Web Interface (Recommended)

Launch the Streamlit web application:

```bash
python run_web.py
```

This will:

- Start a local Streamlit server
- Open your browser to `http://localhost:8501`
- Provide an interactive interface for all operations

**Features:**

- Upload and manage documents
- Generate tags with customizable parameters
- Collect feedback on tag quality
- Learn from feedback to improve future tags
- Run complete pipeline in one click
- View detailed statistics and results

### Option 2: Command Line Interface

Run the complete pipeline from command line:

```bash
python pipeline.py
```

This will:

1. Generate tags for all documents in `documents/` folder
2. Collect feedback on the generated tags
3. Learn from feedback to derive tag weights
4. Save results to `outputs/` folder

**With Options:**

```bash
# Specify different parameters
python pipeline.py --top-k 10 --documents-folder documents
```

**Available options:**

- `--top-k`: Number of tags to generate per document (default: 5)
- `--documents-folder`: Path to documents folder (default: documents)
- `--output-dir`: Path to output directory (default: outputs)

### Output Files

All results are saved in the `outputs/` folder:

- `outputs/tags.json`: Generated tags with scores
- `outputs/feedback.json`: Feedback results for each tag
- `outputs/tag_weights.json`: Learned tag weights for improvement

---

## Design Decisions

### 1. Core Architecture

- Three-step pipeline for clear separation of concerns
- Modular design for easy testing and maintenance
- Clear data flow: Generate → Collect → Learn

### 2. Tag Generation Approach

- **Why TF-IDF**: Fast, deterministic baseline without external dependencies
- **Weighting**: Learned weights applied to improve future generations
- **Future**: Extensible to LLM/vector embeddings (see `SYSTEM_EXTENSION_PROPOSAL.md`)

### 3. Feedback Collection Strategy

- **Multi-signal scoring**: Combines TF-IDF, frequency, and position signals
- **Weight distribution**: 50% TF-IDF, 20% frequency, 30% position
- **Simulated feedback**: Rule-based approach for demonstration (easily replaceable with real user feedback)

### 4. Learning Mechanism

- **Rule-based approach**: Transparent and interpretable learning rules
- **Four categories**: Strong boost (≥80%), Mild boost (50-79%), Mild penalty (20-49%), Strong penalty (<20%)
- **Weight application**: Learned weights improve future tag generation

### 5. State Management

- **Dual persistence**: Session state for speed + file system for persistence
- **Smart reloading**: Timestamp-based reload to stay current
- **Benefits**: Works across tabs and survives page refreshes

### 6. Web Interface Design

- **Streamlit**: Rapid development with built-in UI components
- **Component-based**: Modular architecture for maintainability
- **Session awareness**: Visual indicators show data availability

---

## How Feedback Improves the System

### The Learning Loop

The system continuously learns from feedback to improve tag quality over time.

**First Run:**

1. System generates tags using TF-IDF (statistical word importance)
2. Some tags are relevant, others are not
3. Feedback collects approval/rejection for each tag

**Learning Phase:**

4. System analyzes which tags were approved/rejected
5. High approval rate (>80%) → Tag gets boosted (weight > 1.0)
6. Low approval rate (<20%) → Tag gets penalized (weight < 1.0)

**Second Run:**

7. System uses learned weights when generating tags
8. Boosted tags appear more often, penalized tags appear less often
9. Results improve because the system learned what works

### Example

**Before Learning:**

- Document: "Introduction to machine learning algorithms and neural networks..."
- Tags: "machine", "learning", "algorithms", "neural", "networks"

**After Learning (feedback showed "machine learning" and "neural networks" were approved):**

- Same document now generates: "machine learning", "neural networks", "algorithms"
- Quality improved because the system learned compound terms score better

### Key Insight

The more feedback you provide, the better the system gets at understanding what makes a good tag for your specific domain and use case.

---

## Testing

Run the test suite to verify the installation:

```bash
pytest tests/ -v
```

---

## Project Structure

```
Intelligent-Text-Tagger/
├── documents/          # Input documents (add your files here)
├── outputs/            # Generated results (tags, feedback, weights)
├── src/               # Source code
│   ├── generate_tags.py
│   ├── collect_feedback.py
│   └── learn_from_feedback.py
├── web/               # Web application
│   ├── app.py         # Main Streamlit app
│   ├── components/    # UI components
│   └── utils/         # Utilities
├── tests/             # Test files
├── config.py          # Configuration parameters
├── pipeline.py        # CLI pipeline
├── run_web.py         # Web app launcher
└── requirements.txt   # Python dependencies
```

---

## Configuration

Adjust system parameters in `config.py`:

- **TFIDF_WEIGHT**: Weight for TF-IDF scores (default: 0.5)
- **FREQUENCY_WEIGHT**: Weight for frequency scores (default: 0.2)
- **POSITION_WEIGHT**: Weight for position scores (default: 0.3)
- **APPROVAL_THRESHOLD**: Threshold for tag approval (default: 0.6)
- **LEARNING_WEIGHTS**: Learning rules for tag weights

---

## Sample Input/Output

### Input Document

```
documents/meeting_notes_2025_10_21.txt

Team Meeting - October 21, 2025

Agenda:
- Review Q4 roadmap progress
- Discuss LLM evaluation automation
- Plan next sprint deliverables

Key Points:
- QA automation reduced manual review time by 90%.
- Dockerized LLM pipeline scaled to 5K+ users.
- New Supabase dashboard in testing phase.

Next Steps:
- Finalize evaluation metrics.
- Prepare internal demo for Halluminate leadership.
```

### Generated Tags (Output)

**File: `outputs/tags.json`**

```json
{
  "filename": "meeting_notes_2025_10_21.txt",
  "tags": [
    { "tag": "automation", "tfidf_score": 0.295, "adjusted_tfidf_score": 0.384 },
    { "tag": "evaluation", "tfidf_score": 0.295, "adjusted_tfidf_score": 0.384 },
    { "tag": "llm", "tfidf_score": 0.295, "adjusted_tfidf_score": 0.384 },
    { "tag": "review", "tfidf_score": 0.295, "adjusted_tfidf_score": 0.384 },
    { "tag": "discus", "tfidf_score": 0.147, "adjusted_tfidf_score": 0.147 }
  ]
}
```

### Feedback Results (Output)

**File: `outputs/feedback.json`**

```json
{
  "filename": "meeting_notes_2025_10_21.txt",
  "feedback": [
    { "tag": "automation", "status": "approved", "relevance_score": 0.407 },
    { "tag": "evaluation", "status": "approved", "relevance_score": 0.407 },
    { "tag": "llm", "status": "approved", "relevance_score": 0.407 },
    { "tag": "review", "status": "approved", "relevance_score": 0.407 },
    { "tag": "discus", "status": "rejected", "relevance_score": 0.254 }
  ]
}
```

### Learned Weights (Output)

**File: `outputs/tag_weights.json`**

```json
{
  "automation": 1.3,
  "evaluation": 1.3,
  "llm": 1.3,
  "review": 1.3,
  "discus": 0.5
}
```

**Interpretation:**

- Scores > 1.0 mean the tag was approved frequently (gets boosted in future runs)
- Score of 1.3 = strong boost (approval rate ≥80%)

---

## Next Steps

- See `SYSTEM_EXTENSION_PROPOSAL.md` for future extension ideas
- Explore the web interface for interactive features
- Review test files to understand the system behavior
- Modify `config.py` to tune the learning parameters
