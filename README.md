# Customer Feedback & Sentiment Analysis System

A comprehensive feedback intelligence and sentiment analysis system built on the **Amazon Reviews 2023** dataset (McAuley Lab).

## Project Overview

This project analyzes customer reviews across two primary Amazon product domains:
1. **Electronics**
2. **Fashion** (specifically mapped to Amazon's canonical category **Clothing, Shoes & Jewelry**)

### Dataset Verification

From McAuley Lab's Hugging Face dataset ([`McAuley-Lab/Amazon-Reviews-2023`](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)):

| Domain | Canonical Amazon Category | Exact Review Config Name | Exact Metadata Config Name |
| :--- | :--- | :--- | :--- |
| **Electronics** | Electronics | `raw_review_Electronics` | `raw_meta_Electronics` |
| **Fashion** | Clothing, Shoes & Jewelry | `raw_review_Clothing_Shoes_and_Jewelry` | `raw_meta_Clothing_Shoes_and_Jewelry` |

> **Note on Fashion:** In the Amazon Reviews 2023 dataset, the comprehensive fashion category is named `Clothing_Shoes_and_Jewelry` (rather than generic `Fashion` or `Amazon_Fashion`), containing over 23M ratings and 715k items.

---

## Directory Structure

```text
FeedbackIQ_amazon reviews/
├── data/
│   ├── raw/
│   │   ├── electronics/           # raw_sample.parquet (50,000 reviews)
│   │   └── fashion/               # raw_sample.parquet (50,000 reviews)
│   └── processed/
│       ├── electronics_processed.parquet  # 46,191 clean reviews
│       ├── fashion_processed.parquet      # 46,471 clean reviews
│       └── combined_processed.parquet     # 92,662 clean reviews
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configurations, paths, schema, cleaning params
│   ├── data_loader.py             # Streaming pipeline (load, sample, save)
│   └── preprocessing.py          # Cleaning, deduplication, length capping, sentiment
├── models/
├── app/
├── tests/
│   ├── __init__.py
│   └── test_dataset_loading.py    # Unit tests for loading, cleaning & sentiment
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+ (tested with Python 3.11)
- VS Code or compatible editor

### 2. Virtual Environment Setup

Create and activate a virtual environment:

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

Install the core data loading and preprocessing dependencies:
```bash
pip install -r requirements.txt
```

Installed packages:
- `datasets>=2.20.0,<3.0.0` (Hugging Face)
- `pandas`
- `pyarrow`
- `numpy`
- `jupyter`

---

## Pipelines & Usage

### 1. Configuration
Adjust parameters in [`src/config.py`](src/config.py):
```python
SAMPLE_SIZE = 50000
MIN_REVIEW_WORDS = 3
MIN_REVIEW_CHARS = 10
MAX_REVIEW_LENGTH = 1000

CATEGORY_CONFIG = {
    "Electronics": "raw_review_Electronics",
    "Fashion": "raw_review_Clothing_Shoes_and_Jewelry"
}
```

### 2. Data Loading Pipeline
Streams reviews without pulling multi-gigabyte files into memory, retaining essential schema fields and persisting raw samples:
```bash
python src/data_loader.py
```

### 3. Review Preprocessing Pipeline
Cleans text (strips HTML, unescapes entities, normalizes whitespace, caps length, preserves punctuation and stopwords), handles missing data, removes empty/short reviews, removes duplicates, and generates sentiment labels (`1-2` negative, `3` neutral, `4-5` positive):
```bash
python src/preprocessing.py
```

### 4. Running Test Suite
Execute unit tests verifying configuration, live streaming, cleaning, and sentiment mapping:
```bash
python -m unittest discover -s tests
```

---

## Next Steps
- Exploratory data analysis (EDA) in `notebooks/`
- Sentiment classification modeling (VADER, RoBERTa, or fine-tuned Transformer)
- RAG-assisted customer feedback retrieval
- API service (`app/`) and user-facing dashboards
