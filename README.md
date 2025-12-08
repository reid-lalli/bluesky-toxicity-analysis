# Bluesky Toxicity Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and analysis for studying the evolution of toxicity on Bluesky social media platform from its launch (February 6, 2024) through December 1, 2025. Our research examines how toxicity patterns develop over time and how they relate to platform growth.

**Paper:** "The Evolution of Toxicity on Social Media: Analyzing Bluesky from its Inception"  
**Authors:** Reid Lalli, Luke McAdams, Ethan Schoen (Occidental College)

## Key Features

- Automated scraping of 60,247+ posts over 665 days
- Uses `unitary/unbiased-toxic-roberta` transformer model
- Analyzes using ARIMA modeling, cohort analysis, and sensitivity testing
- Visualizes time series plots, correlation matrices, and event impact analysis
- Retry logic for API failures, graceful degradation

## Installation

### Prerequisites
- Python 3.11 or higher
- 8GB+ RAM (for transformer model)
- GPU recommended (optional, for faster inference)

### Setup

```bash
# Clone the repository
git clone https://github.com/[your-username]/bluesky-toxicity-analysis.git
cd bluesky-toxicity-analysis

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional, can also edit config in script)
export BLUESKY_USERNAME="your_username.bsky.social"
export BLUESKY_PASSWORD="your_password"
export PERSPECTIVE_API_KEY="your_api_key"  # Optional for comparison analysis
```

## Usage

### Basic Analysis

```bash
python hackathon_playground_final.py
```

The script will:
1. Authenticate with Bluesky API
2. Collect posts from Feb 6, 2024 to Dec 1, 2025 (configurable)
3. Classify toxicity using ML model
4. Generate 8+ visualizations and reports
5. Export results to CSV

### Configuration

Edit the configuration section in `hackathon_playground_final.py`:

```python
# Configuration
OUTPUT_DIR = r'C:\Users\reide\Downloads'  # Change output directory
START_DATE = datetime.date(2024, 2, 6)    # Platform launch
END_DATE = datetime.date(2025, 12, 1)     # Analysis end date
BATCH_SIZE = 64                            # Posts per ML batch
MAX_WORKERS = 10                           # Parallel API workers
ROLLING_WINDOW = 7                         # Days for smoothing trends
```

### Output Files

The script generates the following files in `OUTPUT_DIR`:

**Visualizations (PNG, 300 DPI):**
- `daily_toxicity_trend.png` - Time series with rolling averages and event markers
- `sensitivity_analysis.png` - Comparison across toxicity thresholds (k=0.5, 0.75, 0.9)
- `arima_forecast.png` - ARIMA diagnostics and predictions
- `platform_growth_analysis.png` - Correlation between growth and toxicity
- `cohort_analysis.png` - User cohort behavior patterns
- `event_impact_analysis.png` - Before/after event comparisons
- `method_comparison.png` - RoBERTa vs Perspective API (if enabled)

**Data Files:**
- `toxicity_results.csv` - Complete dataset with scores and metadata
- `theoretical_framework.txt` - Academic interpretation and theory

## Methodology

### Data Collection
- **Source:** Bluesky public API (`app.bsky.feed.searchPosts`)
- **Sampling:** Top 100 English posts per day
- **Period:** 665 days (Feb 6, 2024 - Dec 1, 2025)
- **Total:** 60,247 posts

### Toxicity Classification
- **Model:** `unitary/unbiased-toxic-roberta` (BERT-based)
- **Threshold:** 0.5 probability score (with sensitivity analysis at 0.75, 0.9)
- **Batch Processing:** 64 posts/batch for efficiency

### Statistical Methods
- **Time Series:** 7-day rolling averages, ARIMA(1,1,1) modeling
- **Correlation:** Pearson and Spearman tests for growth relationship
- **Hypothesis Testing:** Mann-Whitney U tests for event impacts
- **Robustness:** Multi-threshold sensitivity analysis, cohort segmentation

## Project Structure

```
bluesky-toxicity-analysis/
├── hackathon_playground_final.py   # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── bluesky_toxicity_paper.tex     # Research paper (LaTeX)
├── references.bib                  # Bibliography
└── outputs/                        # Generated visualizations (not tracked)
    ├── *.png
    ├── toxicity_results.csv
    └── theoretical_framework.txt
```

## Research Questions

**RQ1:** How has the level of toxicity on Bluesky changed over time?  
**RQ2:** Does platform population growth correlate with toxicity changes?

**Hypotheses:**
- H₀: No temporal change in toxicity (slope = 0)
- Hₐ: Toxicity changes over time (slope ≠ 0)

## Key Findings

[To be filled after analysis runs - add your specific results here]

## Dependencies

Core libraries:
- `requests` - API communication
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` / `seaborn` - Visualization
- `transformers` / `torch` - ML model
- `statsmodels` - Time series analysis
- `scipy` / `scikit-learn` - Statistical tests

See `requirements.txt` for complete list with versions.

## Contributing

This is a research project. For questions or collaboration inquiries:
- Email: rlalli@oxy.edu
- Bluesky: @sds-hackathon.bsky.social

## License

MIT License - See LICENSE file for details

```bibtex
@misc{lalli2025bluesky,
  author = {Lalli, Reid and McAdams, Luke and Schoen, Ethan},
  title = {The Evolution of Toxicity on Social Media: Analyzing Bluesky from its Inception},
  year = {2025},
  institution = {Occidental College},
  url = {https://github.com/[your-username]/bluesky-toxicity-analysis}
}
```

## Acknowledgments

- Bluesky team for maintaining an open API
- Hugging Face for the `unitary/unbiased-toxic-roberta` model
- Oxy Swim & Dive team for always being there to support me :)
- Meiqing Zhang, Justin Li, and the Occidental Computer Science program for making this available

## Data Availability

Due to Bluesky's Terms of Service, raw post content is not shared. Aggregated daily statistics are available in our companion data repository:  
- https://github.com/reid-lalli/bluesky-toxicity-data

---

**Note:** This research analyzes public posts only and does not store personally identifiable information. All data collection complies with Bluesky's API terms and research ethics standards.
