# LLM Temporal Stability in Healthcare
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Prompts](https://img.shields.io/badge/prompts-15%20×%203%20models-blueviolet)
![Responses](https://img.shields.io/badge/responses%20collected-1%2C350-green)
![License](https://img.shields.io/badge/licence-MIT-orange)

> *Ask a large language model the same clinical question on a Monday and again on a Friday. Will the answer be the same?*

This repository contains the full data collection, analysis, and clustering pipeline for a study investigating the **temporal stability of Large Language Model responses to healthcare prompts** over a 30-day observation period.

---

## What This Study Does

Most LLM evaluation frameworks measure performance at a single point in time. This study takes a different approach — treating stability as a **temporal property** that must be observed over time, not assumed from a one-off benchmark.

Three large language models were queried with the same 15 healthcare prompts every day for 30 consecutive days. The lexical and semantic similarity between consecutive-day responses was calculated for each model–prompt combination, producing **temporal trajectories** of response stability. Nine stability characteristics were then extracted from these trajectories and used to cluster them into **three distinct temporal stability signatures** — recurring behavioural patterns that reveal how models behave over time, not just how well they perform on any given day.

---

## Models

| Model | Provider |
|---|---|
| GPT-4o-mini | OpenAI |
| Gemini-2.5-Flash | Google DeepMind |
| Claude-Haiku-4-5 | Anthropic |

---

## Dataset

| Metric | Value |
|---|---|
| Observation period | 30 days |
| Healthcare prompts | 15 |
| Models | 3 |
| Total responses collected | 1,350 |
| Consecutive-day comparisons per model | 435 |
| Total paired observations | 1,305 |

Prompts were designed using the healthcare taxonomy from Paruchuri et al. (2025), derived from 11,000 real-world ChatGPT healthcare conversations. Categories covered: Emotional Support, Disease Understanding, Medical Guidance, Symptom Assessment, Medical Findings Interpretation, Lifestyle Changes, and Care Navigation.

---

## Methodology

### Similarity Metrics
- **Lexical similarity** — TF-IDF cosine similarity (`TfidfVectorizer` + `cosine_similarity`, scikit-learn), capturing consistency in response wording
- **Semantic similarity** — `all-MiniLM-L6-v2` sentence transformer embeddings + cosine similarity, capturing consistency in response meaning

### Temporal Trajectories
Each unique model–prompt combination produces a trajectory of 29 consecutive-day similarity values. 45 trajectories in total (15 prompts × 3 models).

### Temporal Stability Characteristics
Nine features were engineered from each trajectory:

| Feature | Description |
|---|---|
| `mean_semantic_similarity` | Average consecutive-day semantic similarity |
| `std_semantic_similarity` | Variability in semantic similarity |
| `mean_lexical_similarity` | Average consecutive-day lexical similarity |
| `semantic_lexical_gap` | Divergence between semantic and lexical scores |
| `mean_absolute_daily_change` | Average day-to-day semantic shift |
| `largest_drop` | Largest single-day semantic decrease |
| `largest_recovery` | Largest single-day semantic increase |
| `daily_change_std` | Variability of day-to-day changes |
| `trend` | Overall directional slope over 30 days |

### Clustering
- K-means clustering (k=3, determined via Elbow method + Silhouette analysis)
- Validated against Ward hierarchical clustering
- Post-hoc Random Forest feature importance for interpretability

---

## Key Findings

**1. Semantic > Lexical stability across all models**
All three models maintained higher semantic than lexical similarity (mean difference = 0.149, W = 26,866, p < 0.001). Models consistently preserved the meaning of responses even as wording changed — but this distinction would have been invisible with a single metric.

**2. Three temporal stability signatures identified**

| Signature | Pattern | Risk Profile |
|---|---|---|
| Signature 1 | High semantic stability, low lexical stability | Meaning preserved; wording consistently changes |
| Signature 2 | High variability, large drops and recoveries | Unpredictable — most concerning for healthcare use |
| Signature 3 | High stability in both dimensions | Most consistent; ideal for high-stakes repeated use |

**3. Stability is a model–prompt property, not a model property**
No signature consisted exclusively of trajectories from one model. The same model exhibited different stability behaviours depending on the prompt — meaning that selecting a "stable model" is not sufficient without also considering prompt type.

---

## Repository Structure

```
├── master_dataset.csv                   # Raw responses collected over 30 days (1,350 rows)
├── lexical_similarity_results.csv       # Lexical similarity scores (1,305 paired observations)
├── semantic_similarity_scores.csv       # Semantic similarity scores (1,305 paired observations)
├── lexical_similarity_scores.ipynb      # Lexical similarity computation (TF-IDF cosine)
├── semantic_similarity_scores.ipynb     # Semantic similarity computation (sentence transformers)
├── Exploratory_Data_Analysis.ipynb      # EDA — response length, distributions, model comparison
├── temporal_stability_clustering.ipynb  # Trajectory construction, feature engineering, clustering, PCA, RF
└── README.md
```
---

## Requirements

```
pandas
numpy
scikit-learn
sentence-transformers
scipy
matplotlib
seaborn
```
---

## Author

**Noor Khilji**  
[github.com/noorfkhilji](https://github.com/noorfkhilji)
