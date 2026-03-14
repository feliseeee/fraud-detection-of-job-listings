# Fraudulent Job Ad Detection
### GovTech Anti-Scam Products Team (GASP) — Take-Home Assessment

---

## Overview

This project develops an automated pipeline to detect and block fraudulent job advertisements before they reach users. Working with a real-world dataset of 17,880 job postings, the solution progresses through exploratory data analysis, classical NLP modelling, deep learning, and LLM-based reasoning — culminating in a two-stage production-ready architecture recommended for GASP deployment.

---

## Repository Structure

```
├── data/
│   └── DataSet.csv 
│   └── bert_test_predictions.csv     # BERT test set predictions
├── model/
│   └── bert/                         # Local bert-base-uncased checkpoint
├── outputs/
│   └── bert-fraud/                   # BERT training checkpoints
├── eda.ipynb                         # Exploratory data analysis
├── nlp_text_processing.ipynb         # Classical NLP models (LR, RF, XGBoost)
├── deep_learning_method.ipynb        # Fine-tuned BERT classifier
├── llm_fraud_2.ipynb                 # LLM-based zero-shot detection (Gemma)
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Total records | 17,880 job postings |
| Features | 18 columns (text + structured metadata) |
| Fraudulent postings | 866 (4.84%) |
| Legitimate postings | 17,014 (95.16%) |
| Key text fields | title, description, requirements, benefits, company_profile |
| Key metadata fields | industry, employment_type, salary_range, has_company_logo, has_questions |

**Key challenge:** Severe class imbalance (95/5 split) — a naive model predicting "legitimate" every time achieves 95% accuracy while catching zero scams.

---

## Notebooks

### 1. `eda.ipynb` — Exploratory Data Analysis

Investigates the structure, quality, and fraud signals in the dataset.

**Key findings:**
- Fraudulent posts have ~3× shorter company profiles (224 vs 622 chars avg)
- Missing company logo and absence of screening questions correlate with fraud
- Part-time postings carry ~9.3% fraud rate vs 4.2% for full-time
- Oil & Energy (~38%) and Accounting (~36%) show elevated fraud rates
- Fraudulent benefits language clusters around vague phrases: *"work-life balance"*, *"online training"*, *"data entry"*
- Duplicate postings are an **anti-fraud** signal — only 8 of 275 duplicates were fraudulent
- Missing salary range is **not** a reliable fraud signal — 84% of all postings lack salary info

---

### 2. `nlp_text_processing.ipynb` — Classical NLP Models

Builds a full feature engineering pipeline and trains three classifiers.

**Pipeline:**
```
Raw text → HTML clean → normalise → combine fields
       → TF-IDF (5,000 features, 1–2 grams)
       → + 8 structured features (logo, questions, industry, employment type, etc.)
       → 5,008 total features
       → stratified 80/20 train/test split
       → class imbalance handling
       → model training + evaluation
```

**Models trained:**

| Model | Imbalance Handling | Recall | Precision | F1 | AUC |
|---|---|---|---|---|---|
| Logistic Regression | `class_weight='balanced'` | 84.3% | 45.3% | 58.9% | 97.1% |
| Random Forest | `class_weight='balanced'`, 100 estimators | 51.7% | 96.7% | 67.4% | 97.5% |
| **XGBoost** | `scale_pos_weight ≈ 19:1` | **72.1%** | **79.5%** | **75.6%** | **97.3%** |

**Why XGBoost was selected:** Best balance of recall and precision. Logistic Regression's high recall is offset by a false alarm rate of nearly 1 in 2 flagged posts, making it operationally unsustainable. XGBoost's F1 of 75.6% is the strongest overall.

---

### 3. `deep_learning_method.ipynb` — Fine-tuned BERT

Fine-tunes `bert-base-uncased` for sequence classification using HuggingFace Transformers.

**Architecture:**
```
Job posting text (title + company profile + description + requirements)
       → BERT tokeniser (WordPiece, max 256 tokens)
       → [CLS] token + 12 transformer encoder layers (768 hidden dims, 12 attention heads)
       → [CLS] representation (768-dim vector)
       → Classification head (Linear 768 → 2, softmax)
       → Fraud / Legitimate
```

**Training configuration:**

| Parameter | Value |
|---|---|
| Base model | `bert-base-uncased` |
| Max token length | 256 |
| Learning rate | 2e-5 |
| Epochs | 1 |
| Train batch size | 8 |
| Weight decay | 0.01 |
| Best model metric | F1 |
| Split | 70% train / 15% val / 15% test (stratified) |

**Results:**

| Metric | Score |
|---|---|
| Recall | 90.0% |
| Precision | 74.5% |
| F1 | 81.5% |
| ROC-AUC | 98.5% |
| Accuracy | 98.0% |

BERT significantly outperforms all classical models by understanding posting language contextually — detecting suspicious patterns that only emerge from reading the full text, not just keyword frequencies.

---

### 4. `llm_fraud_2.ipynb` — LLM Zero-Shot Detection (Gemma)

Uses a locally-run quantised LLM as a zero-shot fraud reasoner — no training on labelled data required.

**Model:** Gemma 3 4B (`gemma-3-4b-it-q4_0.gguf`) via `llama_cpp`

**Approach:**
```
Job posting fields (truncated to 1,500 chars each)
       → Structured prompt with 8 fraud indicators from EDA
       → Gemma 3 4B (4-bit quantised, CPU, 8,192 context)
       → JSON output: fraud_score + red_flags + reasoning
       → is_fraud label (score ≥ 0.5)
```

**Prompt design** embeds EDA-derived fraud indicators:
1. Excessive promotional language
2. Vague claims without specifics
3. URL placeholders instead of real links
4. Missing or generic contact information
5. Unrealistic salary promises
6. Generic benefits descriptions
7. Overemphasis on soft skills without technical requirements
8. Geographical appeal without verifiable details

**Output example:**
```json
{
  "fraud_score": 0.75,
  "red_flags": [
    "Excessive promotional language ('dynamic startup', 'fast release cycles')",
    "Vague claims without specifics (diverse team from top companies, no names)",
    "Missing contact information"
  ],
  "reasoning": "The posting relies heavily on buzzwords without concrete details..."
}
```

**Note:** Currently tested on 10 sampled postings — formal metrics (Recall, F1, AUC) pending full evaluation run.

---

## Recommended Production Architecture

A two-stage pipeline combining XGBoost's speed with the LLM's reasoning capability:

```
Incoming job posting
        ↓
   [Stage 1] XGBoost
   Fast real-time scoring
        ↓
  ┌─────┴──────────┐
High confidence    Borderline / flagged
  legitimate              ↓
      ↓           [Stage 2] Gemma LLM
  Auto-approve    Contextual reasoning
                  + red flags + explanation
                          ↓
                   Human reviewer queue
                          ↓
                   Approve / Block
```

**Why this design:**
- XGBoost handles bulk volume instantly on CPU — no GPU required
- Gemma only invoked on borderline cases — keeps latency and cost low
- LLM output provides a documented audit trail for every blocked posting
- Zero-shot LLM generalises to novel scam patterns without retraining
- Human review remains in the loop at pilot stage

---

## Evaluation Strategy

**Primary metric: Recall** — missing a scam is costlier than over-flagging. A missed fraud exposes real job seekers to harm.

**Secondary metric: F1-score** — prevents optimising recall by flagging everything. Ensures the system remains operationally viable for reviewers.

**Additional metrics:** Precision, ROC-AUC (threshold flexibility), confusion matrix.

**Validation approach:** Stratified train/test splits preserving the 4.84% fraud ratio. Recommended for production readiness: 5-fold stratified cross-validation + time-based holdout to assess temporal drift.

---

## Setup & Dependencies

```bash
pip install pandas numpy scikit-learn xgboost transformers datasets
pip install beautifulsoup4 nltk tqdm scipy
pip install llama-cpp-python  # for LLM notebook
```

**For BERT training (GPU recommended):**
```bash
pip install torch torchvision torchaudio
```

**Models required (not included in repo):**
- `model/bert/` — download `bert-base-uncased` from HuggingFace
- `model/gemma-3-4b-it-q4_0.gguf` — download quantised Gemma 3 4B from HuggingFace

---

## Key Limitations & Future Work

- **Model drift:** All models trained on a static dataset. Periodic retraining on newly confirmed fraud cases is essential as scam tactics evolve
- **LLM evaluation:** Gemma was only tested on 10 postings — a full evaluation run with formal metrics is needed before production consideration
- **GPU dependency:** BERT requires GPU infrastructure for production-scale throughput
- **Label encoding:** Nominal categorical features (industry, function) were label-encoded, imposing false ordinal relationships — one-hot or target encoding recommended for future iterations
- **Threshold tuning:** The XGBoost decision threshold should be tuned on a validation set to optimise the recall/precision trade-off for GASP's specific risk tolerance

---

## Results Summary

| Approach | Recall | F1 | Infrastructure | Explainable |
|---|---|---|---|---|
| Logistic Regression | 84.3% | 58.9% | CPU | No |
| Random Forest | 51.7% | 67.4% | CPU | Partial |
| XGBoost | 72.1% | 75.6% | CPU | Partial |
| BERT | 90.0% | 81.5% | GPU | No |
| Gemma LLM | TBD | TBD | CPU | Yes |


