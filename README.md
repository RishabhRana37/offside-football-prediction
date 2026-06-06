# Offside: Football Player Goal Prediction Pipeline

An end-to-end, state-of-the-art machine learning pipeline built for the **Offside Football Analytics Datathon** hosted by IEEE Computer Society MUJ. 

The task is to predict the probability that a football player will score at least one goal in a given appearance (`scored_flag`), optimizing for the **Average Precision (AP)** metric.

---

## 🏆 Model Performance & Validation
We evaluate model performance using **Stratified 5-Fold Cross Validation** on the training dataset:

| Model Configuration | Validation Setup | Mean Out-of-Fold (OOF) AP |
| :--- | :--- | :--- |
| **LightGBM Classifier (Baseline)** | 5-Fold CV | 0.4090 |
| **CatBoost Classifier (Baseline)** | 5-Fold CV | 0.4498 |
| **CatBoost Classifier (Champion Model)** | 5-Fold CV + Smooth Target Encoding + Post-Proc | **0.4619** |

### Champion Model Features:
1. **Football Domain Ratios**: Scales expected stats by match minutes ratio (e.g., `expected_xG_in_match`, `expected_xA_in_match`, `xG_per_shot`) and international cap goal scoring efficiency.
2. **Smoothed Out-of-Fold Target Encoding**: Target encodes player (`name_y`), home/away clubs (`home_club_name`, `away_club_name`), competition types, and referees with a smoothing factor ($m=20$) to prevent data leakage.
3. **Domain Post-Processing**: Automatically forces predictions to `0.0` for any appearances where the player logged exactly `0` minutes of playing time.

---

## 📂 Repository Structure

```text
├── src/
│   ├── eda.py               # Exploratory Data Analysis plotting functions
│   ├── features.py          # Preprocessing pipeline & target encoding utilities
│   ├── models.py            # Optuna tuning, LightGBM, and CatBoost trainers
│   └── explain.py           # SHAP analysis and feature importance plotting
├── notebooks/
│   └── offside_eda_and_modeling.ipynb  # Jupyter notebook walk-through of the ML pipeline
├── plots/                   # Saved EDA and explainability charts
│   ├── target_distribution.png
│   ├── missing_values_train.png
│   ├── scoring_probability_by_position.png
│   ├── feature_importance_plot.png
│   └── shap_summary_plot.png
├── solution.csv             # Final champion model prediction results (19.8 MB)
├── data_dictionary.csv      # Features glossary mapping file
├── feature_catalog.csv      # Engineered features catalog listing
├── app.py                   # Streamlit web application dashboard for player predictions
├── main.py                  # Orchestrator script for end-to-end pipeline run
└── improve_and_submit.py    # Training execution script for the 0.4619 AP champion model
```

---

## 🚀 Getting Started & Setup

### 1. Environment and Dependencies
Set up your virtual environment and install the required libraries:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### 2. Datasets
Download the Datathon competition files from [Kaggle](https://www.kaggle.com/competitions/offside-data-thon/data).
Place the raw files (`train.csv`, `test.csv`) in the repository root directory.

### 3. Pipeline Execution
To train the champion model and generate `solution.csv`:
```bash
python improve_and_submit.py
```

To run the interactive Streamlit player profiling application:
```bash
streamlit run app.py
```
