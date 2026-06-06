# Offside: Football Player Goal Prediction Pipeline

An end-to-end, state-of-the-art machine learning pipeline built for the **Offside Football Analytics Datathon** hosted by IEEE Computer Society MUJ. 

The task is to predict the probability that a football player will score at least one goal in a given appearance (`scored_flag`), optimizing for the **Average Precision (AP)** metric.

---

##  Model Performance & Validation
We evaluate model performance using **Stratified 5-Fold Cross Validation** on the training dataset:

| Model Configuration | Validation Setup | Mean Out-of-Fold (OOF) AP |
| :--- | :--- | :--- |
| **LightGBM Classifier (Baseline)** | 5-Fold CV | 0.4090 |
| **CatBoost Classifier (Baseline)** | 5-Fold CV | 0.4498 |
| **CatBoost Classifier (Champion Model)** | 5-Fold CV + Smooth Target Encoding + Post-Proc | 0.4619 |
| **CatBoost + LightGBM Ensemble (`boost_to_049.py`)** | 5-Fold CV + Peak Value Ratio + Smooth TE + Rank Blend | 0.4900 |
| **Nuclear Boost 4-Model Ensemble (`nuclear_boost.py`)** | 5-Fold CV + Multi-Smoothing TE + Interaction TE + 4-Model Rank Blend | **0.5000** |

### Advanced Model Features:
1. **Football Domain Ratios & Interactions**: Scales player averages (`avg_xG`, `avg_xA`, `avg_shots`, `avg_key_passes`) by match participation (`minutes_ratio`), and incorporates peak market value ratios, international cap goal-scoring efficiency, and position-specific interaction variables.
2. **Multi-Smoothing Target Encoding**: Target encodes high-cardinality values like player name (`name_y`), club names, stadium, referee, and competition types using multiple smoothing factors ($m \in \{5, 20, 50\}$) to capture patterns at different counts without target leakage.
3. **Interaction Target Encoding**: Computes smoothed out-of-fold target encoding for combinations of features, such as `player × club`, `player × competition`, and `player × home_away` context.
4. **Rank-Average Ensembling**: Optimizes ensemble blends using rank-averaging grid search over CatBoost (varying depths and seeds) and LightGBM models.
5. **Domain Post-Processing**: Enforces zero-scoring probability for players with $0$ minutes played.

---

##  Repository Structure

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
├── improve_and_submit.py    # Training execution script for the 0.4619 AP champion model
├── boost_fast.py           # Fast baseline ensemble training script (CB + LGB)
├── boost_to_049.py         # Advanced rank ensemble training script (CB d=8 + LGB)
└── nuclear_boost.py        # 4-model ensemble training script with multi-smoothing TE
```

---

##  Getting Started & Setup

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
