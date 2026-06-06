# Offside: Football Player Goal Prediction Pipeline

An end-to-end, grandmaster-level machine learning pipeline built for the **Offside Football Analytics Datathon** hosted by IEEE Computer Society MUJ. 

The task is to predict the probability that a player will score at least one goal in a given match (`scored_flag`), optimizing for the **Average Precision (AP)** metric.

## 📈 Model Performance & Validation
We used **Stratified 5-Fold Cross Validation** on the 1.3 million row training set to validate model performance:
* **LightGBM Classifier**: Mean Out-of-Fold (OOF) AP = **0.4090**
* **CatBoost Classifier**: Mean Out-of-Fold (OOF) AP = **0.4498**
* **Final Ensemble**: 100% CatBoost selected by OOF blending grid search.

---

## 📁 Repository Structure
```text
├── src/
│   ├── eda.py          # Exploratory Data Analysis & plotting functions
│   ├── features.py     # Feature engineering & out-of-fold target encoding
│   ├── models.py       # LightGBM, CatBoost model training & Optuna tuning
│   └── explain.py      # SHAP analysis and feature importance plotting
├── plots/              # Visualizations (Target distribution, SHAP summary, feature importances)
├── main.py             # End-to-end pipeline orchestrator
├── notebook.ipynb      # Step-by-step Jupyter Notebook wrapper
└── unzip_and_run.py    # Auto-extraction & execution script
```

---

## 🛠️ Feature Engineering & Preprocessing
* **Temporal Context**: Parses match dates into year, month, day of week, day of year, and weekend flags.
* **Football Domain Ratios**: Scales player average expected goals (`avg_xG`) and expected assists (`avg_xA`) by match minutes played, and computes international caps scoring efficiency.
* **Robust Target Encoding**: Performs out-of-fold target encoding for players (`name_y`) and clubs (`home_club_name`, `away_club_name`) to prevent data leakage.
* **Missing Value Indicators**: Creates binary flags for high-missingness columns and imputes using median values.

---

## 🚀 Getting Started & Execution

1. **Set up the virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/env/activate  # On macOS/Linux
   pip install pandas numpy scikit-learn lightgbm catboost optuna shap matplotlib seaborn notebook
   ```

2. **Download dataset**:
   Download the competition files (`train.csv`, `test.csv`, etc.) from [Kaggle](https://www.kaggle.com/competitions/offside-data-thon/data) and place them in the root of the project.

3. **Run the pipeline**:
   * **Via CLI**:
     ```bash
     python main.py
     ```
   * **Via Jupyter**: Open `notebook.ipynb` to step through the code and visualize plots interactively.
