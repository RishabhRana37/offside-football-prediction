# ⚽ Offside — Football Player Goal Prediction

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/CatBoost-FF6600?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-02A8A8?style=for-the-badge&logoColor=white"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/IEEE%20CS%20MUJ-Datathon-00629B?style=for-the-badge&logo=ieee&logoColor=white"/>
</p>

End-to-end ML pipeline built for the **Offside Football Analytics Datathon** by **IEEE Computer Society, MUJ** — my first datathon.

**Task:** Predict the probability a player scores at least one goal (`scored_flag`) on a 1.3M row dataset, optimized for **Average Precision (AP)**.

---

## Results

| Model | OOF Average Precision |
|---|---|
| LightGBM | 0.4090 |
| CatBoost | **0.4498** ✅ |
| Final | 100% CatBoost via OOF blending |

Validated using **Stratified 5-Fold Cross Validation**.

---

## Key Engineering Decisions

- **OOF Target Encoding** for player/club names — prevents leakage across folds
- **xG & xA per minute** instead of raw averages — better reflects actual performance
- **Optuna** for Bayesian hyperparameter search across both models
- **SHAP** for feature importance analysis and model explainability
- **CatBoost** chosen over LightGBM for superior handling of high-cardinality categoricals

---

## Project Structure

```
├── src/
│   ├── eda.py          # Exploratory data analysis
│   ├── features.py     # Feature engineering & OOF encoding
│   ├── models.py       # Model training & Optuna tuning
│   └── explain.py      # SHAP analysis
├── plots/              # Feature importance & SHAP visualizations
├── main.py             # Pipeline orchestrator
└── app.py              # Streamlit prediction interface
```

---

## Run Locally

```bash
git clone https://github.com/RishabhRana37/offside-football-prediction.git
cd offside-football-prediction
pip install -r requirements.txt

# Add train.csv & test.csv from https://www.kaggle.com/competitions/offside-data-thon/data
python main.py
```

---

**Rishabh Rana** · [LinkedIn](https://linkedin.com/in/rishabh-rana37) · [GitHub](https://github.com/RishabhRana37)
