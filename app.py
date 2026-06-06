import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration with a modern dark theme and custom icon
st.set_page_config(
    page_title="Offside AI | Football Goal Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS for styling
st.markdown("""
<style>
    /* Premium background and typography styling */
    .reportview-container {
        background: #0f172a;
    }
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #3b82f6 50%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Cards and Glassmorphism */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 1.5rem;
        backdrop-filter: blur(12px);
        margin-bottom: 1rem;
    }
    .highlight-card {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .custom-label {
        font-weight: 600;
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model and pipeline resources safely
@st.cache_resource
def load_ml_resources():
    pipeline_path = 'fitted_pipeline.pkl'
    model_path = 'catboost_model.cbm'
    
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        return None, None
        
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    model = CatBoostClassifier()
    model.load_model(model_path)
    return pipeline, model

@st.cache_resource
def load_player_profiles():
    profiles_path = 'player_profiles.pkl'
    if os.path.exists(profiles_path):
        with open(profiles_path, 'rb') as f:
            return pickle.load(f)
    return {}

# Load resources
pipeline, model = load_ml_resources()
player_profiles = load_player_profiles()

# Sidebar branding
with st.sidebar:
    st.image("https://img.icons8.com/color/120/000000/soccer-ball.png", width=80)
    st.markdown("<h2 style='margin-top: 0;'>Offside AI</h2>", unsafe_allow_html=True)
    st.markdown("Predictive Goal Scoring Analytics Engine built for the **Offside Football Datathon**.")
    st.markdown("---")
    st.markdown("### Model Architecture")
    st.info("🤖 **CatBoost Classifier**\n* 5-Fold Stratified CV\n* Tuned via Optuna\n* Validation AP: **0.4498**")
    st.markdown("---")
    st.markdown("Created by **RishabhRana37**")

# Check if model files are missing
if pipeline is None or model is None:
    st.error("⚠️ **Model & Pipeline Files Missing**")
    st.warning("The project model files (`fitted_pipeline.pkl` and `catboost_model.cbm`) could not be found. If the model training task is still running, please wait a minute and refresh this page. Otherwise, please run `save_model.py` to generate them.")
    st.stop()

# Header
st.markdown("<div class='main-title'>⚽ Offside Goal Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A grandmaster-level football analytics engine to predict goal-scoring probabilities.</div>", unsafe_allow_html=True)

# Tabs
tab_single, tab_batch, tab_eda, tab_about = st.tabs([
    "🎯 Player Goal Predictor", 
    "📁 Batch Predictor", 
    "📈 Model Insights & EDA", 
    "ℹ️ Methodology & About"
])

# ================= TAB 1: SINGLE PLAYER PREDICTOR =================
with tab_single:
    st.subheader("Select Player and Match Context")
    
    # Check if we have players list
    player_names = sorted(list(pipeline.global_target_means.get('name_y', {}).keys()))
    if not player_names:
        player_names = ["Cristiano Ronaldo", "Lionel Messi", "Erling Haaland", "Kylian Mbappé"]
        
    col_p, col_c = st.columns([1, 1])
    
    with col_p:
        selected_player = st.selectbox("👤 Select Player Profile", ["-- Custom Profile --"] + player_names)
        
    # Pre-populate variables based on selected player profile
    profile = {}
    if selected_player != "-- Custom Profile --" and selected_player in player_profiles:
        profile = player_profiles[selected_player]
        
    # Standard fallback values
    default_age = profile.get('age', 26.0)
    default_foot = profile.get('foot', 'right')
    default_pos = profile.get('position', 'Attack')
    default_subpos = profile.get('sub_position', 'Center-Forward')
    default_citizen = profile.get('country_of_citizenship', 'England')
    default_club = profile.get('home_club_name', 'Arsenal FC')
    default_mv = profile.get('market_value_before_match', 15000000.0)
    default_highest_mv = profile.get('highest_market_value_in_eur', 25000000.0)
    default_caps = profile.get('international_caps', 10.0)
    default_goals = profile.get('international_goals', 2.0)
    default_xg = profile.get('avg_xG', 0.25)
    default_xa = profile.get('avg_xA', 0.12)
    default_shots = profile.get('avg_shots', 2.1)
    default_min_ratio = profile.get('minutes_ratio', 0.75)
    default_is_attacker = int(default_pos == 'Attack')
    
    st.markdown("---")
    
    # Organize input fields into grids
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='custom-label'>👤 Player Bio & Details</div>", unsafe_allow_html=True)
        age = st.slider("Age", 16, 42, int(default_age))
        foot = st.selectbox("Preferred Foot", ['right', 'left', 'both'], index=['right', 'left', 'both'].index(default_foot) if default_foot in ['right', 'left', 'both'] else 0)
        position = st.selectbox("Position Group", ['Attack', 'Midfield', 'Defender', 'Goalkeeper'], index=['Attack', 'Midfield', 'Defender', 'Goalkeeper'].index(default_pos) if default_pos in ['Attack', 'Midfield', 'Defender', 'Goalkeeper'] else 0)
        sub_position = st.text_input("Detailed Sub-position", default_subpos)
        citizenship = st.text_input("Country of Citizenship", default_citizen)
        
    with col2:
        st.markdown("<div class='custom-label'>📊 Performance Statistics</div>", unsafe_allow_html=True)
        avg_xg = st.number_input("Average xG (per 90 min)", min_value=0.0, max_value=2.0, value=float(default_xg), step=0.01)
        avg_xa = st.number_input("Average xA (per 90 min)", min_value=0.0, max_value=2.0, value=float(default_xa), step=0.01)
        avg_shots = st.number_input("Average Shots (per 90 min)", min_value=0.0, max_value=15.0, value=float(default_shots), step=0.1)
        minutes_ratio = st.slider("Season Minutes Played Ratio", 0.0, 1.0, float(default_min_ratio))
        minutes_played = st.slider("Expected Minutes in this Match", 0, 90, 90)
        starter_flag = st.checkbox("Starting the Match?", value=True)
        
    with col3:
        st.markdown("<div class='custom-label'>🏟️ Match Context & Market Info</div>", unsafe_allow_html=True)
        home_club = st.text_input("Player's Club Name", default_club)
        opponent_club = st.text_input("Opponent Club Name", "Manchester City FC")
        home_away = st.selectbox("Match Venue", ['Home', 'Away'], index=0)
        
        market_val = st.number_input("Market Value (EUR)", min_value=0.0, value=float(default_mv), step=500000.0)
        highest_mv = st.number_input("Highest Market Value (EUR)", min_value=0.0, value=float(default_highest_mv), step=500000.0)
        
        caps = st.number_input("International Caps", min_value=0, value=int(default_caps))
        goals = st.number_input("International Goals", min_value=0, value=int(default_goals))

    # Extra hidden flags derived automatically
    is_attacker = 1 if position == 'Attack' else 0
    full_match_flag = 1 if minutes_played == 90 else 0
    substitute_flag = 1 if not starter_flag else 0
    prime_age_flag = 1 if 24 <= age <= 29 else 0
    veteran_flag = 1 if age >= 32 else 0
    has_national_team_experience = 1 if caps > 0 else 0
    finisher_flag = 1 if (avg_xg > 0.35 and is_attacker) else 0
    creative_player_flag = 1 if (avg_xa > 0.20) else 0

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔮 Predict Goal Probability", type="primary", use_container_width=True):
        # Construct input raw dataframe
        input_data = {
            'date': ['2026-06-06'],
            'foot': [foot],
            'position': [position],
            'sub_position': [sub_position],
            'country_of_citizenship': [citizenship],
            'home_club_name': [home_club if home_away == 'Home' else opponent_club],
            'away_club_name': [opponent_club if home_away == 'Home' else home_club],
            'stadium': ['Emirates Stadium'], # default
            'referee': ['Michael Oliver'],   # default
            'competition_type': ['domestic_league'],
            'confederation': ['uefa'],
            'market_value_tier': ['High' if market_val > 20000000 else ('Medium' if market_val > 5000000 else 'Low')],
            'age_bucket': ['24-29' if prime_age_flag else ('30+' if age >= 30 else 'U23')],
            'name_x': ['Premier League'],
            'home_away': [home_away],
            'country_name': [citizenship],
            'name_y': [selected_player if selected_player != "-- Custom Profile --" else "Custom Player"],
            'avg_xG': [avg_xg],
            'avg_xA': [avg_xa],
            'avg_shots': [avg_shots],
            'minutes_ratio': [minutes_ratio],
            'market_value_before_match': [market_val],
            'highest_market_value_in_eur': [highest_mv],
            'age': [age],
            'international_caps': [caps],
            'international_goals': [goals],
            'starter_flag': [starter_flag],
            'minutes_played': [minutes_played],
            'is_attacker': [is_attacker],
            'full_match_flag': [full_match_flag],
            'substitute_flag': [substitute_flag],
            'card_flag': [0],
            'prime_age_flag': [prime_age_flag],
            'veteran_flag': [veteran_flag],
            'has_national_team_experience': [has_national_team_experience],
            'finisher_flag': [finisher_flag],
            'creative_player_flag': [creative_player_flag],
            'has_understat': [1],
            'analytics_coverage_flag': [1]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Preprocess features
        processed_df = pipeline.transform(input_df)
        
        # Align features to model expectations
        X_infer = processed_df[pipeline.feature_names]
        
        # Run inference
        cat_features = [col for col in pipeline.cat_cols if col in X_infer.columns]
        cat_indices = [X_infer.columns.get_loc(col) for col in cat_features]
        
        pool = Pool(X_infer, cat_features=cat_indices)
        prob = model.predict_proba(pool)[0, 1]
        
        # Visual feedback with styling
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.markdown("<div class='highlight-card'>", unsafe_allow_html=True)
            st.metric("Goal Scorer Probability", f"{prob*100:.2f}%")
            
            # Show gauge color coding
            if prob < 0.10:
                st.error("🛑 **Low Goal Threat** (Probability < 10%)")
            elif prob < 0.25:
                st.warning("⚠️ **Moderate Goal Threat** (Probability 10% - 25%)")
            elif prob < 0.45:
                st.success("🟢 **High Goal Threat** (Probability 25% - 45%)")
            else:
                st.markdown("<h4 style='color: #a855f7; margin-top:0;'>🔥 Extreme Goal Threat</h4>", unsafe_allow_html=True)
                st.markdown("**Probability exceeds 45%! Exceptionally high scoring chance.**")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_res2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Feature Valuation Contribution")
            
            # Display simulated prediction contributors
            st.write(f"**Player**: {input_df['name_y'].values[0]} ({position})")
            st.write(f"**Historical Avg xG**: {avg_xg:.2f} | **Expected Match Minutes**: {minutes_played} min")
            st.write(f"**International Caps/Goals**: {caps} / {goals} (Ratio: {goals/(caps+1e-5):.2%})")
            st.write(f"**Current Valuation**: €{market_val:,.0f} (Ratio to Peak: {market_val/(highest_mv+1e-5):.1%})")
            
            # Target Encoded Baseline Score
            player_te = processed_df['name_y_target_enc'].values[0]
            st.write(f"**Player Base Goal-Scoring Rate**: {player_te:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 2: BATCH PREDICTION =================
with tab_batch:
    st.subheader("Generate Batch Predictions")
    st.markdown("Upload a CSV file containing player match statistics to get predictions for all entries in one run.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully! Parsing columns...")
            st.dataframe(batch_df.head())
            
            if st.button("🚀 Process Batch Predictions", type="primary"):
                with st.spinner("Analyzing and calculating predictions..."):
                    # Process columns using pipeline
                    processed_batch = pipeline.transform(batch_df)
                    
                    # Align features
                    X_batch = processed_batch[pipeline.feature_names]
                    
                    # Inference
                    cat_features = [col for col in pipeline.cat_cols if col in X_batch.columns]
                    cat_indices = [X_batch.columns.get_loc(col) for col in cat_features]
                    batch_pool = Pool(X_batch, cat_features=cat_indices)
                    
                    preds = model.predict_proba(batch_pool)[:, 1]
                    
                    # Join predictions
                    results_df = batch_df.copy()
                    results_df['scored_probability'] = preds
                    
                    # Reorder to keep identifier columns at the front
                    id_cols = ['appearance_id', 'name_y', 'scored_probability']
                    other_cols = [c for c in results_df.columns if c not in id_cols]
                    results_df = results_df[id_cols + other_cols]
                    
                    st.success("Batch predictions successfully completed!")
                    st.dataframe(results_df.head(20))
                    
                    # Convert to CSV for download
                    csv_data = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_data,
                        file_name="offside_batch_predictions.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"Error reading file or generating predictions: {str(e)}")

# ================= TAB 3: MODEL INSIGHTS & EDA =================
with tab_eda:
    st.subheader("Model Validation & Explanations")
    st.markdown("Review the validation metrics, feature importances, and SHAP explainability summaries generated during pipeline training.")
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🧬 SHAP Feature Attribution")
        st.markdown("The SHAP summary plots show how features drive the predictions. High values of `name_y_target_enc` (a player's historical scoring rate) and `expected_xG_in_match` are the strongest positive contributors.")
        if os.path.exists("plots/shap_summary_plot.png"):
            st.image("plots/shap_summary_plot.png", use_container_width=True)
        else:
            st.caption("SHAP plot image not found in plots/ folder.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Scoring Probability by Position")
        st.markdown("Attackers have a significantly higher baseline goal-scoring probability compared to midfielders and defenders.")
        if os.path.exists("plots/scoring_probability_by_position.png"):
            st.image("plots/scoring_probability_by_position.png", use_container_width=True)
        else:
            st.caption("Position plot image not found in plots/ folder.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_e2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🌳 Split-Based Feature Importance")
        st.markdown("Feature importance scores indicating the total number of splits created on each variable. High-cardinality values like referee and club mappings are highly informative.")
        if os.path.exists("plots/feature_importance_plot.png"):
            st.image("plots/feature_importance_plot.png", use_container_width=True)
        else:
            st.caption("Feature importance plot image not found in plots/ folder.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("### 🎯 Target Variable Distribution")
        st.markdown("Only **8.54%** of player appearances in the dataset result in a goal. The dataset is heavily imbalanced, requiring Average Precision for objective model validation.")
        if os.path.exists("plots/target_distribution.png"):
            st.image("plots/target_distribution.png", use_container_width=True)
        else:
            st.caption("Target distribution plot image not found in plots/ folder.")
        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 4: ABOUT & METHODOLOGY =================
with tab_about:
    st.markdown("""
    ### 📖 Methodology & Validation
    
    This application deploys a machine learning model optimized to predict player-level goal scoring.
    
    #### Cross-Validation Framework
    We utilized **Stratified 5-Fold Cross-Validation** to prevent local validation leakage.
    - Folds are stratified based on the target `scored_flag` to maintain the 8.54% positive class ratio across train and test splits.
    - Predictions are validated using **Average Precision (AP)**, which captures the Area Under the Precision-Recall Curve.
    
    #### Model Performance
    * **LightGBM Baseline**: Mean Out-of-Fold (OOF) AP = **0.4090**
    * **CatBoost Classifier**: Mean Out-of-Fold (OOF) AP = **0.4498** (Selected as the final model due to superior categorical handling)
    
    #### Features Engineered
    1. **Target Encoding**: Out-of-fold target means for player (`name_y`), home club, and away club.
    2. **Attacking Ratios**: `expected_xG_in_match` scales historical expected goals by match minutes played.
    3. **International Profile**: `intl_goal_efficiency` represents goals scored per international cap.
    4. **Market Valuation**: Tracks current market value relative to career peak (`market_value_ratio_peak`).
    """)
