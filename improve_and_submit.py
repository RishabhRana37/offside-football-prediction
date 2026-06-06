"""
improve_and_submit.py  — final fast version
============================================
Strategy: use proven src/features.py FeaturePipeline as base (fast),
then stack improvements on top:
  1. Larger dataset: train_cleaned.csv (1,185,335 rows)
  2. Extra football features added separately (vectorised)
  3. Smoothed OOF target encoding (pre-computed into arrays, then concat)
  4. CatBoost: depth=6, lr=0.05, 1000 iters, class_weights=[1,3]
  5. Post-processing: zero where minutes_played==0 or team_goals==0
"""

import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

from src.features import FeaturePipeline, generate_out_of_fold_target_encoding

t0 = time.time()

WORKSPACE  = "/Users/rana/OFF SIDE"
TRAIN_NEW  = "/Users/rana/Downloads/train_cleaned.csv"
TRAIN_WS   = os.path.join(WORKSPACE, "train.csv")
TEST_PATH  = os.path.join(WORKSPACE, "test.csv")
SOL_PATH   = os.path.join(WORKSPACE, "solution.csv")
SUB_PATH   = os.path.join(WORKSPACE, "submission.csv")
PIP_PATH   = os.path.join(WORKSPACE, "fitted_pipeline.pkl")
MDL_PATH   = os.path.join(WORKSPACE, "catboost_model.cbm")
PRF_PATH   = os.path.join(WORKSPACE, "player_profiles.pkl")
TARGET     = "scored_flag"

# ──────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────────────────────
print("="*60)
print("Loading data ...")
train_src = TRAIN_NEW if os.path.exists(TRAIN_NEW) else TRAIN_WS
train_raw = pd.read_csv(train_src, low_memory=False)
test_df   = pd.read_csv(TEST_PATH,  low_memory=False)
print(f"  train: {train_raw.shape}   test: {test_df.shape}   [{time.time()-t0:.1f}s]")

y_raw       = train_raw[TARGET].astype(int).copy()
GLOBAL_MEAN = y_raw.mean()
print(f"  positive rate: {GLOBAL_MEAN:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 2. Base Feature Pipeline (handles imputation, encoding, date, football feats)
# ──────────────────────────────────────────────────────────────────────────
print("\nRunning FeaturePipeline ...")
pipeline    = FeaturePipeline(random_state=42)
train_feat  = pipeline.fit_transform(train_raw, target_col=TARGET)
test_feat   = pipeline.transform(test_df)
print(f"  pipeline done  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 3. Additional vectorised features (pre-compute as dicts then assign once)
# ──────────────────────────────────────────────────────────────────────────
print("Adding extra features ...")

def extra_features(df_src, df_feat):
    """Compute extra columns and return as a dict of arrays."""
    cols = {}

    mr  = df_src.get('minutes_ratio',  pd.Series(0.0, index=df_src.index))
    xg  = df_src.get('avg_xG',         pd.Series(0.0, index=df_src.index))
    xa  = df_src.get('avg_xA',         pd.Series(0.0, index=df_src.index))
    sh  = df_src.get('avg_shots',      pd.Series(0.0, index=df_src.index))
    npx = df_src.get('avg_npxG',       pd.Series(0.0, index=df_src.index))
    xgc = df_src.get('avg_xGChain',    pd.Series(0.0, index=df_src.index))
    xgb = df_src.get('avg_xGBuildup',  pd.Series(0.0, index=df_src.index))
    kp  = df_src.get('avg_key_passes', pd.Series(0.0, index=df_src.index))

    cols['expected_npxG_in_match']        = (npx * mr).values
    cols['expected_xGChain_in_match']     = (xgc * mr).values
    cols['expected_xGBuildup_in_match']   = (xgb * mr).values
    cols['expected_key_passes_in_match']  = (kp  * mr).values
    cols['xG_per_shot_ext']               = (xg  / (sh.fillna(0) + 1e-5)).values

    # Market value vs peak
    mv  = df_src.get('market_value_before_match',  pd.Series(0.0, index=df_src.index))
    pmv = df_src.get('highest_market_value_in_eur',pd.Series(0.0, index=df_src.index))
    cols['market_value_ratio_peak_ext'] = (mv / (pmv + 1e-5)).values

    # Attacker × market value
    if 'is_attacker' in df_feat.columns:
        cols['attacker_mv'] = (df_feat['is_attacker'].values * mv.values)

    # Intl efficiency (already in pipeline but recompute clean version)
    caps  = df_src.get('international_caps',  pd.Series(0.0, index=df_src.index))
    goals = df_src.get('international_goals', pd.Series(0.0, index=df_src.index))
    cols['intl_efficiency_ext'] = (goals / (caps + 1e-5)).values

    # Team goals (only for train; test won't have home/away club goals)
    if 'home_club_goals' in df_src.columns and 'away_club_goals' in df_src.columns:
        is_home = df_src['home_away'].str.upper() == 'HOME'
        cols['team_goals']      = np.where(is_home, df_src['home_club_goals'], df_src['away_club_goals'])
        cols['opponent_goals']  = np.where(is_home, df_src['away_club_goals'], df_src['home_club_goals'])
        cols['match_total_goals'] = (df_src['home_club_goals'] + df_src['away_club_goals']).values

    return cols


tr_extra = extra_features(train_raw, train_feat)
te_extra = extra_features(test_df,   test_feat)

# Add columns to feature DataFrames at once via pd.concat
tr_extra_df = pd.DataFrame(tr_extra, index=train_feat.index)
te_extra_df = pd.DataFrame(te_extra, index=test_feat.index)

train_feat = pd.concat([train_feat, tr_extra_df], axis=1)
test_feat  = pd.concat([test_feat,  te_extra_df],  axis=1)
print(f"  extra features done  train:{train_feat.shape}  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 4. Smoothed OOF Target Encoding  (pre-compute arrays, concat at end)
# ──────────────────────────────────────────────────────────────────────────
print("Smoothed OOF target encoding ...")
ENCODE_COLS = [c for c in ['name_y','home_club_name','away_club_name',
                            'competition_type','referee']
               if c in train_raw.columns]
SMOOTH_M    = 20
SKF5        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

te_arrays_train = {}  # col -> array of OOF encoded values
te_arrays_test  = {}

for col in ENCODE_COLS:
    enc = f'{col}_te_smooth'
    arr_train = np.full(len(train_raw), GLOBAL_MEAN, dtype=np.float32)

    for tr_i, va_i in SKF5.split(train_raw, y_raw):
        fold_data = train_raw.iloc[tr_i]
        stats     = fold_data.groupby(col)[TARGET].agg(['sum','count'])
        sm        = (stats['sum'] + SMOOTH_M * GLOBAL_MEAN) / (stats['count'] + SMOOTH_M)
        mapped    = train_raw.iloc[va_i][col].map(sm).fillna(GLOBAL_MEAN).values.astype(np.float32)
        arr_train[va_i] = mapped

    te_arrays_train[enc] = arr_train

    # Test: full-train smoothed mapping
    fs = train_raw.groupby(col)[TARGET].agg(['sum','count'])
    sm_full = (fs['sum'] + SMOOTH_M * GLOBAL_MEAN) / (fs['count'] + SMOOTH_M)
    if col in test_df.columns:
        te_arrays_test[enc] = test_df[col].map(sm_full).fillna(GLOBAL_MEAN).values.astype(np.float32)
    else:
        te_arrays_test[enc] = np.full(len(test_df), GLOBAL_MEAN, dtype=np.float32)
    print(f"  {col} ✓  [{time.time()-t0:.1f}s]")

# Concat all TE columns at once
te_train_df = pd.DataFrame(te_arrays_train, index=train_feat.index)
te_test_df  = pd.DataFrame(te_arrays_test,  index=test_feat.index)
train_feat  = pd.concat([train_feat, te_train_df], axis=1)
test_feat   = pd.concat([test_feat,  te_test_df],  axis=1)
print(f"  target encoding done  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 5. Final Feature Matrix
# ──────────────────────────────────────────────────────────────────────────
DROP_COLS = {'appearance_id', 'name_y', TARGET, 'date',
             'home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id'}

FEATURES = [c for c in train_feat.columns
            if c not in DROP_COLS and train_feat[c].dtype != object]

X_train = train_feat[FEATURES].copy()
X_test  = test_feat.reindex(columns=FEATURES, fill_value=0).copy()
y_train = y_raw.reset_index(drop=True)

print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")

cat_in_x    = [c for c in pipeline.cat_cols if c in FEATURES]
cat_indices = [FEATURES.index(c) for c in cat_in_x]

# ──────────────────────────────────────────────────────────────────────────
# 6. CatBoost 5-Fold CV
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CatBoost 5-Fold Training ...")
print("="*60)

PARAMS = {
    'loss_function':  'Logloss',
    'eval_metric':    'AUC',
    'learning_rate':  0.05,
    'depth':          6,
    'l2_leaf_reg':    3.0,
    'iterations':     1000,
    'random_seed':    42,
    'verbose':        100,
    'task_type':      'CPU',
    'thread_count':   -1,
    'class_weights':  [1.0, 3.0],
    'od_type':        'Iter',
    'od_wait':        80,
}

oof_preds  = np.zeros(len(X_train))
test_preds = np.zeros(len(X_test))
fold_scores = []
models = []

for fold, (tr_i, va_i) in enumerate(SKF5.split(X_train, y_train)):
    t1 = time.time()
    X_tr, y_tr = X_train.iloc[tr_i], y_train.iloc[tr_i]
    X_va, y_va = X_train.iloc[va_i], y_train.iloc[va_i]

    pool_tr = Pool(X_tr, y_tr, cat_features=cat_indices)
    pool_va = Pool(X_va, y_va, cat_features=cat_indices)
    pool_te = Pool(X_test,     cat_features=cat_indices)

    m = CatBoostClassifier(**PARAMS)
    m.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=80, verbose=100)

    vp = m.predict_proba(pool_va)[:, 1]
    oof_preds[va_i] = vp
    ap = average_precision_score(y_va, vp)
    fold_scores.append(ap)
    test_preds += m.predict_proba(pool_te)[:, 1] / 5
    models.append(m)
    print(f"\n  >>> Fold {fold+1}  AP={ap:.4f}  best_iter={m.best_iteration_}  [{time.time()-t1:.0f}s] <<<\n")

raw_oof_ap = average_precision_score(y_train, oof_preds)
print(f"  Mean Fold AP : {np.mean(fold_scores):.4f}")
print(f"  Overall OOF AP: {raw_oof_ap:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 7. Post-Processing Rules
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Post-Processing ===")

mask_min  = (train_raw['minutes_played'] == 0).values
is_home   = train_raw['home_away'].str.upper() == 'HOME'
team_goals = np.where(is_home, train_raw['home_club_goals'], train_raw['away_club_goals'])
mask_team  = (team_goals == 0)

oof_pp1 = oof_preds.copy(); oof_pp1[mask_min] = 0.0
oof_pp2 = oof_preds.copy(); oof_pp2[mask_min | mask_team] = 0.0

ap1 = average_precision_score(y_train, oof_pp1)
ap2 = average_precision_score(y_train, oof_pp2)
print(f"  minutes_played==0 rule  → AP={ap1:.4f} (Δ{ap1-raw_oof_ap:+.4f})")
print(f"  Both rules              → AP={ap2:.4f} (Δ{ap2-raw_oof_ap:+.4f})")

if ap2 >= ap1 and ap2 > raw_oof_ap:
    final_oof = oof_pp2; final_ap = ap2; rule = "BOTH"
    test_preds[test_df['minutes_played'] == 0] = 0.0
elif ap1 > raw_oof_ap:
    final_oof = oof_pp1; final_ap = ap1; rule = "minutes_played==0"
    test_preds[test_df['minutes_played'] == 0] = 0.0
else:
    final_oof = oof_preds; final_ap = raw_oof_ap; rule = "none"

print(f"  Applied rule: {rule}   Final OOF AP: {final_ap:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 8. Save Artifacts
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Saving Artifacts ===")

# Best-fold CatBoost model
best_idx = int(np.argmax(fold_scores))
models[best_idx].save_model(MDL_PATH)
print(f"  Model saved (fold {best_idx+1}, AP={fold_scores[best_idx]:.4f})")

# Rebuild pipeline for app.py (uses original FeaturePipeline with OOF TE)
pipeline2 = FeaturePipeline(random_state=42)
tf2 = pipeline2.fit_transform(train_raw, target_col=TARGET)
skf_p = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
enc_c = [c for c in ['name_y','home_club_name','away_club_name'] if c in train_raw.columns]
if enc_c:
    enc_df = generate_out_of_fold_target_encoding(train_raw, skf_p, target_col=TARGET, cols_to_encode=enc_c)
    for c in enc_c:
        tf2[f'{c}_target_enc'] = enc_df[f'{c}_target_enc']
drop_p = ['appearance_id','name_y',TARGET,'date']
pipeline2.feature_names = [c for c in tf2.columns if c not in drop_p]
with open(PIP_PATH, 'wb') as f: pickle.dump(pipeline2, f)
print("  Pipeline saved")

# Player profiles
pc = [c for c in ['name_y','age','foot','position','sub_position',
    'country_of_citizenship','home_club_name','market_value_before_match',
    'highest_market_value_in_eur','international_caps','international_goals',
    'avg_xG','avg_xA','avg_shots','minutes_ratio'] if c in train_raw.columns]
pf = train_raw[pc].dropna(subset=['name_y'])
ag = pf.groupby('name_y').agg({
    'age':'median',
    'foot': lambda x: x.mode().iloc[0] if not x.mode().empty else 'right',
    'position': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Attack',
    'sub_position': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Center-Forward',
    'country_of_citizenship': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
    'home_club_name': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown',
    'market_value_before_match':'median','highest_market_value_in_eur':'median',
    'international_caps':'median','international_goals':'median',
    'avg_xG':'median','avg_xA':'median','avg_shots':'median','minutes_ratio':'median'
}).fillna({'age':26.,'market_value_before_match':1e6,'highest_market_value_in_eur':2e6,
           'international_caps':0.,'international_goals':0.,'avg_xG':0.1,
           'avg_xA':0.05,'avg_shots':1.,'minutes_ratio':0.5})
with open(PRF_PATH, 'wb') as f: pickle.dump(ag.to_dict(orient='index'), f)
print(f"  {len(ag)} player profiles saved")

# ──────────────────────────────────────────────────────────────────────────
# 9. Generate Submission
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Generating Submission ===")
sub = pd.DataFrame({'appearance_id': test_df['appearance_id'], 'scored_flag': test_preds})
assert sub['scored_flag'].between(0,1).all() and not sub.isnull().any().any()
sub.to_csv(SOL_PATH, index=False)
sub.to_csv(SUB_PATH, index=False)

print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  Train rows    : {len(X_train):,}")
print(f"  Features      : {len(FEATURES)}")
print(f"  Mean Fold AP  : {np.mean(fold_scores):.4f}")
print(f"  OOF AP (raw)  : {raw_oof_ap:.4f}")
print(f"  Post-proc     : {rule}")
print(f"  Final OOF AP  : {final_ap:.4f}")
print(f"  Total time    : {(time.time()-t0)/60:.1f} min")
print("="*60)
print("✅ Done! Submit solution.csv to Kaggle.")
