"""
boost_to_049.py — MAXIMUM AP push
===================================
Strategy:
  1. Full train_cleaned.csv (1.18M rows)
  2. Extended feature engineering (frequency encoding, more interactions)
  3. More target encoding columns (sub_position, country_name, stadium, season, confederation)
  4. CatBoost (depth=8, lr=0.03, 2000 iters) — deeper, slower LR
  5. LightGBM (num_leaves=63, lr=0.03, 1500 iters) — diverse model
  6. Rank-average ensemble of CatBoost + LightGBM
  7. Post-processing: zero-minutes rule
"""

import os, pickle, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

from src.features import FeaturePipeline

t0 = time.time()

WORKSPACE  = "/Users/rana/OFF SIDE"
TRAIN_PATH = "/Users/rana/Downloads/train_cleaned.csv"
if not os.path.exists(TRAIN_PATH):
    TRAIN_PATH = os.path.join(WORKSPACE, "train.csv")
TEST_PATH  = os.path.join(WORKSPACE, "test.csv")
SOL_PATH   = os.path.join(WORKSPACE, "solution.csv")
TARGET     = "scored_flag"

# ──────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data ...")
train_raw = pd.read_csv(TRAIN_PATH, low_memory=False)
test_df   = pd.read_csv(TEST_PATH, low_memory=False)
print(f"  train: {train_raw.shape}   test: {test_df.shape}   [{time.time()-t0:.1f}s]")

y_raw       = train_raw[TARGET].astype(int).copy()
GLOBAL_MEAN = y_raw.mean()
print(f"  positive rate: {GLOBAL_MEAN:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 2. Base Feature Pipeline
# ──────────────────────────────────────────────────────────────────────────
print("\nRunning FeaturePipeline ...")
pipeline    = FeaturePipeline(random_state=42)
train_feat  = pipeline.fit_transform(train_raw, target_col=TARGET)
test_feat   = pipeline.transform(test_df)
print(f"  pipeline done  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 3. EXTENDED extra features
# ──────────────────────────────────────────────────────────────────────────
print("Adding extended features ...")

def extra_features(df_src, df_feat):
    cols = {}
    mr  = df_src.get('minutes_ratio',  pd.Series(0.0, index=df_src.index))
    xg  = df_src.get('avg_xG',         pd.Series(0.0, index=df_src.index))
    xa  = df_src.get('avg_xA',         pd.Series(0.0, index=df_src.index))
    sh  = df_src.get('avg_shots',      pd.Series(0.0, index=df_src.index))
    npx = df_src.get('avg_npxG',       pd.Series(0.0, index=df_src.index))
    xgc = df_src.get('avg_xGChain',    pd.Series(0.0, index=df_src.index))
    xgb = df_src.get('avg_xGBuildup',  pd.Series(0.0, index=df_src.index))
    kp  = df_src.get('avg_key_passes', pd.Series(0.0, index=df_src.index))

    cols['expected_npxG_in_match']       = (npx * mr).values
    cols['expected_xGChain_in_match']    = (xgc * mr).values
    cols['expected_xGBuildup_in_match']  = (xgb * mr).values
    cols['expected_key_passes_in_match'] = (kp  * mr).values
    cols['xG_per_shot_ext']              = (xg  / (sh.fillna(0) + 1e-5)).values

    # npxG fraction of xG — penalty dependence
    cols['npxG_fraction'] = (npx / (xg + 1e-5)).values

    # xG minus npxG — penalty xG component
    cols['penalty_xG'] = (xg - npx).values

    # Shots per key pass — conversion pressure
    cols['shots_per_kp'] = (sh / (kp + 1e-5)).values

    # Market value
    mv  = df_src.get('market_value_before_match',  pd.Series(0.0, index=df_src.index))
    pmv = df_src.get('highest_market_value_in_eur', pd.Series(0.0, index=df_src.index))
    cols['market_value_ratio_peak_ext'] = (mv / (pmv + 1e-5)).values

    # Log market value squared (captures non-linearity)
    lmv = df_src.get('log_market_value', pd.Series(0.0, index=df_src.index))
    cols['log_mv_sq'] = (lmv ** 2).values

    # Attacker interactions
    if 'is_attacker' in df_feat.columns:
        cols['attacker_mv']     = (df_feat['is_attacker'].values * mv.values)
        cols['attacker_xG']     = (df_feat['is_attacker'].values * xg.values)
        cols['attacker_shots']  = (df_feat['is_attacker'].values * sh.values)
        cols['attacker_mr']     = (df_feat['is_attacker'].values * mr.values)

    # Midfielder interactions
    if 'is_midfielder' in df_feat.columns:
        cols['midfielder_xG']   = (df_feat['is_midfielder'].values * xg.values)
        cols['midfielder_shots'] = (df_feat['is_midfielder'].values * sh.values)

    # Defender interactions
    if 'is_defender' in df_feat.columns:
        cols['defender_xG'] = (df_feat['is_defender'].values * xg.values)

    # Starter interactions
    if 'starter_flag' in df_feat.columns:
        cols['starter_xG']  = (df_feat['starter_flag'].values * xg.values)
        cols['starter_mr']  = (df_feat['starter_flag'].values * mr.values)

    # International efficiency
    caps  = df_src.get('international_caps',  pd.Series(0.0, index=df_src.index))
    goals = df_src.get('international_goals', pd.Series(0.0, index=df_src.index))
    cols['intl_efficiency_ext'] = (goals / (caps + 1e-5)).values
    cols['intl_goals_sq']       = (goals ** 2).values

    # Age interactions
    age = df_src.get('age', pd.Series(26.0, index=df_src.index))
    cols['age_xG']        = (age * xg).values
    cols['age_sq']        = (age ** 2).values
    cols['prime_xG']      = np.where((age >= 24) & (age <= 30), xg.values, 0.0)

    # Height interaction
    height = df_src.get('height_in_cm', pd.Series(180.0, index=df_src.index))
    cols['height_xG'] = (height * xg).values

    # Attendance (proxy for match importance)
    att = df_src.get('attendance', pd.Series(0.0, index=df_src.index)).fillna(0)
    cols['log_attendance'] = np.log1p(att).values

    # Team goals (train only)
    if 'home_club_goals' in df_src.columns and 'away_club_goals' in df_src.columns:
        is_home = df_src['home_away'].str.upper() == 'HOME'
        cols['team_goals']        = np.where(is_home, df_src['home_club_goals'], df_src['away_club_goals'])
        cols['opponent_goals']    = np.where(is_home, df_src['away_club_goals'], df_src['home_club_goals'])
        cols['match_total_goals'] = (df_src['home_club_goals'] + df_src['away_club_goals']).values

    return cols

tr_extra = extra_features(train_raw, train_feat)
te_extra = extra_features(test_df,   test_feat)
train_feat = pd.concat([train_feat, pd.DataFrame(tr_extra, index=train_feat.index)], axis=1)
test_feat  = pd.concat([test_feat,  pd.DataFrame(te_extra, index=test_feat.index)],  axis=1)
print(f"  extra features done  train:{train_feat.shape}  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 4. FREQUENCY ENCODING for high-cardinality categoricals
# ──────────────────────────────────────────────────────────────────────────
print("Frequency encoding ...")
FREQ_COLS = [c for c in ['name_y', 'home_club_name', 'away_club_name', 'stadium',
                          'referee', 'country_name', 'sub_position', 'country_of_citizenship']
             if c in train_raw.columns]

for col in FREQ_COLS:
    freq_map = train_raw[col].value_counts(normalize=True).to_dict()
    train_feat[f'{col}_freq'] = train_raw[col].map(freq_map).fillna(0).values
    test_feat[f'{col}_freq']  = test_df[col].map(freq_map).fillna(0).values if col in test_df.columns else 0

# ──────────────────────────────────────────────────────────────────────────
# 5. EXTENDED Smoothed OOF Target Encoding
# ──────────────────────────────────────────────────────────────────────────
print("Smoothed OOF target encoding (extended) ...")
ENCODE_COLS = [c for c in ['name_y', 'home_club_name', 'away_club_name',
                            'competition_type', 'referee', 'sub_position',
                            'country_name', 'confederation', 'stadium']
               if c in train_raw.columns]
SMOOTH_M    = 20
SKF5        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

te_arrays_train = {}
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

# Also do INTERACTION target encoding: player × club
print("  Player × club interaction TE ...")
if 'name_y' in train_raw.columns and 'home_club_name' in train_raw.columns:
    train_raw['_player_club'] = train_raw['name_y'].astype(str) + '_' + train_raw['home_club_name'].astype(str)
    test_df['_player_club']   = test_df['name_y'].astype(str) + '_' + test_df['home_club_name'].astype(str)
    
    enc = '_player_club_te_smooth'
    arr_train = np.full(len(train_raw), GLOBAL_MEAN, dtype=np.float32)
    for tr_i, va_i in SKF5.split(train_raw, y_raw):
        fold_data = train_raw.iloc[tr_i]
        stats     = fold_data.groupby('_player_club')[TARGET].agg(['sum','count'])
        sm        = (stats['sum'] + SMOOTH_M * GLOBAL_MEAN) / (stats['count'] + SMOOTH_M)
        arr_train[va_i] = train_raw.iloc[va_i]['_player_club'].map(sm).fillna(GLOBAL_MEAN).values.astype(np.float32)
    te_arrays_train[enc] = arr_train
    fs = train_raw.groupby('_player_club')[TARGET].agg(['sum','count'])
    sm_full = (fs['sum'] + SMOOTH_M * GLOBAL_MEAN) / (fs['count'] + SMOOTH_M)
    te_arrays_test[enc] = test_df['_player_club'].map(sm_full).fillna(GLOBAL_MEAN).values.astype(np.float32)
    print(f"  player_club ✓  [{time.time()-t0:.1f}s]")

te_train_df = pd.DataFrame(te_arrays_train, index=train_feat.index)
te_test_df  = pd.DataFrame(te_arrays_test,  index=test_feat.index)
train_feat  = pd.concat([train_feat, te_train_df], axis=1)
test_feat   = pd.concat([test_feat,  te_test_df],  axis=1)
print(f"  target encoding done  [{time.time()-t0:.1f}s]")

# ──────────────────────────────────────────────────────────────────────────
# 6. Final Feature Matrix
# ──────────────────────────────────────────────────────────────────────────
DROP_COLS = {'appearance_id', 'name_y', TARGET, 'date', '_player_club',
             'home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id'}

FEATURES = [c for c in train_feat.columns
            if c not in DROP_COLS and train_feat[c].dtype != object]

X_train = train_feat[FEATURES].copy()
X_test  = test_feat.reindex(columns=FEATURES, fill_value=0).copy()
y_train = y_raw.reset_index(drop=True)

print(f"\nX_train: {X_train.shape}   X_test: {X_test.shape}")

cat_in_x    = [c for c in pipeline.cat_cols if c in FEATURES]
cat_indices = [FEATURES.index(c) for c in cat_in_x]

# For LightGBM — need column names not indices
lgb_cat_cols = cat_in_x

# ──────────────────────────────────────────────────────────────────────────
# 7. CatBoost 5-Fold CV (DEEPER, SLOWER LR)
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CatBoost 5-Fold Training (depth=8, lr=0.03) ...")
print("=" * 60)

CB_PARAMS = {
    'loss_function':  'Logloss',
    'eval_metric':    'AUC',
    'learning_rate':  0.03,
    'depth':          8,
    'l2_leaf_reg':    3.0,
    'iterations':     2000,
    'random_seed':    42,
    'verbose':        200,
    'task_type':      'CPU',
    'thread_count':   -1,
    'class_weights':  [1.0, 3.0],
    'od_type':        'Iter',
    'od_wait':        100,
    'random_strength': 0.5,
    'bagging_temperature': 0.8,
}

oof_cb     = np.zeros(len(X_train))
test_cb    = np.zeros(len(X_test))
cb_scores  = []

for fold, (tr_i, va_i) in enumerate(SKF5.split(X_train, y_train)):
    t1 = time.time()
    X_tr, y_tr = X_train.iloc[tr_i], y_train.iloc[tr_i]
    X_va, y_va = X_train.iloc[va_i], y_train.iloc[va_i]

    pool_tr = Pool(X_tr, y_tr, cat_features=cat_indices)
    pool_va = Pool(X_va, y_va, cat_features=cat_indices)
    pool_te = Pool(X_test,     cat_features=cat_indices)

    m = CatBoostClassifier(**CB_PARAMS)
    m.fit(pool_tr, eval_set=pool_va, early_stopping_rounds=100, verbose=200)

    vp = m.predict_proba(pool_va)[:, 1]
    oof_cb[va_i] = vp
    ap = average_precision_score(y_va, vp)
    cb_scores.append(ap)
    test_cb += m.predict_proba(pool_te)[:, 1] / 5
    print(f"\n  >>> Fold {fold+1}  AP={ap:.4f}  best_iter={m.best_iteration_}  [{time.time()-t1:.0f}s] <<<\n")

cb_oof_ap = average_precision_score(y_train, oof_cb)
print(f"  CatBoost Mean Fold AP : {np.mean(cb_scores):.4f}")
print(f"  CatBoost OOF AP      : {cb_oof_ap:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 8. LightGBM 5-Fold CV
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LightGBM 5-Fold Training ...")
print("=" * 60)

LGB_PARAMS = {
    'objective':       'binary',
    'metric':          'binary_logloss',
    'boosting_type':   'gbdt',
    'learning_rate':   0.03,
    'num_leaves':      63,
    'max_depth':       -1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq':    1,
    'min_child_samples': 20,
    'n_estimators':    2000,
    'verbose':         -1,
    'random_state':    42,
    'n_jobs':          -1,
    'scale_pos_weight': 3.0,
    'reg_alpha':       0.1,
    'reg_lambda':      1.0,
}

oof_lgb     = np.zeros(len(X_train))
test_lgb    = np.zeros(len(X_test))
lgb_scores  = []

for fold, (tr_i, va_i) in enumerate(SKF5.split(X_train, y_train)):
    t1 = time.time()
    X_tr, y_tr = X_train.iloc[tr_i], y_train.iloc[tr_i]
    X_va, y_va = X_train.iloc[va_i], y_train.iloc[va_i]

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        categorical_feature=lgb_cat_cols,
        callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)]
    )

    vp = model.predict_proba(X_va)[:, 1]
    oof_lgb[va_i] = vp
    ap = average_precision_score(y_va, vp)
    lgb_scores.append(ap)
    test_lgb += model.predict_proba(X_test)[:, 1] / 5
    print(f"  >>> Fold {fold+1}  AP={ap:.4f}  best_iter={model.best_iteration_}  [{time.time()-t1:.0f}s] <<<\n")

lgb_oof_ap = average_precision_score(y_train, oof_lgb)
print(f"  LightGBM Mean Fold AP : {np.mean(lgb_scores):.4f}")
print(f"  LightGBM OOF AP      : {lgb_oof_ap:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 9. RANK-AVERAGE ENSEMBLE
# ──────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Ensemble: Rank-Averaging CatBoost + LightGBM ...")
print("=" * 60)

# Try different blending weights on OOF
best_w, best_ap = 0.5, 0.0
for w in np.arange(0.0, 1.01, 0.01):
    blend = w * oof_cb + (1 - w) * oof_lgb
    ap = average_precision_score(y_train, blend)
    if ap > best_ap:
        best_ap = ap
        best_w = w
print(f"  Best linear blend: w_cb={best_w:.2f}  OOF AP={best_ap:.4f}")

# Also try rank averaging
rank_cb  = rankdata(oof_cb)  / len(oof_cb)
rank_lgb = rankdata(oof_lgb) / len(oof_lgb)

best_rw, best_rank_ap = 0.5, 0.0
for w in np.arange(0.0, 1.01, 0.01):
    blend = w * rank_cb + (1 - w) * rank_lgb
    ap = average_precision_score(y_train, blend)
    if ap > best_rank_ap:
        best_rank_ap = ap
        best_rw = w
print(f"  Best rank blend:   w_cb={best_rw:.2f}  OOF AP={best_rank_ap:.4f}")

# Use whichever ensemble is better
if best_rank_ap >= best_ap:
    print(f"  → Using RANK ensemble (AP={best_rank_ap:.4f})")
    oof_final = best_rw * rank_cb + (1 - best_rw) * rank_lgb
    rank_cb_test  = rankdata(test_cb)  / len(test_cb)
    rank_lgb_test = rankdata(test_lgb) / len(test_lgb)
    test_final = best_rw * rank_cb_test + (1 - best_rw) * rank_lgb_test
    final_ap = best_rank_ap
else:
    print(f"  → Using LINEAR ensemble (AP={best_ap:.4f})")
    oof_final = best_w * oof_cb + (1 - best_w) * oof_lgb
    test_final = best_w * test_cb + (1 - best_w) * test_lgb
    final_ap = best_ap

# ──────────────────────────────────────────────────────────────────────────
# 10. Post-Processing
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Post-Processing ===")

mask_min = (train_raw['minutes_played'] == 0).values
oof_pp = oof_final.copy()
oof_pp[mask_min] = 0.0
ap_pp = average_precision_score(y_train, oof_pp)
print(f"  minutes_played==0 rule → AP={ap_pp:.4f} (Δ{ap_pp-final_ap:+.4f})")

if ap_pp > final_ap:
    test_final_out = test_final.copy()
    test_final_out[test_df['minutes_played'] == 0] = 0.0
    final_ap = ap_pp
    print(f"  ✓ Applied post-processing. Final OOF AP: {final_ap:.4f}")
else:
    test_final_out = test_final.copy()
    print(f"  ✗ Post-processing did not help. Final OOF AP: {final_ap:.4f}")

# ──────────────────────────────────────────────────────────────────────────
# 11. Generate Submission
# ──────────────────────────────────────────────────────────────────────────
print("\n=== Generating Submission ===")
sub = pd.DataFrame({'appearance_id': test_df['appearance_id'], 'scored_flag': test_final_out})
assert not sub.isnull().any().any(), "Submission contains nulls!"
sub.to_csv(SOL_PATH, index=False)
sub.to_csv(os.path.join(WORKSPACE, "submission.csv"), index=False)

# Also save to Desktop
sub.to_csv("/Users/rana/Desktop/solution4.csv", index=False)

print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Train rows     : {len(X_train):,}")
print(f"  Features       : {len(FEATURES)}")
print(f"  CatBoost OOF AP: {cb_oof_ap:.4f}")
print(f"  LightGBM OOF AP: {lgb_oof_ap:.4f}")
print(f"  Ensemble OOF AP: {final_ap:.4f}")
print(f"  Total time     : {(time.time()-t0)/60:.1f} min")
print("=" * 60)
print("✅ Done! Submit solution.csv to Kaggle.")
