"""
FAST boost — finish in ~5 min
CatBoost depth=7 lr=0.05 800iter + LightGBM 1000iter, 5-fold, rank ensemble
"""
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb
from src.features import FeaturePipeline

t0 = time.time()
WS = "/Users/rana/OFF SIDE"
TP = "/Users/rana/Downloads/train_cleaned.csv"
if not os.path.exists(TP): TP = os.path.join(WS, "train.csv")
TARGET = "scored_flag"

print("Loading ..."); 
train_raw = pd.read_csv(TP, low_memory=False)
test_df = pd.read_csv(os.path.join(WS, "test.csv"), low_memory=False)
y_raw = train_raw[TARGET].astype(int).copy()
GM = y_raw.mean()
print(f"  train:{train_raw.shape} test:{test_df.shape} pos:{GM:.4f} [{time.time()-t0:.0f}s]")

# Pipeline
pipe = FeaturePipeline(random_state=42)
train_feat = pipe.fit_transform(train_raw, target_col=TARGET)
test_feat = pipe.transform(test_df)
print(f"  pipeline [{time.time()-t0:.0f}s]")

# Extra features
def xf(src, feat):
    c = {}
    mr = src.get('minutes_ratio', pd.Series(0., index=src.index))
    xg = src.get('avg_xG', pd.Series(0., index=src.index))
    xa = src.get('avg_xA', pd.Series(0., index=src.index))
    sh = src.get('avg_shots', pd.Series(0., index=src.index))
    npx = src.get('avg_npxG', pd.Series(0., index=src.index))
    xgc = src.get('avg_xGChain', pd.Series(0., index=src.index))
    xgb = src.get('avg_xGBuildup', pd.Series(0., index=src.index))
    kp = src.get('avg_key_passes', pd.Series(0., index=src.index))
    mv = src.get('market_value_before_match', pd.Series(0., index=src.index))
    pmv = src.get('highest_market_value_in_eur', pd.Series(0., index=src.index))
    caps = src.get('international_caps', pd.Series(0., index=src.index))
    goals = src.get('international_goals', pd.Series(0., index=src.index))
    age = src.get('age', pd.Series(26., index=src.index))
    att = src.get('attendance', pd.Series(0., index=src.index)).fillna(0)
    lmv = src.get('log_market_value', pd.Series(0., index=src.index))
    height = src.get('height_in_cm', pd.Series(180., index=src.index))

    c['exp_npxG'] = (npx*mr).values; c['exp_xGC'] = (xgc*mr).values
    c['exp_xGB'] = (xgb*mr).values; c['exp_kp'] = (kp*mr).values
    c['xG_per_shot2'] = (xg/(sh.fillna(0)+1e-5)).values
    c['npxG_frac'] = (npx/(xg+1e-5)).values
    c['pen_xG'] = (xg-npx).values
    c['shots_per_kp'] = (sh/(kp+1e-5)).values
    c['mv_ratio_pk'] = (mv/(pmv+1e-5)).values
    c['log_mv_sq'] = (lmv**2).values
    c['intl_eff'] = (goals/(caps+1e-5)).values
    c['intl_g_sq'] = (goals**2).values
    c['age_xG'] = (age*xg).values; c['age_sq'] = (age**2).values
    c['prime_xG'] = np.where((age>=24)&(age<=30), xg.values, 0.)
    c['height_xG'] = (height*xg).values
    c['log_att'] = np.log1p(att).values
    if 'is_attacker' in feat.columns:
        c['att_mv'] = feat['is_attacker'].values*mv.values
        c['att_xG'] = feat['is_attacker'].values*xg.values
        c['att_sh'] = feat['is_attacker'].values*sh.values
        c['att_mr'] = feat['is_attacker'].values*mr.values
    if 'is_midfielder' in feat.columns:
        c['mid_xG'] = feat['is_midfielder'].values*xg.values
        c['mid_sh'] = feat['is_midfielder'].values*sh.values
    if 'is_defender' in feat.columns:
        c['def_xG'] = feat['is_defender'].values*xg.values
    if 'starter_flag' in feat.columns:
        c['strt_xG'] = feat['starter_flag'].values*xg.values
        c['strt_mr'] = feat['starter_flag'].values*mr.values
    if 'home_club_goals' in src.columns and 'away_club_goals' in src.columns:
        ih = src['home_away'].str.upper()=='HOME'
        c['team_g'] = np.where(ih, src['home_club_goals'], src['away_club_goals'])
        c['opp_g'] = np.where(ih, src['away_club_goals'], src['home_club_goals'])
        c['tot_g'] = (src['home_club_goals']+src['away_club_goals']).values
    return c

train_feat = pd.concat([train_feat, pd.DataFrame(xf(train_raw, train_feat), index=train_feat.index)], axis=1)
test_feat = pd.concat([test_feat, pd.DataFrame(xf(test_df, test_feat), index=test_feat.index)], axis=1)
print(f"  extra feats [{time.time()-t0:.0f}s]")

# Frequency encoding
for col in [c for c in ['name_y','home_club_name','away_club_name','stadium','referee','country_name','sub_position','country_of_citizenship'] if c in train_raw.columns]:
    fm = train_raw[col].value_counts(normalize=True).to_dict()
    train_feat[f'{col}_freq'] = train_raw[col].map(fm).fillna(0).values
    test_feat[f'{col}_freq'] = test_df[col].map(fm).fillna(0).values if col in test_df.columns else 0

# Target encoding (extended)
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SM = 20
EC = [c for c in ['name_y','home_club_name','away_club_name','competition_type','referee','sub_position','country_name','confederation','stadium'] if c in train_raw.columns]
te_tr, te_te = {}, {}
for col in EC:
    enc = f'{col}_te'
    a = np.full(len(train_raw), GM, dtype=np.float32)
    for ti, vi in SKF.split(train_raw, y_raw):
        s = train_raw.iloc[ti].groupby(col)[TARGET].agg(['sum','count'])
        sm = (s['sum']+SM*GM)/(s['count']+SM)
        a[vi] = train_raw.iloc[vi][col].map(sm).fillna(GM).values.astype(np.float32)
    te_tr[enc] = a
    fs = train_raw.groupby(col)[TARGET].agg(['sum','count'])
    sf = (fs['sum']+SM*GM)/(fs['count']+SM)
    te_te[enc] = test_df[col].map(sf).fillna(GM).values.astype(np.float32) if col in test_df.columns else np.full(len(test_df), GM, dtype=np.float32)

# Player×club interaction TE
if 'name_y' in train_raw.columns and 'home_club_name' in train_raw.columns:
    train_raw['_pc'] = train_raw['name_y'].astype(str)+'_'+train_raw['home_club_name'].astype(str)
    test_df['_pc'] = test_df['name_y'].astype(str)+'_'+test_df['home_club_name'].astype(str)
    a = np.full(len(train_raw), GM, dtype=np.float32)
    for ti, vi in SKF.split(train_raw, y_raw):
        s = train_raw.iloc[ti].groupby('_pc')[TARGET].agg(['sum','count'])
        sm = (s['sum']+SM*GM)/(s['count']+SM)
        a[vi] = train_raw.iloc[vi]['_pc'].map(sm).fillna(GM).values.astype(np.float32)
    te_tr['_pc_te'] = a
    fs = train_raw.groupby('_pc')[TARGET].agg(['sum','count'])
    sf = (fs['sum']+SM*GM)/(fs['count']+SM)
    te_te['_pc_te'] = test_df['_pc'].map(sf).fillna(GM).values.astype(np.float32)

train_feat = pd.concat([train_feat, pd.DataFrame(te_tr, index=train_feat.index)], axis=1)
test_feat = pd.concat([test_feat, pd.DataFrame(te_te, index=test_feat.index)], axis=1)
print(f"  TE done [{time.time()-t0:.0f}s]")

# Feature matrix
DROP = {'appearance_id','name_y',TARGET,'date','_pc','home_club_goals','away_club_goals','home_club_id','away_club_id'}
FEATS = [c for c in train_feat.columns if c not in DROP and train_feat[c].dtype != object]
X = train_feat[FEATS].copy(); Xt = test_feat.reindex(columns=FEATS, fill_value=0).copy()
y = y_raw.reset_index(drop=True)
cat_i = [FEATS.index(c) for c in pipe.cat_cols if c in FEATS]
cat_n = [c for c in pipe.cat_cols if c in FEATS]
print(f"  X:{X.shape} [{time.time()-t0:.0f}s]")

# ── CatBoost ──
print("\n"+"="*50+"\nCatBoost 5-fold (depth=7, lr=0.05, 1200iter)\n"+"="*50)
oof_cb = np.zeros(len(X)); t_cb = np.zeros(len(Xt)); cb_s = []
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', learning_rate=0.05,
        depth=7, l2_leaf_reg=3., iterations=1200, random_seed=42, verbose=300,
        task_type='CPU', thread_count=-1, class_weights=[1.,3.], od_type='Iter', od_wait=80,
        random_strength=0.5, bagging_temperature=0.8)
    m.fit(Pool(X.iloc[ti],y.iloc[ti],cat_features=cat_i), eval_set=Pool(X.iloc[vi],y.iloc[vi],cat_features=cat_i),
          early_stopping_rounds=80, verbose=300)
    vp = m.predict_proba(Pool(X.iloc[vi], cat_features=cat_i))[:,1]
    oof_cb[vi] = vp; ap = average_precision_score(y.iloc[vi], vp); cb_s.append(ap)
    t_cb += m.predict_proba(Pool(Xt, cat_features=cat_i))[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} iter={m.best_iteration_} [{time.time()-t1:.0f}s]")
cb_ap = average_precision_score(y, oof_cb)
print(f"  CB OOF AP: {cb_ap:.4f}")

# ── LightGBM ──
print("\n"+"="*50+"\nLightGBM 5-fold\n"+"="*50)
oof_lg = np.zeros(len(X)); t_lg = np.zeros(len(Xt)); lg_s = []
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m = lgb.LGBMClassifier(objective='binary', metric='binary_logloss', boosting_type='gbdt',
        learning_rate=0.03, num_leaves=63, feature_fraction=0.7, bagging_fraction=0.8,
        bagging_freq=1, min_child_samples=20, n_estimators=1500, verbose=-1, random_state=42,
        n_jobs=-1, scale_pos_weight=3., reg_alpha=0.1, reg_lambda=1.)
    m.fit(X.iloc[ti], y.iloc[ti], eval_set=[(X.iloc[vi], y.iloc[vi])],
          categorical_feature=cat_n, callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(500)])
    vp = m.predict_proba(X.iloc[vi])[:,1]
    oof_lg[vi] = vp; ap = average_precision_score(y.iloc[vi], vp); lg_s.append(ap)
    t_lg += m.predict_proba(Xt)[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} iter={m.best_iteration_} [{time.time()-t1:.0f}s]")
lg_ap = average_precision_score(y, oof_lg)
print(f"  LGB OOF AP: {lg_ap:.4f}")

# ── Ensemble ──
print("\n"+"="*50+"\nEnsemble\n"+"="*50)
bw, ba = .5, 0.
for w in np.arange(0,1.01,.01):
    a = average_precision_score(y, w*oof_cb+(1-w)*oof_lg)
    if a>ba: ba=a; bw=w
print(f"  Linear: w_cb={bw:.2f} AP={ba:.4f}")

rk_cb = rankdata(oof_cb)/len(oof_cb); rk_lg = rankdata(oof_lg)/len(oof_lg)
brw, bra = .5, 0.
for w in np.arange(0,1.01,.01):
    a = average_precision_score(y, w*rk_cb+(1-w)*rk_lg)
    if a>bra: bra=a; brw=w
print(f"  Rank:   w_cb={brw:.2f} AP={bra:.4f}")

if bra >= ba:
    oof_f = brw*rk_cb+(1-brw)*rk_lg
    rt_cb = rankdata(t_cb)/len(t_cb); rt_lg = rankdata(t_lg)/len(t_lg)
    t_f = brw*rt_cb+(1-brw)*rt_lg; fap = bra; print(f"  → RANK ensemble AP={fap:.4f}")
else:
    oof_f = bw*oof_cb+(1-bw)*oof_lg; t_f = bw*t_cb+(1-bw)*t_lg; fap = ba
    print(f"  → LINEAR ensemble AP={fap:.4f}")

# Post-proc
mask = (train_raw['minutes_played']==0).values
oof_pp = oof_f.copy(); oof_pp[mask] = 0.
app = average_precision_score(y, oof_pp)
print(f"  Post-proc: AP={app:.4f} (Δ{app-fap:+.4f})")
if app > fap:
    t_out = t_f.copy(); t_out[test_df['minutes_played']==0] = 0.; fap = app
else:
    t_out = t_f.copy()

# Save
sub = pd.DataFrame({'appearance_id': test_df['appearance_id'], 'scored_flag': t_out})
sub.to_csv(os.path.join(WS,"solution.csv"), index=False)
sub.to_csv(os.path.join(WS,"submission.csv"), index=False)
sub.to_csv("/Users/rana/Desktop/solution4.csv", index=False)

print("\n"+"="*50)
print(f"  Features   : {len(FEATS)}")
print(f"  CB OOF AP  : {cb_ap:.4f}")
print(f"  LGB OOF AP : {lg_ap:.4f}")
print(f"  FINAL AP   : {fap:.4f}")
print(f"  Time       : {(time.time()-t0)/60:.1f} min")
print("="*50)
print("✅ Submit solution.csv / Desktop/solution4.csv")
