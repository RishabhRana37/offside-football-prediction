"""
NUCLEAR OPTION: 4-model diverse ensemble, multi-smoothing TE, count features
Target: 0.50 AP. Must finish in ~10 min.
"""
import os,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata
from catboost import CatBoostClassifier,Pool
import lightgbm as lgb
from src.features import FeaturePipeline

t0=time.time()
WS="/Users/rana/OFF SIDE"
TP="/Users/rana/Downloads/train_cleaned.csv"
if not os.path.exists(TP): TP=os.path.join(WS,"train.csv")
TARGET="scored_flag"

print("="*60); print("NUCLEAR BOOST — 4-model ensemble"); print("="*60)
print("Loading...")
train_raw=pd.read_csv(TP,low_memory=False)
test_df=pd.read_csv(os.path.join(WS,"test.csv"),low_memory=False)
y_raw=train_raw[TARGET].astype(int).copy()
GM=y_raw.mean()
print(f"  train:{train_raw.shape} test:{test_df.shape} pos:{GM:.4f} [{time.time()-t0:.0f}s]")

# Pipeline
pipe=FeaturePipeline(random_state=42)
train_feat=pipe.fit_transform(train_raw,target_col=TARGET)
test_feat=pipe.transform(test_df)
print(f"  pipeline [{time.time()-t0:.0f}s]")

# ── EXTRA FEATURES ──
def xf(s,f):
    c={}
    mr=s.get('minutes_ratio',pd.Series(0.,index=s.index))
    xg=s.get('avg_xG',pd.Series(0.,index=s.index))
    xa=s.get('avg_xA',pd.Series(0.,index=s.index))
    sh=s.get('avg_shots',pd.Series(0.,index=s.index))
    npx=s.get('avg_npxG',pd.Series(0.,index=s.index))
    xgc=s.get('avg_xGChain',pd.Series(0.,index=s.index))
    xgb=s.get('avg_xGBuildup',pd.Series(0.,index=s.index))
    kp=s.get('avg_key_passes',pd.Series(0.,index=s.index))
    mv=s.get('market_value_before_match',pd.Series(0.,index=s.index))
    pmv=s.get('highest_market_value_in_eur',pd.Series(0.,index=s.index))
    caps=s.get('international_caps',pd.Series(0.,index=s.index))
    goals=s.get('international_goals',pd.Series(0.,index=s.index))
    age=s.get('age',pd.Series(26.,index=s.index))
    att=s.get('attendance',pd.Series(0.,index=s.index)).fillna(0)
    lmv=s.get('log_market_value',pd.Series(0.,index=s.index))
    ht=s.get('height_in_cm',pd.Series(180.,index=s.index))

    c['exp_npxG']=(npx*mr).values;c['exp_xGC']=(xgc*mr).values
    c['exp_xGB']=(xgb*mr).values;c['exp_kp']=(kp*mr).values
    c['xG_per_shot2']=(xg/(sh.fillna(0)+1e-5)).values
    c['npxG_frac']=(npx/(xg+1e-5)).values
    c['pen_xG']=(xg-npx).values
    c['shots_per_kp']=(sh/(kp+1e-5)).values
    c['mv_rat']=(mv/(pmv+1e-5)).values
    c['log_mv_sq']=(lmv**2).values
    c['intl_eff']=(goals/(caps+1e-5)).values
    c['intl_g_sq']=(goals**2).values
    c['age_xG']=(age*xg).values;c['age_sq']=(age**2).values
    c['prime_xG']=np.where((age>=24)&(age<=30),xg.values,0.)
    c['ht_xG']=(ht*xg).values
    c['log_att']=np.log1p(att).values
    # xG + xA total threat
    c['total_threat']=(xg+xa).values
    c['threat_in_match']=((xg+xa)*mr).values
    # xGChain minus xGBuildup = final third contribution
    c['final_third']=(xgc-xgb).values
    c['final_third_match']=((xgc-xgb)*mr).values
    # Shots on target proxy
    c['xG_x_shots']=(xg*sh).values
    c['xG_x_shots_match']=(xg*sh*mr).values
    # Position interactions
    if 'is_attacker' in f.columns:
        c['att_mv']=f['is_attacker'].values*mv.values
        c['att_xG']=f['is_attacker'].values*xg.values
        c['att_sh']=f['is_attacker'].values*sh.values
        c['att_mr']=f['is_attacker'].values*mr.values
        c['att_threat']=f['is_attacker'].values*(xg+xa).values
    if 'is_midfielder' in f.columns:
        c['mid_xG']=f['is_midfielder'].values*xg.values
        c['mid_sh']=f['is_midfielder'].values*sh.values
        c['mid_threat']=f['is_midfielder'].values*(xg+xa).values
    if 'is_defender' in f.columns:
        c['def_xG']=f['is_defender'].values*xg.values
    if 'starter_flag' in f.columns:
        c['strt_xG']=f['starter_flag'].values*xg.values
        c['strt_mr']=f['starter_flag'].values*mr.values
    if 'home_club_goals' in s.columns and 'away_club_goals' in s.columns:
        ih=s['home_away'].str.upper()=='HOME'
        c['team_g']=np.where(ih,s['home_club_goals'],s['away_club_goals'])
        c['opp_g']=np.where(ih,s['away_club_goals'],s['home_club_goals'])
        c['tot_g']=(s['home_club_goals']+s['away_club_goals']).values
    return c

train_feat=pd.concat([train_feat,pd.DataFrame(xf(train_raw,train_feat),index=train_feat.index)],axis=1)
test_feat=pd.concat([test_feat,pd.DataFrame(xf(test_df,test_feat),index=test_feat.index)],axis=1)
print(f"  extra feats [{time.time()-t0:.0f}s]")

# ── FREQUENCY + COUNT ENCODING ──
FREQ_COLS=[c for c in ['name_y','home_club_name','away_club_name','stadium','referee','country_name','sub_position','country_of_citizenship'] if c in train_raw.columns]
for col in FREQ_COLS:
    fm=train_raw[col].value_counts(normalize=True).to_dict()
    cm=train_raw[col].value_counts().to_dict()
    train_feat[f'{col}_freq']=train_raw[col].map(fm).fillna(0).values
    test_feat[f'{col}_freq']=test_df[col].map(fm).fillna(0).values if col in test_df.columns else 0
    train_feat[f'{col}_count']=train_raw[col].map(cm).fillna(0).values
    test_feat[f'{col}_count']=test_df[col].map(cm).fillna(0).values if col in test_df.columns else 0

# ── MULTI-SMOOTHING TARGET ENCODING ──
SKF=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
EC=[c for c in ['name_y','home_club_name','away_club_name','competition_type','referee','sub_position','country_name','confederation','stadium'] if c in train_raw.columns]

te_tr,te_te={},{}
for SM in [5,20,50]:  # Multiple smoothing factors = more signal
    for col in EC:
        enc=f'{col}_te_m{SM}'
        a=np.full(len(train_raw),GM,dtype=np.float32)
        for ti,vi in SKF.split(train_raw,y_raw):
            st=train_raw.iloc[ti].groupby(col)[TARGET].agg(['sum','count'])
            sm=(st['sum']+SM*GM)/(st['count']+SM)
            a[vi]=train_raw.iloc[vi][col].map(sm).fillna(GM).values.astype(np.float32)
        te_tr[enc]=a
        fs=train_raw.groupby(col)[TARGET].agg(['sum','count'])
        sf=(fs['sum']+SM*GM)/(fs['count']+SM)
        te_te[enc]=test_df[col].map(sf).fillna(GM).values.astype(np.float32) if col in test_df.columns else np.full(len(test_df),GM,dtype=np.float32)
    print(f"  TE m={SM} [{time.time()-t0:.0f}s]")

# Player×club, Player×competition, Player×home_away interaction TE
INTERACTIONS = []
if 'name_y' in train_raw.columns:
    if 'home_club_name' in train_raw.columns:
        INTERACTIONS.append(('name_y','home_club_name'))
    if 'competition_type' in train_raw.columns:
        INTERACTIONS.append(('name_y','competition_type'))
    if 'home_away' in train_raw.columns:
        INTERACTIONS.append(('name_y','home_away'))

for c1,c2 in INTERACTIONS:
    key=f'{c1}_{c2}'
    train_raw[f'_i_{key}']=train_raw[c1].astype(str)+'_'+train_raw[c2].astype(str)
    test_df[f'_i_{key}']=test_df[c1].astype(str)+'_'+test_df[c2].astype(str)
    enc=f'_i_{key}_te'
    a=np.full(len(train_raw),GM,dtype=np.float32)
    for ti,vi in SKF.split(train_raw,y_raw):
        st=train_raw.iloc[ti].groupby(f'_i_{key}')[TARGET].agg(['sum','count'])
        sm=(st['sum']+20*GM)/(st['count']+20)
        a[vi]=train_raw.iloc[vi][f'_i_{key}'].map(sm).fillna(GM).values.astype(np.float32)
    te_tr[enc]=a
    fs=train_raw.groupby(f'_i_{key}')[TARGET].agg(['sum','count'])
    sf=(fs['sum']+20*GM)/(fs['count']+20)
    te_te[enc]=test_df[f'_i_{key}'].map(sf).fillna(GM).values.astype(np.float32)
    print(f"  interaction TE: {key} [{time.time()-t0:.0f}s]")

train_feat=pd.concat([train_feat,pd.DataFrame(te_tr,index=train_feat.index)],axis=1)
test_feat=pd.concat([test_feat,pd.DataFrame(te_te,index=test_feat.index)],axis=1)
print(f"  ALL TE done [{time.time()-t0:.0f}s]")

# ── Feature matrix ──
DROP={'appearance_id','name_y',TARGET,'date','home_club_goals','away_club_goals','home_club_id','away_club_id'}
DROP.update([c for c in train_feat.columns if c.startswith('_i_')])
FEATS=[c for c in train_feat.columns if c not in DROP and train_feat[c].dtype!=object]
X=train_feat[FEATS].copy();Xt=test_feat.reindex(columns=FEATS,fill_value=0).copy()
y=y_raw.reset_index(drop=True)
cat_i=[FEATS.index(c) for c in pipe.cat_cols if c in FEATS]
cat_n=[c for c in pipe.cat_cols if c in FEATS]
print(f"\n  FEATURES: {len(FEATS)}   [{time.time()-t0:.0f}s]")

# ══════════════════════════════════════════════════════════════
# 4 DIVERSE MODELS
# ══════════════════════════════════════════════════════════════
all_oof=[]
all_test=[]
all_names=[]

# MODEL 1: CatBoost depth=7, seed=42
print("\n"+"="*50+"\nMODEL 1: CatBoost d=7 s=42\n"+"="*50)
oof1=np.zeros(len(X));t1p=np.zeros(len(Xt));s1=[]
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m=CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',learning_rate=0.05,
        depth=7,l2_leaf_reg=3.,iterations=1000,random_seed=42,verbose=500,
        task_type='CPU',thread_count=-1,class_weights=[1.,3.],od_type='Iter',od_wait=80,
        random_strength=0.5,bagging_temperature=0.8)
    m.fit(Pool(X.iloc[ti],y.iloc[ti],cat_features=cat_i),eval_set=Pool(X.iloc[vi],y.iloc[vi],cat_features=cat_i),
          early_stopping_rounds=80,verbose=500)
    vp=m.predict_proba(Pool(X.iloc[vi],cat_features=cat_i))[:,1]
    oof1[vi]=vp;ap=average_precision_score(y.iloc[vi],vp);s1.append(ap)
    t1p+=m.predict_proba(Pool(Xt,cat_features=cat_i))[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} [{time.time()-t1:.0f}s]")
print(f"  CB1 OOF AP: {average_precision_score(y,oof1):.4f}")
all_oof.append(oof1);all_test.append(t1p);all_names.append('CB1')

# MODEL 2: CatBoost depth=6, seed=123, different regularization
print("\n"+"="*50+"\nMODEL 2: CatBoost d=6 s=123\n"+"="*50)
oof2=np.zeros(len(X));t2p=np.zeros(len(Xt));s2=[]
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m=CatBoostClassifier(loss_function='Logloss',eval_metric='AUC',learning_rate=0.05,
        depth=6,l2_leaf_reg=5.,iterations=1000,random_seed=123,verbose=500,
        task_type='CPU',thread_count=-1,class_weights=[1.,3.5],od_type='Iter',od_wait=80,
        random_strength=1.0,bagging_temperature=1.0)
    m.fit(Pool(X.iloc[ti],y.iloc[ti],cat_features=cat_i),eval_set=Pool(X.iloc[vi],y.iloc[vi],cat_features=cat_i),
          early_stopping_rounds=80,verbose=500)
    vp=m.predict_proba(Pool(X.iloc[vi],cat_features=cat_i))[:,1]
    oof2[vi]=vp;ap=average_precision_score(y.iloc[vi],vp);s2.append(ap)
    t2p+=m.predict_proba(Pool(Xt,cat_features=cat_i))[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} [{time.time()-t1:.0f}s]")
print(f"  CB2 OOF AP: {average_precision_score(y,oof2):.4f}")
all_oof.append(oof2);all_test.append(t2p);all_names.append('CB2')

# MODEL 3: LightGBM num_leaves=63
print("\n"+"="*50+"\nMODEL 3: LightGBM nl=63\n"+"="*50)
oof3=np.zeros(len(X));t3p=np.zeros(len(Xt));s3=[]
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',boosting_type='gbdt',
        learning_rate=0.03,num_leaves=63,feature_fraction=0.7,bagging_fraction=0.8,
        bagging_freq=1,min_child_samples=20,n_estimators=1200,verbose=-1,random_state=42,
        n_jobs=-1,scale_pos_weight=3.,reg_alpha=0.1,reg_lambda=1.)
    m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],
          categorical_feature=cat_n,callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(500)])
    vp=m.predict_proba(X.iloc[vi])[:,1]
    oof3[vi]=vp;ap=average_precision_score(y.iloc[vi],vp);s3.append(ap)
    t3p+=m.predict_proba(Xt)[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} [{time.time()-t1:.0f}s]")
print(f"  LGB1 OOF AP: {average_precision_score(y,oof3):.4f}")
all_oof.append(oof3);all_test.append(t3p);all_names.append('LGB1')

# MODEL 4: LightGBM num_leaves=31, different config
print("\n"+"="*50+"\nMODEL 4: LightGBM nl=31\n"+"="*50)
oof4=np.zeros(len(X));t4p=np.zeros(len(Xt));s4=[]
for f,(ti,vi) in enumerate(SKF.split(X,y)):
    t1=time.time()
    m=lgb.LGBMClassifier(objective='binary',metric='binary_logloss',boosting_type='gbdt',
        learning_rate=0.05,num_leaves=31,max_depth=7,feature_fraction=0.8,bagging_fraction=0.9,
        bagging_freq=1,min_child_samples=30,n_estimators=1200,verbose=-1,random_state=99,
        n_jobs=-1,scale_pos_weight=2.5,reg_alpha=0.05,reg_lambda=0.5)
    m.fit(X.iloc[ti],y.iloc[ti],eval_set=[(X.iloc[vi],y.iloc[vi])],
          categorical_feature=cat_n,callbacks=[lgb.early_stopping(60,verbose=False),lgb.log_evaluation(500)])
    vp=m.predict_proba(X.iloc[vi])[:,1]
    oof4[vi]=vp;ap=average_precision_score(y.iloc[vi],vp);s4.append(ap)
    t4p+=m.predict_proba(Xt)[:,1]/5
    print(f"  F{f+1} AP={ap:.4f} [{time.time()-t1:.0f}s]")
print(f"  LGB2 OOF AP: {average_precision_score(y,oof4):.4f}")
all_oof.append(oof4);all_test.append(t4p);all_names.append('LGB2')

# ══════════════════════════════════════════════════════════════
# OPTIMAL ENSEMBLE via greedy weight search
# ══════════════════════════════════════════════════════════════
print("\n"+"="*50+"\nENSEMBLE OPTIMIZATION\n"+"="*50)

# Convert to ranks for rank averaging
ranks_oof=[rankdata(o)/len(o) for o in all_oof]
ranks_test=[rankdata(t)/len(t) for t in all_test]

# Greedy weight search over rank averages
best_ap=0;best_w=None
for w0 in np.arange(0.1,0.6,0.05):
    for w1 in np.arange(0.05,0.4,0.05):
        for w2 in np.arange(0.05,0.4,0.05):
            w3=1.0-w0-w1-w2
            if w3<0.01: continue
            blend=w0*ranks_oof[0]+w1*ranks_oof[1]+w2*ranks_oof[2]+w3*ranks_oof[3]
            ap=average_precision_score(y,blend)
            if ap>best_ap: best_ap=ap;best_w=[w0,w1,w2,w3]

print(f"  Best weights: {[f'{w:.2f}' for w in best_w]}")
print(f"  Best rank ensemble OOF AP: {best_ap:.4f}")

# Also try simple average
simple_rank=sum(ranks_oof)/4
simple_ap=average_precision_score(y,simple_rank)
print(f"  Simple average OOF AP: {simple_ap:.4f}")

# Also try linear blend optimization
best_lin_ap=0;best_lw=None
for w0 in np.arange(0.1,0.6,0.05):
    for w1 in np.arange(0.05,0.4,0.05):
        for w2 in np.arange(0.05,0.4,0.05):
            w3=1.0-w0-w1-w2
            if w3<0.01: continue
            blend=w0*all_oof[0]+w1*all_oof[1]+w2*all_oof[2]+w3*all_oof[3]
            ap=average_precision_score(y,blend)
            if ap>best_lin_ap: best_lin_ap=ap;best_lw=[w0,w1,w2,w3]
print(f"  Best linear ensemble OOF AP: {best_lin_ap:.4f}")

# Use the best approach
if best_ap>=best_lin_ap and best_ap>=simple_ap:
    oof_f=best_w[0]*ranks_oof[0]+best_w[1]*ranks_oof[1]+best_w[2]*ranks_oof[2]+best_w[3]*ranks_oof[3]
    t_f=best_w[0]*ranks_test[0]+best_w[1]*ranks_test[1]+best_w[2]*ranks_test[2]+best_w[3]*ranks_test[3]
    fap=best_ap;method="RANK-WEIGHTED"
elif best_lin_ap>=simple_ap:
    oof_f=best_lw[0]*all_oof[0]+best_lw[1]*all_oof[1]+best_lw[2]*all_oof[2]+best_lw[3]*all_oof[3]
    t_f=best_lw[0]*all_test[0]+best_lw[1]*all_test[1]+best_lw[2]*all_test[2]+best_lw[3]*all_test[3]
    fap=best_lin_ap;method="LINEAR-WEIGHTED"
else:
    oof_f=simple_rank;t_f=sum(ranks_test)/4;fap=simple_ap;method="SIMPLE-AVG"

print(f"  → {method} AP={fap:.4f}")

# Post-proc
mask=(train_raw['minutes_played']==0).values
oof_pp=oof_f.copy();oof_pp[mask]=0.
app=average_precision_score(y,oof_pp)
print(f"  Post-proc: AP={app:.4f} (Δ{app-fap:+.4f})")
if app>fap:
    t_out=t_f.copy();t_out[test_df['minutes_played']==0]=0.;fap=app
else:
    t_out=t_f.copy()

# SAVE
sub=pd.DataFrame({'appearance_id':test_df['appearance_id'],'scored_flag':t_out})
sub.to_csv(os.path.join(WS,"solution.csv"),index=False)
sub.to_csv(os.path.join(WS,"submission.csv"),index=False)
sub.to_csv("/Users/rana/Desktop/solution4.csv",index=False)

print("\n"+"="*60)
for i,n in enumerate(all_names):
    print(f"  {n} OOF AP: {average_precision_score(y,all_oof[i]):.4f}")
print(f"  ENSEMBLE ({method}): {fap:.4f}")
print(f"  Features: {len(FEATS)}")
print(f"  Time: {(time.time()-t0)/60:.1f} min")
print("="*60)
print("✅ SAVED: solution.csv + Desktop/solution4.csv")
