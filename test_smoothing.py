import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier, Pool

def generate_smoothed_target_encoding_oof(train_df, kfold, target_col='scored_flag', cols_to_encode=['name_y', 'home_club_name', 'away_club_name'], m=10):
    encoded_train = train_df.copy()
    global_mean = train_df[target_col].mean()
    
    for col in cols_to_encode:
        encoded_train[f'{col}_target_enc'] = global_mean
        
    for train_idx, val_idx in kfold.split(train_df, train_df[target_col]):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        for col in cols_to_encode:
            # Smoothed target encoding
            stats = train_fold.groupby(col)[target_col].agg(['sum', 'count'])
            col_means = (stats['sum'] + m * global_mean) / (stats['count'] + m)
            encoded_train.iloc[val_idx, encoded_train.columns.get_loc(f'{col}_target_enc')] = val_fold[col].map(col_means).fillna(global_mean)
            
    return encoded_train

def evaluate_m(m):
    workspace_dir = "/Users/rana/OFF SIDE"
    train_path = os.path.join(workspace_dir, 'train.csv')
    train_df = pd.read_csv(train_path)
    target_col = 'scored_flag'
    y_train = train_df[target_col].copy()
    
    # Simple feature prep
    from src.features import FeaturePipeline
    pipeline = FeaturePipeline(random_state=42)
    train_feat = pipeline.fit_transform(train_df, target_col=target_col)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols_to_encode = ['name_y', 'home_club_name', 'away_club_name']
    
    train_feat_encoded = generate_smoothed_target_encoding_oof(train_df, skf, target_col=target_col, cols_to_encode=cols_to_encode, m=m)
    for col in cols_to_encode:
        train_feat[f'{col}_target_enc'] = train_feat_encoded[f'{col}_target_enc']
        
    cols_to_drop = ['appearance_id', 'name_y', target_col, 'date']
    features_to_use = [col for col in train_feat.columns if col not in cols_to_drop]
    X_train = train_feat[features_to_use]
    
    cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'learning_rate': 0.05,
        'depth': 6,
        'iterations': 500,
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU',
        'thread_count': -1
    }
    
    # We can do 3 folds for speed in finding the best m
    skf_eval = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cat_features = [col for col in pipeline.cat_cols if col in X_train.columns]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_features]
    
    oof_preds = np.zeros(len(X_train))
    scores = []
    
    for train_idx, val_idx in skf_eval.split(X_train, y_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_va, y_va, cat_features=cat_indices)
        
        model = CatBoostClassifier(**cat_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = val_pred
        scores.append(average_precision_score(y_va, val_pred))
        
    mean_ap = np.mean(scores)
    # Apply post-processing rule
    train_team_goals = train_df.apply(lambda r: r['home_club_goals'] if r['home_away'] == 'HOME' else r['away_club_goals'], axis=1)
    oof_preds_rule = oof_preds.copy()
    oof_preds_rule[(train_team_goals == 0) | (train_df['minutes_played'] == 0)] = 0.0
    rule_ap = average_precision_score(y_train, oof_preds_rule)
    
    print(f"Smoothing m={m} | Mean 3-Fold AP: {mean_ap:.6f} | Post-processed AP: {rule_ap:.6f}")

def main():
    for m in [0, 5, 10, 20, 50]:
        evaluate_m(m)

if __name__ == '__main__':
    main()
