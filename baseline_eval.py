import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

from src.features import FeaturePipeline, generate_out_of_fold_target_encoding
from src.models import train_lgbm, train_catboost, blend_predictions

def main():
    workspace_dir = "/Users/rana/OFF SIDE"
    train_path = os.path.join(workspace_dir, 'train.csv')
    test_path = os.path.join(workspace_dir, 'test.csv')
    
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    target_col = 'scored_flag'
    y_train = train_df[target_col].copy()
    
    print("Fitting feature pipeline...")
    pipeline = FeaturePipeline(random_state=42)
    train_feat = pipeline.fit_transform(train_df, target_col=target_col)
    test_feat = pipeline.transform(test_df)
    
    print("Generating target encodings...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols_to_encode = []
    for col in ['name_y', 'home_club_name', 'away_club_name']:
        if col in train_df.columns:
            cols_to_encode.append(col)
            
    if len(cols_to_encode) > 0:
        train_feat_encoded = generate_out_of_fold_target_encoding(train_df, skf, target_col=target_col, cols_to_encode=cols_to_encode)
        for col in cols_to_encode:
            train_feat[f'{col}_target_enc'] = train_feat_encoded[f'{col}_target_enc']
            
    cols_to_drop = ['appearance_id', 'name_y', target_col, 'date']
    features_to_use = [col for col in train_feat.columns if col not in cols_to_drop]
    X_train = train_feat[features_to_use]
    X_test = test_feat[features_to_use]
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    # Baseline hyperparameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'n_estimators': 800,
        'n_jobs': -1
    }
    
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
    
    # Train LGBM
    oof_lgb, test_lgb, _, lgb_ap = train_lgbm(
        X_train, y_train, X_test,
        cat_cols=pipeline.cat_cols,
        params=lgb_params,
        n_splits=5,
        random_state=42
    )
    
    # Train CatBoost
    oof_cat, test_cat, _, cat_ap = train_catboost(
        X_train, y_train, X_test,
        cat_cols=pipeline.cat_cols,
        params=cat_params,
        n_splits=5,
        random_state=42
    )
    
    # Blend predictions
    oof_blend, test_blend, blend_weight = blend_predictions(
        oof_lgb, oof_cat,
        test_lgb, test_cat,
        y_train
    )
    
    print("\n=== Evaluating Post-Processing Rules on OOF Blend ===")
    
    # 1. Evaluate baseline OOF blend score
    baseline_ap = average_precision_score(y_train, oof_blend)
    print(f"Baseline Blend OOF AP: {baseline_ap:.6f}")
    
    # 2. Evaluate with team_goals == 0 rule
    # In training, team goals can be calculated
    train_team_goals = train_df.apply(lambda r: r['home_club_goals'] if r['home_away'] == 'HOME' else r['away_club_goals'], axis=1)
    
    oof_rule_team = oof_blend.copy()
    oof_rule_team[train_team_goals == 0] = 0.0
    team_rule_ap = average_precision_score(y_train, oof_rule_team)
    print(f"OOF AP with team_goals == 0 rule: {team_rule_ap:.6f} (Diff: {team_rule_ap - baseline_ap:+.6f})")
    
    # 3. Evaluate with minutes_played == 0 rule
    oof_rule_min = oof_blend.copy()
    oof_rule_min[train_df['minutes_played'] == 0] = 0.0
    min_rule_ap = average_precision_score(y_train, oof_rule_min)
    print(f"OOF AP with minutes_played == 0 rule: {min_rule_ap:.6f} (Diff: {min_rule_ap - baseline_ap:+.6f})")
    
    # 4. Combined rules
    oof_rule_combined = oof_blend.copy()
    oof_rule_combined[(train_team_goals == 0) | (train_df['minutes_played'] == 0)] = 0.0
    combined_rule_ap = average_precision_score(y_train, oof_rule_combined)
    print(f"OOF AP with both rules: {combined_rule_ap:.6f} (Diff: {combined_rule_ap - baseline_ap:+.6f})")

if __name__ == '__main__':
    main()
