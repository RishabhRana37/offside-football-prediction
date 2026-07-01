import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from catboost import CatBoostClassifier, Pool

# Import custom modules
from src.features import FeaturePipeline, generate_out_of_fold_target_encoding

def main():
    workspace_dir = "/Users/ashwanikumar/code/Goal scoring probability ML/probabiltiy code/offside-football-prediction"
    train_path = os.path.join(workspace_dir, 'train.csv')
    
    print("Loading training data...")
    train_df = pd.read_csv(train_path)
    target_col = 'scored_flag'
    y_train = train_df[target_col].copy()
    
    print("Fitting feature pipeline...")
    pipeline = FeaturePipeline(random_state=42)
    train_feat = pipeline.fit_transform(train_df, target_col=target_col)
    
    print("Generating target encodings...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols_to_encode = []
    for col in ['name_y', 'home_club_name', 'away_club_name', 'referee', 'position', 'stadium']:
        if col in train_df.columns:
            cols_to_encode.append(col)
            
    if len(cols_to_encode) > 0:
        train_feat_encoded = generate_out_of_fold_target_encoding(train_df, skf, target_col=target_col, cols_to_encode=cols_to_encode)
        for col in cols_to_encode:
            train_feat[f'{col}_target_enc'] = train_feat_encoded[f'{col}_target_enc']
            
        # Player scoring efficiency per shot
        train_feat['scoring_rate_per_shot'] = train_feat['name_y_target_enc'] / (train_feat['avg_shots'] + 1e-5)
            
    # Drop columns that are IDs, strings, or target column
    cols_to_drop = ['appearance_id', 'name_y', target_col, 'date']
    features_to_use = [col for col in train_feat.columns if col not in cols_to_drop]
    X_train = train_feat[features_to_use]
    
    # Save feature names to pipeline so we can align columns at inference time
    pipeline.feature_names = features_to_use
    
    print("Saving fitted feature pipeline...")
    pipeline_path = os.path.join(workspace_dir, 'fitted_pipeline.pkl')
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    print("Training final CatBoost model...")
    # Best parameters from Optuna tuning:
    best_cat_params = {
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'learning_rate': 0.0284,
        'depth': 6,
        'l2_leaf_reg': 3.002,
        'iterations': 800,
        'random_seed': 42,
        'verbose': 50,
        'task_type': 'CPU',
        'thread_count': -1
    }
    
    cat_features = [col for col in pipeline.cat_cols if col in X_train.columns]
    cat_indices = [X_train.columns.get_loc(col) for col in cat_features]
    
    train_pool = Pool(X_train, y_train, cat_features=cat_indices)
    
    model = CatBoostClassifier(**best_cat_params)
    model.fit(train_pool)
    
    print("Saving CatBoost model...")
    model_path = os.path.join(workspace_dir, 'catboost_model.cbm')
    model.save_model(model_path)
    
    print("Successfully saved fitted_pipeline.pkl and catboost_model.cbm!")

if __name__ == '__main__':
    main()
