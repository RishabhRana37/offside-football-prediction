import os
import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Import custom modules
from src.eda import run_eda_pipeline
from src.features import FeaturePipeline, generate_out_of_fold_target_encoding
from src.models import (
    train_lgbm, train_catboost, train_xgboost,
    tune_lgbm_with_optuna, tune_catboost_with_optuna, tune_xgboost_with_optuna,
    blend_predictions
)
from src.explain import run_explainability_pipeline

def find_and_copy_datasets(workspace_dir):
    """
    Scans typical directories (Downloads, Desktop, workspace) for the competition files
    and copies them to the workspace if they are found elsewhere.
    """
    search_dirs = [
        "/Users/rana/Downloads",
        "/Users/rana/Desktop",
        workspace_dir
    ]
    
    target_files = ['train.csv', 'test.csv', 'sample_submission.csv', 'data_dictionary.csv', 'feature_catalog.csv']
    found_files = {}
    
    print("=== [Data Loader] Scanning for Competition Files ===")
    
    # Check if they are already in the workspace
    all_in_workspace = all(os.path.exists(os.path.join(workspace_dir, f)) for f in target_files)
    if all_in_workspace:
        print("  All competition files are already in the workspace.")
        return True
        
    # Search in directories
    for filename in target_files:
        for s_dir in search_dirs:
            file_path = os.path.join(s_dir, filename)
            if os.path.exists(file_path):
                found_files[filename] = file_path
                break
                
    # If all found, copy them to workspace
    if len(found_files) == len(target_files):
        print("  Found all competition files! Copying them to the workspace...")
        for filename, src_path in found_files.items():
            dest_path = os.path.join(workspace_dir, filename)
            if src_path != dest_path:
                shutil.copy(src_path, dest_path)
                print(f"    Copied {filename} from {os.path.dirname(src_path)}")
        return True
    else:
        missing = [f for f in target_files if f not in found_files]
        print(f"  [ERROR] Missing dataset files: {missing}")
        print("  Please download them from Kaggle and place them in the workspace or Downloads folder.")
        return False


def main():
    workspace_dir = "/Users/ashwanikumar/code/Goal scoring probability ML/probabiltiy code/offside-football-prediction"
    
    # 1. Locate and copy dataset files
    if not find_and_copy_datasets(workspace_dir):
        print("\n[STOP] Dataset files not found. Pipeline cannot run.")
        return
        
    print("\n=== Loading Datasets ===")
    train_df = pd.read_csv(os.path.join(workspace_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(workspace_dir, 'test.csv'))
    
    # 2. Run EDA Pipeline
    run_eda_pipeline(train_df, test_df, output_dir=os.path.join(workspace_dir, "plots"))
    
    # 3. Fit and Transform Features
    print("\n=== Running Preprocessing and Feature Engineering ===")
    pipeline = FeaturePipeline(random_state=42)
    
    # Separate features and target
    target_col = 'scored_flag'
    y_train = train_df[target_col].copy()
    
    # Apply initial transformations (Label Encoding, Imputation, Date Extract, Football Features)
    train_feat = pipeline.fit_transform(train_df, target_col=target_col)
    test_feat = pipeline.transform(test_df)
    
    # 4. Out-of-Fold Target Encoding (preventing leakage)
    print("  Generating Out-of-Fold Target Encoding for Player/Club/Referee/Position/Stadium...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cols_to_encode = []
    for col in ['name_y', 'home_club_name', 'away_club_name', 'referee', 'position', 'stadium']:
        if col in train_df.columns:
            cols_to_encode.append(col)
            
    if len(cols_to_encode) > 0:
        train_feat_encoded = generate_out_of_fold_target_encoding(train_df, skf, target_col=target_col, cols_to_encode=cols_to_encode)
        # Add target encoding columns back to train features
        for col in cols_to_encode:
            train_feat[f'{col}_target_enc'] = train_feat_encoded[f'{col}_target_enc']
            
        # Player scoring efficiency per shot
        train_feat['scoring_rate_per_shot'] = train_feat['name_y_target_enc'] / (train_feat['avg_shots'] + 1e-5)
            
    # Drop columns that are IDs, strings, or target column
    cols_to_drop = ['appearance_id', 'name_y', target_col, 'date']
    features_to_use = [col for col in train_feat.columns if col not in cols_to_drop]
    
    X_train = train_feat[features_to_use]
    X_test = test_feat[features_to_use]
    
    # Ensure test columns align perfectly with train columns
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print(f"  Final training features shape: {X_train.shape}")
    print(f"  Final testing features shape: {X_test.shape}")
    
    # 5. Hyperparameter Tuning using Optuna
    # We will do a quick tuning (e.g. 5 trials) to demonstrate the Optuna integration.
    # Increase n_trials for better performance.
    best_lgb_params = tune_lgbm_with_optuna(X_train, y_train, cat_cols=pipeline.cat_cols, n_trials=5, random_state=42)
    best_cat_params = tune_catboost_with_optuna(X_train, y_train, cat_cols=pipeline.cat_cols, n_trials=5, random_state=42)
    best_xgb_params = tune_xgboost_with_optuna(X_train, y_train, cat_cols=pipeline.cat_cols, n_trials=5, random_state=42)
    
    # 6. Train Models with Optimal Parameters
    oof_lgb, test_lgb, lgb_models, lgb_ap = train_lgbm(
        X_train, y_train, X_test, 
        cat_cols=pipeline.cat_cols, 
        params=best_lgb_params, 
        n_splits=5, 
        random_state=42
    )
    
    oof_cat, test_cat, cat_models, cat_ap = train_catboost(
        X_train, y_train, X_test, 
        cat_cols=pipeline.cat_cols, 
        params=best_cat_params, 
        n_splits=5, 
        random_state=42
    )

    oof_xgb, test_xgb, xgb_models, xgb_ap = train_xgboost(
        X_train, y_train, X_test,
        cat_cols=pipeline.cat_cols,
        params=best_xgb_params,
        n_splits=5,
        random_state=42
    )
    
    # Save feature names to pipeline so we can align columns at inference time in the app
    pipeline.feature_names = features_to_use
    
    # 7. Ensemble Blending
    oof_blend, test_blend, blend_weights = blend_predictions(
        oof_lgb, oof_cat, oof_xgb,
        test_lgb, test_cat, test_xgb,
        y_train
    )
    
    # 8. Run Explainability (SHAP on LightGBM)
    run_explainability_pipeline(lgb_models, X_train, output_dir=os.path.join(workspace_dir, "plots"))
    
    # 9. Generate Submission Files
    print("\n=== Generating Submission Files ===")
    sub_df = pd.DataFrame({
        'appearance_id': test_df['appearance_id'],
        'scored_flag': test_blend
    })
    
    # Validate predictions range
    assert sub_df['scored_flag'].min() >= 0.0 and sub_df['scored_flag'].max() <= 1.0, "Predictions are out of range [0, 1]!"
    assert not sub_df.isnull().any().any(), "Submission contains null values!"
    
    # Write to solution.csv (Kaggle expected filename) and submission.csv (alternative)
    solution_path = os.path.join(workspace_dir, 'solution.csv')
    sub_df.to_csv(solution_path, index=False)
    sub_df.to_csv(os.path.join(workspace_dir, 'submission.csv'), index=False)
    
    print(f"  Submission file saved to {solution_path} (Shape: {sub_df.shape})")
    print("  First 5 rows of submission:")
    print(sub_df.head())
    print("\n=== Pipeline Successfully Executed! ===")


if __name__ == "__main__":
    main()
