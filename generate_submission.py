import os
import pickle
import pandas as pd
from catboost import CatBoostClassifier, Pool

def main():
    workspace_dir = "/Users/rana/OFF SIDE"
    test_path = os.path.join(workspace_dir, 'test.csv')
    pipeline_path = os.path.join(workspace_dir, 'fitted_pipeline.pkl')
    model_path = os.path.join(workspace_dir, 'catboost_model.cbm')
    solution_path = os.path.join(workspace_dir, 'solution.csv')
    
    print("Loading test data...")
    test_df = pd.read_csv(test_path)
    
    print("Loading fitted feature pipeline and trained model...")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
        
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    print("Preprocessing test data...")
    # Apply initial transformations (Label Encoding, Imputation, etc.)
    processed_test = pipeline.transform(test_df)
    
    # Align features to model expectations
    X_test = processed_test[pipeline.feature_names]
    
    print("Running model inference...")
    cat_features = [col for col in pipeline.cat_cols if col in X_test.columns]
    cat_indices = [X_test.columns.get_loc(col) for col in cat_features]
    
    test_pool = Pool(X_test, cat_features=cat_indices)
    preds = model.predict_proba(test_pool)[:, 1]
    
    print("Generating solution.csv...")
    solution_df = pd.DataFrame({
        'appearance_id': test_df['appearance_id'],
        'scored_flag': preds
    })
    
    # Validation checks
    assert len(solution_df) == len(test_df), "Row counts do not match!"
    assert not solution_df.isnull().any().any(), "Solution contains null values!"
    assert solution_df['scored_flag'].min() >= 0.0 and solution_df['scored_flag'].max() <= 1.0, "Predictions are out of range [0, 1]!"
    
    solution_df.to_csv(solution_path, index=False)
    print(f"Successfully generated and saved Kaggle-ready solution file to: {solution_path}")
    print(solution_df.head())

if __name__ == "__main__":
    main()
