import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

def run_explainability_pipeline(models, X, output_dir="plots"):
    """
    Computes SHAP values and feature importances from trained models (typically LightGBM),
    and saves explanation plots to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n=== [Explain] Starting Model Explainability and SHAP Analysis ===")
    
    # We will use the first model from the cross-validation models list for SHAP
    model = models[0]
    
    # 1. SHAP Feature Attribution
    try:
        # Sample data if it is too large to speed up calculation (SHAP can be slow)
        sample_size = min(1000, len(X))
        X_sample = X.sample(n=sample_size, random_state=42) if len(X) > sample_size else X.copy()
        
        print(f"  Calculating SHAP values using a sample of {len(X_sample)} records...")
        
        # Use TreeExplainer (optimized for tree models)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample)
        
        # Plot SHAP summary
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP Feature Summary (Attribution/Impact on Prediction)", fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=150)
        plt.close()
        print("  SHAP summary plot saved successfully.")
        
    except Exception as e:
        print(f"  [ERROR] SHAP analysis encountered an error: {e}")
        print("  Proceeding with standard tree feature importance instead.")

    # 2. Standard Feature Importance
    try:
        # Check if the model has feature_importances_ property
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = X.columns
            
            feat_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Print top 15 features
            print("\nTop 15 Features by LightGBM Feature Importance:")
            print(feat_imp_df.head(15))
            
            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x=feat_imp_df.head(15)['Importance'], y=feat_imp_df.head(15)['Feature'], palette="mako")
            plt.title("Top 15 Most Important Features (LightGBM)", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Split/Gain Importance Value", fontsize=12)
            plt.ylabel("Features", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "feature_importance_plot.png"), dpi=150)
            plt.close()
            print("  Feature importance plot saved successfully.")
        else:
            print("  Model does not support standard feature importance plotting.")
            
    except Exception as e:
        print(f"  [ERROR] Standard feature importance plotting encountered an error: {e}")
        
    print("=== [Explain] Explainability Pipeline Completed ===")
