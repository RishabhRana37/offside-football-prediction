import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda_pipeline(train_df, test_df, output_dir="plots"):
    """
    Runs a complete Exploratory Data Analysis (EDA) pipeline, printing key statistics
    and saving beautiful visual plots to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=== [EDA] Starting Exploratory Data Analysis ===")
    
    # 1. Dataset Shape and Overview
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Test dataset shape: {test_df.shape}")
    
    # 2. Target Distribution
    if 'scored_flag' in train_df.columns:
        target_counts = train_df['scored_flag'].value_counts()
        target_pct = train_df['scored_flag'].value_counts(normalize=True) * 100
        print("\nTarget Distribution (scored_flag):")
        for val, count in target_counts.items():
            print(f"  Class {val}: {count} appearances ({target_pct[val]:.2f}%)")
            
        # Plot target distribution
        plt.figure(figsize=(6, 5))
        sns.set_theme(style="whitegrid")
        colors = ["#4A90E2", "#E94E77"]  # Premium blue and coral pink
        ax = sns.barplot(x=target_counts.index, y=target_counts.values, palette=colors, hue=target_counts.index, legend=False)
        plt.title("Target Distribution (scored_flag)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Player Scored Flag (0 = No, 1 = Yes)", fontsize=12)
        plt.ylabel("Number of Appearances", fontsize=12)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}\n({height/len(train_df)*100:.1f}%)',
                        (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', color='white', fontweight='bold',
                        xytext=(0, 0), textcoords='offset points')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_distribution.png"), dpi=150)
        plt.close()
    else:
        print("\n[WARNING] 'scored_flag' not found in training dataset.")
        
    # 3. Missing Value Analysis
    print("\nMissing Value Summary (Top Columns in Train):")
    missing_train = train_df.isnull().sum()
    missing_train_pct = (missing_train / len(train_df)) * 100
    missing_train_df = pd.DataFrame({'Missing Count': missing_train, 'Percentage': missing_train_pct})
    missing_train_df = missing_train_df[missing_train_df['Missing Count'] > 0].sort_values(by='Percentage', ascending=False)
    
    if len(missing_train_df) > 0:
        print(missing_train_df.head(15))
        
        # Plot top missing values
        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_train_df.head(15)['Percentage'], y=missing_train_df.head(15).index, palette="viridis")
        plt.title("Top Columns by Percentage of Missing Values (Train)", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Percentage Missing (%)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "missing_values_train.png"), dpi=150)
        plt.close()
    else:
        print("  No missing values found in training set!")

    # 4. Correlation Analysis of Numeric Columns
    print("\nCorrelation Analysis with Target (Top Attacking features):")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'scored_flag' in numeric_cols:
        correlations = train_df[numeric_cols].corr()['scored_flag'].sort_values(ascending=False)
        print("Top 10 Positively Correlated Features with scored_flag:")
        print(correlations.head(11))
        print("\nTop 5 Negatively Correlated Features with scored_flag:")
        print(correlations.tail(5))
        
        # Plot Correlation Heatmap for Top Attacking Features
        attacking_features = [col for col in ['scored_flag', 'avg_xG', 'avg_shots', 'avg_xGChain', 
                                             'avg_npxG', 'minutes_played', 'minutes_ratio', 'starter_flag', 
                                             'avg_xA', 'avg_key_passes', 'market_value_before_match'] if col in train_df.columns]
        if len(attacking_features) > 1:
            plt.figure(figsize=(10, 8))
            corr_matrix = train_df[attacking_features].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title("Attacking Features Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "attacking_features_correlation.png"), dpi=150)
            plt.close()

    # 5. Position scoring analysis
    position_cols = [col for col in ['is_attacker', 'is_midfielder', 'is_defender', 'is_goalkeeper'] if col in train_df.columns]
    if len(position_cols) > 0 and 'scored_flag' in train_df.columns:
        print("\nScoring Probability by Position:")
        pos_probs = {}
        for pos in position_cols:
            subset = train_df[train_df[pos] == 1]
            if len(subset) > 0:
                prob = subset['scored_flag'].mean()
                pos_probs[pos.replace('is_', '').capitalize()] = prob
                print(f"  {pos.replace('is_', '').capitalize()}: {prob:.2f} scoring probability (N = {len(subset)})")
        
        if len(pos_probs) > 0:
            plt.figure(figsize=(8, 5))
            sns.barplot(x=list(pos_probs.keys()), y=list(pos_probs.values()), palette="pastel")
            plt.title("Scoring Probability by Playing Position Group", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Position Group", fontsize=12)
            plt.ylabel("Scoring Probability", fontsize=12)
            plt.ylim(0, 1.0)
            for i, val in enumerate(pos_probs.values()):
                plt.text(i, val + 0.02, f"{val:.2f}", ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "scoring_probability_by_position.png"), dpi=150)
            plt.close()
            
    print("=== [EDA] EDA Completed and Plots Saved to 'plots/' ===")
