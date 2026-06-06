import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeaturePipeline:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.cat_cols = [
            'foot', 'position', 'sub_position', 'country_of_citizenship',
            'home_club_name', 'away_club_name', 'stadium', 'referee',
            'competition_type', 'confederation', 'market_value_tier', 'age_bucket',
            'name_x', 'home_away', 'country_name'
        ]
        self.num_cols = []
        self.imputers = {}
        self.label_encoders = {}
        self.global_target_means = {}
        self.missing_flag_cols = []

    def fit_transform(self, train_df, target_col='scored_flag'):
        """
        Fits the pipeline on the training set and returns the engineered training set.
        """
        df = train_df.copy()
        
        # Convert boolean and flag columns to 1/0 integers
        bool_cols = [
            'full_match_flag', 'starter_flag', 'substitute_flag', 
            'card_flag', 'prime_age_flag', 'veteran_flag', 
            'has_national_team_experience', 'finisher_flag', 'creative_player_flag',
            'has_understat', 'analytics_coverage_flag'
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool).astype(int)
        
        # 1. Parse Date Features
        df = self._process_dates(df)
        
        # 2. Football-Specific Feature Engineering
        df = self._engineer_football_features(df)
        
        # Identify numeric and categorical columns
        all_cols = df.columns.tolist()
        exclude_cols = ['appearance_id', 'name_y', target_col, 'date']
        
        self.num_cols = [col for col in all_cols if col not in self.cat_cols and col not in exclude_cols]
        
        # 3. Missing Value Imputation
        for col in self.num_cols:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            self.imputers[col] = median_val
            # Add missing flag if missingness is high (>5%)
            missing_rate = df[col].isnull().mean()
            if missing_rate > 0.05:
                df[f'{col}_is_missing'] = df[col].isnull().astype(int)
                self.missing_flag_cols.append(col)
            df[col] = df[col].fillna(median_val)
            
        for col in self.cat_cols:
            if col in df.columns:
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                self.imputers[col] = mode_val
                df[col] = df[col].fillna(mode_val).astype(str)
                
                # Fit Label Encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le

        # 4. Target Encoding Baseline (Global Means for fallback)
        if target_col in train_df.columns:
            # We save the global mean target encoding for players, clubs, and positions
            self.global_target_means['name_y'] = train_df.groupby('name_y')[target_col].mean().to_dict()
            self.global_target_means['home_club_name'] = train_df.groupby('home_club_name')[target_col].mean().to_dict()
            self.global_target_means['away_club_name'] = train_df.groupby('away_club_name')[target_col].mean().to_dict()
            self.global_target_means['global_mean'] = train_df[target_col].mean()

        return df

    def transform(self, test_df):
        """
        Transforms the testing/validation set using the fitted pipeline parameters.
        """
        df = test_df.copy()
        
        # Convert boolean and flag columns to 1/0 integers
        bool_cols = [
            'full_match_flag', 'starter_flag', 'substitute_flag', 
            'card_flag', 'prime_age_flag', 'veteran_flag', 
            'has_national_team_experience', 'finisher_flag', 'creative_player_flag',
            'has_understat', 'analytics_coverage_flag'
        ]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool).astype(int)
        
        # 1. Parse Date Features
        df = self._process_dates(df)
        
        # 2. Football-Specific Feature Engineering
        df = self._engineer_football_features(df)
        
        # 3. Impute and Encode
        for col in self.num_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputers.get(col, 0.0))
            else:
                df[col] = self.imputers.get(col, 0.0)
                
        # Recreate missingness indicators that were added during training
        for col in self.missing_flag_cols:
            if col in df.columns:
                df[f'{col}_is_missing'] = df[col].isnull().astype(int)
            else:
                df[f'{col}_is_missing'] = 0
            
        for col in self.cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(self.imputers.get(col, 'Unknown')).astype(str)
                le = self.label_encoders[col]
                # Handle unseen categories in test
                unseen_mask = ~df[col].isin(le.classes_)
                if unseen_mask.any():
                    # Map unseen classes to the most frequent class or add to classes
                    most_freq = le.classes_[0]
                    df.loc[unseen_mask, col] = most_freq
                df[col] = le.transform(df[col])
            else:
                # If category column is missing in test, fill with 0 (default label)
                df[col] = 0

        # Apply target encoding fallback mapping for test set
        global_mean = self.global_target_means.get('global_mean', 0.0)
        df['name_y_target_enc'] = df['name_y'].map(self.global_target_means.get('name_y', {})).fillna(global_mean)
        df['home_club_name_target_enc'] = df['home_club_name'].map(self.global_target_means.get('home_club_name', {})).fillna(global_mean)
        df['away_club_name_target_enc'] = df['away_club_name'].map(self.global_target_means.get('away_club_name', {})).fillna(global_mean)

        return df

    def _process_dates(self, df):
        """
        Extracts year, month, day of week, day of year, and weekend flag from the date column.
        """
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['year'] = df['date'].dt.year.fillna(2025).astype(int)
            df['month'] = df['date'].dt.month.fillna(6).astype(int)
            df['day_of_week'] = df['date'].dt.dayofweek.fillna(5).astype(int)
            df['day_of_year'] = df['date'].dt.dayofyear.fillna(160).astype(int)
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        else:
            # Fallback if date is not present
            df['year'] = 2025
            df['month'] = 6
            df['day_of_week'] = 5
            df['day_of_year'] = 160
            df['is_weekend'] = 1
        return df

    def _engineer_football_features(self, df):
        """
        Creates domain-specific football analytics features.
        """
        # 1. Attacking and Attacking-scaled metrics
        # Combine expected goals (xG) and expected assists (xA)
        if 'avg_xG' in df.columns and 'avg_xA' in df.columns:
            df['avg_xG_plus_xA'] = df['avg_xG'] + df['avg_xA']
        
        # Expected Goals contribution scaled by playing time ratio
        if 'avg_xG' in df.columns and 'minutes_ratio' in df.columns:
            df['expected_xG_in_match'] = df['avg_xG'] * df['minutes_ratio']
            df['expected_shots_in_match'] = df.get('avg_shots', 0) * df['minutes_ratio']
            
        # Creative capability scaled by playing time ratio
        if 'avg_xA' in df.columns and 'minutes_ratio' in df.columns:
            df['expected_xA_in_match'] = df['avg_xA'] * df['minutes_ratio']

        # Attacking threat ratio
        if 'avg_shots' in df.columns and 'avg_xG' in df.columns:
            df['xG_per_shot'] = df['avg_xG'] / (df['avg_shots'] + 1e-5)

        # 2. Market Value interaction features
        if 'market_value_before_match' in df.columns and 'highest_market_value_in_eur' in df.columns:
            df['market_value_diff_peak'] = df['highest_market_value_in_eur'] - df['market_value_before_match']
            df['market_value_ratio_peak'] = df['market_value_before_match'] / (df['highest_market_value_in_eur'] + 1e-5)
            
        if 'market_value_before_match' in df.columns and 'age' in df.columns:
            df['value_age_interaction'] = df['market_value_before_match'] * df['age']
            
        # 3. International profile efficiency
        if 'international_caps' in df.columns and 'international_goals' in df.columns:
            df['intl_goal_efficiency'] = df['international_goals'] / (df['international_caps'] + 1e-5)

        # 4. Starting status and minutes interaction
        if 'starter_flag' in df.columns and 'minutes_played' in df.columns:
            df['starter_minutes'] = df['starter_flag'] * df['minutes_played']

        # 5. Position and Attacking potential
        # Attackers tend to score more; create custom attack threat category
        if 'is_attacker' in df.columns:
            df['attacker_value'] = df['is_attacker'] * df.get('market_value_before_match', 0)

        return df


def generate_out_of_fold_target_encoding(train_df, kfold, target_col='scored_flag', cols_to_encode=['name_y', 'home_club_name', 'away_club_name']):
    """
    Computes out-of-fold target encoding for high cardinality features to prevent target leakage.
    """
    encoded_train = train_df.copy()
    global_mean = train_df[target_col].mean()
    
    # Initialize target encoding columns
    for col in cols_to_encode:
        encoded_train[f'{col}_target_enc'] = global_mean
        
    for train_idx, val_idx in kfold.split(train_df, train_df[target_col]):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        
        for col in cols_to_encode:
            # Calculate target encoding on training fold
            col_means = train_fold.groupby(col)[target_col].mean()
            # Map to validation fold
            encoded_train.iloc[val_idx, encoded_train.columns.get_loc(f'{col}_target_enc')] = val_fold[col].map(col_means).fillna(global_mean)
            
    return encoded_train
