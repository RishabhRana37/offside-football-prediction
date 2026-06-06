import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import optuna

# Hide Optuna verbose logging by default
optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_lgbm(X_train, y_train, X_test, cat_cols, params=None, n_splits=5, random_state=42):
    """
    Trains a LightGBM Classifier using Stratified K-Fold.
    Returns out-of-fold predictions, test predictions, and the trained models.
    """
    print("\n--- Training LightGBM Model ---")
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1,
            'random_state': random_state,
            'n_estimators': 1500,
            'n_jobs': -1
        }
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models = []
    scores = []
    
    # Identify categorical column names
    cat_features = [col for col in cat_cols if col in X_train.columns]
    
    # Ensure test columns align with train columns
    X_test_aligned = X_test[X_train.columns]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        
        # Fit with early stopping
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            categorical_feature=cat_features,
            callbacks=callbacks
        )
        
        val_pred = model.predict_proba(X_va)[:, 1]
        oof_preds[val_idx] = val_pred
        
        # Compute Average Precision for the validation set
        fold_score = average_precision_score(y_va, val_pred)
        scores.append(fold_score)
        print(f"  Fold {fold+1} Average Precision: {fold_score:.4f}")
        
        # Predict on test set
        test_preds += model.predict_proba(X_test_aligned)[:, 1] / n_splits
        models.append(model)
        
    mean_score = np.mean(scores)
    oof_score = average_precision_score(y_train, oof_preds)
    print(f"Mean Fold AP: {mean_score:.4f} | Overall OOF AP: {oof_score:.4f}")
    return oof_preds, test_preds, models, oof_score


def train_catboost(X_train, y_train, X_test, cat_cols, params=None, n_splits=5, random_state=42):
    """
    Trains a CatBoost Classifier using Stratified K-Fold.
    Returns out-of-fold predictions, test predictions, and the trained models.
    """
    print("\n--- Training CatBoost Model ---")
    if params is None:
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'learning_rate': 0.05,
            'depth': 6,
            'iterations': 800,
            'random_seed': random_state,
            'verbose': False,
            'task_type': 'CPU',
            'thread_count': -1
        }
        
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    models = []
    scores = []
    
    # CatBoost expects categorical columns to be type int or str. Let's make sure they are integers
    cat_features = [col for col in cat_cols if col in X_train.columns]
    # Get index positions of categorical columns
    cat_indices = [X_train.columns.get_loc(col) for col in cat_features]
    
    # Ensure test columns align with train columns
    X_test_aligned = X_test[X_train.columns]

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]
        
        # Create Pools
        train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
        val_pool = Pool(X_va, y_va, cat_features=cat_indices)
        test_pool = Pool(X_test_aligned, cat_features=cat_indices)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
        
        val_pred = model.predict_proba(val_pool)[:, 1]
        oof_preds[val_idx] = val_pred
        
        # Compute Average Precision
        fold_score = average_precision_score(y_va, val_pred)
        scores.append(fold_score)
        print(f"  Fold {fold+1} Average Precision: {fold_score:.4f}")
        
        # Predict on test set
        test_preds += model.predict_proba(test_pool)[:, 1] / n_splits
        models.append(model)
        
    mean_score = np.mean(scores)
    oof_score = average_precision_score(y_train, oof_preds)
    print(f"Mean Fold AP: {mean_score:.4f} | Overall OOF AP: {oof_score:.4f}")
    return oof_preds, test_preds, models, oof_score


def tune_lgbm_with_optuna(X_train, y_train, cat_cols, n_trials=10, random_state=42):
    """
    Tunes LightGBM hyperparameters using Optuna, optimizing for Average Precision.
    """
    print("\n--- Starting Optuna Tuning for LightGBM ---")
    
    if len(X_train) > 150000:
        print(f"  Sampling 150,000 rows from {len(X_train)} for faster Optuna tuning...")
        sample_idx = X_train.sample(n=150000, random_state=random_state).index
        X_tune = X_train.loc[sample_idx].reset_index(drop=True)
        y_tune = y_train.loc[sample_idx].reset_index(drop=True)
    else:
        X_tune = X_train.reset_index(drop=True)
        y_tune = y_train.reset_index(drop=True)

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': 1,
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'n_estimators': 500,
            'verbose': -1,
            'random_state': random_state,
            'n_jobs': -1
        }
        
        # Simple 3-fold cross validation for speed in tuning
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        cat_features = [col for col in cat_cols if col in X_tune.columns]
        
        for train_idx, val_idx in skf.split(X_tune, y_tune):
            X_tr, y_tr = X_tune.iloc[train_idx], y_tune.iloc[train_idx]
            X_va, y_va = X_tune.iloc[val_idx], y_tune.iloc[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                categorical_feature=cat_features,
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            val_pred = model.predict_proba(X_va)[:, 1]
            scores.append(average_precision_score(y_va, val_pred))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best Trial Value (Average Precision): {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'bagging_freq': 1,
        'n_estimators': 1500,
        'verbose': -1,
        'random_state': random_state,
        'n_jobs': -1
    })
    return best_params


def tune_catboost_with_optuna(X_train, y_train, cat_cols, n_trials=8, random_state=42):
    """
    Tunes CatBoost hyperparameters using Optuna, optimizing for Average Precision.
    """
    print("\n--- Starting Optuna Tuning for CatBoost ---")
    
    if len(X_train) > 100000:
        print(f"  Sampling 100,000 rows from {len(X_train)} for faster Optuna tuning...")
        sample_idx = X_train.sample(n=100000, random_state=random_state).index
        X_tune = X_train.loc[sample_idx].reset_index(drop=True)
        y_tune = y_train.loc[sample_idx].reset_index(drop=True)
    else:
        X_tune = X_train.reset_index(drop=True)
        y_tune = y_train.reset_index(drop=True)

    def objective(trial):
        params = {
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.1, log=True),
            'depth': trial.suggest_int('depth', 4, 8),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_seed': random_state,
            'verbose': False,
            'iterations': 500,
            'task_type': 'CPU',
            'thread_count': -1
        }
        
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        
        cat_features = [col for col in cat_cols if col in X_tune.columns]
        cat_indices = [X_tune.columns.get_loc(col) for col in cat_features]
        
        for train_idx, val_idx in skf.split(X_tune, y_tune):
            X_tr, y_tr = X_tune.iloc[train_idx], y_tune.iloc[train_idx]
            X_va, y_va = X_tune.iloc[val_idx], y_tune.iloc[val_idx]
            
            train_pool = Pool(X_tr, y_tr, cat_features=cat_indices)
            val_pool = Pool(X_va, y_va, cat_features=cat_indices)
            
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, verbose=False)
            
            val_pred = model.predict_proba(val_pool)[:, 1]
            scores.append(average_precision_score(y_va, val_pred))
            
        return np.mean(scores)
        
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best Trial Value (Average Precision): {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    best_params = study.best_params
    best_params.update({
        'loss_function': 'Logloss',
        'eval_metric': 'Logloss',
        'iterations': 800,
        'random_seed': random_state,
        'verbose': False,
        'task_type': 'CPU',
        'thread_count': -1
    })
    return best_params


def blend_predictions(oof_lgb, oof_cat, test_lgb, test_cat, y_train):
    """
    Finds the optimal blending weights for LGBM and CatBoost out-of-fold predictions
    to maximize Average Precision, then applies them to test predictions.
    """
    print("\n--- Ensembling Models via Blending ---")
    best_weight = 0.5
    best_score = 0.0
    
    # Grid search for best blend weight
    for w in np.linspace(0, 1, 101):
        blend_oof = w * oof_lgb + (1 - w) * oof_cat
        score = average_precision_score(y_train, blend_oof)
        if score > best_score:
            best_score = score
            best_weight = w
            
    print(f"Optimal Blending Weight: LGBM = {best_weight:.2f}, CatBoost = {1 - best_weight:.2f}")
    print(f"Ensemble OOF Average Precision: {best_score:.4f}")
    
    # Recalculate blend_oof using the optimal weight
    blend_oof = best_weight * oof_lgb + (1 - best_weight) * oof_cat
    blend_test = best_weight * test_lgb + (1 - best_weight) * test_cat
    return blend_oof, blend_test, best_weight

