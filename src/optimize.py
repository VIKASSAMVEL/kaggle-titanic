import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.data_loader import load_raw_data
from src.preprocessor import get_preprocessor

def optimize_models():
    train_df, _ = load_raw_data()
    y = train_df['Survived']
    X = train_df.drop('Survived', axis=1)
    
    preprocessor = get_preprocessor()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 1. Random Forest Optimization
    print("Optimizing Random Forest...")
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    rf_param_grid = {
        'classifier__n_estimators': [100, 200, 300, 500],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__bootstrap': [True, False]
    }
    
    rf_search = RandomizedSearchCV(rf_pipeline, rf_param_grid, n_iter=20, cv=cv, verbose=1, n_jobs=-1, random_state=42, scoring='accuracy')
    rf_search.fit(X, y)
    print(f"Best RF Params: {rf_search.best_params_}")
    print(f"Best RF Score: {rf_search.best_score_:.4f}")
    
    # 2. XGBoost Optimization
    print("\nOptimizing XGBoost...")
    xgb_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
    ])
    
    xgb_param_grid = {
        'classifier__n_estimators': [100, 200, 500],
        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7, 10],
        'classifier__subsample': [0.6, 0.8, 1.0],
        'classifier__colsample_bytree': [0.6, 0.8, 1.0],
        'classifier__gamma': [0, 0.1, 0.2]
    }
    
    xgb_search = RandomizedSearchCV(xgb_pipeline, xgb_param_grid, n_iter=20, cv=cv, verbose=1, n_jobs=-1, random_state=42, scoring='accuracy')
    xgb_search.fit(X, y)
    print(f"Best XGB Params: {xgb_search.best_params_}")
    print(f"Best XGB Score: {xgb_search.best_score_:.4f}")
    
    # Save best params to file
    with open('models/best_params.txt', 'w') as f:
        f.write(f"Random Forest Best Params: {rf_search.best_params_}\n")
        f.write(f"XGBoost Best Params: {xgb_search.best_params_}\n")
        f.write(f"Best RF Score: {rf_search.best_score_}\n")
        f.write(f"Best XGB Score: {xgb_search.best_score_}\n")

if __name__ == "__main__":
    optimize_models()
