import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_raw_data
from src.preprocessor import get_preprocessor

def train_and_evaluate():
    # Load data
    train_df, _ = load_raw_data()
    y = train_df['Survived']
    X = train_df.drop('Survived', axis=1)
    
    # Preprocessor
    preprocessor = get_preprocessor()
    
    # Models to test
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=200, min_samples_split=10, min_samples_leaf=1, max_depth=10, bootstrap=True, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500, learning_rate=0.01, max_depth=7, subsample=0.8, colsample_bytree=0.8, gamma=0.2, random_state=42)
    }
    
    results = {}
    best_score = 0
    best_model_name = ""
    best_pipeline = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Training and evaluating models...")
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        results[name] = mean_score
        print(f"{name}: {mean_score:.4f} (+/- {std_score:.4f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            pipeline.fit(X, y) # Retrain on full data
            best_pipeline = pipeline
            
    print(f"\nBest Model: {best_model_name} with accuracy: {best_score:.4f}")
    
    # Train Voting Classifier (Ensemble)
    print("\nTraining Voting Classifier (Ensemble)...")
    from sklearn.ensemble import VotingClassifier
    
    estimators = []
    for name, model in models.items():
        estimators.append((name, model))
        
    voting_clf = VotingClassifier(estimators=estimators, voting='soft')
    voting_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', voting_clf)
    ])
    
    voting_scores = cross_val_score(voting_pipeline, X, y, cv=cv, scoring='accuracy')
    voting_mean = np.mean(voting_scores)
    print(f"VotingClassifier: {voting_mean:.4f} (+/- {np.std(voting_scores):.4f})")
    
    # Check if Voting is better
    if voting_mean >= best_score:
        print("VotingClassifier is the best model!")
        voting_pipeline.fit(X, y)
        best_pipeline = voting_pipeline
    else:
        print(f"Best model remains {best_model_name}")

    # Save best model
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'best_model.pkl')
    joblib.dump(best_pipeline, model_path)
    print(f"Saved best model to {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
