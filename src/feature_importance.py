import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_feature_importance():
    model_path = os.path.join('models', 'best_model.pkl')
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        return

    pipeline = joblib.load(model_path)
    
    # Check if it is a VotingClassifier
    if hasattr(pipeline.named_steps['classifier'], 'estimators_'):
        voting_clf = pipeline.named_steps['classifier']
        # ... (existing voting logic) ...
        print("Analyzing Ensemble...")
        # (This part requires the existing logic, but for brevity I'll just refactor to handle both)
        
    classifier = pipeline.named_steps['classifier']
    preprocessor = pipeline.named_steps['preprocessor']
    
    feature_names = [
        'Age', 'Fare', 'FamilySize', 
        'Embarked_C', 'Embarked_Q', 'Embarked_S', 
        'Sex_female', 'Sex_male', 
        'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare'
    ]

    # Helper function to plot
    def plot_imp(model, name):
        if hasattr(model, 'feature_importances_'):
            print(f"\n{name} Feature Importance:")
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # Print top 10
            for i in range(min(10, len(indices))):
                print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

    # Handle Voting Classifier
    if hasattr(classifier, 'estimators_'):
        for name, model in classifier.named_estimators_.items():
            plot_imp(model, name)
            
    # Handle Single Model (e.g. XGBoost, RandomForest)
    else:
        plot_imp(classifier, type(classifier).__name__)

if __name__ == "__main__":
    plot_feature_importance()
