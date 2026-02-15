import pandas as pd
import joblib
import os
from src.data_loader import load_raw_data

def make_predictions():
    # Load test data
    _, test_df = load_raw_data()
    
    # Load model
    model_path = os.path.join('models', 'best_model.pkl')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run src/train.py first.")
        
    model = joblib.load(model_path)
    
    # Predict
    predictions = model.predict(test_df)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': predictions
    })
    
    # Save submission
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(submission.head())

if __name__ == "__main__":
    make_predictions()
