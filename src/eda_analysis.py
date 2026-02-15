import pandas as pd
import numpy as np
import os

def load_data():
    train_path = os.path.join('data', 'raw', 'train.csv')
    test_path = os.path.join('data', 'raw', 'test.csv')
    
    if not os.path.exists(train_path):
        print(f"Error: File not found at {train_path}")
        return None, None
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def analyze_data(df, name="Training Data"):
    print(f"--- {name} Analysis ---")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nSurvival Rate by Sex:")
    if 'Survived' in df.columns:
        print(df.groupby('Sex')['Survived'].mean())
    print("-" * 30)

if __name__ == "__main__":
    train_df, test_df = load_data()
    if train_df is not None:
        analyze_data(train_df, "Train")
        analyze_data(test_df, "Test")
