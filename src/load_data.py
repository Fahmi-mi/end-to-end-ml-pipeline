import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        pass

    def load_data(filename, folder):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)
    
    def save_data(df, filename, folder):
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
    def data_info(df):
        print(f"{df} Info:")
        df.info()
        print(f"\n{df} Describe:")
        print(df.describe())
        
    def preview_data(df, num_rows=5):
        return df.head(num_rows)
    
    def split_data(df, target_column, test_size=0.2, random_state=42):
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test