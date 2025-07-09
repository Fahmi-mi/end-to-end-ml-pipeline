import pandas as pd
import os

def load_data(filename, folder='data/raw'):
    """Load CSV file from the given folder"""
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)
