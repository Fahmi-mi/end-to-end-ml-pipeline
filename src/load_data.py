import pandas as pd
import os
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self):
        pass

    def load_data(self, filename, folder):
        """
        Parameters:
            filename (str): Nama file CSV yang ingin dimuat
            folder (str): Folder tempat file CSV berada
        
        Returns:
            pd.DataFrame: DataFrame yang berisi data dari file CSV
        """
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return pd.read_csv(filepath)
    
    def save_data(self, df, filename, folder):
        """
        Parameters:
            df (pd.DataFrame): DataFrame yang ingin disimpan
            filename (str): Nama file CSV untuk menyimpan DataFrame
            folder (str): Folder tempat file CSV akan disimpan
            
        Returns:
            None: DataFrame disimpan ke file CSV
        """
        filepath = os.path.join(folder, filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")
        
    def data_info(self, df):
        """
        Parameters:
            df (pd.DataFrame): DataFrame yang ingin ditampilkan informasinya
            
        Returns:
            None: Menampilkan informasi dan deskripsi DataFrame
        """
        print(f"Info:")
        df.info()
        print(f"\nDescribe:")
        return df.describe()
        
    def preview_data(self, df, num_rows=5):
        """
        Parameters:
            df (pd.DataFrame): DataFrame yang ingin ditampilkan preview-nya
            num_rows (int): Jumlah baris yang ingin ditampilkan, default 5
            
        Returns:
            pd.DataFrame: DataFrame yang berisi preview dari data
        """
        return df.head(num_rows)
    
    def split_data(self, df, id_column, target_column, test_size=0.2, random_state=42, stratify=None):
        """
        Parameters:
            df (pd.DataFrame): DataFrame yang ingin dibagi menjadi data latih dan u
            target_column (str): Nama kolom target yang ingin diprediksi
            test_size (float): Proporsi data yang digunakan untuk testing, default 0.2
            random_state (int): Seed untuk reprodusibilitas, default 42
            
        Returns:
            tuple: Tuple yang berisi DataFrame fitur latih, DataFrame fitur uji,
        """
        X = df.drop(columns=[id_column, target_column])
        y = df[target_column]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=df[stratify] if stratify else None)
        return X_train, X_val, y_train, y_val