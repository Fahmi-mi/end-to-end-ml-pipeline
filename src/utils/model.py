import joblib
import os

class ModelLoader:
    def __init__(self):
        pass
    
    def save_model(self, model, filename, folder='models'):
        """
        Parameters:
            model: Model yang akan disimpan
            filename (str): Nama file untuk menyimpan model
            folder (str): Direktori tempat menyimpan model
            
        Returns:
            None: Model disimpan ke file
        """
        path = os.path.join(folder, filename)
        joblib.dump(model, path)
        print(f"Model saved to {path}")
        
    def load_model(self, filename, folder='models'):
        """
        Parameters:
            filename (str): Nama file model yang akan dimuat
            folder (str): Direktori tempat model disimpan
            
        Returns:
            model: Model yang dimuat dari file
        """
        path = os.path.join(folder, filename)
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model