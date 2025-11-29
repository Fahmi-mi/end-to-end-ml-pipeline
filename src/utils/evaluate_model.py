from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

class Evaluator:
    def __init__(self):
        pass

    def evaluate_classification(self, y_true, y_pred):
        """
        Parameters:
            y_true (list or np.array): Nilai sebenarnya dari target
            y_pred (list or np.array): Nilai prediksi dari model
            
        Returns:
            None: Menampilkan metrik evaluasi untuk klasifikasi
        """
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("Precision:", precision_score(y_true, y_pred, average='weighted'))
        print("Recall:", recall_score(y_true, y_pred, average='weighted'))
        print("F1 Score:", f1_score(y_true, y_pred, average='weighted'))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    
    def evaluate_regression(self, y_true, y_pred):
        """
        Parameters:
            y_true (list or np.array): Nilai sebenarnya dari target
            y_pred (list or np.array): Nilai prediksi dari model
            
        Returns:
            None: Menampilkan metrik evaluasi untuk regresi
        """
        print("MSE:", mean_squared_error(y_true, y_pred))
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("R2 Score:", r2_score(y_true, y_pred))