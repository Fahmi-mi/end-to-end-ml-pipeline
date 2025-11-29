import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings, joblib, os
warnings.filterwarnings("ignore")
sns.set_style(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classification
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import shap
import sweetviz as sv

class ModelSelector:
    def __init__(self, data_path, target, is_classification=True, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.target = target
        self.is_classification = is_classification
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        
    def load_data(self):
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xls', '.xlsx')):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Format file belum didukung, pakai csv atau xlsx")
    
    def auto_eda(self):
        report = sv.analyze(self.df)
        report.show_notebook()
        
    def preprocess(self):
        X = self.df.drop(self.target, axis=1)
        y = self.df[self.target].copy()
        if self.is_classification and y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), name=self.target, index=y.index)
            joblib.dump(le, 'label_encoder.pkl')
            print({list(le.classes_)})
        if self.is_classification:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=y)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
    
    def run_lazypredict(self, X_train, X_test, y_train, y_test):
        if self.is_classification:
            lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        else:
            lazy = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
        return models
    
    def manual_benchmark(self, X_train, X_test, y_train, y_test):
        if self.is_classification:
            models_dict = {
                "LGBM": LGBMClassifier(random_state=self.random_state, n_jobs=-1),
                "XGBoost": XGBClassifier(random_state=self.random_state, n_jobs=-1),
                "CatBoost": CatBoostClassifier(random_state=self.random_state, verbose=False, thread_count=-1),
                "RandomForest": RandomForestClassifier(n_estimators=500, random_state=self.random_state, n_jobs=-1),
                "ExtraTrees": ExtraTreesClassifier(n_estimators=500, random_state=self.random_state, n_jobs=-1)
            }
        else:
            models_dict = {
                "LGBM": LGBMRegressor(random_state=self.random_state, n_jobs=-1),
                "XGBoost": XGBRegressor(random_state=self.random_state, n_jobs=-1),
                "CatBoost": CatBoostRegressor(random_state=self.random_state, verbose=False, thread_count=-1),
                "RandomForest": RandomForestRegressor(n_estimators=500, random_state=self.random_state, n_jobs=-1),
                "ExtraTrees": ExtraTreesRegressor(n_estimators=500, random_state=self.random_state, n_jobs=-1)
            }
        results = []
        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            if self.is_classification:
                acc = accuracy_score(y_test, pred)
                f1 = f1_score(y_test, pred, average='weighted')
                results.append({"Model": name, "Accuracy": acc, "F1_Weighted": f1})
            else:
                r2 = r2_score(y_test, pred)
                mae = mean_absolute_error(y_test, pred)
                results.append({"Model": name, "R2": r2, "MAE": mae})
        result_df = pd.DataFrame(results)
        if self.is_classification:
            result_df = result_df.sort_values(by='F1_Weighted', ascending=False)
        else:
            result_df = result_df.sort_values(by='R2', ascending=False)
        best_model_name = result_df.iloc[0]['Model']
        return result_df, models_dict[best_model_name]
    
    def shap_summary(self, final_model, X_test):
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_test)
        if self.is_classification and isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, show=False)
        else:
            shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('SHAP_summary.png', dpi=150, bbox_inches='tight')
        
    def run(self):
        self.load_data()
        self.auto_eda()
        print("\n================ PIPELINE ================")
        X_train, X_test, y_train, y_test = self.preprocess()
        self.run_lazypredict(X_train, X_test, y_train, y_test)
        result_df, best_model_obj = self.manual_benchmark(X_train, X_test, y_train, y_test)
        best_model = result_df.iloc[0]["Model"]
        print(f"\nModel baseline terbaik: {best_model}")
        self.shap_summary(best_model_obj, X_test)