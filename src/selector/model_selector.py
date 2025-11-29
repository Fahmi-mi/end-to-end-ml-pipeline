import pandas as pd
import warnings, joblib, os, contextlib
warnings.filterwarnings("ignore")
os.environ["CATBOOST_INFO_DIR"] = os.devnull

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from lazypredict.Supervised import LazyClassifier, LazyRegressor

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
        with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            models, predictions = lazy.fit(X_train, X_test, y_train, y_test)
        return models
    
        
    def run(self):
        self.load_data()
        print("\n================ PIPELINE ================")
        X_train, X_test, y_train, y_test = self.preprocess()
        models = self.run_lazypredict(X_train, X_test, y_train, y_test)
        print(models)