import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import TargetEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, RFE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


class Preprocessor:
    def __init__(self):
        pass
    
    def one_hot_encode(self, df):
        """
        Parameters:
            df (pd.Dataframe): DataFrame input
            
        Returns:
            pd.DataFrame: DataFrame yang sudah di one-hot encode
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) == 0:
            return df

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(cat_cols, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def label_encode(self, df):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
        
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di label encode
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        le = LabelEncoder()
        for col in cat_cols:
            df[col] = le.fit_transform(df[col])
        return df
    
    def target_encode(self, df, target):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            target (str): Nama kolom target untuk encoding
            
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di target encode
        """
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        encoder = TargetEncoder()
        for col in cat_cols:
            df[col] = encoder.fit_transform(df[col], df[target])
        return df
    
    def impute_missing_categorical(self, df, column, strategy, fill_value=None):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (str or list): Kolom yang ingin diimpute
            strategy (str): Pilihan ['most_frequent', 'constant', 'mean', 'median']
            fill_value (any, optional): Nilai pengganti jika strategy='constant'

        Returns:
            pd.DataFrame: DataFrame hasil imputasi
        """
        if strategy == 'constant':
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)
        df[column] = imputer.fit_transform(df[column])
        return df
    
    def impute_missing_numerical(self, df, column, strategy):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (str or list): Kolom yang ingin diimpute
            strategy (str): Pilihan ['mean', 'median', 'most_frequent', 'constant', 'knn']
                - 'mean', 'median', 'most_frequent', 'constant' menggunakan SimpleImputer
                - 'knn' menggunakan KNNImputer

        Returns:
            pd.DataFrame: DataFrame hasil imputasi
        """
        if strategy == 'knn':
            imputer = KNNImputer()
            df[column] = imputer.fit_transform(df[column])
        else:
            imputer = SimpleImputer(strategy=strategy)
            df[column] = imputer.fit_transform(df[column])
        return df
    
class FeatureEngineering:
    def __init__(self):
        pass

    def scale(self, df, columns, method='standard'):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin di-scale
            method (str): Pilihan ['standard', 'minmax', 'robust']
            
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di-scale
        """
        scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }[method]
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def polynomial_features(self, df, columns, degree=2):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin dibuatkan fitur polinomial
            degree (int): Derajat polinomial yang ingin dibuatkan fitur
            
        Returns:
            pd.DataFrame: DataFrame dengan fitur polinomial
        """
        poly = PolynomialFeatures(degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns), index=df.index)
        df = df.drop(columns, axis=1)
        return pd.concat([df, poly_df], axis=1)

    def pca(self, df, columns, n_components=2):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin di-reduce dengan PCA
            n_components (int): Jumlah komponen utama yang ingin diambil
            
        Returns:
            pd.DataFrame: DataFrame dengan komponen utama hasil PCA
        """
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(df[columns])
        pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
        df = df.drop(columns, axis=1)
        return pd.concat([df, pca_df], axis=1)

    def select_k_best(self, X, y, k=10, score_func=None):
        """
        Parameters:
            X (pd.DataFrame): DataFrame fitur input
            y (pd.Series): Series target
            k (int): Jumlah fitur terbaik yang ingin dipilih
            score_func (callable): Fungsi skor untuk memilih fitur, default adalah f_classif untuk klasifikasi
            Pilihan umum:
                - f_classif (ANOVA F-value, untuk klasifikasi)
                - chi2 (chi-square, untuk klasifikasi)
                - mutual_info_classif (untuk klasifikasi)
                - f_regression (untuk regresi)
                - mutual_info_regression (untuk regresi)
                Default: f_classif untuk klasifikasi, f_regression untuk regresi
            
        Returns:
            pd.DataFrame: DataFrame dengan fitur yang sudah dipilih
        """
        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new

    def remove_constant_features(self, df):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            
        Returns:
            pd.DataFrame: DataFrame tanpa fitur konstan (hanya memiliki satu nilai unik)
        """
        return df.loc[:, df.nunique() > 1]
    
    def log_transform(self, df, columns):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin di-log transform
        
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di-log transform
        """
        df[columns] = np.log1p(df[columns])
        return df

    def binning(self, df, column, bins, labels=None, include_lowest=True):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            column (str): Nama kolom yang ingin di-binning
            bins (list): Daftar batas bin
            labels (list, optional): Daftar label untuk setiap bin, jika None maka akan menggunakan interval default
            
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di-binning
        """
        df[column + '_bin'] = pd.cut(df[column], bins=bins, labels=labels, include_lowest=include_lowest)
        return df

    def cap_outliers(self, df, column, lower_quantile=0.01, upper_quantile=0.99):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            column (str): Nama kolom yang ingin di-cap outliers
            lower_quantile (float): Kuantil bawah untuk outlier, default 0.01
            upper_quantile (float): Kuantil atas untuk outlier, default 0.99
            
        Returns:
            pd.DataFrame: DataFrame dengan kolom yang sudah di-cap outliers
        """
        lower = df[column].quantile(lower_quantile)
        upper = df[column].quantile(upper_quantile)
        df[column] = np.clip(df[column], lower, upper)
        return df

    def drop_outliers(self, df, column, lower_quantile=0.01, upper_quantile=0.99):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            column (str): Nama kolom yang ingin di-drop outliers
            lower_quantile (float): Kuantil bawah untuk outlier, default 0.01
            upper_quantile (float): Kuantil atas untuk outlier, default 0.99
            
        Returns:
            pd.DataFrame: DataFrame tanpa baris yang merupakan outliers pada kolom tersebut
        """
        lower = df[column].quantile(lower_quantile)
        upper = df[column].quantile(upper_quantile)
        return df[(df[column] >= lower) & (df[column] <= upper)]

    def create_interaction_features(self, df, columns, operation='multiply'):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Dua kolom yang ingin dibuatkan fitur interaksi
            operation (str): Pilihan ['multiply', 'divide', 'subtract', 'add']
        
        Returns:
            pd.DataFrame: DataFrame dengan fitur interaksi baru
        """
        col1, col2 = columns
        if operation == 'multiply':
            df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        elif operation == 'divide':
            df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-9)
        elif operation == 'subtract':
            df[f'{col1}_sub_{col2}'] = df[col1] - df[col2]
        elif operation == 'add':
            df[f'{col1}_add_{col2}'] = df[col1] + df[col2]
        return df

    def extract_datetime_features(self, df, column):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            column (str): Nama kolom datetime yang ingin diekstrak fitur
            
        Returns:
            pd.DataFrame: DataFrame dengan fitur datetime yang sudah diekstrak
        """
        df[column] = pd.to_datetime(df[column])
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_weekday'] = df[column].dt.weekday
        return df

    def aggregate_features(self, df, groupby_cols, agg_dict):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            groupby_cols (list): Kolom yang ingin digunakan untuk grouping
            agg_dict (dict): Dictionary yang berisi kolom dan fungsi agregasi, contoh {'col1': 'mean', 'col2': 'sum'}
        
        Returns:
            pd.DataFrame: DataFrame dengan fitur agregasi
        """
        agg_df = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        return agg_df

    def rfe_selection(self, estimator, X, y, n_features_to_select=5):
        """
        Parameters:
            estimator (object): Estimator yang digunakan untuk RFE, seperti LinearRegression, RandomForestClassifier, dll.
            X (pd.DataFrame): DataFrame fitur input
            y (pd.Series): Series target
            n_features_to_select (int): Jumlah fitur yang ingin dipilih
            
        Returns:
            pd.DataFrame: DataFrame dengan fitur yang sudah dipilih menggunakan RFE
        """
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector = selector.fit(X, y)
        return selector.transform(X)

    def tsne(self, df, columns, n_components=2, random_state=42):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin di-reduce dengan t-SNE
            n_components (int): Jumlah komponen yang ingin diambil, default 2
            random_state (int): Seed untuk reprodusibilitas
            
        Returns:
            pd.DataFrame: DataFrame dengan komponen t-SNE
        """
        tsne = TSNE(n_components=n_components, random_state=random_state)
        tsne_features = tsne.fit_transform(df[columns])
        tsne_df = pd.DataFrame(tsne_features, columns=[f'tSNE_{i+1}' for i in range(n_components)], index=df.index)
        return tsne_df

    def umap(self, df, columns, n_components=2, random_state=42):
        """
        Parameters:
            df (pd.DataFrame): DataFrame input
            columns (list): Kolom yang ingin di-reduce dengan UMAP
            n_components (int): Jumlah komponen yang ingin diambil, default 2
            random_state (int): Seed untuk reprodusibilitas
            
        Returns:
            pd.DataFrame: DataFrame dengan komponen UMAP
        """
        umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
        umap_features = umap_model.fit_transform(df[columns])
        umap_df = pd.DataFrame(umap_features, columns=[f'UMAP_{i+1}' for i in range(n_components)], index=df.index)
        return umap_df