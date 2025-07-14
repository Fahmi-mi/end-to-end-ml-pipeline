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
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        if len(cat_cols) == 0:
            return df

        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[cat_cols])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols), index=df.index)
        df = df.drop(cat_cols, axis=1)
        return pd.concat([df, encoded_df], axis=1)

    def label_encode(self, df, column):
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        return df
    
    def target_encode(self, df, column, target):
        encoder = TargetEncoder()
        df[column] = encoder.fit_transform(df[column], df[target])
        return df
    
    def impute_missing_categorical(self, df, column, strategy, fill_value=None):
        """
        Impute missing values for categorical columns.

        Parameters:
            df (pd.DataFrame): DataFrame input.
            columns (str or list): Kolom yang ingin diimpute.
            strategy (str): Pilihan ['most_frequent', 'constant', 'mean', 'median'].
            fill_value (any, optional): Nilai pengganti jika strategy='constant'.

        Returns:
            pd.DataFrame: DataFrame hasil imputasi.
        """
        if strategy == 'constant':
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)
        df[column] = imputer.fit_transform(df[column])
        return df
    
    def impute_missing_numerical(self, df, column, strategy):
        """
        Impute missing values for numerical columns.

        Parameters:
            df (pd.DataFrame): DataFrame input.
            columns (str or list): Kolom yang ingin diimpute.
            strategy (str): Pilihan ['mean', 'median', 'most_frequent', 'constant', 'knn'].
                - 'mean', 'median', 'most_frequent', 'constant' menggunakan SimpleImputer.
                - 'knn' menggunakan KNNImputer.

        Returns:
            pd.DataFrame: DataFrame hasil imputasi.
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
        Scaling/Normalization: 'standard', 'minmax', 'robust'
        """
        scaler = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }[method]
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def polynomial_features(self, df, columns, degree=2):
        poly = PolynomialFeatures(degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        poly_df = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(columns), index=df.index)
        df = df.drop(columns, axis=1)
        return pd.concat([df, poly_df], axis=1)

    def pca(self, df, columns, n_components=2):
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(df[columns])
        pca_df = pd.DataFrame(pca_features, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
        df = df.drop(columns, axis=1)
        return pd.concat([df, pca_df], axis=1)

    def select_k_best(self, X, y, k=10, score_func=None):
        selector = SelectKBest(score_func=score_func, k=k)
        X_new = selector.fit_transform(X, y)
        return X_new

    def remove_constant_features(self, df):
        return df.loc[:, df.nunique() > 1]
    
    def log_transform(self, df, columns):
        df[columns] = np.log1p(df[columns])
        return df

    def binning(self, df, column, bins, labels=None):
        df[column + '_bin'] = pd.cut(df[column], bins=bins, labels=labels)
        return df

    def cap_outliers(self, df, column, lower_quantile=0.01, upper_quantile=0.99):
        lower = df[column].quantile(lower_quantile)
        upper = df[column].quantile(upper_quantile)
        df[column] = np.clip(df[column], lower, upper)
        return df

    def drop_outliers(self, df, column, lower_quantile=0.01, upper_quantile=0.99):
        lower = df[column].quantile(lower_quantile)
        upper = df[column].quantile(upper_quantile)
        return df[(df[column] >= lower) & (df[column] <= upper)]

    def create_interaction_features(self, df, columns, operation='multiply'):
        """
        Operation: 'multiply', 'divide', 'subtract', 'add'
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
        df[column] = pd.to_datetime(df[column])
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        df[f'{column}_weekday'] = df[column].dt.weekday
        return df

    def aggregate_features(self, df, groupby_cols, agg_dict):
        """
        Aggregasi (mean, sum, count pada grouped data).
        agg_dict: {'col1': 'mean', 'col2': 'sum', ...}
        """
        agg_df = df.groupby(groupby_cols).agg(agg_dict).reset_index()
        return agg_df

    def rfe_selection(self, estimator, X, y, n_features_to_select=5):
        selector = RFE(estimator, n_features_to_select=n_features_to_select)
        selector = selector.fit(X, y)
        return selector.transform(X)

    def tsne(self, df, columns, n_components=2, random_state=42):
        tsne = TSNE(n_components=n_components, random_state=random_state)
        tsne_features = tsne.fit_transform(df[columns])
        tsne_df = pd.DataFrame(tsne_features, columns=[f'tSNE_{i+1}' for i in range(n_components)], index=df.index)
        return tsne_df

    def umap(self, df, columns, n_components=2, random_state=42):
        umap_model = umap.UMAP(n_components=n_components, random_state=random_state)
        umap_features = umap_model.fit_transform(df[columns])
        umap_df = pd.DataFrame(umap_features, columns=[f'UMAP_{i+1}' for i in range(n_components)], index=df.index)
        return umap_df