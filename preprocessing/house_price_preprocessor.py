import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# custom transformers 
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ = X.copy()
        X_['age'] = 2025 - X_['yr_built']
        X_['was_renovated'] = np.where(X_['yr_renovated'] > 0, 1, 0)
        X_.drop(columns=['yr_built', 'yr_renovated', 'sqft_above'], inplace=True)
        return X_

class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, clip_cols=None, lower_q=0.01, upper_q=0.99):
        self.clip_cols = clip_cols
        self.lower_q = lower_q
        self.upper_q = upper_q
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_ = pd.DataFrame(X).copy()
        if self.clip_cols:
            for col in self.clip_cols:
                lower, upper = X_[col].quantile([self.lower_q, self.upper_q])
                self.bounds_[col] = (lower, upper)
        return self

    def transform(self, X):
        X_ = X.copy()
        if self.clip_cols:
            for col, (lower, upper) in self.bounds_.items():
                X_[col] = X_[col].clip(lower, upper)
        return X_

# column definitions
log_features = ['sqft_lot', 'sqft_living', 'sqft_basement']
non_log_features = [
    'bedrooms', 'bathrooms', 'floors', 'view',
    'condition', 'age',
]
categorical_features = ['city']


# sub-transformers
log_numeric_transformer = Pipeline(steps=[
    ('clipper', OutlierClipper(clip_cols=['sqft_lot'], lower_q=0.01, upper_q=0.99)),
    ('log', FunctionTransformer(np.log1p, validate=False)),
    ('scaler', StandardScaler())
])

normal_numeric_transformer = Pipeline(steps=[
    ('clipper', OutlierClipper(clip_cols=['view'], lower_q=0.01, upper_q=0.99)),
    ('scaler', StandardScaler())
])

numeric_transformer = ColumnTransformer(transformers=[
    ('log_num', log_numeric_transformer, log_features),
    ('norm_num', normal_numeric_transformer, non_log_features)
])

categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

# full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, log_features + non_log_features),
    ('cat', categorical_transformer, categorical_features)
])

# full preprocessing pipeline
full_preprocessor = Pipeline(steps=[
    ('feature_engineering', FeatureEngineer()),
    ('preprocess', preprocessor),
])
