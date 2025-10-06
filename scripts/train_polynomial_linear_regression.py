import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing.house_price_preprocessor import full_preprocessor
from sklearn.preprocessing import PolynomialFeatures


# load data
df = pd.read_csv("./data/raw_data.csv")

# keep required columns
columns = [
    'bedrooms','bathrooms','floors','waterfront','view',
    'condition','yr_renovated','yr_built','sqft_lot',
    'sqft_living','sqft_above','sqft_basement','city','price'
]
df = df[columns]

# drop invalid rows
mask = (df['bathrooms'] != 0) & (df['bedrooms'] != 0) & (df['price'] != 0)
df = df[mask]

# separate target variable
y = np.log1p(df['price'])
X = df.drop(columns=['price'])

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# build full model pipeline 
pipeline = Pipeline(steps=[
    ('preprocess', full_preprocessor),
    ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
    ('model', LinearRegression())
])

# train
pipeline.fit(X_train, y_train)

# evaluate model
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Model evaluation:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# Save trained pipeline
dump(pipeline, './models/model_polynomial_linear_regression.joblib')
print("Saved model to './models/model_polynomial_linear_regression.joblib'")