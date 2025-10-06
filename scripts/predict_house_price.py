import pandas as pd
import numpy as np
from joblib import load
import os

# load trained model
model_path = './models/model_support_vector_regression.joblib'  
pipeline = load(model_path)

# load new data 
new_data = pd.read_csv('./data/new_houses.csv')

# keep required columns
columns = [
    'bedrooms','bathrooms','floors','waterfront','view',
    'condition','yr_renovated','yr_built','sqft_lot',
    'sqft_living','sqft_above','sqft_basement','city','price'
]
new_data = new_data[columns]

# drop invalid rows
mask = (new_data['bathrooms'] != 0) & (new_data['bedrooms'] != 0) & (new_data['price'] != 0)
new_data = new_data[mask]

# separate target variable
#y = np.log1p(new_data['price'])
y = new_data['price'].values
X = new_data.drop(columns=['price'])

# predict
# y_pred_log = pipeline.predict(X)
# y_pred_price = np.expm1(y_pred_log)
y_pred = pipeline.predict(X)

# file path
pred_path = './data/predicted_prices.csv'

if not os.path.exists(pred_path):
    new_data['price_svr'] = y_pred.round(2)
    new_data.to_csv(pred_path, index=False)
    print(f"Created new file: {pred_path}")
else:
    existing = pd.read_csv(pred_path)
    if len(existing) == len(new_data):
        existing['price_svr'] = y_pred.round(2)
        existing.to_csv(pred_path, index=False)
        print(f"Added 'price_svr' predictions to existing file: {pred_path}")
    else:
        print("Warning: row counts differ — not updating the file.")