import numpy as np

def clean_data(df):
    df = df.copy()

    required_columns = ['bedrooms','bathrooms','floors', 'waterfront','view', 'condition', 
                        'yr_renovated', 'yr_built', 'sqft_lot', 'sqft_living', 'sqft_above',
                        'sqft_basement', 'city', 'price' ]
    df = df[required_columns].copy()

    # drop invalid rows
    mask = (df['bathrooms'] != 0) & (df['bedrooms'] != 0) & (df['price'] != 0)
    df = df[mask]

    # separate target and features
    y = np.log1p(df['price'])   
    X = df.drop(columns=['price'])

    return X, y