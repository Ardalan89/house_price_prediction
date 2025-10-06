import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing.preprocessors import full_preprocessor_no_scale
from preprocessing.data_cleaning import clean_data


# load data
df = pd.read_csv("./data/raw_data.csv")

# Clean the data
X,y = clean_data(df)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# build full model pipeline 
pipeline = Pipeline(steps=[
    ('preprocess', full_preprocessor_no_scale),
    ('model', DecisionTreeRegressor(max_depth=10, random_state=0))
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
print(f"MSE  : {mse:.3f}")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")

# Save trained pipeline
dump(pipeline, './models/model_decision_tree_regression.joblib')
print("Saved model to './models/model_decision_tree_regression.joblib'")