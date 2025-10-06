# House Price Prediction

This project predicts house prices using various regression models. The dataset used for this project is sourced from Kaggle and includes house sales data from King County, USA.

## Dataset

The dataset used in this project is the [House Sales in King County, USA](https://www.kaggle.com/datasets/shree1992/housedata/data) dataset. It contains information on house sales, including features such as the number of bedrooms, bathrooms, square footage, and price. All rights to the dataset belong to the original author.


## Models

The following regression models are implemented in this project:

1. **Multiple Linear Regression**
2. **Polynomial Linear Regression**
3. **Decision Tree Regression**
4. **Random Forest Regression**
5. **Support Vector Regression**

Each model is trained, evaluated, and saved as a `.joblib` file in the `models/` directory.

## Preprocessing

The preprocessing pipeline includes:
- Cleaning the data (removing invalid rows, handling missing values, etc.)
- Feature engineering
- Splitting the data into training and testing sets
- Saving processed data in `.npz` format for reuse

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for exploratory data analysis and model training. These notebooks provide a step-by-step walkthrough of the data preprocessing, model training, and evaluation processes.

## Scripts

The `scripts/` directory contains Python scripts for training models and making predictions:
- `train_<model_name>_regression.py`: Scripts for training specific models.
- `predict_house_price.py`: Script for predicting house prices using trained models.

## How to Run

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd house_price_prediction

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Preprocess the data: Run the preprocessing notebook or script to generate processed data files.
4. Train models: Use the scripts in the scripts/ directory to train models. For example:
   ```bash
   python -m scripts.train_multiple_linear_regression
5. Make predictions: Use the predict_house_price.py script to predict house prices for new data:
   ```bash
   python -m scripts.predict_house_price

## Results
The trained models are evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Evaluation results are printed in the console during training.