from create_model import load_data

from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

import numpy as np

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/kc_house_data.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def evaluate_model():
    """Load data, train model, and evaluate using cross-validation."""
    print("Loading data...")
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)

    # Define the model pipeline (same as in create_model.py)
    model_pipeline = pipeline.make_pipeline(
        preprocessing.RobustScaler(),
        neighbors.KNeighborsRegressor()
    )

    print("Performing Cross-Validation...")
    cv_scores_mae = model_selection.cross_val_score(model_pipeline, x, y, cv=5, scoring='neg_mean_absolute_error')
    cv_scores_mse = model_selection.cross_val_score(model_pipeline, x, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_r2 = model_selection.cross_val_score(model_pipeline, x, y, cv=5, scoring='r2')

    # Convert scores back to positive for MAE and MSE, get square root for RMSE
    mae_scores = -cv_scores_mae
    mse_scores = -cv_scores_mse
    rmse_scores = np.sqrt(mse_scores)

    print("\n--- Full Cross-Validation Results ---")
    print(f"Mean Absolute Error (MAE): {mae_scores.mean():,.2f} (+/- {mae_scores.std() * 2:.2f})")
    print(f"Root Mean Squared Error (RMSE): {rmse_scores.mean():,.2f} (+/- {rmse_scores.std() * 2:.2f})")
    print(f"R-squared (R2): {cv_scores_r2.mean():.4f} (+/- {cv_scores_r2.std() * 2:.4f})")

    print(f"The model's average absolute error is around ${mae_scores.mean():,.0f}.")
    print(f"The R-squared value of ~{cv_scores_r2.mean():.3f} indicates that the model explains approximately {cv_scores_r2.mean()*100:.1f}% of the variance in house prices.")

if __name__ == "__main__":
    evaluate_model()
