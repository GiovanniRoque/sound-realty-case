import json
import pathlib
import pickle
from typing import List
from typing import Tuple

import pandas
from sklearn import model_selection
from sklearn import neighbors
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import metrics

import numpy as np

SALES_PATH = "data/kc_house_data.csv"  # path to CSV with home sale data
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"  # path to CSV with demographics
# List of columns (subset) that will be taken from home sale data
SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]
OUTPUT_DIR = "model"  # Directory where output artifacts will be saved


def load_data(
    sales_path: str, demographics_path: str, sales_column_selection: List[str]
) -> Tuple[pandas.DataFrame, pandas.Series]:
    """Load the target and feature data by merging sales and demographics.

    Args:
        sales_path: path to CSV file with home sale data
        demographics_path: path to CSV file with home sale data
        sales_column_selection: list of columns from sales data to be used as
            features

    Returns:
        Tuple containg with two elements: a DataFrame and a Series of the same
        length.  The DataFrame contains features for machine learning, the
        series contains the target variable (home sale price).

    """
    data = pandas.read_csv(sales_path,
                           usecols=sales_column_selection,
                           dtype={'zipcode': str})
    demographics = pandas.read_csv(DEMOGRAPHICS_PATH,
                                   dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    # Remove the target variable from the dataframe, features will remain
    y = merged_data.pop('price')
    x = merged_data

    return x, y


def evaluate_model_test_set(model_pipeline: pipeline.Pipeline, x_test, y_test):
    """Evaluate using the test set."""

    # Optional: Train on full set and evaluate on a hold-out test set for final metrics
    print("\n--- Hold-Out Test Set Evaluation ---")
    y_pred = model_pipeline.predict(x_test)

    test_mae = metrics.mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    test_r2 = metrics.r2_score(y_test, y_pred)

    print(f"Test Set Mean Absolute Error: {test_mae:,.2f}.")
    print(f"Test Set Root Mean Square Error: {test_rmse:,.2f}.")
    print(f"Test Set R-squared value: {test_r2:.4f}. The model explains approximately {test_r2*100:.1f}% of the variance in house prices in the test set.")


def main():
    """Load data, train model, and export artifacts."""
    x, y = load_data(SALES_PATH, DEMOGRAPHICS_PATH, SALES_COLUMN_SELECTION)
    x_train, _x_test, y_train, _y_test = model_selection.train_test_split(
        x, y, random_state=42)

    model = pipeline.make_pipeline(preprocessing.RobustScaler(),
                                   neighbors.KNeighborsRegressor()).fit(x_train, y_train)

    output_dir = pathlib.Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    # Output model artifacts: pickled model and JSON list of features
    pickle.dump(model, open(output_dir / "model.pkl", 'wb'))
    json.dump(list(x_train.columns),
              open(output_dir / "model_features.json", 'w'))
    
    evaluate_model_test_set(model, _x_test, _y_test)

if __name__ == "__main__":
    main()
