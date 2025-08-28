import logging
import os
import pickle 
import json

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

import pandas as pd

# Start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config variables
MODEL_DIR = 'model'
DATA_DIR = 'data'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.json')
DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, 'zipcode_demographics.csv')

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float

# Request model
class HouseDataRequest(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str


class ModelLoader:
    """Encapsulates the loaded model, features and demographics data."""
    def __init__(self):
        self.model = None
        self.model_features = []
        self.demographics_df = pd.DataFrame()
        self.loaded = False

    def load_model_and_data(self):
        """Loads the model, feature list, and demographics data."""
        try:
            with open(MODEL_PATH, 'rb') as f:
                self.model: Pipeline = pickle.load(f)
            logger.info(f"Model loaded: {MODEL_PATH}")

            with open(FEATURES_PATH, 'r') as f:
                self.model_features = json.load(f)
            logger.info(f"Model features loaded: {FEATURES_PATH}")

            # using zipcode as str since it's a str in kc_house_data, index for merging
            self.demographics_df = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
            self.demographics_df.set_index('zipcode', inplace=True)
            logger.info(f"Demographics data loaded: {DEMOGRAPHICS_PATH}")

            self.loaded = True
            logger.info(f"All model data loaded successfully!")

        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            raise e
        
        except Exception as e:
            raise e

# Create an instance of the loader
model_loader = ModelLoader()

# Start FastAPI
app = FastAPI(title="Sound Realty House Price Prediction API",
              description="API for predicting house prices using a KNeighborsRegressor model from SKLearn.",
              version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Load model when model starts
    logger.info("Starting up application")
    model_loader.load_model_and_data()
    logger.info("Application startup complete!")

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(house_data: HouseDataRequest):

    if not model_loader.loaded:
        logger.error("Model and data not loaded")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Model or data not initialized")

    # Access model from loader
    model = model_loader.model
    model_features = model_loader.model_features
    demographics_df = model_loader.demographics_df

    try:
        # Merge the input data to the demographics dataframe as it was done on create_model
        input_house_data = pd.DataFrame([house_data.model_dump()])
        merged_house_data = input_house_data.merge(demographics_df, how="left", left_on='zipcode', right_index=True).drop(columns=['zipcode'])
        logger.info(f"Merged dataframe shape: {merged_house_data.shape}")

        prediction = model.predict(merged_house_data)[0]
        logger.info("Prediction successful!")

        return PredictionResponse(predicted_price=float(prediction))
    except Exception as e:
        # Generic error for initial debugging
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An internal error occurred during prediction")