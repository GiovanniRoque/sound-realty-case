import logging
import os
import pickle 
import json
import threading
import datetime

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

class ModelNotLoadedException:
    pass

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    metadata: dict

# Request model
# If new base features are added, we need to add them here.
class HouseDataRequest(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    sqft_above: float
    sqft_basement: float
    zipcode: str

# Promotion request model
class ModelPromotionRequest(BaseModel):
    model_filename: str
    features_filename: str

class ModelLoader:
    """Encapsulates the loaded model, features and demographics data."""
    def __init__(self):
        self.model = None
        self.model_features = []
        self.demographics_df = pd.DataFrame()
        self.model_filename = ""
        self.last_updated = datetime.datetime.now()
        self.loaded = False
        self._lock = threading.Lock()

    def load_model_and_data(self, model_path=MODEL_PATH, features_path=FEATURES_PATH):
        """Loads the model, feature list, and demographics data."""
        with self._lock:
            try:
                with open(model_path, 'rb') as f:
                    self.model: Pipeline = pickle.load(f)
                logger.info(f"Model loaded: {model_path}")

                with open(features_path, 'r') as f:
                    self.model_features = json.load(f)
                logger.info(f"Model features loaded: {features_path}")

                # using zipcode as str since it's a str in kc_house_data, index for merging
                self.demographics_df = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
                self.demographics_df.set_index('zipcode', inplace=True)
                logger.info(f"Demographics data loaded: {DEMOGRAPHICS_PATH}")

                self.loaded = True
                self.model_filename =  os.path.basename(model_path) # Extract filename from modelpath
                self.last_updated = datetime.datetime.now().isoformat()
                logger.info(f"All model data loaded successfully!")

            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                raise e
            
            except Exception as e:
                raise e
            
    def is_loaded(self):
        """Check if model is loaded for health checks."""
        return self.loaded


    def get_model_data(self):
        """Get current model data safely."""
        with self._lock:
            if not self.is_loaded():
                raise ModelNotLoadedException("Model not loaded")
            
            return {
                "model": self.model,
                "model_features": self.model_features.copy(),
                "demographics_df": self.demographics_df.copy(),
                "model_filename": self.model_filename,
                "loaded": self.loaded,
                "last_updated": self.last_updated
            }

# Create an instance of the loader
model_loader = ModelLoader()

# Start FastAPI
app = FastAPI(title="Sound Realty House Price Prediction API",
              description="API for predicting house prices using a KNeighborsRegressor model from SKLearn.",
              version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Load model when model starts
    logger.info("Starting up application with base model")
    model_loader.load_model_and_data()
    logger.info("Application startup complete!")

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(house_data: HouseDataRequest):
    """Runs the predictions for the house price prediction model."""
    try:
        predict_start = datetime.datetime.now()
        # Access model from loader
        model_data = model_loader.get_model_data()
        model = model_data['model']
        demographics_df = model_data['demographics_df']

        # Merge the input data to the demographics dataframe as it was done on create_model
        input_house_data = pd.DataFrame([house_data.model_dump()])
        merged_house_data = input_house_data.merge(demographics_df, how="left", left_on='zipcode', right_index=True).drop(columns=['zipcode'])
        logger.info(f"Merged dataframe shape: {merged_house_data.shape}")

        prediction = model.predict(merged_house_data)[0]
        predict_end = datetime.datetime.now()
        predict_timedelta = predict_end - predict_start

        logger.info(f"Prediction done with: {model_loader.model_filename} in {predict_timedelta.microseconds} microseconds")

        return PredictionResponse(predicted_price=float(prediction), metadata={"last_updated": model_data["last_updated"], "model_filename": model_data["model_filename"], "prediction_time_microseconds": predict_timedelta.microseconds})
    
    except ModelNotLoadedException: 
        logger.error("Model and data not loaded")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Model or data not initialized")
    except Exception as e:
        # Generic error for initial debugging
        logger.exception(f"Unexpected error during prediction: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="An internal error occurred during prediction")
    
@app.post("/promote", status_code=status.HTTP_200_OK)
async def promote(model_promotion: ModelPromotionRequest): 
    """Promotes a new model in runtime."""
    # Currently this only ensures the promotion of the state in one guvicorn worker. When deploying, scale resources horizontally instead of vertically.
    # Ideally, promotion should also be registered in a database to persist information when restoring the cluster. Saving locally won't be enough as it would only be inside the container.

    new_model_path = os.path.join(MODEL_DIR, model_promotion.model_filename)
    new_features_path = os.path.join(MODEL_DIR, model_promotion.features_filename)

    try:
        model_loader.load_model_and_data(model_path=new_model_path, features_path=new_features_path)
        return {"status": "promoted"}
    except FileNotFoundError as e:
        logger.error(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not find files")
    except Exception as e:
      logger.exception(f"Unexpected error during promotion: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not load new model")
    
@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Simple health check endpoint."""
    if model_loader.is_loaded():
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"status": "unhealthy"})
