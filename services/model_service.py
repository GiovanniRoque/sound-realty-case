import logging
import os
import pickle 
import json
import threading
import datetime
from abc import ABC, abstractmethod

from sklearn.pipeline import Pipeline

import pandas as pd

from contracts.models import PredictionResponse, HouseDataRequest

# Start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config variables
MODEL_DIR = 'model'
DATA_DIR = 'data'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.pkl')
FEATURES_PATH = os.path.join(MODEL_DIR, 'model_features.json')
DEMOGRAPHICS_PATH = os.path.join(DATA_DIR, 'zipcode_demographics.csv')


class ModelNotLoadedException(Exception):
    pass


class AbstractModelService(ABC):
    """
    Abstract base class defining the interface for a model prediction service.
    If you want to use a different ModelService, implement the following functions.~
    """
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        pass

    @abstractmethod
    def predict(self, house_data: HouseDataRequest) -> PredictionResponse:
        """Make a prediction based on the provided house data."""
        pass

    @abstractmethod
    def load_model_and_data(self):
        """Loads the model, feature list, and demographics data."""
        pass


class ModelService(AbstractModelService):
    """Encapsulates the loaded model, features and demographics data, generating predictions from the model."""
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
        
    def predict(self, house_data: HouseDataRequest) -> PredictionResponse:
        """Make a prediction using the loaded model."""
        predict_start = datetime.datetime.now()

        with self._lock:
            # Access model from service
            model_data = self.get_model_data()
            model: Pipeline = model_data['model']
            demographics_df = model_data['demographics_df']
            model_filename = model_data['model_filename']

            # Merge the input data to the demographics dataframe as it was done on create_model
            input_house_data = pd.DataFrame([house_data.model_dump()])
            merged_house_data = input_house_data.merge(demographics_df, how="left", left_on='zipcode', right_index=True).drop(columns=['zipcode'])
            logger.info(f"Merged dataframe shape: {merged_house_data.shape}")

            prediction = model.predict(merged_house_data)[0]
            
        predict_end = datetime.datetime.now()
        predict_timedelta = predict_end - predict_start

        logger.info(f"Prediction done with: {model_filename} in {predict_timedelta.microseconds} microseconds")

        return PredictionResponse(predicted_price=float(prediction), metadata={"last_updated": model_data["last_updated"], "model_filename": model_data["model_filename"], "prediction_time_microseconds": predict_timedelta.microseconds})

