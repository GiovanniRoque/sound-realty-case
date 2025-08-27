import logging
import os
import pickle 
import json

from fastapi import FastAPI
from pydantic import BaseModel

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
                self.model = pickle.load(f)
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


