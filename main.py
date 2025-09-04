import logging
import os

from fastapi import FastAPI, HTTPException, status

from contracts.models import PredictionResponse, HouseDataRequest, ModelPromotionRequest
from services.model_service import ModelService, ModelNotLoadedException, MODEL_DIR

# Start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create an instance of the service
model_service = ModelService()

# Start FastAPI
app = FastAPI(title="Sound Realty House Price Prediction API",
              description="API for predicting house prices using a KNeighborsRegressor model from SKLearn.",
              version="1.0.0")

@app.on_event("startup")
async def startup_event():
    # Load model when model starts
    logger.info("Starting up application with base model")
    model_service.load_model_and_data()
    logger.info("Application startup complete!")

@app.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(house_data: HouseDataRequest):
    """Runs the predictions for the house price prediction model."""
    try:
        return model_service.predict(house_data)
    
    except ModelNotLoadedException: 
        logger.error("Model and data not loaded")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
        model_service.load_model_and_data(model_path=new_model_path, features_path=new_features_path)
        return {"status": "promoted"}
    except FileNotFoundError as e:
        logger.error(e)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail="Specified model or features file not found")
    except Exception as e:
      logger.exception(f"Unexpected error during promotion: {e}")
      raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Could not load new model")
    
@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """Simple health check endpoint."""
    if model_service.is_loaded():
        return {"status": "healthy"}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"status": "unhealthy"})
