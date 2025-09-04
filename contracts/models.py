from pydantic import BaseModel

# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    metadata: dict

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

# Promotion request model
class ModelPromotionRequest(BaseModel):
    model_filename: str
    features_filename: str
