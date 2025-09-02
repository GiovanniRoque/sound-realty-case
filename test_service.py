import requests
import pandas as pd
import logging 

# Start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERVICE_URL = "http://localhost:5000" 
DATA_FILE = "data/future_unseen_examples.csv"
NUM_EXAMPLES = 100 # Number of examples to test

def test_service():
    """Test the prediction service."""
    try:
        # Load data
        df = pd.read_csv(DATA_FILE, dtype={'zipcode': str})
        logger.info(f"Loaded {len(df)} examples from {DATA_FILE}")

        # Test Health Check
        health_resp = requests.get(f"{SERVICE_URL}/health")
        if health_resp.status_code == 200 and health_resp.json().get('status') == 'healthy':
            logger.info("Health check: PASSED")
        else:
            logger.error(f"Health check: FAILED - {health_resp.text}")
            return

        # Test Prediction Endpoint
        for i, row in df.head(NUM_EXAMPLES).iterrows():
            data = row.to_dict()
            logger.info(f"Sending example {i+1}: {data}")

            # Because our endpoint uses pydantic for validation, the endpoint already filters the subset of the columns used in the model, ignoring the extra ones.
            response = requests.post(f"{SERVICE_URL}/predict", json=data)

            # Check response
            if response.status_code == 200:
                result = response.json()
                predicted_price = result.get('predicted_price')
                logger.info(f"  -> Predicted Price: ${predicted_price:,.2f}")
            else:
                logger.info(f"  -> Error: {response.status_code} - {response.text}")

    except FileNotFoundError:
        print(f"Error: Could not find data file {DATA_FILE}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to service: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during testing: {e}")

if __name__ == "__main__":
    test_service()
