import pandas as pd
from locust import HttpUser, task, between, events
import logging
import json
import random

random.seed(42)

# Start logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATA_FILE = "data/future_unseen_examples.csv"
test_data = []

#  Load data once at the start of the test
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Load test data when the Locust test starts."""
    global test_data
    try:
        # Load data into a list of dictionaries
        # Ensure zipcode is treated as string
        df = pd.read_csv(DATA_FILE, dtype={'zipcode': str})
        test_data = df.to_dict('records')
        logging.info(f"Locust: Loaded {len(test_data)} examples from {DATA_FILE}")
        if not test_data:
             logging.error("Locust: No data loaded. Test data is empty.")
    except FileNotFoundError:
        logging.error(f"Locust: Error - Could not find data file {DATA_FILE}")
    except Exception as e:
        logging.error(f"Locust: Unexpected error loading data: {e}")


class HousePricePredictionUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Called when a Locust user starts running."""
        # Check health once when the user starts to ensure the service is up
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if data.get("status") == "healthy":
                        response.success()
                        logging.info("Health check passed for user.")
                    else:
                        response.failure(f"Health check failed: {data}")
                        logging.warning(f"Health check failed for user: {data}")
                except json.JSONDecodeError:
                    response.failure("Health check response was not valid JSON")
                    logging.error("Health check response was not valid JSON")
            else:
                 response.failure(f"Health check failed with status {response.status_code}")
                 logging.warning(f"Health check failed with status {response.status_code}")


    @task
    def predict_house_price(self):
        """Task to send a prediction request."""
        # Select a row of data randomly
        data = random.choice(test_data)

        # Send POST request to the /predict endpoint
        # Using catch_response=True allows for manually marking success/failure
        with self.client.post("/predict", json=data, catch_response=True) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    predicted_price = result.get('predicted_price')
                    if predicted_price is not None:
                        response.success()
                        logging.info(f"Prediction successful: ${predicted_price:,.2f}")
                    else:
                        response.failure("Response missing 'predicted_price'")
                        logging.warning(f"Prediction response missing 'predicted_price': {result}")
                except json.JSONDecodeError:
                    response.failure("Response was not valid JSON")
                    logging.error("Prediction response was not valid JSON")
            else:
                # Mark the request as failed in Locust's statistics
                response.failure(f"Request failed with status {response.status_code}: {response.text}")
                logging.info(f"Prediction request failed: {response.status_code} - {response.text}")
