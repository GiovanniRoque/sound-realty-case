# sound-realty-case
Case for Sound Realty.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

*   **Docker:** Install Docker Desktop or Docker Engine by following the official guide: [https://docs.docker.com/get-started/](https://docs.docker.com/get-started/)
*   **(Alternative) Conda:** If you prefer running locally without Docker, install Miniconda or Anaconda: [https://docs.conda.io/en/latest/](https://docs.conda.io/en/latest/)


### Setup & Running

#### Running Using Docker (Recommended)

1.  **Clone or Download the Repository:**
    Obtain the project files.

2.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and change to the project's root directory.

3.  **Build the Docker Image:**
    ```bash
    docker build -t house-price-api .
    ```

4.  **Run the Docker Container:**
    ```bash
    docker run -p 5000:5000 house-price-api
    ```
    This command maps port 5000 on your local machine to port 5000 inside the container.

5.  **Access the API:**
    The API will be available at `http://localhost:5000`.
    *   Interactive API documentation (Swagger UI): `http://localhost:5000/docs`


#### Running Locally (with Conda)

1.  **Clone or Download the Repository:**
    Obtain the project files.

2.  **Navigate to the Project Directory:**
    Open your terminal or command prompt and change to the project's root directory.

3.  **Create and Activate the environment:**
    ```bash
    conda env create -f conda_environment.yml
    conda activate housing
    ```
    This command maps creates and activates the local environment.

4.  **Create and Evaluate the model:**
    Once you've created and activated the environment, you can run the script which
    creates and evaluates the fitted model with the hold-out test set:
    ```sh
    python create_model.py
    ```

5.  **Cross-Validate the model:**
    You can also run a cross-validation script to evaluate the model with more detail:
    ```sh
    python evaluate_model.py
    ```

6.  **Run the API:**
    Once the model is created locally, run the fastapi app:
    ```sh
    python -m fastapi run main.py
    ```

7.  **Access the API:**
    The API will be available at `http://localhost:5000`.
    *   Interactive API documentation (Swagger UI): `http://localhost:5000/docs`


## API Endpoints

*   **`POST /predict`**
    *   **Description:** Predicts the price of a house based on a specific set of features. The API automatically fetches and merges relevant demographic data based on the provided `zipcode`.
    *   **Request Body:** JSON object conforming to `HouseDataRequest`.
        ```json
        {
          "bedrooms": 3.0,
          "bathrooms": 2.0,
          "sqft_living": 1500.0,
          "sqft_lot": 5000.0,
          "floors": 2.0,
          "sqft_above": 1300.0,
          "sqft_basement": 200.0,
          "zipcode": "90201"
        }
        ```
    *   **Response:** JSON object `PredictionResponse`.
        ```json
        {
          "predicted_price": 450000.50,
          "metadata": {
            "last_updated": "2025-09-02T10:00:00.123456",
            "model_filename": "model.pkl",
            "prediction_time_microseconds": 1234
          }
        }
        ```

*   **`POST /promote`**
    *   **Description:** Promotes a new model version by loading specified model and feature files from the `model/` directory. **Important:** This change only affects the current application instance (e.g., one worker).
    *   **Request Body:** JSON object `ModelPromotionRequest`.
        ```json
        {
          "model_filename": "new_model_v2.pkl",
          "features_filename": "new_model_v2_features.json"
        }
        ```
    *   **Response:** JSON object indicating success.
        ```json
        {
          "status": "promoted"
        }
        ```
*   **`GET /health`**
    *   **Description:** Checks if the API is running and the default model is loaded.
    *   **Response:** `{"status": "healthy"}` (200 OK) or error (503 Service Unavailable).

## Example Usage (with `curl`)

Assuming the API is running at `http://localhost:5000`:

**Using `/predict` endpoint:**

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "bedrooms": 4.0,
       "bathrooms": 2.25,
       "sqft_living": 2070.0,
       "sqft_lot": 8893.0,
       "floors": 2.0,
       "sqft_above": 2070.0,
       "sqft_basement": 0.0,
       "zipcode": "90210"
     }'