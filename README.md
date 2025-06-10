# Content-Based Recommendation System

This project implements a lightweight, simple content-based recommendation system using Python, TensorFlow/Keras for neural network embeddings, and FastAPI for the RESTful API.

## Features

*   **Content-Based Recommendations:** Recommends items similar to a given item based on their content (title, description, tags).
*   **Neural Network Embeddings:** Uses a simple Multi-Layer Perceptron (MLP) trained like an autoencoder on TF-IDF features to generate item embeddings.
*   **TF-IDF:** Utilizes TF-IDF for initial text feature extraction.
*   **RESTful API:** Exposes recommendation functionality via a FastAPI application.
*   **API Key Authentication:** Secures the recommendation endpoint using API keys stored in a `.env` file.
*   **Automatic Key Generation:** Includes an endpoint (or initial setup) to generate API keys.

## Project Structure

```
recommendation_system/
├── data/
│   └── items.csv           # Example item data
├── models/                 # Saved models and artifacts (created by train.py)
│   ├── tfidf_vectorizer.pkl
│   ├── embedding_model.keras
│   └── item_embeddings.pkl
├── src/
│   ├── __init__.py
│   ├── auth.py             # API key handling and authentication
│   ├── data_loader.py      # Loads and prepares data
│   ├── main.py             # FastAPI application
│   ├── model.py            # Keras model definition
│   ├── preprocessing.py    # Text preprocessing and TF-IDF
│   ├── recommender.py      # Recommendation logic (cosine similarity)
│   ├── train.py            # Training script
│   └── utils.py            # Utility functions (saving/loading artifacts)
├── venv/                   # Virtual environment
├── .env                    # Stores API keys (created automatically)
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup

1.  **Clone the repository (or place the project files):**
    ```bash
    # Assuming you have the project folder
    cd recommendation_system
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: The `requirements.txt` file needs to be generated first if not provided. See step below.)*

## Training

Run the training script to process the data, train the model, and generate the necessary artifacts (`tfidf_vectorizer.pkl`, `embedding_model.keras`, `item_embeddings.pkl`) in the `models/` directory.

```bash
cd src
python train.py
cd ..
```

This script performs the following steps:
*   Loads data from `data/items.csv`.
*   Preprocesses the text content.
*   Creates and fits a TF-IDF vectorizer.
*   Builds and trains the neural network model (autoencoder-style).
*   Generates embeddings for all items using the trained base model.
*   Saves the vectorizer, the base embedding model, and the item embeddings to the `models/` directory.

## Running the API Server

1.  **Ensure artifacts are generated:** Make sure you have run the `train.py` script successfully and the `models/` directory contains the required files.

2.  **Generate an initial API Key (if needed):** The API server will automatically generate an initial API key and save it to the `.env` file if no keys are found upon startup. Check the console output when starting the server for the generated key.

3.  **Start the FastAPI server using Uvicorn:**
    ```bash
    cd src
    uvicorn main:app --host 0.0.0.0 --port 8000
    ```
    *   `--host 0.0.0.0` makes the server accessible within the network.
    *   `--port 8000` specifies the port.
    *   Use `--reload` during development for automatic code reloading.

## API Usage

The API provides two main endpoints:

### 1. Get Recommendations

*   **URL:** `/recommend`
*   **Method:** `GET`
*   **Headers:**
    *   `X-API-Key`: `<your_api_key>` (Required)
*   **Query Parameters:**
    *   `item_id` (string, required): The ID of the item you want recommendations for (e.g., `item_1`).
    *   `n` (integer, optional, default=5): The number of recommendations to return.
*   **Example Request (using curl):**
    ```bash
    curl -X GET "http://localhost:8000/recommend?item_id=item_3&n=3" -H "X-API-Key: <your_api_key>"
    ```
*   **Success Response (200 OK):**
    ```json
    {
      "input_item_id": "item_3",
      "recommendations": [
        "item_4",
        "item_5",
        "item_1"
      ]
    }
    ```
*   **Error Responses:**
    *   `401 Unauthorized`: If the API key is missing or invalid.
    *   `404 Not Found`: If the provided `item_id` does not exist.
    *   `422 Unprocessable Entity`: If query parameters are invalid.
    *   `503 Service Unavailable`: If the model artifacts haven't been loaded correctly.

### 2. Generate New API Key (Admin)

*   **URL:** `/admin/generate-key`
*   **Method:** `POST`
*   **Authentication:** *Note: This endpoint is not secured by default in this simple implementation. Secure it appropriately in a production environment (e.g., require a master admin key, restrict access).* 
*   **Example Request (using curl):**
    ```bash
    curl -X POST "http://localhost:8000/admin/generate-key"
    ```
*   **Success Response (200 OK):**
    ```json
    {
      "api_key": "<newly_generated_api_key>",
      "message": "Key generated and added successfully."
    }
    ```
    The new key is automatically added to the `.env` file.

## Notes

*   The neural network training is minimal, primarily serving to create embeddings from TF-IDF. More sophisticated training (e.g., using contrastive loss) could improve embedding quality.
*   Text preprocessing is simplified (lowercase, remove punctuation/numbers, remove stopwords, basic split). Stemming was removed due to NLTK resource issues during testing.
*   API key storage uses a simple `.env` file. For production, consider more secure storage like a database or secrets manager.
*   The `/admin/generate-key` endpoint lacks robust security in this example.


