# tf-recommender-lightweight-system
A lightweight content-based recommendation system built with Python and TensorFlow/Keras, exposed via a simple REST API with automatic API key generation.
# Content-Based Recommendation System

This project implements a lightweight, simple content-based recommendation system using Python. It leverages TF-IDF for feature extraction, TensorFlow/Keras for generating neural network embeddings, and FastAPI for serving a RESTful API.

## 🌟 Features

*   **Content-Based Recommendations**: Recommends items similar to a given item based on their content (e.g., title, description, tags).
*   **Neural Network Embeddings**: Uses a simple Multi-Layer Perceptron (MLP), trained in an autoencoder-like fashion on TF-IDF features, to generate item embeddings.
*   **TF-IDF**: Utilizes Term Frequency-Inverse Document Frequency for initial text feature extraction.
*   **RESTful API**: Exposes recommendation functionality via a FastAPI application.
*   **API Key Authentication**: Secures the recommendation endpoint using API keys.
*   **Automatic Key Generation**: Includes an endpoint to generate new API keys, which are stored in a `.env` file.

## 📂 Project Structure
```bash
recommendation_system/
├── data/                            # Contains input datasets
│   └── items.csv                    # Example item data (item_id, title, description)
│
├── models/                          # Trained models and saved artifacts
│   ├── tfidf_vectorizer.pkl         # Saved TF-IDF vectorizer
│   ├── embedding_model.keras        # Trained Keras embedding model
│   └── item_embeddings.pkl          # Pickled item embeddings
│
├── src/                             # Source code for the application
│   ├── __init__.py                  # Package initializer
│   ├── auth.py                      # API key handling and authentication
│   ├── data_loader.py               # Loads and prepares item data
│   ├── main.py                      # FastAPI application entry point
│   ├── model.py                     # Defines the Keras model
│   ├── preprocessing.py             # Text preprocessing and TF-IDF logic
│   ├── recommender.py               # Core recommendation logic (e.g., cosine similarity)
│   ├── train.py                     # Model training and artifact generation
│   └── utils.py                     # Utility functions (e.g., saving/loading artifacts)
│
├── venv/                            # Python virtual environment (recommended)
│
├── .env                             # Environment variables (e.g., API keys)
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

*   `data/`: Contains the raw data for items.
*   `models/`: Stores the pre-trained TF-IDF vectorizer, the Keras embedding model, and the generated item embeddings.
*   `src/`: Contains all the Python source code for the application.

## 🚀 Setup

1.  **Clone the Repository (or place project files):**
    ```bash
    # If you have the project folder named 'recommendation_system'
    cd recommendation_system
    ```

2.  **Create and Activate a Virtual Environment:**
    (Recommended to keep dependencies isolated)
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file with all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, you'll need to create it, e.g., `pip freeze > requirements.txt` after installing packages like `tensorflow`, `scikit-learn`, `fastapi`, `uvicorn`, `python-dotenv`, `pandas`)*

## 🏋️ Training the Model

Run the training script from the project's root directory to process the data, train the embedding model, and generate the necessary artifacts in the `models/` directory.

```bash
python src/train.py
```

### This script performs the following steps:
* Loads data from data/items.csv.
* Preprocesses the text content (e.g., title, description).
* Creates and fits a TF-IDF vectorizer.
* Builds and trains the neural network model (autoencoder-style) to learn item embeddings.
* Generates and saves embeddings for all items using the trained base model.
* Saves the TF-IDF vectorizer, the Keras embedding model, and the final item embeddings to the models/ directory.

# 🚀 FastAPI Recommendation Server

A lightweight FastAPI-based API server that delivers item recommendations using a trained TF-IDF + embedding model.

---

## 📦 Prerequisites

Before starting the server, ensure you have successfully run the training script:

```bash
python src/train.py
```
### This should generate the following files in the models/ directory:
* tfidf_vectorizer.pkl
* embedding_model.keras
* item_embeddings.pkl
# 🔑 API Key Setup
On the first run, if no .env file or API keys exist, the server will automatically generate an initial API key and store it in a new .env file.

#### 👉 Check the console log during the first server startup to copy the generated key.

# 🖥 Starting the Server

From the project root, start the FastAPI server using Uvicorn:
```bash

```
