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
