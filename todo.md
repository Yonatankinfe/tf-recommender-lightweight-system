## Content-Based Recommendation System Todo List

### 1. Project Setup
- [X] Create project directory structure (`src`, `data`, `models`).
- [X] Create example input data file (`data/items.csv`).

### 2. Data Processing
- [X] Implement function to load data from CSV (`src/data_loader.py`).
- [X] Implement text preprocessing (cleaning, tokenization) (`src/preprocessing.py`).
- [X] Implement TF-IDF feature extraction (`src/preprocessing.py`).

### 3. Model Implementation (TensorFlow/Keras)
- [X] Define simple MLP neural network architecture (`src/model.py`).
- [ ] Implement model training logic (`src/train.py`).
- [X] Implement function to generate embeddings using the trained model (`src/model.py`).
- [X] Implement model and TF-IDF vectorizer persistence (saving/loading) (`src/utils.py`).

### 4. Recommendation Logic
- [X] Implement cosine similarity calculation (`src/recommender.py`).
- [X] Implement function to find top N similar items (`src/recommender.py`).

### 5. API Development (FastAPI)
- [X] Set up basic FastAPI application structure (`src/main.py`).
- [X] Implement API key generation logic (`src/auth.py`).
- [X] Implement API key storage/validation (using `.env` file and `python-dotenv`) (`src/auth.py`).
- [X] Implement API key authentication dependency (`src/auth.py`).
- [X] Implement `/recommend` endpoint in `src/main.py`:
    - [X] Handle GET request with `item_id` and `n`.
    - [X] Integrate API key authentication.
    - [X] Load model, vectorizer, and embeddings.
    - [X] Call recommendation logic.
    - [X] Return JSON response.
    - [X] Implement error handling (400, 401, 404).
- [X] Implement `/admin/generate-key` endpoint (or simple script) (`src/main.py` or `generate_key.py`):
    - [X] Handle POST request or run script.
    - [X] Implement basic security/confirmation.
    - [X] Generate and store a new API key in `.env`.
    - [X] Output the new key.

### 6. Scripts & Orchestration
- [X] Finalize the main training script (`src/train.py`).
- [X] Finalize the API server script (`src/main.py`).
- [X] Create `.env` file for API keys.

### 7. Documentation & Packaging
- [X] Write inline comments throughout the code.
- [X] Create `README.md` with setup, training, and API usage instructions.
- [X] Generate final `requirements.txt` (`pip freeze > requirements.txt`).
- [X] Ensure logical code organization.
- [ ] Package the project folder for delivery.
