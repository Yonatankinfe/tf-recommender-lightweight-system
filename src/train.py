import os
import sys
import pandas as pd
import numpy as np

# Ensure the src directory is in the Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(src_dir)) # Add project root to path

from src.data_loader import load_data
from src.preprocessing import preprocess_text, create_tfidf_vectorizer
# Updated import for new model structure
from src.model import build_base_embedding_model, build_training_model, generate_embeddings 
from src.utils import save_vectorizer, save_model, save_embeddings

# Configuration
DATA_PATH = "../data/items.csv" # Relative to the script location (src)
MAX_FEATURES = 5000 # Max features for TF-IDF
EMBEDDING_DIM = 128 # Output dimension for embeddings
EPOCHS = 10 # Increased epochs slightly for potentially better reconstruction
BATCH_SIZE = 32

def main():
    """Main function to orchestrate the training and artifact generation process."""
    print("--- Starting Training Pipeline ---")

    # 1. Load Data
    print("\nStep 1: Loading data...")
    items_df = load_data(DATA_PATH)
    if items_df.empty:
        print("Error: Failed to load data. Exiting.")
        return
    item_ids = items_df["item_id"].tolist()

    # 2. Preprocess Text
    print("\nStep 2: Preprocessing text data...")
    items_df["content_clean"] = items_df["content"].astype(str).apply(preprocess_text)
    print("Text preprocessing complete.")
    print("Sample processed text:")
    print(items_df["content_clean"].head())

    # 3. Create and Fit TF-IDF Vectorizer
    print("\nStep 3: Creating and fitting TF-IDF vectorizer...")
    vectorizer = create_tfidf_vectorizer(items_df["content_clean"].tolist(), max_features=MAX_FEATURES)

    # 4. Save Vectorizer
    print("\nStep 4: Saving TF-IDF vectorizer...")
    save_vectorizer(vectorizer, filename="tfidf_vectorizer.pkl")

    # 5. Transform Text to TF-IDF Vectors
    print("\nStep 5: Transforming text to TF-IDF vectors...")
    tfidf_matrix = vectorizer.transform(items_df["content_clean"].tolist())
    tfidf_matrix_dense = tfidf_matrix.toarray().astype(np.float32)
    print(f"TF-IDF matrix shape: {tfidf_matrix_dense.shape}")
    input_dim = tfidf_matrix_dense.shape[1]

    # 6. Build Base Embedding Model
    print("\nStep 6: Building the base embedding model...")
    base_model = build_base_embedding_model(input_dim=input_dim, embedding_dim=EMBEDDING_DIM)

    # 7. Build Training Model (Autoencoder-like)
    print("\nStep 7: Building the training model...")
    training_model = build_training_model(base_model, input_dim=input_dim)

    # 8. Train the Training Model
    print("\nStep 8: Training the autoencoder model...")
    # Use the TF-IDF vectors as both input and target for reconstruction
    training_model.fit(tfidf_matrix_dense, tfidf_matrix_dense, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    print("Model training complete.")

    # 9. Generate Embeddings using the Base Model
    print("\nStep 9: Generating embeddings for all items using the base model...")
    # Use the base_model (which shares trained weights) for generating final embeddings
    embeddings = generate_embeddings(base_model, tfidf_matrix_dense)

    # 10. Save the Base Model (for inference)
    print("\nStep 10: Saving the base embedding model...")
    save_model(base_model, filename="embedding_model.keras") # Save the base model

    # 11. Save Embeddings
    print("\nStep 11: Saving item embeddings...")
    save_embeddings(embeddings, item_ids, filename="item_embeddings.pkl")

    print("\n--- Training Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()

