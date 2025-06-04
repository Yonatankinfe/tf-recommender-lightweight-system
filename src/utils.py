import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
import os

MODELS_DIR = "../models"

def save_vectorizer(vectorizer: TfidfVectorizer, filename: str = "tfidf_vectorizer.pkl"):
    """Saves the TF-IDF vectorizer to a file using pickle.

    Args:
        vectorizer: The fitted TfidfVectorizer object.
        filename: The name of the file to save the vectorizer to.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(vectorizer, f)
        print(f"Vectorizer saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving vectorizer: {e}")

def load_vectorizer(filename: str = "tfidf_vectorizer.pkl") -> TfidfVectorizer | None:
    """Loads the TF-IDF vectorizer from a file.

    Args:
        filename: The name of the file containing the vectorizer.

    Returns:
        The loaded TfidfVectorizer object, or None if loading fails.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"Vectorizer loaded successfully from {filepath}")
        return vectorizer
    except FileNotFoundError:
        print(f"Error: Vectorizer file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading vectorizer: {e}")
        return None

def save_model(model: keras.Model, filename: str = "embedding_model.keras"):
    """Saves the Keras model.

    Args:
        model: The trained Keras model.
        filename: The name of the file to save the model to (use .keras format).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        model.save(filepath)
        print(f"Model saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename: str = "embedding_model.keras") -> keras.Model | None:
    """Loads the Keras model.

    Args:
        filename: The name of the file containing the model.

    Returns:
        The loaded Keras model, or None if loading fails.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        model = keras.models.load_model(filepath)
        print(f"Model loaded successfully from {filepath}")
        # Re-compile is often needed after loading, although for inference it might not be strictly necessary
        # model.compile(optimizer='adam', loss='mse') # Use the same compile settings as during training
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_embeddings(embeddings: tf.Tensor, item_ids: list, filename: str = "item_embeddings.pkl"):
    """Saves the generated embeddings and corresponding item IDs.

    Args:
        embeddings: A TensorFlow Tensor or NumPy array of embeddings.
        item_ids: A list of item IDs corresponding to the embeddings.
        filename: The name of the file to save the embeddings to.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        # Convert tensor to numpy array for pickling if needed
        if isinstance(embeddings, tf.Tensor):
            embeddings_np = embeddings.numpy()
        else:
            embeddings_np = embeddings

        data_to_save = {"item_ids": item_ids, "embeddings": embeddings_np}
        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)
        print(f"Embeddings saved successfully to {filepath}")
    except Exception as e:
        print(f"Error saving embeddings: {e}")

def load_embeddings(filename: str = "item_embeddings.pkl") -> tuple[list, tf.Tensor | None] | tuple[None, None]:
    """Loads the item embeddings and corresponding item IDs.

    Args:
        filename: The name of the file containing the embeddings.

    Returns:
        A tuple containing (list of item_ids, TensorFlow Tensor of embeddings), or (None, None) if loading fails.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        item_ids = data["item_ids"]
        embeddings_np = data["embeddings"]
        embeddings_tf = tf.convert_to_tensor(embeddings_np, dtype=tf.float32)
        print(f"Embeddings loaded successfully from {filepath}")
        return item_ids, embeddings_tf
    except FileNotFoundError:
        print(f"Error: Embeddings file not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None, None

if __name__ == '__main__':
    # Example Usage (requires dummy model and vectorizer)
    print("\n--- Testing Persistence Utilities ---")
    # Dummy data
    dummy_vectorizer = TfidfVectorizer()
    dummy_vectorizer.fit(["sample text"])
    dummy_model = tf.keras.Sequential([tf.keras.layers.Dense(10, input_shape=(1,)), tf.keras.layers.Dense(5)])
    dummy_model.compile(optimizer='adam', loss='mse')
    dummy_embeddings = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=tf.float32)
    dummy_ids = ["id1", "id2"]

    # Save
    save_vectorizer(dummy_vectorizer, "test_vectorizer.pkl")
    save_model(dummy_model, "test_model.keras")
    save_embeddings(dummy_embeddings, dummy_ids, "test_embeddings.pkl")

    # Load
    loaded_vectorizer = load_vectorizer("test_vectorizer.pkl")
    loaded_model = load_model("test_model.keras")
    loaded_ids, loaded_embeddings = load_embeddings("test_embeddings.pkl")

    print("\nLoaded Vectorizer:", loaded_vectorizer)
    # print("\nLoaded Model Summary:")
    # loaded_model.summary() # Requires graphviz potentially
    print("\nLoaded IDs:", loaded_ids)
    print("\nLoaded Embeddings:", loaded_embeddings)

    # Clean up test files
    print("\nCleaning up test files...")
    if os.path.exists(os.path.join(MODELS_DIR, "test_vectorizer.pkl")):
        os.remove(os.path.join(MODELS_DIR, "test_vectorizer.pkl"))
    if os.path.exists(os.path.join(MODELS_DIR, "test_model.keras")):
        # Keras save format might create a directory
        import shutil
        if os.path.isdir(os.path.join(MODELS_DIR, "test_model.keras")):
            shutil.rmtree(os.path.join(MODELS_DIR, "test_model.keras"))
        else:
             os.remove(os.path.join(MODELS_DIR, "test_model.keras"))
    if os.path.exists(os.path.join(MODELS_DIR, "test_embeddings.pkl")):
        os.remove(os.path.join(MODELS_DIR, "test_embeddings.pkl"))
    print("Cleanup complete.")

