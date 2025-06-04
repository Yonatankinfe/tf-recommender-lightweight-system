import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_base_embedding_model(input_dim: int, embedding_dim: int = 128) -> keras.Model:
    """Builds the core MLP model to generate embeddings (up to the embedding layer).

    Args:
        input_dim: The dimensionality of the input TF-IDF vectors (number of features).
        embedding_dim: The desired dimensionality of the output embeddings.

    Returns:
        A Keras model that outputs embeddings.
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_tfidf"),
            layers.Dense(512, activation="relu", name="dense_1"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="relu", name="dense_2"),
            layers.Dropout(0.2),
            layers.Dense(embedding_dim, activation=None, name="embedding_output") # No activation for embeddings
        ],
        name="base_embedding_generator"
    )
    # No compilation needed for the base model if only used for inference or as part of another model
    print("Base Embedding Model Summary:")
    model.summary()
    return model

def build_training_model(base_model: keras.Model, input_dim: int) -> keras.Model:
    """Builds a model for training by adding a reconstruction layer to the base embedding model.

    Args:
        base_model: The pre-built base embedding model.
        input_dim: The dimensionality of the original input (TF-IDF features), used for the output layer.

    Returns:
        A compiled Keras model suitable for autoencoder-like training.
    """
    # Add a final layer to reconstruct the input dimension
    reconstruction_layer = layers.Dense(input_dim, activation=None, name="reconstruction_output")(base_model.layers[-1].output)
    
    # Define the training model using the base model's input and the reconstruction layer's output
    training_model = keras.Model(inputs=base_model.inputs[0], outputs=reconstruction_layer, name="training_autoencoder")

    # Compile the training model with MSE loss
    training_model.compile(optimizer='adam', loss='mse') # Corrected this line

    print("Training Model Summary:")
    training_model.summary()

    return training_model

def generate_embeddings(model: keras.Model, tfidf_matrix) -> tf.Tensor:
    """Generates embeddings for the given TF-IDF matrix using the base embedding model.

    Args:
        model: The base Keras embedding model (outputs embeddings).
        tfidf_matrix: The input TF-IDF matrix (can be sparse or dense).

    Returns:
        A TensorFlow Tensor containing the generated embeddings.
    """
    # Keras models generally expect dense input. Convert sparse matrix if necessary.
    if hasattr(tfidf_matrix, "toarray"):
        tfidf_matrix_dense = tfidf_matrix.toarray()
    else:
        tfidf_matrix_dense = tfidf_matrix

    embeddings = model.predict(tfidf_matrix_dense)
    print(f"Generated embeddings with shape: {embeddings.shape}")
    return tf.convert_to_tensor(embeddings, dtype=tf.float32)

if __name__ == "__main__":
    # Example Usage (requires dummy data)
    import numpy as np

    dummy_input_dim = 100 # Example TF-IDF feature count
    dummy_embedding_dim = 64
    num_samples = 10

    # Create dummy TF-IDF data
    dummy_tfidf = np.random.rand(num_samples, dummy_input_dim).astype(np.float32)

    # Build the base model
    base_model = build_base_embedding_model(dummy_input_dim, dummy_embedding_dim)

    # Build the training model
    training_model = build_training_model(base_model, dummy_input_dim)

    # Generate embeddings (using the untrained base model for demonstration)
    dummy_embeddings = generate_embeddings(base_model, dummy_tfidf)

    print("\nExample Embeddings (first 2):")
    print(dummy_embeddings[:2])

