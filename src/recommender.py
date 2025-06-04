import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def find_top_n_similar(target_item_id: str, item_ids: list[str], embeddings: tf.Tensor, n: int = 5) -> list[str]:
    """Finds the top N most similar items to a target item based on embedding cosine similarity.

    Args:
        target_item_id: The ID of the item to find recommendations for.
        item_ids: A list of all item IDs, corresponding to the rows in the embeddings tensor.
        embeddings: A TensorFlow Tensor where each row is the embedding for the corresponding item_id.
        n: The number of similar items to return.

    Returns:
        A list of the top N most similar item IDs (excluding the target item itself).
        Returns an empty list if the target_item_id is not found.
    """
    try:
        target_index = item_ids.index(target_item_id)
    except ValueError:
        print(f"Error: Target item ID 	'{target_item_id}	' not found in the provided item list.")
        return []

    # Ensure embeddings are numpy array for scikit-learn compatibility
    if isinstance(embeddings, tf.Tensor):
        embeddings_np = embeddings.numpy()
    else:
        embeddings_np = embeddings

    target_embedding = embeddings_np[target_index].reshape(1, -1)

    # Calculate cosine similarity between the target item and all items
    similarities = cosine_similarity(target_embedding, embeddings_np)[0] # Get the first (and only) row

    # Get indices of top N similar items (excluding the item itself)
    # Add a small offset to n because argsort includes the item itself
    # Ensure we don't request more items than available (minus the target itself)
    num_available = len(item_ids) - 1
    k = min(n + 1, num_available + 1) # +1 because argsort includes self

    # Get indices sorted by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter out the target item's index and take top N
    top_n_indices = [idx for idx in sorted_indices if idx != target_index][:n]

    # Get the corresponding item IDs
    recommended_item_ids = [item_ids[i] for i in top_n_indices]

    return recommended_item_ids

if __name__ == "__main__":
    # Example Usage
    print("--- Testing Recommendation Logic ---")
    dummy_ids = [f"item_{i}" for i in range(1, 6)] # item_1 to item_5
    # Dummy embeddings (5 items, 4 features)
    dummy_embeddings_tf = tf.constant([
        [0.1, 0.9, 0.2, 0.3], # item_1
        [0.8, 0.2, 0.1, 0.4], # item_2 (similar to item_4)
        [0.3, 0.3, 0.8, 0.7], # item_3
        [0.7, 0.3, 0.2, 0.5], # item_4 (similar to item_2)
        [0.1, 0.8, 0.3, 0.2]  # item_5 (similar to item_1)
    ], dtype=tf.float32)

    target = "item_1"
    num_recommendations = 2
    recommendations = find_top_n_similar(target, dummy_ids, dummy_embeddings_tf, n=num_recommendations)
    print(f"\nRecommendations for 	'{target}	' (top {num_recommendations}): {recommendations}")
    # Expected: item_5 should be most similar, then maybe item_4 or item_2

    target = "item_2"
    num_recommendations = 3
    recommendations = find_top_n_similar(target, dummy_ids, dummy_embeddings_tf, n=num_recommendations)
    print(f"\nRecommendations for 	'{target}	' (top {num_recommendations}): {recommendations}")
    # Expected: item_4 should be most similar

    target_nonexistent = "item_99"
    recommendations = find_top_n_similar(target_nonexistent, dummy_ids, dummy_embeddings_tf, n=num_recommendations)
    print(f"\nRecommendations for 	'{target_nonexistent}	' (top {num_recommendations}): {recommendations}")
    # Expected: []

