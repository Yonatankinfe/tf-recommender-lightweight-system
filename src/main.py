from fastapi import FastAPI, HTTPException, Depends, status, Query
from typing import List, Dict, Optional
import uvicorn
import os
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer

# Adjust sys.path if necessary, or ensure modules are installed/discoverable
# Assuming src is the working directory or in PYTHONPATH
from .auth import get_api_key, initialize_api_keys, generate_api_key, add_api_key_to_env
from .utils import load_vectorizer, load_model, load_embeddings
from .recommender import find_top_n_similar

app = FastAPI(
    title="Content-Based Recommendation API",
    description="Provides item recommendations based on content similarity using TF-IDF and a Neural Network.",
    version="0.1.0"
)

# Global variables to hold loaded models and data
# These will be populated during startup
vectorizer: Optional[TfidfVectorizer] = None
model: Optional[tf.keras.Model] = None
item_ids: Optional[List[str]] = None
embeddings: Optional[tf.Tensor] = None

@app.on_event("startup")
def load_artifacts():
    """Load necessary artifacts (vectorizer, model, embeddings) on startup."""
    global vectorizer, model, item_ids, embeddings

    print("Loading artifacts...")
    # Adjust paths if needed, assuming models are in ../models relative to src
    vectorizer = load_vectorizer(filename="tfidf_vectorizer.pkl")
    model = load_model(filename="embedding_model.keras") # Ensure this matches the saved model name
    item_ids, embeddings = load_embeddings(filename="item_embeddings.pkl")

    if not all([vectorizer, model, item_ids is not None, embeddings is not None]):
        print("Warning: Not all artifacts were loaded successfully. Recommendation endpoint might fail.")
        # Depending on requirements, you might want to raise an error or prevent startup
    else:
        print("Artifacts loaded successfully.")

    # Initialize API keys (generate one if none exist)
    initialize_api_keys()

@app.get("/", tags=["General"], summary="Health Check")
def read_root():
    """Basic health check endpoint."""
    return {"status": "OK", "message": "Recommendation API is running."}

@app.get(
    "/recommend",
    tags=["Recommendations"],
    summary="Get Content-Based Recommendations",
    response_model=Dict[str, List[str] | str]
)
def get_recommendations(
    item_id: str = Query(..., description="The ID of the item to get recommendations for."),
    n: int = Query(5, ge=1, le=50, description="The number of recommendations to return."),
    api_key: str = Depends(get_api_key)
):
    """Returns a list of `n` item IDs similar to the given `item_id`.

    Requires a valid API key via the `X-API-Key` header.
    """
    if not all([item_ids is not None, embeddings is not None]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation service is not ready. Artifacts not loaded."
        )

    if item_id not in item_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item ID 	"{item_id}	" not found."
        )

    recommendations = find_top_n_similar(target_item_id=item_id, item_ids=item_ids, embeddings=embeddings, n=n)

    return {"input_item_id": item_id, "recommendations": recommendations}

@app.post(
    "/admin/generate-key",
    tags=["Admin"],
    summary="Generate a New API Key",
    response_model=Dict[str, str],
    # Note: In a real application, this endpoint should be properly secured (e.g., admin roles, IP whitelist).
    # For this exercise, we allow generation but rely on secure deployment/access control.
)
def generate_new_api_key(
    # Here you might add a dependency for admin authentication if needed
    # admin_user: str = Depends(get_admin_user) # Example
):
    """Generates a new API key and adds it to the .env file.

    **Warning:** This endpoint should be secured in a production environment.
    """
    new_key = generate_api_key()
    success = add_api_key_to_env(new_key)
    if success:
        return {"api_key": new_key, "message": "Key generated and added successfully."}
    else:
        # This might happen if the key somehow already exists (very unlikely) or file write fails
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate or save the new API key."
        )

# Example of how to run the server (for local testing)
if __name__ == "__main__":
    # Get the directory of the current script (src)
    src_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root)
    project_root = os.path.dirname(src_dir)
    # Construct the path to the .env file
    dotenv_path = os.path.join(project_root, ".env")

    print(f"Attempting to load .env from: {dotenv_path}")
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=dotenv_path)

    print("Starting Uvicorn server...")
    # Run uvicorn. Note: For production, use a process manager like Gunicorn + Uvicorn workers.
    # Host 0.0.0.0 makes it accessible externally (within sandbox network or if port exposed)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=src_dir)
    # reload=True is useful for development, disable in production

