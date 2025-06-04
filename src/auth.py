import secrets
import os
from dotenv import load_dotenv, set_key, find_dotenv
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

# Load environment variables from .env file
# Use find_dotenv to locate .env in the project root or parent directories
dotenv_path = find_dotenv()
if not dotenv_path:
    # If .env doesn't exist, create one in the project root
    project_root = os.path.dirname(os.path.dirname(__file__)) # Assumes src is one level down
    dotenv_path = os.path.join(project_root, ".env")
    with open(dotenv_path, "w") as f:
        f.write("# API Keys will be stored here\n")
    print(f"Created .env file at: {dotenv_path}")

load_dotenv(dotenv_path=dotenv_path)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Store keys in memory for simplicity during runtime, loaded from .env
# A more robust solution might use a database or a dedicated key management service.
VALID_API_KEYS = set(key.strip() for key in os.getenv("VALID_API_KEYS", "").split(",") if key.strip())

def generate_api_key(length: int = 32) -> str:
    """Generates a cryptographically secure random API key."""
    return secrets.token_urlsafe(length)

def add_api_key_to_env(new_key: str) -> bool:
    """Adds a new API key to the .env file and updates the in-memory set."""
    global VALID_API_KEYS
    if new_key in VALID_API_KEYS:
        print(f"Key 	'{new_key[:8]}...	' already exists.")
        return False

    current_keys = os.getenv("VALID_API_KEYS", "")
    if current_keys:
        updated_keys = current_keys + "," + new_key
    else:
        updated_keys = new_key

    # Use set_key to update the .env file
    # find_dotenv() ensures we write to the correct .env file
    success = set_key(find_dotenv(raise_error_if_not_found=True), "VALID_API_KEYS", updated_keys)

    if success:
        VALID_API_KEYS.add(new_key)
        print(f"Successfully added key 	'{new_key[:8]}...	' to .env and runtime set.")
        return True
    else:
        print("Error updating .env file.")
        return False

def get_api_key(api_key: str = Security(api_key_header)) -> str:
    """Dependency function to validate the API key from the header."""
    if api_key in VALID_API_KEYS:
        return api_key
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

# Function to initialize at least one key if none exist
def initialize_api_keys():
    """Checks if any API keys exist in .env, generates one if not."""
    if not VALID_API_KEYS:
        print("No API keys found in .env. Generating initial key...")
        initial_key = generate_api_key()
        if add_api_key_to_env(initial_key):
            print(f"Initial API Key generated: {initial_key}")
            print(f"Please store this key securely. It has been added to {dotenv_path}")
        else:
            print("Failed to generate and store initial API key.")
    else:
        print(f"Loaded {len(VALID_API_KEYS)} API key(s) from {dotenv_path}")

if __name__ == "__main__":
    print("--- Testing API Key Utilities ---")
    # Ensure .env exists (it should after import)
    if not os.path.exists(dotenv_path):
        print("Error: .env file not found or created.")
    else:
        print(f"Using .env file: {dotenv_path}")
        initialize_api_keys()

        # Test generation
        new_key_1 = generate_api_key()
        print(f"\nGenerated Key 1: {new_key_1}")

        # Test adding
        print("\nAttempting to add Key 1...")
        add_api_key_to_env(new_key_1)
        print(f"Current keys in memory: {VALID_API_KEYS}")

        # Test adding duplicate
        print("\nAttempting to add Key 1 again...")
        add_api_key_to_env(new_key_1)

        # Test validation (requires running a dummy FastAPI app or mocking)
        print("\nSimulating validation:")
        print(f"Is 	'{new_key_1[:8]}...	' valid? {new_key_1 in VALID_API_KEYS}")
        print(f"Is 	'invalid-key	' valid? {"invalid-key" in VALID_API_KEYS}")

        # Note: To fully test get_api_key, it needs to be used within a FastAPI route.

