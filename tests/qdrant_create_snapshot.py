import os
import sys
import requests

# Add the parent directory to sys.path to access config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_API_KEY

# Define the full URL for Qdrant API
qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/collections/{QDRANT_COLLECTION_NAME}"

# Define collection properties
collection_payload = {
    "vector_size": 1536,  # Typically 1536 for OpenAI embeddings, adjust as necessary
    "distance": "Cosine",  # Or other distance metric like Euclidean or DotProduct
}

# Create the collection
response = requests.put(
    qdrant_url,
    headers={
        "api-key": QDRANT_API_KEY  # Use your actual API key here
    },
    json=collection_payload,
)

# Check if the collection was created successfully
if response.status_code == 200:
    print(f"Collection '{QDRANT_COLLECTION_NAME}' created successfully.")
else:
    print(f"Failed to create collection. Error: {response.text}")
