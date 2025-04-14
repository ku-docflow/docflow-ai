import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_MODEL = os.getenv("LANGCHAIN_MODEL", "gpt-3.5-turbo")

PORT = int(os.getenv("PORT", 8081))

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documents")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "your-qdrant-api-key")
QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


class CATEGORY:
    DEV_DOC = "DEV_DOC"
    MEETING_DOC = "MEETING_DOC"
