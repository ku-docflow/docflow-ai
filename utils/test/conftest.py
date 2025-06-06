import pytest
import os
import sys
import logging
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("QDRANT_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("FLASK_ENV", "testing")
    monkeypatch.setenv("FLASK_DEBUG", "1")

@pytest.fixture(autouse=True)
def mock_openai():
    with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
        mock_embeddings.return_value.embed_query.return_value = [0.1] * 1024
        with patch('langchain.llms.OpenAI') as mock_llm:
            mock_llm.return_value.invoke.return_value = "Mocked response"
            yield mock_llm

@pytest.fixture(autouse=True)
def mock_qdrant():
    with patch('qdrant_client.QdrantClient') as mock_client:
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        yield mock_instance 