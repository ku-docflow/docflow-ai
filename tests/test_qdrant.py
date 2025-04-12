# backend/tests/test_qdrant_service.py
import pytest
from unittest.mock import MagicMock, patch
from services.qdrant_service import store_document_embedding
from langchain_core.documents import Document
import logging

# Sample payload for tests
SAMPLE_PAYLOAD = {
    "title": "API 설계 문서",
    "summary": "요약된 내용입니다",
    "userId": "user123",
    "createdBy": "John Doe",
    "keywords": ["API", "JWT", "설계"],
    "category": "DEV_DOC",
}

@pytest.fixture
def mock_qdrant():
    """Fixture to mock Qdrant.from_documents."""
    with patch("services.qdrant_service.Qdrant") as mock_qdrant:
        yield mock_qdrant

@pytest.fixture
def mock_embeddings():
    """Fixture to mock embeddings_model."""
    with patch("services.qdrant_service.embeddings_model") as mock_emb:
        yield mock_emb

@pytest.fixture
def mock_logger():
    """Fixture to mock logging.exception."""
    with patch("services.qdrant_service.logging.exception") as mock_log:
        yield mock_log

def test_store_document_embedding_success(mock_qdrant, mock_embeddings):
    """Test storing a document embedding successfully."""
    document_id = "doc123"
    
    # Call the function
    store_document_embedding(document_id, SAMPLE_PAYLOAD)
    
    # Verify Qdrant.from_documents was called with correct arguments
    expected_doc = Document(
        page_content="API 설계 문서 요약된 내용입니다",
        metadata={
            "title": "API 설계 문서",
            "summary": "요약된 내용입니다",
            "userId": "user123",
            "createdBy": "John Doe",
            "keywords": ["API", "JWT", "설계"],
            "category": "DEV_DOC",
            "docId": "doc123",
        }
    )
    
    mock_qdrant.from_documents.assert_called_once_with(
        documents=[expected_doc],
        embedding=mock_embeddings,
        url="http://localhost:6333",
        collection_name="documents",
        prefer_grpc=True
    )

def test_store_document_embedding_missing_fields(mock_qdrant, mock_embeddings):
    """Test storing a document with missing payload fields."""
    document_id = "doc456"
    partial_payload = {
        "userId": "user456",
        "createdBy": "Jane Doe"
        # title, summary, keywords, category omitted
    }
    
    # Call the function
    store_document_embedding(document_id, partial_payload)
    
    # Verify Qdrant.from_documents was called with correct arguments
    expected_doc = Document(
        page_content="",  # title and summary are empty
        metadata={
            "title": None,
            "summary": None,
            "userId": "user456",
            "createdBy": "Jane Doe",
            "keywords": None,
            "category": None,
            "docId": "doc456",
        }
    )
    
    mock_qdrant.from_documents.assert_called_once_with(
        documents=[expected_doc],
        embedding=mock_embeddings,
        url="http://localhost:6333",
        collection_name="documents",
        prefer_grpc=True
    )

def test_store_document_embedding_qdrant_failure(mock_qdrant, mock_embeddings, mock_logger):
    """Test handling of Qdrant errors."""
    document_id = "doc789"
    mock_qdrant.from_documents.side_effect = Exception("Qdrant connection error")
    
    # Call the function and expect an exception
    with pytest.raises(Exception, match="문서 임베딩 저장 실패"):
        store_document_embedding(document_id, SAMPLE_PAYLOAD)
    
    # Verify logging was called
    mock_logger.assert_called_once_with("Error storing document in Qdrant")
    
    # Verify Qdrant.from_documents was attempted
    mock_qdrant.from_documents.assert_called()

def test_store_document_embedding_empty_payload(mock_qdrant, mock_embeddings):
    """Test storing a document with an empty payload."""
    document_id = "doc999"
    empty_payload = {}
    
    # Call the function
    store_document_embedding(document_id, empty_payload)
    
    # Verify Qdrant.from_documents was called with correct arguments
    expected_doc = Document(
        page_content="",
        metadata={
            "title": None,
            "summary": None,
            "userId": None,
            "createdBy": None,
            "keywords": None,
            "category": None,
            "docId": "doc999",
        }
    )
    
    mock_qdrant.from_documents.assert_called_once_with(
        documents=[expected_doc],
        embedding=mock_embeddings,
        url="http://localhost:6333",
        collection_name="documents",
        prefer_grpc=True
    )