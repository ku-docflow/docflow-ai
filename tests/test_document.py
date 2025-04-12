import sys
import os
import json
import pytest
from unittest.mock import patch
from config import CATEGORY


# Add root to path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Sample POST payload
POST_BODY = {
    "documentId": "b6f1b2e0-3c3e-4f9a-9439-4e87df75e732",
    "chatContext": "김영수: API 설계 시작합니다.\n정하나: 로그인 방식은 JWT로 가시죠.",
    "userId": "user-1923",
    "createdBy": "김영수",
}

@patch("services.llm_service.extract_keywords_and_category")
@patch("services.llm_service.generate_document_summary")
@patch("services.qdrant_service.store_document_embedding")
def test_process_success(mock_store, mock_generate, mock_extract, client):
    mock_extract.return_value = {
        "keywords": ["API", "JWT", "설계"],
        "category": CATEGORY.DEV_DOC,
    }
    mock_generate.return_value = {
        "title": "API 설계 문서",
        "summary": "요약된 내용입니다",
        "document": "생성된 문서 원문",
    }
    mock_store.return_value = None

    response = client.post("/api/process-document", json=POST_BODY)
    data = response.get_json()

    assert response.status_code == 200
    assert data["message"] == "성공했습니다"
    assert data["data"]["documentId"] == POST_BODY["documentId"]
    assert data["data"]["category"] == CATEGORY.DEV_DOC
    assert "document" in data["data"]

@patch("services.llm_service.extract_keywords_and_category", side_effect=Exception("LLM 실패"))
def test_process_llm_failure(mock_extract, client):
    response = client.post("/api/process-document", json=POST_BODY)
    data = response.get_json()

    assert response.status_code == 500
    assert data["error"] == "AgentFailed"
    assert "LLM 응답 생성에 실패했습니다" in data["message"]

def test_process_missing_fields(client):
    response = client.post("/api/process-document", json={})
    data = response.get_json()

    assert response.status_code == 400
    assert data["error"] == "InvalidInput"
    assert "필수 필드가 누락되었습니다" in data["message"]
