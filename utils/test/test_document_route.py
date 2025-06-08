import pytest
from unittest.mock import patch
from app import create_app  # Adjust if create_app is in a different module

@pytest.fixture
def app():
    app = create_app()
    # Establish application context for things like `request`, `jsonify`, etc.
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    return app.test_client()

@patch("routes.document_route.extract_keyword.extract_keywords_and_category")
@patch("routes.document_route.generate_document.generate_document")
@patch("routes.document_route.generate_summary.generate_document_summary")
@patch("routes.document_route.qdrant_service.store_document_embedding")
def test_process_document_success(
    mock_store_document,
    mock_generate_summary,
    mock_generate_doc,
    mock_extract_keywords,
    client
):
    payload = {
        "documentId": 123,
        "organizationId": 456,
        "userId": 789,
        "chatContext": "회의에서 논의된 주요 내용입니다.",
        "createdBy": "홍길동",
        "createdAt": "2023-10-01T12:00:00Z"
    }

    mock_extract_keywords.return_value = {
        "keywords": ["회의", "논의", "결정"],
        "category": "회의록"
    }

    mock_generate_doc.return_value = "회의 전체 문서 내용입니다."
    mock_generate_summary.return_value = {
        "title": "회의 요약 제목",
        "summary": "요약된 회의 내용입니다."
    }

    response = client.post("/api/process-document", json=payload)

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["message"] == "성공했습니다"
    assert json_data["data"]["title"] == "회의 요약 제목"
    assert json_data["data"]["document"] == "회의 전체 문서 내용입니다."
    assert json_data["data"]["summary"] == "요약된 회의 내용입니다."
    mock_store_document.assert_called_once()

def test_process_document_missing_fields(client):
    payload = {
        "documentId": 123,
        "organizationId": 456,
        "userId": 789,
        "createdBy": "홍길동",
        "createdAt": "2023-10-01T12:00:00Z"
    }

    response = client.post("/api/process-document", json=payload)
    assert response.status_code == 400
    json_data = response.get_json()
    assert "생성봇 필수 필드가 누락되었습니다." in json_data["message"]
