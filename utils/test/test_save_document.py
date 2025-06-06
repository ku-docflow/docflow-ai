import pytest
from app import create_app
import json
from datetime import datetime
import logging
from unittest.mock import patch

logger = logging.getLogger(__name__)

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_save_document_missing_fields(client):
    test_data = {
        "documentId": 123,
        "organizationId": 456
    }
    
    response = client.post(
        '/api/save-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data['statusCode'] == 500
    assert "문서 저장 중 오류가 발생했습니다" in data['message']

def test_save_document_invalid_document_id(client):
    test_data = {
        "documentId": "invalid_id",
        "organizationId": 456,
        "content": "Test content",
        "userId": "user123",
        "createdBy": "test_user",
        "createdAt": datetime.utcnow().isoformat()
    }
    
    response = client.post(
        '/api/save-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 500
    data = json.loads(response.data)
    assert data['statusCode'] == 500
    assert "문서 저장 중 오류가 발생했습니다" in data['message'] 