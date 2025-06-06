import pytest
from app import create_app
import json

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_search_document_with_references(client):
    test_data = {
        "references": [
            {
                "title": "Test Document 1",
                "content": "This is a test document content for testing purposes."
            }
        ],
        "userQuery": "What is this document about?"
    }
    
    response = client.post(
        '/api/search-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'data' in data
    assert 'ragResponse' in data['data']

def test_search_document_without_references(client):
    test_data = {
        "userQuery": "What is this document about?"
    }
    
    response = client.post(
        '/api/search-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'data' in data
    assert 'ragResponse' in data['data']

def test_search_document_missing_query(client):
    test_data = {
        "references": [
            {
                "title": "Test Document 1",
                "content": "This is a test document content."
            }
        ]
    }
    
    response = client.post(
        '/api/search-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'message' in data
    assert 'userQuery가 누락되었습니다' in data['message']

def test_search_document_invalid_reference(client):
    test_data = {
        "references": [
            {
                "title": "Test Document 1"
            }
        ],
        "userQuery": "What is this document about?"
    }
    
    response = client.post(
        '/api/search-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'message' in data
    assert 'reference에는 title과 content가 모두 포함되어야 합니다' in data['message'] 