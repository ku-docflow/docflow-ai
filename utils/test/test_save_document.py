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

def test_save_document_success(client):
    test_data = {
    "documentId": 101,
    "organizationId": 2001,
    "content": "## Overview\nThis document outlines the main objectives for Q3 2025...\n\n### Goals\n1. Improve user retention\n2. Launch beta features\n3. Optimize backend performance",
    "userId": "34567",
    "createdBy": "Jane Doe",
    "createdAt": "2025-05-12T10:30:00Z"
    }

    response = client.post( 
        '/api/save-document',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['statusCode'] == 200
    assert data['message'] == "성공했습니다"