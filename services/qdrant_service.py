import logging
from typing import Dict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import OpenAIEmbeddings
from config import QDRANT_URL, QDRANT_COLLECTION_NAME

def store_document_embedding(document_id: str, payload: Dict) -> None:
    """
    Store a document embedding into Qdrant with the required structure.
    """
    try:
        # Initialize embeddings model
        embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        # Combine title and summary for embedding
        combined_text = f"{payload.get('title', '')} {payload.get('document', '')}".strip()
        
        # Compute the embedding vector
        vector = embeddings_model.embed_query(combined_text)
        
        # Initialize Qdrant client
        client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=True,
        )
        
        # Check if the collection exists, create if it doesn't
        if not client.collection_exists(QDRANT_COLLECTION_NAME):
            client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.DOT),
            )
        
        # Create the point with the required structure
        point = models.PointStruct(
            id=document_id,
            vector=vector,
            payload={
                "title": payload.get("title"),
                "summary": payload.get("summary"),
                "userId": payload.get("userId"),
                "createdBy": payload.get("createdBy"),
                "keywords": payload.get("keywords"),
                "category": payload.get("category"),
                "createdAt": payload.get("createdAt"),
                "organizationId": payload.get("OrganizationId"),
            }
        )
        
        # Upsert the point into Qdrant
        client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[point]
        )
        
        print(f"Document with ID {document_id} stored successfully in Qdrant.")

    except Exception as e:
        logging.exception("Error storing document in Qdrant")
        raise Exception("문서 임베딩 저장 실패") from e