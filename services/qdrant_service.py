import logging
from typing import Dict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from config import QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME, QDRANT_URL

embeddings_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions=1024  #uncomment when needed for specific model
)


def store_document_embedding(document_id: str, payload: Dict) -> None:
    """
    Store a document embedding into Qdrant using LangChain integration.
    """
    try:
        combined_text = f"{payload.get('title', '')} {payload.get('summary', '')}".strip()

        doc = Document(
            page_content=combined_text,
            metadata={
                "title": payload.get("title"),
                "summary": payload.get("summary"),
                "userId": payload.get("userId"),
                "createdBy": payload.get("createdBy"),
                "keywords": payload.get("keywords"),
                "category": payload.get("category"),
                "docId": document_id
            }
        )

        # Store the doc in Qdrant (force recreation to avoid vector name conflict)
        # FIX REQUIRED :
        Qdrant.from_documents(
            documents=[doc],
            embedding=embeddings_model,
            url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION_NAME,
            prefer_grpc=True,
            force_recreate=True  # Fix: Avoid named vector reuse error -> FIX REQUIRED
        )

        print(f"Document with ID {document_id} stored successfully in Qdrant.")

    except Exception as e:
        logging.exception("Error storing document in Qdrant")
        raise Exception("문서 임베딩 저장 실패") from e