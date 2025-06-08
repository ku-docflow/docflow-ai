from typing import List, Dict, Optional
import json
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from langchain_openai import OpenAIEmbeddings
import logging
import uuid
from utils.error_handler import handle_error

logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str = "interaction_memory"):
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.encoder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024
        )
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """qdrant에 지정된 컬렉션이 존재하는지 확인하고, 없으면 생성"""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"신규 컬렉션 생성: {self.collection_name}")
        except Exception as e:
            error_response, _ = handle_error(
                "Error ensuring collection exists",
                f"Failed to ensure collection {self.collection_name} exists: {str(e)}",
                500
            )
            logger.error(error_response["message"])
            raise Exception(error_response["message"])

    def store_interaction(self, query: str, response: str, metadata: Optional[Dict] = None) -> str:
        """
        querty와 response를 저장하고, 메타데이터를 포함하여 Qdrant에 상호작용 기록을 저장
        """
        try:
            # Create interaction record
            interaction = {
                "query": query,
                "response": response,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
            
            # Encode the query for vector search
            query_vector = self.encoder.embed_query(query)
            
            # Generate a unique ID using UUID
            interaction_id = str(uuid.uuid4())
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=interaction_id,
                        vector=query_vector,
                        payload=interaction
                    )
                ]
            )
            
            return interaction_id
            
        except Exception as e:
            error_response, _ = handle_error(
                "Error storing interaction",
                f"Failed to store interaction: {str(e)}",
                500
            )
            logger.error(error_response["message"])
            raise Exception(error_response["message"])

    def retrieve_relevant_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """
        기존 상호작용에서 관련된 기억을 검색하고 반환
        """
        try:
            # Encode the query
            query_vector = self.encoder.embed_query(query)
            
            # Search for similar interactions
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # Extract and return the interactions
            return [hit.payload for hit in search_result]
            
        except Exception as e:
            error_response, _ = handle_error(
                "Error retrieving memories",
                f"Failed to retrieve relevant memories: {str(e)}",
                500
            )
            logger.error(error_response["message"])
            raise Exception(error_response["message"])

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        프롬프트에 사용할 수 있도록 string 형식으로 memories를 포맷
        """
        if not memories:
            return ""
            
        formatted_memories = []
        for memory in memories:
            formatted_memory = f"""
이전 상호작용:
질문: {memory['query']}
답변: {memory['response']}
시간: {memory['timestamp']}
"""
            formatted_memories.append(formatted_memory)
            
        return "\n".join(formatted_memories)