from typing import List, Dict, Optional
import json
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from langchain_openai import OpenAIEmbeddings
import logging
import uuid

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
        """Ensure the memory collection exists in Qdrant."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1024,  # Size for text-embedding-3-large
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def store_interaction(self, query: str, response: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a query-response interaction in memory.
        
        Args:
            query: The user's query
            response: The system's response
            metadata: Additional metadata to store
            
        Returns:
            str: The ID of the stored interaction
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
            logger.error(f"Error storing interaction: {str(e)}")
            raise

    def retrieve_relevant_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve relevant past interactions based on semantic similarity.
        
        Args:
            query: The current query to find relevant memories for
            limit: Maximum number of memories to retrieve
            
        Returns:
            List[Dict]: List of relevant interactions
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
            logger.error(f"Error retrieving memories: {str(e)}")
            return []

    def format_memories_for_prompt(self, memories: List[Dict]) -> str:
        """
        Format retrieved memories into a string suitable for inclusion in a prompt.
        
        Args:
            memories: List of memory dictionaries
            
        Returns:
            str: Formatted memory context
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