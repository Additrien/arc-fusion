import os
import weaviate
import weaviate.classes as wvc
from typing import Dict, List, Any, Optional
from weaviate.classes.init import Auth
import asyncio
from concurrent.futures import ThreadPoolExecutor

class VectorStore:
    """Handles Weaviate vector database operations."""
    
    def __init__(self):
        self.weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.collection_name = "DocumentChunks"
        self.client = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._connected = False
    
    async def _ensure_connected(self):
        """Ensure Weaviate client is connected."""
        if not self._connected:
            await self._connect()
    
    async def _connect(self):
        """Initialize Weaviate client and create schema if needed."""
        try:
            # Connect to Weaviate
            self.client = weaviate.connect_to_local(
                host=self.weaviate_url.split("://")[1].split(":")[0],
                port=int(self.weaviate_url.split(":")[-1])
            )
            
            # Create collection if it doesn't exist
            await self._create_collection()
            self._connected = True
            
        except Exception as e:
            raise Exception(f"Failed to connect to Weaviate: {str(e)}")
    
    async def _create_collection(self):
        """Create the DocumentChunks collection with proper schema."""
        def _create_sync():
            try:
                # Check if collection exists
                existing_collection = self.client.collections.get(self.collection_name)
                # Collection exists, no need to create
                return
            except Exception:
                # Collection doesn't exist, create it
                try:
                    self.client.collections.create(
                        name=self.collection_name,
                        properties=[
                            wvc.config.Property(
                                name="content",
                                data_type=wvc.config.DataType.TEXT,
                                description="Child chunk content"
                            ),
                            wvc.config.Property(
                                name="parent_id",
                                data_type=wvc.config.DataType.TEXT,
                                description="Parent chunk identifier"
                            ),
                            wvc.config.Property(
                                name="document_id",
                                data_type=wvc.config.DataType.TEXT,
                                description="Document identifier"
                            ),
                            wvc.config.Property(
                                name="filename",
                                data_type=wvc.config.DataType.TEXT,
                                description="Original filename"
                            ),
                            wvc.config.Property(
                                name="parent_index",
                                data_type=wvc.config.DataType.INT,
                                description="Parent chunk index"
                            ),
                            wvc.config.Property(
                                name="child_index",
                                data_type=wvc.config.DataType.INT,
                                description="Child chunk index"
                            )
                        ]
                    )
                except Exception as e:
                    raise Exception(f"Failed to create collection: {str(e)}")
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _create_sync)
    
    async def store_document_chunks(self, result: Dict[str, Any]):
        """Store child chunks with embeddings in Weaviate."""
        await self._ensure_connected()
        
        def _store_sync():
            collection = self.client.collections.get(self.collection_name)
            
            # Prepare batch data
            objects = []
            for chunk_data in result["child_chunks"]:
                objects.append(
                    wvc.data.DataObject(
                        properties={
                            "content": chunk_data["content"],
                            "parent_id": chunk_data["parent_id"],
                            "document_id": chunk_data["document_id"],
                            "filename": chunk_data["filename"],
                            "parent_index": chunk_data["parent_index"],
                            "child_index": chunk_data["child_index"]
                        },
                        vector=chunk_data["embedding"],
                        uuid=chunk_data["id"]
                    )
                )
            
            # Batch insert
            collection.data.insert_many(objects)
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _store_sync)
    
    async def hybrid_search(self, query: str, query_embedding: List[float], 
                           limit: int = 20) -> List[Dict[str, Any]]:
        """Perform hybrid search (vector + BM25) on child chunks."""
        await self._ensure_connected()
        
        def _search_sync():
            collection = self.client.collections.get(self.collection_name)
            
            response = collection.query.hybrid(
                query=query,
                vector=query_embedding,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(distance=True, score=True)
            )
            
            results = []
            for obj in response.objects:
                results.append({
                    "id": str(obj.uuid),
                    "content": obj.properties["content"],
                    "parent_id": obj.properties["parent_id"],
                    "document_id": obj.properties["document_id"],
                    "filename": obj.properties["filename"],
                    "parent_index": obj.properties["parent_index"],
                    "child_index": obj.properties["child_index"],
                    "distance": obj.metadata.distance,
                    "score": obj.metadata.score
                })
            
            return results
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _search_sync)
    
    async def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all unique documents in the database."""
        await self._ensure_connected()
        
        def _get_docs_sync():
            collection = self.client.collections.get(self.collection_name)
            
            # Query for unique document IDs
            response = collection.query.fetch_objects(
                return_properties=["document_id", "filename"],
                limit=1000  # Adjust based on expected document count
            )
            
            # Group by document_id to get unique documents
            documents = {}
            for obj in response.objects:
                doc_id = obj.properties["document_id"]
                if doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": obj.properties["filename"]
                    }
            
            return list(documents.values())
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_docs_sync)
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and metrics."""
        await self._ensure_connected()
        
        def _get_stats_sync():
            collection = self.client.collections.get(self.collection_name)
            
            # Get total object count
            response = collection.aggregate.over_all(total_count=True)
            total_chunks = response.total_count
            
            # Get unique document count (approximate)
            doc_response = collection.query.fetch_objects(
                return_properties=["document_id"],
                limit=1000
            )
            
            unique_docs = len(set(
                obj.properties["document_id"] 
                for obj in doc_response.objects
            ))
            
            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
                "collection_name": self.collection_name,
                "weaviate_url": self.weaviate_url
            }
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_stats_sync)
    
    async def clear_all_documents(self):
        """Clear all documents from the database."""
        await self._ensure_connected()
        
        def _clear_sync():
            collection = self.client.collections.get(self.collection_name)
            # Delete all documents - use delete_all for simplicity
            try:
                result = collection.data.delete_all()
                return result
            except Exception:
                # Fallback: delete by batch if delete_all not available
                return collection.data.delete_many(
                    where=wvc.query.Filter.by_property("document_id").contains_any([""])
                )
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _clear_sync)
    
    def __del__(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
        self.executor.shutdown(wait=True) 