import os
import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Configure
from typing import Dict, List, Any, Optional
from weaviate.classes.init import Auth
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.utils.performance import time_async_function, time_async_block
from app.core.config.services import VectorStoreConfig

class VectorStore:
    """Handles Weaviate vector database operations."""
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        self.config = config or VectorStoreConfig()
        self.weaviate_url = self.config.url
        self.collection_name = self.config.collection_name
        self.parent_collection_name = self.config.parent_collection_name
        self.client = None
        
        # Dynamic worker pool sizing - use half of available CPU cores
        cpu_count = os.cpu_count() or 4
        max_workers = max(2, cpu_count // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
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
        """Create the DocumentChunks and ParentChunks collections with proper schema."""
        def _create_sync():
            from app.utils.logger import get_logger
            logger = get_logger('arc_fusion.vector_store.collection_creation')
            
            # Create child chunks collection
            try:
                logger.info("Checking if DocumentChunks collection exists...")
                # Check if collection exists
                existing_collection = self.client.collections.get(self.collection_name)
                logger.info("DocumentChunks collection already exists, skipping creation")
                # Collection exists, no need to create
            except Exception:
                # Collection doesn't exist, create it
                try:
                    logger.info("Creating DocumentChunks collection with vector configuration...")
                    logger.info(f"Using Configure.Vectors.self_provided: {Configure.Vectors.self_provided}")
                    
                    collection = self.client.collections.create(
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
                        ],
                        # Configure for manual vector storage
                        vector_config=Configure.Vectors.self_provided
                    )
                    logger.info(f"Successfully created DocumentChunks collection: {collection}")
                except Exception as e:
                    logger.error(f"Failed to create child chunks collection: {str(e)}")
                    raise Exception(f"Failed to create child chunks collection: {str(e)}")
            
            # Create parent chunks collection
            try:
                logger.info("Checking if ParentChunks collection exists...")
                # Check if parent collection exists
                existing_parent_collection = self.client.collections.get(self.parent_collection_name)
                logger.info("ParentChunks collection already exists, skipping creation")
                # Collection exists, no need to create
            except Exception:
                # Collection doesn't exist, create it
                try:
                    logger.info("Creating ParentChunks collection...")
                    self.client.collections.create(
                        name=self.parent_collection_name,
                        properties=[
                            wvc.config.Property(
                                name="content",
                                data_type=wvc.config.DataType.TEXT,
                                description="Parent chunk content"
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
                            )
                        ]
                    )
                    logger.info("Successfully created ParentChunks collection")
                except Exception as e:
                    logger.error(f"Failed to create parent chunks collection: {str(e)}")
                    raise Exception(f"Failed to create parent chunks collection: {str(e)}")
        
        await asyncio.get_event_loop().run_in_executor(self.executor, _create_sync)
    
    async def store_document_chunks(self, result: Dict[str, Any]):
        """Store child chunks with embeddings and parent chunks in Weaviate with robust error handling."""
        await self._ensure_connected()
        
        from app.utils.logger import get_logger
        logger = get_logger('arc_fusion.vector_store')
        
        total_chunks = len(result["child_chunks"])
        
        # Use configurable batch size
        from app.config import WEAVIATE_BATCH_SIZE, BATCH_DELAY_SECONDS
        batch_size = WEAVIATE_BATCH_SIZE
        batch_delay = BATCH_DELAY_SECONDS
        
        logger.info(f"Starting batch storage of {total_chunks} child chunks in batches of {batch_size}")
        
        # First, store parent chunks
        await self._store_parent_chunks(result, logger)
        
        def _store_batch_sync(batch_objects: List, batch_num: int, total_batches: int):
            try:
                collection = self.client.collections.get(self.collection_name)
                
                logger.info(f"Storing batch {batch_num}/{total_batches} with {len(batch_objects)} objects")
                
                # Insert batch and capture response
                response = collection.data.insert_many(batch_objects)
                
                # Check for errors in the response
                if hasattr(response, 'errors') and response.errors:
                    for error in response.errors:
                        logger.error(f"Batch insertion error: {error}")
                    raise Exception(f"Batch insertion had {len(response.errors)} errors")
                
                # Check if all objects were inserted
                if hasattr(response, 'uuids'):
                    inserted_count = len(response.uuids)
                    expected_count = len(batch_objects)
                    if inserted_count != expected_count:
                        logger.warning(f"Partial batch insertion: {inserted_count}/{expected_count} objects stored")
                    else:
                        logger.info(f"Batch {batch_num} stored successfully: {inserted_count} objects")
                else:
                    logger.info(f"Batch {batch_num} completed (no UUID response available)")
                
                return len(batch_objects)
                
            except Exception as e:
                logger.error(f"Failed to store batch {batch_num}: {str(e)}")
                raise
        
        # Process child chunks in batches
        stored_count = 0
        total_batches = (total_chunks + batch_size - 1) // batch_size
        
        for i in range(0, total_chunks, batch_size):
            batch_chunks = result["child_chunks"][i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # Prepare batch data
            batch_objects = []
            for chunk_data in batch_chunks:
                # Validate embedding dimensions
                embedding = chunk_data["embedding"]
                if not isinstance(embedding, list) or len(embedding) != 768:
                    logger.error(f"Invalid embedding dimensions: {len(embedding) if isinstance(embedding, list) else 'not a list'}")
                    raise ValueError(f"Embedding must be a list of 768 floats, got {type(embedding)} with length {len(embedding) if isinstance(embedding, list) else 'N/A'}")
                
                batch_objects.append(
                    wvc.data.DataObject(
                        properties={
                            "content": chunk_data["content"],
                            "parent_id": chunk_data["parent_id"],
                            "document_id": chunk_data["document_id"],
                            "filename": chunk_data["filename"],
                            "parent_index": chunk_data["parent_index"],
                            "child_index": chunk_data["child_index"]
                        },
                        vector=embedding,
                        uuid=chunk_data["id"]
                    )
                )
                
                # Debug: Log embedding info
                logger.info(f"Adding object with embedding: dim={len(embedding)}, first_vals={embedding[:3]}, uuid={chunk_data['id']}")
            
            # Store batch with error handling
            try:
                batch_stored = await asyncio.get_event_loop().run_in_executor(
                    self.executor, 
                    _store_batch_sync, 
                    batch_objects, 
                    batch_num, 
                    total_batches
                )
                stored_count += batch_stored
                
                # Small delay between batches to avoid overwhelming the connection
                if batch_num < total_batches:
                    await asyncio.sleep(batch_delay)
                    
            except Exception as e:
                logger.error(f"Batch {batch_num} failed completely: {str(e)}")
                # Continue with next batch instead of failing completely
                continue
        
        logger.info(f"Batch storage completed: {stored_count}/{total_chunks} child chunks stored")
        
        # Verify storage
        stats = await self.get_database_stats()
        logger.info(f"Database now contains {stats['total_chunks']} total child chunks")
        
        if stored_count < total_chunks:
            logger.warning(f"Incomplete storage: {stored_count}/{total_chunks} chunks stored")
        
        return {
            "total_chunks": total_chunks,
            "stored_chunks": stored_count,
            "success_rate": stored_count / total_chunks if total_chunks > 0 else 0
        }
    
    async def _store_parent_chunks(self, result: Dict[str, Any], logger):
        """Store parent chunks in Weaviate for persistent access."""
        # Get parent chunks data from the result
        parent_chunks_data = result.get("parent_chunks", [])
        
        if not parent_chunks_data:
            logger.warning("No parent chunks data found in result")
            return
        
        def _store_parents_sync():
            parent_collection = self.client.collections.get(self.parent_collection_name)
            
            parent_objects = []
            
            for parent_data in parent_chunks_data:
                parent_objects.append(
                    wvc.data.DataObject(
                        properties={
                            "content": parent_data["content"],
                            "document_id": parent_data["document_id"],
                            "filename": parent_data["filename"],
                            "parent_index": parent_data["parent_index"]
                        },
                        uuid=parent_data["id"]  # Use parent_id as UUID
                    )
                )
            
            if parent_objects:
                logger.info(f"Storing {len(parent_objects)} parent chunks")
                response = parent_collection.data.insert_many(parent_objects)
                
                # Check for errors
                if hasattr(response, 'errors') and response.errors:
                    for error in response.errors:
                        logger.error(f"Parent chunk insertion error: {error}")
                else:
                    logger.info(f"Successfully stored {len(parent_objects)} parent chunks")
            
            return len(parent_objects)
        
        try:
            stored_parents = await asyncio.get_event_loop().run_in_executor(
                self.executor, 
                _store_parents_sync
            )
            logger.info(f"Parent chunk storage completed: {stored_parents} parent chunks stored")
        except Exception as e:
            logger.error(f"Failed to store parent chunks: {str(e)}")
            # Don't fail the entire operation if parent storage fails
    
    @time_async_function("vector_store.hybrid_search")
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
    
    @time_async_function("vector_store.get_parent_chunk")
    async def get_parent_chunk_by_id(self, parent_id: str) -> Optional[str]:
        """Retrieve parent chunk content by parent_id from Weaviate."""
        await self._ensure_connected()
        
        def _get_parent_sync():
            try:
                parent_collection = self.client.collections.get(self.parent_collection_name)
                
                # Get parent chunk by UUID (which is the parent_id)
                response = parent_collection.query.fetch_object_by_id(parent_id)
                
                if response and hasattr(response, 'properties'):
                    return response.properties.get("content")
                
                return None
                
            except Exception as e:
                # Parent not found or other error
                return None
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_parent_sync)
    
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
            # Get child chunks stats
            child_collection = self.client.collections.get(self.collection_name)
            child_response = child_collection.aggregate.over_all(total_count=True)
            total_child_chunks = child_response.total_count
            
            # Get parent chunks stats
            try:
                parent_collection = self.client.collections.get(self.parent_collection_name)
                parent_response = parent_collection.aggregate.over_all(total_count=True)
                total_parent_chunks = parent_response.total_count
            except Exception:
                total_parent_chunks = 0
            
            # Get unique document count (approximate)
            doc_response = child_collection.query.fetch_objects(
                return_properties=["document_id"],
                limit=1000
            )
            
            unique_docs = len(set(
                obj.properties["document_id"] 
                for obj in doc_response.objects
            ))
            
            return {
                "total_chunks": total_child_chunks,  # Keep for backward compatibility
                "child_chunks": total_child_chunks,
                "parent_chunks": total_parent_chunks,
                "unique_documents": unique_docs,
                "collection_name": self.collection_name,
                "parent_collection_name": self.parent_collection_name,
                "weaviate_url": self.weaviate_url
            }
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _get_stats_sync)

    @time_async_function("vector_store.get_all_parent_chunks")
    async def get_all_parent_chunks(self) -> List[str]:
        """Retrieve the content of all parent chunks from Weaviate."""
        await self._ensure_connected()
        
        from app.utils.logger import get_logger
        logger = get_logger('arc_fusion.vector_store')

        def _fetch_all_sync():
            try:
                parent_collection = self.client.collections.get(self.parent_collection_name)
                
                # Fetch all objects. For very large databases, you might need pagination.
                # For this project's scale, a high limit is sufficient.
                response = parent_collection.query.fetch_objects(limit=10000)
                
                if response and hasattr(response, 'objects'):
                    logger.info(f"Fetched {len(response.objects)} parent chunks from the database.")
                    return [obj.properties.get("content") for obj in response.objects if obj.properties]
                
                logger.warning("No parent chunks found in the database.")
                return []
                
            except Exception as e:
                logger.error(f"Failed to fetch all parent chunks: {str(e)}")
                return []
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _fetch_all_sync)

    async def clear_all_documents(self):
        """Clear all documents from the database with robust error handling."""
        await self._ensure_connected()
        
        from app.utils.logger import get_logger
        logger = get_logger('arc_fusion.vector_store')
        
        def _clear_sync():
            total_deleted = 0
            
            # Clear both child chunks and parent chunks collections
            collections_to_clear = [
                (self.collection_name, "child chunks"),
                (self.parent_collection_name, "parent chunks")
            ]
            
            for collection_name, description in collections_to_clear:
                try:
                    collection = self.client.collections.get(collection_name)
                    
                    # First try to get all objects to delete them properly
                    logger.info(f"Fetching all {description} for deletion...")
                    response = collection.query.fetch_objects(limit=10000)  # Get all objects
                    
                    if not response.objects:
                        logger.info(f"No {description} found to delete")
                        continue
                    
                    # Delete objects by UUID
                    uuids_to_delete = [obj.uuid for obj in response.objects]
                    logger.info(f"Deleting {len(uuids_to_delete)} {description}...")
                    
                    # Delete in batches to avoid issues
                    batch_size = 100
                    deleted_count = 0
                    
                    for i in range(0, len(uuids_to_delete), batch_size):
                        batch_uuids = uuids_to_delete[i:i + batch_size]
                        try:
                            result = collection.data.delete_many(
                                where=wvc.query.Filter.by_id().contains_any(batch_uuids)
                            )
                            if hasattr(result, 'successful'):
                                deleted_count += result.successful
                            else:
                                deleted_count += len(batch_uuids)
                            logger.info(f"Deleted {description} batch {i//batch_size + 1}: {len(batch_uuids)} objects")
                        except Exception as batch_error:
                            logger.warning(f"{description.capitalize()} batch deletion failed: {str(batch_error)}")
                            # Try individual deletion for this batch
                            for uuid in batch_uuids:
                                try:
                                    collection.data.delete_by_id(uuid)
                                    deleted_count += 1
                                except Exception:
                                    pass  # Skip individual failures
                    
                    logger.info(f"Successfully deleted {deleted_count} {description}")
                    total_deleted += deleted_count
                    
                except Exception as e:
                    logger.error(f"Error during {description} deletion: {str(e)}")
                    
                    # Fallback: try to recreate the collection
                    try:
                        logger.info(f"Attempting to recreate {description} collection as fallback...")
                        self.client.collections.delete(collection_name)
                        logger.info(f"{description.capitalize()} collection deleted")
                        
                        # Recreate collection based on type
                        if collection_name == self.collection_name:
                            # Child chunks collection
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
                                    ],
                                    # Configure for manual vector storage
                                    vector_config=wvc.config.Configure.Vectors.self_provided()
                                )
                            except Exception:
                                pass  # Collection might already exist
                        else:
                            # Parent chunks collection
                            try:
                                self.client.collections.create(
                                    name=self.parent_collection_name,
                                    properties=[
                                        wvc.config.Property(
                                            name="content",
                                            data_type=wvc.config.DataType.TEXT,
                                            description="Parent chunk content"
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
                                        )
                                    ]
                                )
                            except Exception:
                                pass  # Collection might already exist
                                
                        logger.info(f"{description.capitalize()} collection recreated")
                        
                    except Exception as recreate_error:
                        logger.error(f"{description.capitalize()} collection recreation failed: {str(recreate_error)}")
            
            return {"deleted": total_deleted}
        
        return await asyncio.get_event_loop().run_in_executor(self.executor, _clear_sync)
    
    def __del__(self):
        """Close Weaviate connection."""
        if self.client:
            self.client.close()
        self.executor.shutdown(wait=True)
