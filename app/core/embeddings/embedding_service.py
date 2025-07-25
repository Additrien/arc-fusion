"""
Embedding service for Arc-Fusion.

This module handles the generation of embeddings from text content,
separating this responsibility from the main document processor.
"""

import os
import asyncio
import time
from typing import List, Optional
from google import genai
from google.genai import types
from app.core.config.services import EmbeddingConfig
from app.utils.logger import get_logger
from app.utils.cache import embedding_cache

logger = get_logger('arc_fusion.embeddings.embedding_service')


class EmbeddingService:
    """Service for generating embeddings from text content."""
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with rate limiting and retry logic.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings corresponding to the input texts
            
        Raises:
            Exception: If embedding generation fails after retries
        """
        embeddings = []
        
        for text in texts:
            # Check cache first
            cache_key = f"embedding:{text}"
            cached_embedding = embedding_cache.get(cache_key)
            if cached_embedding:
                embeddings.append(cached_embedding)
                continue
            
            # Generate embedding with retry logic
            embedding = await self._generate_embedding_with_retry(text)
            embeddings.append(embedding)
            
            # Cache the result
            embedding_cache.set(cache_key, embedding)
        
        return embeddings
    
    async def _generate_embedding_with_retry(self, text: str) -> List[float]:
        """
        Generate embedding for a single text with retry logic.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding for the input text
            
        Raises:
            Exception: If embedding generation fails after retries
        """
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Apply rate limiting if enabled
                if self.config.enable_rate_limiting:
                    await asyncio.sleep(self.config.request_delay)
                
                # Generate embedding with semaphore to limit concurrent requests
                async with self._semaphore:
                    response = await self.client.aio.models.embed_content(
                        model=self.config.model,
                        contents=text,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=768
                        )
                    )
                
                embedding = response.embeddings[0].values
                logger.debug(f"Successfully generated embedding for text (attempt {attempt + 1})")
                return embedding
                
            except Exception as e:
                last_exception = e
                logger.warning(f"Embedding generation failed (attempt {attempt + 1}/{self.config.max_retries}): {str(e)}")
                
                # Exponential backoff
                if attempt < self.config.max_retries - 1:
                    delay = self.config.base_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        
        # If we get here, all retries failed
        logger.error(f"Failed to generate embedding after {self.config.max_retries} attempts")
        raise Exception(f"Embedding generation failed after {self.config.max_retries} attempts: {str(last_exception)}")
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a query (different task type).
        
        Args:
            query: Query text to generate embedding for
            
        Returns:
            Embedding for the query text
        """
        # Check cache first
        cache_key = f"query_embedding:{query}"
        cached_embedding = embedding_cache.get(cache_key)
        if cached_embedding:
            return cached_embedding
        
        try:
            # Apply rate limiting if enabled
            if self.config.enable_rate_limiting:
                await asyncio.sleep(self.config.request_delay)
            
            # Generate query embedding
            async with self._semaphore:
                response = await self.client.aio.models.embed_content(
                    model=self.config.model,
                    contents=query,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=768
                    )
                )
            
            embedding = response.embeddings[0].values
            
            # Cache the result
            embedding_cache.set(cache_key, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            # Return zero embedding as fallback
            return [0.0] * 768
