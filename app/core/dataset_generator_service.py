"""
Dataset Generator Service - Creates golden Q&A datasets from text documents.
"""
import os
import json
import random
import asyncio
from typing import List, Dict, Any
from google import genai
from google.genai import types

from app.utils.logger import get_logger
from app import config

logger = get_logger('arc_fusion.dataset_generator')

# Global semaphore to limit concurrent dataset generation tasks
_generation_semaphore = asyncio.Semaphore(1)  # Only 1 dataset generation at a time

class DatasetGeneratorService:
    """Service for generating golden Q&A datasets from text content."""
    
    def __init__(self):
        self.generation_model = config.DATASET_GENERATION_MODEL
        self.fallback_model = config.PRIMARY_MODEL
        self.client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
        self.rate_limit_delay = 0.5  # 500ms between API calls

    def _create_generation_prompt(self, chunk: str) -> str:
        """Create a prompt for generating Q&A pairs from a text chunk."""
        return f"""
You are an expert at creating high-quality question-and-answer pairs for evaluating a Retrieval-Augmented Generation (RAG) system.

Based on the following text chunk, generate 2-3 high-quality question-and-answer pairs that would be suitable for evaluating a RAG system's ability to retrieve and answer questions about this content.

Guidelines:
- Questions should be specific and answerable based on the provided text
- Answers should be comprehensive but concise
- Focus on important facts, concepts, and relationships in the text
- Avoid questions that are too obvious or too obscure
- Make questions that would realistically be asked by users

Text chunk:
{chunk}

Return your response as a JSON array with each Q&A pair as an object containing "question" and "ground_truth" fields.

Example format:
[
  {{
    "question": "What is the main topic discussed in this text?",
    "ground_truth": "The detailed answer based on the text content."
  }},
  {{
    "question": "How does X relate to Y according to the text?",
    "ground_truth": "The specific relationship described in the text."
  }}
]
"""

    async def _generate_with_retry(self, prompt: str, max_retries: int = 3) -> str:
        """Generate content with retry logic for 503 errors."""
        for attempt in range(max_retries):
            try:
                # Use primary model first, fallback on retries
                model = self.generation_model if attempt == 0 else self.fallback_model
                
                response = await self.client.aio.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                return response.text
                
            except Exception as e:
                error_msg = str(e)
                if "503" in error_msg or "overloaded" in error_msg.lower():
                    wait_time = (2 ** attempt) + random.uniform(0, 1)  # Exponential backoff with jitter
                    logger.warning(f"Gemini API overloaded (attempt {attempt + 1}/{max_retries}), waiting {wait_time:.1f}s...")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        continue
                
                # Re-raise if not a 503 error or max retries reached
                raise e
        
        raise Exception("Max retries exceeded for Gemini API calls")

    async def generate_from_chunks(self, chunks: List[str], num_pairs_total: int = 50) -> List[Dict[str, str]]:
        """
        Generates a list of Q&A pairs from a list of text chunks.
        
        Args:
            chunks: A list of text chunks (e.g., from parent documents in Weaviate).
            num_pairs_total: The target total number of Q&A pairs to generate.
            
        Returns:
            A list of dictionaries, where each dictionary is a Q&A pair.
        """
        # Use semaphore to prevent multiple concurrent dataset generations
        async with _generation_semaphore:
            if not chunks:
                logger.warning("Chunk list is empty, skipping Q&A generation.")
                return []

            # Shuffle the chunks to get a more diverse set of questions
            random.shuffle(chunks)
            
            generated_pairs = []
            
            logger.info(f"Starting Q&A generation from {len(chunks)} chunks, targeting {num_pairs_total} total pairs.")
            
            for i, chunk in enumerate(chunks):
                if len(generated_pairs) >= num_pairs_total:
                    break
                
                # Add delay between API calls to respect rate limits
                if i > 0:  # Skip delay for first call
                    logger.debug(f"Rate limiting: waiting {self.rate_limit_delay}s before next API call")
                    await asyncio.sleep(self.rate_limit_delay)
                
                prompt = self._create_generation_prompt(chunk)
                
                try:
                    response_text = await self._generate_with_retry(prompt)
                    
                    response_json = json.loads(response_text)
                    
                    # Handle both list and object responses
                    if isinstance(response_json, list):
                        qa_pairs = response_json
                    elif isinstance(response_json, dict):
                        qa_pairs = response_json.get("qa_pairs", [])
                    else:
                        logger.warning(f"Unexpected response format from LLM: {type(response_json)}")
                        continue
                    
                    if qa_pairs:
                        for pair in qa_pairs:
                            if len(generated_pairs) >= num_pairs_total:
                                break
                            
                            if isinstance(pair, dict) and "question" in pair and "ground_truth" in pair:
                                generated_pairs.append(pair)
                                logger.debug(f"Generated Q&A pair: {pair['question']}")
                            else:
                                logger.warning(f"Skipping malformed Q&A pair from LLM: {pair}")
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON from LLM response: {e}\nResponse text: {response_text}")
                except Exception as e:
                    logger.error(f"An error occurred during Q&A generation from chunk: {e}")
                    continue
                    
            logger.info(f"Successfully generated {len(generated_pairs)} Q&A pairs from the provided chunks.")
            return generated_pairs