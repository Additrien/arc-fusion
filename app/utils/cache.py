"""
Simple in-memory cache for performance optimization.

Provides caching for query embeddings and other frequently accessed data
to reduce API calls and improve response times.
"""

import hashlib
import time
from typing import Any, Dict, Optional
from app.utils.logger import get_logger

logger = get_logger('arc_fusion.cache')

class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hour default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        # Create a string representation of args and kwargs
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        key_string = "|".join(key_parts)
        
        # Generate hash for consistent key length
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        if time.time() > entry['expires_at']:
            # Remove expired entry
            del self._cache[key]
            return None
        
        logger.debug(f"Cache HIT for key: {key[:16]}...")
        return entry['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        self._cache[key] = {
            'value': value,
            'expires_at': time.time() + ttl
        }
        logger.debug(f"Cache SET for key: {key[:16]}... (TTL: {ttl}s)")
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


# Global cache instance
embedding_cache = SimpleCache(default_ttl=3600)  # 1 hour TTL for embeddings
hyde_cache = SimpleCache(default_ttl=1800)       # 30 minutes TTL for HyDE expansions 