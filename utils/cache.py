"""
Caching utility for GenieBot
Caches embeddings and query results to avoid recomputation
"""

import hashlib
import time
from collections import OrderedDict
from typing import Optional, List, Dict, Any
import numpy as np
from .logger import setup_logger

logger = setup_logger(__name__)


class RateLimiter:
    """Simple per-user fixed-window rate limiter (in-memory, single process).

    GenieBot is a public bot with a shared OpenRouter key - without this,
    one user spamming /ask or image uploads can exhaust the free-tier quota
    for everyone. Not distributed-safe, but the bot only ever runs as one
    process on Render's free tier, so that's not a concern here.
    """

    def __init__(self, limit: int = 10, window_seconds: int = 60, max_users: int = 5000):
        """
        Args:
            limit: Max requests allowed per user per window.
            window_seconds: Length of the fixed window, in seconds.
            max_users: Maximum number of distinct users to keep counters for.
                Least-recently-active user is evicted once this cap is hit,
                so the dict can't grow unbounded on a long-running process
                (mirrors UserMemory's eviction pattern in utils/memory.py).
        """
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_users = max_users
        # user_id -> (window_expires_at, count_in_window). OrderedDict so we
        # can evict the least-recently-touched user in O(1) once at capacity.
        self._counters: "OrderedDict[int, tuple]" = OrderedDict()

    def allow(self, user_id: int) -> bool:
        """Return True if this request is allowed, False if the user is over their limit."""
        now = time.time()

        if user_id in self._counters:
            expires_at, count = self._counters[user_id]
            self._counters.move_to_end(user_id)
        else:
            expires_at, count = now + self.window_seconds, 0
            self._counters[user_id] = (expires_at, count)
            self._counters.move_to_end(user_id)
            if len(self._counters) > self.max_users:
                evicted_id, _ = self._counters.popitem(last=False)
                logger.debug(f"Evicted rate limit counter for least-recently-active user {evicted_id}")

        if expires_at < now:
            expires_at = now + self.window_seconds
            count = 0

        count += 1
        self._counters[user_id] = (expires_at, count)

        allowed = count <= self.limit
        if not allowed:
            logger.warning(f"Rate limit exceeded for user {user_id} ({count}/{self.limit} in window)")
        return allowed

    def seconds_until_reset(self, user_id: int) -> int:
        """Seconds remaining until this user's window resets."""
        expires_at, _ = self._counters.get(user_id, (0, 0))
        return max(0, int(expires_at - time.time()))

    def remaining(self, user_id: int) -> int:
        """Requests this user has left in the current window.

        Read-only - unlike allow(), this does not consume a request or
        touch LRU ordering, so it's safe to call from a status/stats
        command without affecting the user's actual quota.
        """
        expires_at, count = self._counters.get(user_id, (0, 0))
        if expires_at < time.time():
            return self.limit
        return max(0, self.limit - count)


class EmbeddingCache:
    """Cache for text embeddings"""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize embedding cache
        
        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self.cache: Dict[str, np.ndarray] = {}
        self.access_count: Dict[str, int] = {}  # Track access frequency
    
    @staticmethod
    def _hash_text(text: str) -> str:
        """
        Hash text for cache key
        
        Args:
            text: Text to hash
        
        Returns:
            Hash string
        """
        return hashlib.md5(text.lower().encode()).hexdigest()
    
    def get(self, text: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache
        
        Args:
            text: Text to retrieve embedding for
        
        Returns:
            Embedding array or None if not cached
        """
        key = self._hash_text(text)
        
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            logger.debug(f"Cache hit for: {text[:50]}... (count: {self.access_count[key]})")
            return self.cache[key]
        
        return None
    
    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache
        
        Args:
            text: Original text
            embedding: Embedding vector
        """
        if len(self.cache) >= self.max_size:
            # Remove least accessed item
            least_accessed = min(self.access_count, key=self.access_count.get)
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
            logger.debug(f"Cache full. Removed least accessed item.")
        
        key = self._hash_text(text)
        self.cache[key] = embedding
        self.access_count[key] = 0
        logger.debug(f"Cached embedding for: {text[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.cache.clear()
        self.access_count.clear()
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_cached": len(self.cache),
            "max_size": self.max_size,
            "hit_rate_estimate": len([v for v in self.access_count.values() if v > 0]) / max(1, len(self.cache))
        }


class QueryCache:
    """Cache for query responses"""
    
    def __init__(self, max_size: int = 500):
        """
        Initialize query cache
        
        Args:
            max_size: Maximum number of cached queries
        """
        self.max_size = max_size
        self.cache: Dict[str, str] = {}
    
    @staticmethod
    def _hash_query(query: str, context: str = "") -> str:
        """
        Hash query (plus any conversation context) for cache key.

        Keying on (query, context) rather than query alone means a question
        asked with conversation context is cached separately from the same
        question asked without it - so the cache stays correct - while still
        allowing a hit when the same question is asked again with the same
        context, instead of unconditionally bypassing the cache whenever any
        context is present.

        Args:
            query: Query string
            context: Optional conversation context string that was used
                alongside the query (e.g. UserMemory.get_context() output)

        Returns:
            Hash string
        """
        return hashlib.md5(f"{query.lower()}||{context}".encode()).hexdigest()

    def get(self, query: str, context: str = "") -> Optional[str]:
        """
        Retrieve cached response

        Args:
            query: Query string
            context: Optional conversation context used alongside the query

        Returns:
            Cached response or None
        """
        key = self._hash_query(query, context)
        result = self.cache.get(key)

        if result:
            logger.debug(f"Query cache hit for: {query[:50]}...")

        return result

    def put(self, query: str, response: str, context: str = "") -> None:
        """
        Store response in cache

        Args:
            query: Query string
            response: Response text
            context: Optional conversation context used alongside the query
        """
        if len(self.cache) >= self.max_size:
            # Remove first item (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]

        key = self._hash_query(query, context)
        self.cache[key] = response
        logger.debug(f"Cached response for: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached queries"""
        self.cache.clear()
        logger.info("Query cache cleared")
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)
