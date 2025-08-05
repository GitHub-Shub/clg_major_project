# backend/cache_manager.py
"""
Phase 2: Advanced Cache Management System
========================================

This module provides a comprehensive caching infrastructure with multiple cache levels,
TTL management, LRU eviction, and persistent storage capabilities.

Cache Levels Implemented:
- Level 1: Exact Query Cache (hash-based exact matches)
- Level 5: Chunk Retrieval Cache (FAISS search results) 
- Level 6: Query Embedding Cache (computed embeddings)

Features:
- TTL (Time To Live) management
- LRU (Least Recently Used) eviction
- Memory limits per cache level
- Persistent storage with compression
- Usage statistics and analytics
- Automatic cleanup and maintenance
"""

import json
import time
import hashlib
import pickle
import gzip
import os
import threading
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from pathlib import Path
import logging


class CacheEntry:
    """
    Represents a single cache entry with metadata.
    """
    
    def __init__(self, key: str, value: Any, ttl_seconds: int = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 1
        self.ttl_seconds = ttl_seconds
        self.expires_at = self.created_at + ttl_seconds if ttl_seconds else None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'ttl_seconds': self.ttl_seconds,
            'expires_at': self.expires_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create entry from dictionary."""
        entry = cls.__new__(cls)
        entry.key = data['key']
        entry.value = data['value']
        entry.created_at = data['created_at']
        entry.last_accessed = data['last_accessed']
        entry.access_count = data['access_count']
        entry.ttl_seconds = data['ttl_seconds']
        entry.expires_at = data['expires_at']
        return entry


class LRUCache:
    """
    LRU Cache with TTL support and memory management.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = None, name: str = "cache"):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.name = name
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired_cleanups': 0,
            'total_sets': 0,
            'memory_evictions': 0
        }
    
    def _normalize_key(self, key: str) -> str:
        """Normalize cache key."""
        return str(key).strip().lower()
    
    def _cleanup_expired(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            self.stats['expired_cleanups'] += 1
    
    def _evict_lru(self):
        """Evict least recently used entries to maintain size limit."""
        while len(self._cache) >= self.max_size:
            # OrderedDict maintains insertion order, oldest first
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self.stats['evictions'] += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        normalized_key = self._normalize_key(key)
        
        with self._lock:
            entry = self._cache.get(normalized_key)
            
            if entry is None:
                self.stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self._cache[normalized_key]
                self.stats['misses'] += 1
                self.stats['expired_cleanups'] += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(normalized_key)
            entry.touch()
            self.stats['hits'] += 1
            
            return entry.value
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Set value in cache.
        
        Args:
            key (str): Cache key
            value (Any): Value to cache
            ttl (int): Time to live in seconds (optional)
        """
        normalized_key = self._normalize_key(key)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            # Clean up expired entries occasionally
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()
            
            # Evict LRU entries if needed
            if normalized_key not in self._cache:
                self._evict_lru()
            
            # Create and store entry
            entry = CacheEntry(normalized_key, value, ttl)
            self._cache[normalized_key] = entry
            self._cache.move_to_end(normalized_key)
            
            self.stats['total_sets'] += 1
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.
        
        Args:
            key (str): Cache key
            
        Returns:
            bool: True if key was found and deleted
        """
        normalized_key = self._normalize_key(key)
        
        with self._lock:
            if normalized_key in self._cache:
                del self._cache[normalized_key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from cache."""
        with self._lock:
            self._cache.clear()
            # Reset stats except configuration
            hits, misses = self.stats['hits'], self.stats['misses']
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'expired_cleanups': 0,
                'total_sets': 0,
                'memory_evictions': 0
            }
    
    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                'name': self.name,
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': round(hit_rate, 2),
                'total_requests': total_requests,
                **self.stats
            }
    
    def get_all_entries(self) -> List[Dict[str, Any]]:
        """Get all cache entries for serialization."""
        with self._lock:
            return [entry.to_dict() for entry in self._cache.values()]
    
    def load_entries(self, entries: List[Dict[str, Any]]) -> None:
        """Load cache entries from serialized data."""
        with self._lock:
            self._cache.clear()
            for entry_data in entries:
                entry = CacheEntry.from_dict(entry_data)
                if not entry.is_expired():
                    self._cache[entry.key] = entry


class CacheManager:
    """
    Centralized cache management system for multiple cache levels.
    """
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger('CacheManager')
        
        # Cache configurations
        self.cache_configs = {
            'exact_query': {
                'max_size': 10000,
                'default_ttl': 30 * 24 * 3600,  # 30 days
                'file_name': 'exact_query_cache.gz'
            },
            'embedding': {
                'max_size': 20000,
                'default_ttl': 7 * 24 * 3600,   # 7 days
                'file_name': 'embedding_cache.gz'
            },
            'retrieval': {
                'max_size': 5000,
                'default_ttl': 7 * 24 * 3600,   # 7 days
                'file_name': 'retrieval_cache.gz'
            }
        }
        
        # Initialize caches
        self.caches = {}
        for cache_name, config in self.cache_configs.items():
            self.caches[cache_name] = LRUCache(
                max_size=config['max_size'],
                default_ttl=config['default_ttl'],
                name=cache_name
            )
        
        # Load existing caches
        self._load_all_caches()
        
        # Global statistics
        self.global_stats = {
            'total_requests': 0,
            'total_hits': 0,
            'cache_saves': 0,
            'cache_loads': 0,
            'last_cleanup': time.time()
        }
        
        self.logger.info(f"CacheManager initialized with {len(self.caches)} cache levels")
    
    def _get_cache_file_path(self, cache_name: str) -> Path:
        """Get file path for cache persistence."""
        config = self.cache_configs.get(cache_name)
        if not config:
            raise ValueError(f"Unknown cache: {cache_name}")
        return self.cache_dir / config['file_name']
    
    def _save_cache(self, cache_name: str) -> bool:
        """
        Save a specific cache to disk with compression.
        
        Args:
            cache_name (str): Name of cache to save
            
        Returns:
            bool: True if successful
        """
        try:
            cache = self.caches.get(cache_name)
            if not cache:
                return False
            
            file_path = self._get_cache_file_path(cache_name)
            entries = cache.get_all_entries()
            
            # Serialize and compress
            data = pickle.dumps({
                'entries': entries,
                'stats': cache.get_stats(),
                'saved_at': time.time()
            })
            
            with gzip.open(file_path, 'wb') as f:
                f.write(data)
            
            self.global_stats['cache_saves'] += 1
            self.logger.info(f"Saved {cache_name} cache: {len(entries)} entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {cache_name} cache: {str(e)}")
            return False
    
    def _load_cache(self, cache_name: str) -> bool:
        """
        Load a specific cache from disk.
        
        Args:
            cache_name (str): Name of cache to load
            
        Returns:
            bool: True if successful
        """
        try:
            file_path = self._get_cache_file_path(cache_name)
            
            if not file_path.exists():
                self.logger.info(f"No saved cache found for {cache_name}")
                return False
            
            with gzip.open(file_path, 'rb') as f:
                data = pickle.loads(f.read())
            
            cache = self.caches.get(cache_name)
            if cache:
                cache.load_entries(data['entries'])
                self.global_stats['cache_loads'] += 1
                self.logger.info(f"Loaded {cache_name} cache: {len(data['entries'])} entries")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to load {cache_name} cache: {str(e)}")
            
        return False
    
    def _load_all_caches(self):
        """Load all caches from disk."""
        self.logger.info("Loading saved caches...")
        loaded_count = 0
        
        for cache_name in self.caches.keys():
            if self._load_cache(cache_name):
                loaded_count += 1
        
        self.logger.info(f"Loaded {loaded_count}/{len(self.caches)} caches from disk")
    
    def save_all_caches(self):
        """Save all caches to disk."""
        self.logger.info("Saving all caches...")
        saved_count = 0
        
        for cache_name in self.caches.keys():
            if self._save_cache(cache_name):
                saved_count += 1
        
        self.logger.info(f"Saved {saved_count}/{len(self.caches)} caches to disk")
    
    def get_cache(self, cache_name: str) -> Optional[LRUCache]:
        """Get a specific cache instance."""
        return self.caches.get(cache_name)
    
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """
        Get value from a specific cache.
        
        Args:
            cache_name (str): Name of cache
            key (str): Cache key
            
        Returns:
            Optional[Any]: Cached value or None
        """
        cache = self.caches.get(cache_name)
        if not cache:
            return None
        
        self.global_stats['total_requests'] += 1
        result = cache.get(key)
        
        if result is not None:
            self.global_stats['total_hits'] += 1
        
        return result
    
    def set(self, cache_name: str, key: str, value: Any, ttl: int = None) -> bool:
        """
        Set value in a specific cache.
        
        Args:
            cache_name (str): Name of cache
            key (str): Cache key
            value (Any): Value to cache
            ttl (int): Time to live in seconds
            
        Returns:
            bool: True if successful
        """
        cache = self.caches.get(cache_name)
        if not cache:
            return False
        
        try:
            cache.set(key, value, ttl)
            return True
        except Exception as e:
            self.logger.error(f"Failed to set {cache_name} cache: {str(e)}")
            return False
    
    def delete(self, cache_name: str, key: str) -> bool:
        """Delete key from specific cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            return False
        return cache.delete(key)
    
    def clear_cache(self, cache_name: str) -> bool:
        """Clear a specific cache."""
        cache = self.caches.get(cache_name)
        if not cache:
            return False
        
        cache.clear()
        self.logger.info(f"Cleared {cache_name} cache")
        return True
    
    def clear_all_caches(self):
        """Clear all caches."""
        for cache_name in self.caches.keys():
            self.clear_cache(cache_name)
        
        self.logger.info("Cleared all caches")
    
    def cleanup_expired(self):
        """Clean up expired entries from all caches."""
        self.logger.info("Running cache cleanup...")
        
        for cache_name, cache in self.caches.items():
            before_size = cache.size()
            cache._cleanup_expired()
            after_size = cache.size()
            
            if before_size != after_size:
                self.logger.info(f"Cleaned up {cache_name}: {before_size} -> {after_size} entries")
        
        self.global_stats['last_cleanup'] = time.time()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Estimate memory usage of all caches."""
        try:
            import sys
            
            memory_info = {}
            total_size = 0
            
            for cache_name, cache in self.caches.items():
                cache_size = 0
                entry_count = 0
                
                # Estimate memory usage
                for entry in cache._cache.values():
                    entry_count += 1
                    try:
                        cache_size += sys.getsizeof(entry.value)
                        cache_size += sys.getsizeof(entry.key)
                        cache_size += sys.getsizeof(entry)
                    except:
                        cache_size += 1024  # Rough estimate
                
                memory_info[cache_name] = {
                    'size_bytes': cache_size,
                    'size_mb': round(cache_size / 1024 / 1024, 2),
                    'entry_count': entry_count,
                    'avg_entry_size': round(cache_size / entry_count) if entry_count > 0 else 0
                }
                
                total_size += cache_size
            
            memory_info['total'] = {
                'size_bytes': total_size,
                'size_mb': round(total_size / 1024 / 1024, 2)
            }
            
            return memory_info
            
        except Exception as e:
            self.logger.error(f"Failed to calculate memory usage: {str(e)}")
            return {'error': str(e)}
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all caches."""
        cache_stats = {}
        
        for cache_name, cache in self.caches.items():
            cache_stats[cache_name] = cache.get_stats()
        
        # Calculate global statistics
        total_hits = sum(stats['hits'] for stats in cache_stats.values())
        total_requests = sum(stats['total_requests'] for stats in cache_stats.values())
        global_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'global_stats': {
                **self.global_stats,
                'global_hit_rate': round(global_hit_rate, 2),
                'total_cache_requests': total_requests,
                'total_cache_hits': total_hits
            },
            'cache_stats': cache_stats,
            'memory_usage': self.get_memory_usage(),
            'configuration': self.cache_configs
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache system."""
        health = {
            'overall_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'issues': []
        }
        
        try:
            # Check each cache
            for cache_name, cache in self.caches.items():
                cache_health = {
                    'name': cache_name,
                    'status': 'healthy',
                    'size': cache.size(),
                    'max_size': cache.max_size
                }
                
                # Check if cache is near capacity
                usage_ratio = cache.size() / cache.max_size
                if usage_ratio > 0.9:
                    cache_health['status'] = 'warning'
                    health['issues'].append(f"{cache_name} cache is {usage_ratio*100:.1f}% full")
                
                # Check hit rate
                stats = cache.get_stats()
                if stats['hit_rate'] < 10 and stats['total_requests'] > 100:
                    cache_health['status'] = 'warning'
                    health['issues'].append(f"{cache_name} cache has low hit rate: {stats['hit_rate']:.1f}%")
                
                health[f'cache_{cache_name}'] = cache_health
            
            # Set overall status
            if any(issue for issue in health['issues']):
                health['overall_status'] = 'warning'
            
        except Exception as e:
            health['overall_status'] = 'error'
            health['error'] = str(e)
        
        return health
    
    def __del__(self):
        """Save caches when manager is destroyed."""
        try:
            self.save_all_caches()
        except:
            pass  # Ignore errors during cleanup


# Utility functions for cache key generation
def generate_query_hash(query: str) -> str:
    """Generate normalized hash for query caching."""
    # Normalize query text
    normalized = query.lower().strip()
    normalized = ' '.join(normalized.split())  # Normalize whitespace
    
    # Generate SHA-256 hash
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def generate_embedding_hash(query: str, model_name: str) -> str:
    """Generate hash for embedding caching including model version."""
    combined = f"{query.lower().strip()}|{model_name}"
    return hashlib.sha256(combined.encode('utf-8')).hexdigest()


def generate_retrieval_hash(embedding: np.ndarray, k: int = 5) -> str:
    """Generate hash for retrieval result caching."""
    # Convert embedding to bytes and include k parameter
    embedding_bytes = embedding.tobytes()
    combined = embedding_bytes + str(k).encode('utf-8')
    return hashlib.sha256(combined).hexdigest()


# Global cache manager instance (singleton pattern)
_cache_manager_instance = None


def get_cache_manager(cache_dir: str = "data/cache") -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager_instance
    if _cache_manager_instance is None:
        _cache_manager_instance = CacheManager(cache_dir)
    return _cache_manager_instance


def reset_cache_manager():
    """Reset global cache manager (for testing)."""
    global _cache_manager_instance
    if _cache_manager_instance:
        _cache_manager_instance.save_all_caches()
    _cache_manager_instance = None