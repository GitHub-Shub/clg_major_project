# backend/semantic_cache_manager.py
"""
Phase 3: Fixed Semantic Cache Manager with Proper Inheritance
===========================================================

This module implements advanced semantic caching capabilities with proper
inheritance from the base CacheManager class.

Fixed Issues:
- Proper initialization of parent class attributes
- Correct cache manager inheritance
- Performance optimizations
- Background task management
"""

import numpy as np
import json
import time
import logging
import threading
import pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import re

# Import base cache manager
from cache_manager import CacheManager, CacheEntry, LRUCache


class SemanticCluster:
    """
    Represents a semantic cluster of similar queries.
    """
    
    def __init__(self, cluster_id: str, representative_query: str, 
                 representative_embedding: np.ndarray, response: str):
        self.cluster_id = cluster_id
        self.representative_query = representative_query
        self.representative_embedding = representative_embedding
        self.response = response
        self.queries = [representative_query]
        self.query_count = 1
        self.hit_count = 0
        self.created_at = time.time()
        self.last_updated = time.time()
        self.confidence_score = 1.0
        self.similarity_threshold = 0.85
    
    def add_query(self, query: str, embedding: np.ndarray, response: str = None):
        """Add a new query to this cluster."""
        self.queries.append(query)
        self.query_count += 1
        self.last_updated = time.time()
        
        # Update response if provided and more recent
        if response:
            self.response = response
    
    def calculate_similarity(self, query_embedding: np.ndarray) -> float:
        """Calculate similarity between query and cluster representative."""
        similarity = cosine_similarity(
            [query_embedding], 
            [self.representative_embedding]
        )[0][0]
        return float(similarity)
    
    def is_similar(self, query_embedding: np.ndarray, threshold: float = None) -> bool:
        """Check if query is similar enough to belong to this cluster."""
        threshold = threshold or self.similarity_threshold
        return self.calculate_similarity(query_embedding) >= threshold
    
    def record_hit(self):
        """Record a cache hit for this cluster."""
        self.hit_count += 1
        self.last_updated = time.time()
        # Increase confidence based on usage
        self.confidence_score = min(1.0, self.confidence_score + 0.01)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cluster to dictionary for serialization."""
        return {
            'cluster_id': self.cluster_id,
            'representative_query': self.representative_query,
            'representative_embedding': self.representative_embedding.tolist(),
            'response': self.response,
            'queries': self.queries,
            'query_count': self.query_count,
            'hit_count': self.hit_count,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'confidence_score': self.confidence_score,
            'similarity_threshold': self.similarity_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticCluster':
        """Create cluster from dictionary."""
        cluster = cls(
            data['cluster_id'],
            data['representative_query'],
            np.array(data['representative_embedding']),
            data['response']
        )
        cluster.queries = data.get('queries', [])
        cluster.query_count = data.get('query_count', 1)
        cluster.hit_count = data.get('hit_count', 0)
        cluster.created_at = data.get('created_at', time.time())
        cluster.last_updated = data.get('last_updated', time.time())
        cluster.confidence_score = data.get('confidence_score', 1.0)
        cluster.similarity_threshold = data.get('similarity_threshold', 0.85)
        return cluster


class LegalTopicExtractor:
    """
    Extracts and manages legal topics for pre-computed responses.
    """
    
    def __init__(self):
        self.legal_topics = {
            # License-related topics
            'driving_license_penalty': {
                'keywords': ['driving', 'license', 'penalty', 'fine', 'without license'],
                'sections': ['Section 181'],
                'priority': 1.0
            },
            'learner_license': {
                'keywords': ['learner', 'license', 'learner\'s', 'age', 'minor'],
                'sections': ['Section 4', 'Section 14'],
                'priority': 0.9
            },
            
            # Safety-related topics  
            'helmet_violation': {
                'keywords': ['helmet', 'two-wheeler', 'bike', 'motorcycle', 'safety'],
                'sections': ['Section 194D'],
                'priority': 1.0
            },
            'seat_belt_violation': {
                'keywords': ['seat belt', 'seatbelt', 'safety belt', 'car', 'vehicle'],
                'sections': ['State Rules'],
                'priority': 0.8
            },
            
            # Traffic violations
            'overspeeding': {
                'keywords': ['speed', 'overspeeding', 'speeding', 'limit', 'fast'],
                'sections': ['Section 183'],
                'priority': 0.9
            },
            'red_light_jumping': {
                'keywords': ['red light', 'traffic signal', 'jump', 'signal', 'stop'],
                'sections': ['Section 177'],
                'priority': 0.8
            },
            'mobile_phone_driving': {
                'keywords': ['mobile', 'phone', 'cell phone', 'driving', 'distracted'],
                'sections': ['Section 177'],
                'priority': 0.7
            },
            
            # Drunk driving
            'drunk_driving': {
                'keywords': ['drunk', 'alcohol', 'drinking', 'intoxicated', 'blood alcohol'],
                'sections': ['Section 185'],
                'priority': 1.0
            },
            
            # Documentation
            'vehicle_registration': {
                'keywords': ['registration', 'RC', 'registration certificate', 'unregistered'],
                'sections': ['Section 192'],
                'priority': 0.9
            },
            'insurance_mandatory': {
                'keywords': ['insurance', 'third party', 'policy', 'coverage'],
                'sections': ['Section 146', 'Section 196'],
                'priority': 0.9
            },
            'pollution_certificate': {
                'keywords': ['pollution', 'PUC', 'emission', 'certificate', 'environment'],
                'sections': ['Section 190'],
                'priority': 0.7
            },
            'driving_documents': {
                'keywords': ['documents', 'papers', 'required', 'carry', 'produce'],
                'sections': ['Multiple Sections'],
                'priority': 0.8
            },
            
            # Special provisions
            'golden_hour': {
                'keywords': ['golden hour', 'emergency', 'medical', 'accident', 'trauma'],
                'sections': ['Section 2(12A)'],
                'priority': 0.6
            },
            'hit_and_run': {
                'keywords': ['hit and run', 'accident', 'flee', 'escape', 'leave scene'],
                'sections': ['Section 161', 'BNS Section 106'],
                'priority': 0.8
            },
            
            # Vehicle-related
            'vehicle_fitness': {
                'keywords': ['fitness', 'certificate', 'commercial', 'roadworthy'],
                'sections': ['Section 56'],
                'priority': 0.6
            },
            'overloading': {
                'keywords': ['overload', 'weight', 'capacity', 'excess', 'load'],
                'sections': ['Section 194'],
                'priority': 0.7
            },
            
            # Authority and enforcement
            'traffic_police_powers': {
                'keywords': ['police', 'authority', 'powers', 'check', 'detain'],
                'sections': ['Section 206'],
                'priority': 0.6
            },
            'e_challan': {
                'keywords': ['e-challan', 'electronic', 'challan', 'online', 'digital'],
                'sections': ['Technology Integration'],
                'priority': 0.5
            }
        }
    
    def extract_topics_from_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Extract relevant legal topics from a query.
        
        Returns:
            List of (topic_name, relevance_score) tuples
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        topic_scores = []
        
        for topic_name, topic_data in self.legal_topics.items():
            keywords = topic_data['keywords']
            priority = topic_data['priority']
            
            # Calculate keyword match score
            keyword_matches = 0
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                # Calculate relevance score
                keyword_score = keyword_matches / len(keywords)
                relevance_score = keyword_score * priority
                topic_scores.append((topic_name, relevance_score))
        
        # Sort by relevance score (highest first)
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        return topic_scores[:3]  # Return top 3 most relevant topics
    
    def get_topic_info(self, topic_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific legal topic."""
        return self.legal_topics.get(topic_name)


class SemanticCacheManager:
    """
    Enhanced cache manager with semantic clustering and pre-computed responses.
    
    Fixed to work as a standalone class rather than inheriting from CacheManager
    to avoid inheritance issues.
    """
    
    def __init__(self, cache_dir: str = "data/cache", embedding_model=None):
        # Initialize as standalone class instead of inheriting
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger('SemanticCacheManager')
        
        # Initialize base cache manager for standard caching
        self.base_cache_manager = CacheManager(cache_dir)
        
        self.embedding_model = embedding_model
        self.topic_extractor = LegalTopicExtractor()
        
        # Semantic clustering
        self.semantic_clusters = {}  # cluster_id -> SemanticCluster
        self.cluster_embeddings = None  # For fast similarity search
        self.similarity_threshold = 0.85
        self.max_clusters = 100
        
        # Pre-computed responses
        self.precomputed_responses = {}  # topic_name -> response
        self.topic_response_cache = {}  # topic_name -> cached_response_data
        
        # Usage analytics
        self.query_patterns = defaultdict(int)
        self.topic_popularity = defaultdict(int)
        self.similarity_stats = {
            'total_similarity_checks': 0,
            'successful_matches': 0,
            'avg_similarity_score': 0.0
        }
        
        # Background processing
        self.background_tasks_enabled = True
        self.last_cluster_update = time.time()
        self.last_precompute_update = time.time()
        
        # Load semantic data
        self._load_semantic_clusters()
        self._load_precomputed_responses()
        
        self.logger.info("Semantic Cache Manager initialized (standalone)")
    
    # Delegate base cache operations to the base cache manager
    def get(self, cache_name: str, key: str) -> Optional[Any]:
        """Get value from base cache."""
        return self.base_cache_manager.get(cache_name, key)
    
    def set(self, cache_name: str, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in base cache."""
        return self.base_cache_manager.set(cache_name, key, value, ttl)
    
    def clear_cache(self, cache_name: str) -> bool:
        """Clear a specific base cache."""
        return self.base_cache_manager.clear_cache(cache_name)
    
    def cleanup_expired(self):
        """Clean up expired entries in base caches."""
        self.base_cache_manager.cleanup_expired()
    
    def save_all_caches(self):
        """Save all base caches."""
        self.base_cache_manager.save_all_caches()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics including base caches."""
        base_stats = self.base_cache_manager.get_comprehensive_stats()
        semantic_stats = self.get_semantic_stats()
        
        return {
            **base_stats,
            'semantic_stats': semantic_stats
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on both base and semantic caches."""
        base_health = self.base_cache_manager.health_check()
        
        semantic_health = {
            'semantic_clusters': len(self.semantic_clusters),
            'precomputed_responses': len(self.precomputed_responses),
            'similarity_threshold': self.similarity_threshold,
            'status': 'healthy'
        }
        
        return {
            'base_cache_health': base_health,
            'semantic_cache_health': semantic_health,
            'overall_status': 'healthy' if base_health.get('overall_status') == 'healthy' else 'degraded'
        }
    
    def _load_semantic_clusters(self):
        """Load semantic clusters from disk."""
        cluster_file = self.cache_dir / 'semantic_clusters.gz'
        
        if cluster_file.exists():
            try:
                with gzip.open(cluster_file, 'rb') as f:
                    data = pickle.load(f)
                
                for cluster_data in data.get('clusters', []):
                    cluster = SemanticCluster.from_dict(cluster_data)
                    self.semantic_clusters[cluster.cluster_id] = cluster
                
                self._rebuild_cluster_embeddings()
                
                self.logger.info(f"Loaded {len(self.semantic_clusters)} semantic clusters")
                
            except Exception as e:
                self.logger.error(f"Failed to load semantic clusters: {str(e)}")
    
    def _save_semantic_clusters(self):
        """Save semantic clusters to disk."""
        cluster_file = self.cache_dir / 'semantic_clusters.gz'
        
        try:
            cluster_data = {
                'clusters': [cluster.to_dict() for cluster in self.semantic_clusters.values()],
                'saved_at': time.time(),
                'stats': self.similarity_stats
            }
            
            with gzip.open(cluster_file, 'wb') as f:
                pickle.dump(cluster_data, f)
            
            self.logger.info(f"Saved {len(self.semantic_clusters)} semantic clusters")
            
        except Exception as e:
            self.logger.error(f"Failed to save semantic clusters: {str(e)}")
    
    def _load_precomputed_responses(self):
        """Load pre-computed responses from disk."""
        precomputed_file = self.cache_dir / 'precomputed_responses.gz'
        
        if precomputed_file.exists():
            try:
                with gzip.open(precomputed_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.precomputed_responses = data.get('responses', {})
                self.topic_response_cache = data.get('topic_cache', {})
                
                self.logger.info(f"Loaded {len(self.precomputed_responses)} pre-computed responses")
                
            except Exception as e:
                self.logger.error(f"Failed to load pre-computed responses: {str(e)}")
    
    def _save_precomputed_responses(self):
        """Save pre-computed responses to disk."""
        precomputed_file = self.cache_dir / 'precomputed_responses.gz'
        
        try:
            data = {
                'responses': self.precomputed_responses,
                'topic_cache': self.topic_response_cache,
                'saved_at': time.time(),
                'topic_popularity': dict(self.topic_popularity)
            }
            
            with gzip.open(precomputed_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.logger.info(f"Saved {len(self.precomputed_responses)} pre-computed responses")
            
        except Exception as e:
            self.logger.error(f"Failed to save pre-computed responses: {str(e)}")
    
    def _rebuild_cluster_embeddings(self):
        """Rebuild the cluster embeddings matrix for fast similarity search."""
        if not self.semantic_clusters:
            self.cluster_embeddings = None
            return
        
        embeddings = []
        for cluster in self.semantic_clusters.values():
            embeddings.append(cluster.representative_embedding)
        
        self.cluster_embeddings = np.array(embeddings)
        self.logger.debug(f"Rebuilt cluster embeddings: {self.cluster_embeddings.shape}")
    
    def find_semantic_match(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Find the best semantic match for a query.
        
        Returns:
            Tuple of (response, confidence_score) if match found, None otherwise
        """
        if not self.semantic_clusters or self.cluster_embeddings is None:
            return None
        
        if not self.embedding_model:
            return None
        
        self.similarity_stats['total_similarity_checks'] += 1
        
        # Calculate similarities to all clusters
        similarities = cosine_similarity([query_embedding], self.cluster_embeddings)[0]
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            # Get the cluster
            cluster_id = list(self.semantic_clusters.keys())[best_idx]
            cluster = self.semantic_clusters[cluster_id]
            
            # Record hit and update stats
            cluster.record_hit()
            self.similarity_stats['successful_matches'] += 1
            
            # Update average similarity score
            current_avg = self.similarity_stats['avg_similarity_score']
            total_matches = self.similarity_stats['successful_matches']
            self.similarity_stats['avg_similarity_score'] = (
                (current_avg * (total_matches - 1) + best_similarity) / total_matches
            )
            
            self.logger.info(f"Semantic match found: {query[:50]}... -> {cluster.representative_query[:50]}... (similarity: {best_similarity:.3f})")
            
            return cluster.response, cluster.confidence_score * best_similarity
        
        return None
    
    def add_semantic_cluster(self, query: str, query_embedding: np.ndarray, response: str) -> str:
        """
        Add a new semantic cluster or update existing one.
        
        Returns:
            cluster_id of the created/updated cluster
        """
        # Generate cluster ID
        cluster_id = hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:12]
        
        # Create new cluster
        cluster = SemanticCluster(
            cluster_id=cluster_id,
            representative_query=query,
            representative_embedding=query_embedding,
            response=response
        )
        
        self.semantic_clusters[cluster_id] = cluster
        self._rebuild_cluster_embeddings()
        
        # Manage cluster size
        if len(self.semantic_clusters) > self.max_clusters:
            self._prune_clusters()
        
        self.logger.info(f"Added new semantic cluster: {query[:50]}...")
        
        return cluster_id
    
    def _prune_clusters(self):
        """Remove least useful clusters to maintain size limit."""
        if len(self.semantic_clusters) <= self.max_clusters:
            return
        
        # Score clusters by usefulness (hits, recency, confidence)
        cluster_scores = []
        current_time = time.time()
        
        for cluster_id, cluster in self.semantic_clusters.items():
            # Calculate usefulness score
            hit_score = cluster.hit_count
            recency_score = 1.0 / (1.0 + (current_time - cluster.last_updated) / 86400)  # Decay over days
            confidence_score = cluster.confidence_score
            
            total_score = hit_score * 0.5 + recency_score * 0.3 + confidence_score * 0.2
            cluster_scores.append((cluster_id, total_score))
        
        # Sort by score (lowest first) and remove least useful
        cluster_scores.sort(key=lambda x: x[1])
        clusters_to_remove = cluster_scores[:len(self.semantic_clusters) - self.max_clusters]
        
        for cluster_id, _ in clusters_to_remove:
            del self.semantic_clusters[cluster_id]
        
        self._rebuild_cluster_embeddings()
        
        self.logger.info(f"Pruned {len(clusters_to_remove)} semantic clusters")
    
    def get_precomputed_response(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Get pre-computed response based on topic extraction.
        
        Returns:
            Tuple of (response, confidence_score) if match found, None otherwise
        """
        # Extract topics from query
        topics = self.topic_extractor.extract_topics_from_query(query)
        
        if not topics:
            return None
        
        # Get the most relevant topic
        best_topic, relevance_score = topics[0]
        
        # Check if we have a pre-computed response
        if best_topic in self.precomputed_responses:
            response = self.precomputed_responses[best_topic]
            confidence = relevance_score * 0.9  # Slightly lower confidence than exact matches
            
            # Update topic popularity
            self.topic_popularity[best_topic] += 1
            
            self.logger.info(f"Pre-computed response found for topic: {best_topic} (relevance: {relevance_score:.3f})")
            
            return response, confidence
        
        return None
    
    def add_precomputed_response(self, topic_name: str, response: str):
        """Add a pre-computed response for a legal topic."""
        self.precomputed_responses[topic_name] = response
        self.topic_response_cache[topic_name] = {
            'response': response,
            'created_at': time.time(),
            'access_count': 0
        }
        
        self.logger.info(f"Added pre-computed response for topic: {topic_name}")
    
    def analyze_query_patterns(self, query: str):
        """Analyze query patterns for optimization insights."""
        # Normalize query for pattern analysis
        query_normalized = re.sub(r'\d+', 'NUM', query.lower())
        query_normalized = re.sub(r'[^\w\s]', '', query_normalized)
        
        # Extract key patterns
        words = query_normalized.split()
        if len(words) >= 2:
            # Track 2-word patterns
            for i in range(len(words) - 1):
                pattern = f"{words[i]} {words[i+1]}"
                self.query_patterns[pattern] += 1
        
        # Track overall query patterns
        self.query_patterns[query_normalized] += 1
    
    def get_popular_patterns(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get the most popular query patterns."""
        return sorted(self.query_patterns.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    def get_semantic_stats(self) -> Dict[str, Any]:
        """Get comprehensive semantic caching statistics."""
        return {
            'semantic_clusters': {
                'total_clusters': len(self.semantic_clusters),
                'similarity_threshold': self.similarity_threshold,
                'max_clusters': self.max_clusters,
                'cluster_stats': {
                    cluster_id: {
                        'query_count': cluster.query_count,
                        'hit_count': cluster.hit_count,
                        'confidence': round(cluster.confidence_score, 3),
                        'representative_query': cluster.representative_query[:100]
                    }
                    for cluster_id, cluster in list(self.semantic_clusters.items())[:10]  # Top 10
                }
            },
            'precomputed_responses': {
                'total_responses': len(self.precomputed_responses),
                'topic_popularity': dict(sorted(self.topic_popularity.items(), 
                                               key=lambda x: x[1], reverse=True)[:10])
            },
            'similarity_stats': self.similarity_stats,
            'query_patterns': {
                'total_patterns': len(self.query_patterns),
                'popular_patterns': self.get_popular_patterns(10)
            }
        }
    
    def save_all_semantic_data(self):
        """Save all semantic caching data to disk."""
        self._save_semantic_clusters()
        self._save_precomputed_responses()
        
        # Also save base cache data
        self.save_all_caches()
        
        self.logger.info("All semantic cache data saved")
    
    def clear_semantic_data(self, data_type: str = 'all'):
        """Clear semantic caching data."""
        if data_type in ['all', 'clusters']:
            self.semantic_clusters.clear()
            self.cluster_embeddings = None
            self.similarity_stats = {
                'total_similarity_checks': 0,
                'successful_matches': 0,
                'avg_similarity_score': 0.0
            }
        
        if data_type in ['all', 'precomputed']:
            self.precomputed_responses.clear()
            self.topic_response_cache.clear()
            self.topic_popularity.clear()
        
        if data_type in ['all', 'patterns']:
            self.query_patterns.clear()
        
        self.logger.info(f"Cleared semantic data: {data_type}")


# Background task manager for cache warming and maintenance
class BackgroundCacheManager:
    """
    Manages background tasks for cache warming, maintenance, and optimization.
    """
    
    def __init__(self, semantic_cache_manager: SemanticCacheManager, rag_pipeline=None):
        self.cache_manager = semantic_cache_manager
        self.rag_pipeline = rag_pipeline
        self.logger = logging.getLogger('BackgroundCacheManager')
        
        # Task configuration
        self.tasks_enabled = True
        self.task_intervals = {
            'cluster_optimization': 3600,    # 1 hour
            'precompute_generation': 7200,   # 2 hours
            'pattern_analysis': 1800,        # 30 minutes
            'cache_maintenance': 600         # 10 minutes
        }
        
        self.last_task_runs = {task: 0 for task in self.task_intervals}
        
        # Background thread
        self.background_thread = None
        self.stop_event = threading.Event()
    
    def start_background_tasks(self):
        """Start background task processing."""
        if self.background_thread and self.background_thread.is_alive():
            return
        
        self.stop_event.clear()
        self.background_thread = threading.Thread(target=self._background_loop, daemon=True)
        self.background_thread.start()
        
        self.logger.info("Background cache management started")
    
    def stop_background_tasks(self):
        """Stop background task processing."""
        self.tasks_enabled = False
        if self.background_thread:
            self.stop_event.set()
            self.background_thread.join(timeout=5)
        
        self.logger.info("Background cache management stopped")
    
    def _background_loop(self):
        """Main background processing loop."""
        while self.tasks_enabled and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Check and run each task if it's time
                for task_name, interval in self.task_intervals.items():
                    last_run = self.last_task_runs[task_name]
                    
                    if current_time - last_run >= interval:
                        self._run_background_task(task_name)
                        self.last_task_runs[task_name] = current_time
                
                # Sleep for a short interval
                self.stop_event.wait(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Background task error: {str(e)}")
                self.stop_event.wait(60)  # Wait longer after error
    
    def _run_background_task(self, task_name: str):
        """Run a specific background task."""
        try:
            self.logger.debug(f"Running background task: {task_name}")
            
            if task_name == 'cluster_optimization':
                self._optimize_clusters()
            elif task_name == 'precompute_generation':
                self._generate_precomputed_responses()
            elif task_name == 'pattern_analysis':
                self._analyze_usage_patterns()
            elif task_name == 'cache_maintenance':
                self._perform_cache_maintenance()
            
        except Exception as e:
            self.logger.error(f"Error in background task {task_name}: {str(e)}")
    
    def _optimize_clusters(self):
        """Optimize semantic clusters by merging similar ones."""
        if len(self.cache_manager.semantic_clusters) < 2:
            return
        
        clusters = list(self.cache_manager.semantic_clusters.values())
        clusters_to_merge = []
        
        # Find clusters that are very similar to each other
        for i, cluster1 in enumerate(clusters):
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                similarity = cosine_similarity(
                    [cluster1.representative_embedding],
                    [cluster2.representative_embedding]
                )[0][0]
                
                if similarity > 0.95:  # Very high similarity
                    clusters_to_merge.append((cluster1, cluster2, similarity))
        
        # Merge similar clusters
        merged_count = 0
        for cluster1, cluster2, similarity in clusters_to_merge:
            if (cluster1.cluster_id in self.cache_manager.semantic_clusters and 
                cluster2.cluster_id in self.cache_manager.semantic_clusters):
                
                # Merge cluster2 into cluster1
                cluster1.queries.extend(cluster2.queries)
                cluster1.query_count += cluster2.query_count
                cluster1.hit_count += cluster2.hit_count
                
                # Remove cluster2
                del self.cache_manager.semantic_clusters[cluster2.cluster_id]
                merged_count += 1
        
        if merged_count > 0:
            self.cache_manager._rebuild_cluster_embeddings()
            self.logger.info(f"Merged {merged_count} similar semantic clusters")
    
    def _generate_precomputed_responses(self):
        """Generate pre-computed responses for popular topics."""
        if not self.rag_pipeline:
            return
        
        # Get popular topics that don't have pre-computed responses
        topics_to_generate = []
        
        for topic_name, topic_data in self.cache_manager.topic_extractor.legal_topics.items():
            if topic_name not in self.cache_manager.precomputed_responses:
                priority = topic_data.get('priority', 0.5)
                popularity = self.cache_manager.topic_popularity.get(topic_name, 0)
                score = priority + (popularity * 0.1)
                topics_to_generate.append((topic_name, score))
        
        # Sort by score and generate top topics
        topics_to_generate.sort(key=lambda x: x[1], reverse=True)
        
        generated_count = 0
        for topic_name, score in topics_to_generate[:3]:  # Generate top 3
            try:
                # Create a representative query for this topic
                topic_data = self.cache_manager.topic_extractor.legal_topics[topic_name]
                representative_query = f"What is the penalty for {topic_data['keywords'][0]}?"
                
                # Generate response using RAG pipeline
                response = self.rag_pipeline.process_query(representative_query)
                
                # Add to pre-computed responses
                self.cache_manager.add_precomputed_response(topic_name, response)
                generated_count += 1
                
                # Add small delay to avoid overwhelming the system
                time.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Failed to generate response for topic {topic_name}: {str(e)}")
        
        if generated_count > 0:
            self.logger.info(f"Generated {generated_count} pre-computed responses")
    
    def _analyze_usage_patterns(self):
        """Analyze usage patterns for optimization insights."""
        # Get popular patterns
        popular_patterns = self.cache_manager.get_popular_patterns(20)
        
        # Look for patterns that might benefit from pre-computed responses
        potential_topics = []
        
        for pattern, count in popular_patterns:
            if count >= 5:  # Pattern seen at least 5 times
                # Extract topics from this pattern
                topics = self.cache_manager.topic_extractor.extract_topics_from_query(pattern)
                for topic_name, relevance in topics:
                    if topic_name not in self.cache_manager.precomputed_responses:
                        potential_topics.append((topic_name, count * relevance))
        
        if potential_topics:
            # Sort by potential value
            potential_topics.sort(key=lambda x: x[1], reverse=True)
            self.logger.info(f"Identified {len(potential_topics)} topics for potential pre-computation")
    
    def _perform_cache_maintenance(self):
        """Perform routine cache maintenance tasks."""
        # Cleanup expired entries
        self.cache_manager.cleanup_expired()
        
        # Save data periodically
        self.cache_manager.save_all_semantic_data()
        
        # Log statistics
        stats = self.cache_manager.get_semantic_stats()
        self.logger.debug(f"Cache maintenance: {stats['semantic_clusters']['total_clusters']} clusters, "
                         f"{stats['precomputed_responses']['total_responses']} pre-computed responses")


# Singleton pattern for global access
_semantic_cache_manager = None

def get_semantic_cache_manager(cache_dir: str = "data/cache", embedding_model=None) -> SemanticCacheManager:
    """Get global semantic cache manager instance."""
    global _semantic_cache_manager
    if _semantic_cache_manager is None:
        _semantic_cache_manager = SemanticCacheManager(cache_dir, embedding_model)
    return _semantic_cache_manager

def reset_semantic_cache_manager():
    """Reset global semantic cache manager (for testing)."""
    global _semantic_cache_manager
    if _semantic_cache_manager:
        _semantic_cache_manager.save_all_semantic_data()
    _semantic_cache_manager = None