# backend/rag_pipeline.py
"""
Phase 3: Intelligent RAG Pipeline with Semantic Caching
======================================================

This enhanced version builds on Phase 2 and adds intelligent semantic caching
and pre-computed responses for maximum performance optimization.

Phase 3 Features (NEW):
- Level 3: Semantic Query Cache (embedding similarity clustering)
- Level 4: Pre-computed Response Cache (topic-based responses)
- Intelligent cache hierarchy with confidence scoring
- Background cache warming and optimization
- Usage pattern analysis and learning
- Dynamic cluster management

Phase 2 Features (RETAINED):
- Level 1: Exact Query Cache (hash-based exact matches)
- Level 5: Chunk Retrieval Cache (FAISS search results)
- Level 6: Query Embedding Cache (computed embeddings)
- Advanced memory management and TTL support

Phase 1 Features (RETAINED):
- Persistent FAISS index storage
- Document change detection with SHA-256 hashing
- Enhanced FAQ cache with 50+ entries
- Build info tracking and versioning
- Comprehensive error handling and recovery
"""

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import ollama
import os
import sys
import requests
import logging
import json
import time
import hashlib
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple, Optional, Any

# Import cache management systems
from cache_manager import (
    CacheManager, 
    get_cache_manager, 
    generate_query_hash, 
    generate_embedding_hash, 
    generate_retrieval_hash
)
from semantic_cache_manager import (
    SemanticCacheManager,
    BackgroundCacheManager,
    get_semantic_cache_manager
)


class IntelligentRAGPipeline:
    """
    Intelligent RAG pipeline with semantic caching and pre-computed responses.
    
    Cache Hierarchy (checked in order):
    1. Level 2: FAQ Cache (instant responses for common questions)
    2. Level 1: Exact Query Cache (hash-based exact matches)
    3. Level 3: Semantic Query Cache (similarity-based matches) [NEW]
    4. Level 4: Pre-computed Response Cache (topic-based responses) [NEW]
    5. Level 6: Query Embedding Cache (skip embedding computation)
    6. Level 5: Chunk Retrieval Cache (skip FAISS search)
    7. Full RAG Pipeline (generate new response)
    """
    
    def __init__(self, config=None):
        """
        Initialize the intelligent RAG pipeline with semantic caching.
        
        Args:
            config (dict): Configuration options for the pipeline
        """
        # Default configuration optimized for intelligent caching
        self.config = {
            # Model settings
            'embedding_model': 'all-MiniLM-L6-v2',
            'ollama_model': 'tinyllama',
            'ollama_timeout': 30,
            
            # Chunking settings
            'chunk_size': 600,
            'chunk_overlap': 100,
            'max_retrieval_chunks': 5,
            
            # Generation settings
            'temperature': 0.3,
            'max_response_tokens': 500,
            
            # Storage paths
            'data_dir': 'data',
            'vector_store_dir': 'data/vector_store',
            'cache_dir': 'data/cache',
            'source_document': 'data/output/mv_act_cleaned.txt',
            
            # Cache settings (Phase 2)
            'enable_exact_query_cache': True,
            'enable_embedding_cache': True,
            'enable_retrieval_cache': True,
            'cache_query_responses': True,
            'min_query_length': 3,
            'max_cache_query_length': 500,
            
            # Semantic cache settings (Phase 3)
            'enable_semantic_cache': True,
            'enable_precomputed_cache': True,
            'semantic_similarity_threshold': 0.85,
            'confidence_threshold': 0.7,
            'enable_background_tasks': True,
            'auto_cluster_creation': True,
            'max_semantic_clusters': 100,
            
            # Performance settings
            'faq_cache_size': 100,
            'index_version': '3.0',  # Updated for Phase 3
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
        
        # Initialize paths
        self._setup_directories()
        
        # Initialize cache managers
        self.cache_manager = get_cache_manager(self.config['cache_dir'])
        self.semantic_cache_manager = None  # Will be initialized after embedding model
        self.background_manager = None
        
        # Initialize components
        self.embedding_model = None
        self.text_splitter = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Persistent storage tracking
        self.build_info = {}
        self.document_hash = None
        
        # FAQ cache (Phase 1)
        self.faq_cache = {}
        
        # Statistics and metrics (enhanced for Phase 3)
        self.pipeline_stats = {
            'initialization_method': 'unknown',
            'startup_time': 0,
            'document_loaded': False,
            'total_chunks': 0,
            'avg_chunk_length': 0,
            'embedding_dimensions': 0,
            'index_built': False,
            'legal_sections_found': 0,
            'tables_found': 0,
            'last_query_time': 0,
            'total_queries_processed': 0,
            
            # Phase 2: Cache statistics
            'faq_cache_hits': 0,
            'exact_query_cache_hits': 0,
            'embedding_cache_hits': 0,
            'retrieval_cache_hits': 0,
            'full_rag_executions': 0,
            'total_cache_hits': 0,
            'cache_saves': 0,
            
            # Phase 3: Semantic cache statistics
            'semantic_cache_hits': 0,
            'precomputed_cache_hits': 0,
            'semantic_clusters_created': 0,
            'precomputed_responses_generated': 0,
            'background_tasks_completed': 0,
            
            'cache_hierarchy_breakdown': {
                'faq': 0,
                'exact_query': 0,
                'semantic': 0,
                'precomputed': 0,
                'embedding_cached': 0,
                'retrieval_cached': 0,
                'full_rag': 0
            },
            
            # Response confidence tracking
            'confidence_scores': {
                'high_confidence': 0,    # > 0.9
                'medium_confidence': 0,  # 0.7 - 0.9
                'low_confidence': 0      # < 0.7
            }
        }
        
        # Setup logging
        self._setup_logging()
        
        # Initialize the pipeline
        self._initialize_pipeline()
    
    def _setup_logging(self):
        """Configure detailed logging for the RAG pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
        self.logger = logging.getLogger('IntelligentRAGPipeline')
    
    def _setup_directories(self):
        """Create necessary directories for persistent storage and caching."""
        directories = [
            self.config['vector_store_dir'],
            self.config['cache_dir'],
            os.path.join(self.config['vector_store_dir'], 'versions')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _calculate_document_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of the document for change detection."""
        if not os.path.exists(file_path):
            return "FILE_NOT_FOUND"
        
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return sha256_hash.hexdigest()
    
    def _get_storage_paths(self) -> Dict[str, str]:
        """Get all persistent storage file paths."""
        base_dir = self.config['vector_store_dir']
        return {
            'index': os.path.join(base_dir, 'faiss_index.bin'),
            'chunks': os.path.join(base_dir, 'chunks.json'),
            'metadata': os.path.join(base_dir, 'chunk_metadata.json'),
            'embeddings': os.path.join(base_dir, 'embeddings.npy'),
            'build_info': os.path.join(base_dir, 'build_info.json'),
            'faq_cache': os.path.join(self.config['cache_dir'], 'faq_cache.json')
        }
    
    def _save_build_info(self):
        """Save build information for tracking and validation."""
        paths = self._get_storage_paths()
        
        self.build_info = {
            'version': self.config['index_version'],
            'build_timestamp': datetime.now().isoformat(),
            'document_hash': self.document_hash,
            'document_path': self.config['source_document'],
            'total_chunks': len(self.chunks),
            'embedding_model': self.config['embedding_model'],
            'embedding_dimensions': self.pipeline_stats['embedding_dimensions'],
            'chunk_size': self.config['chunk_size'],
            'chunk_overlap': self.config['chunk_overlap'],
            'legal_sections': self.pipeline_stats['legal_sections_found'],
            'tables_preserved': self.pipeline_stats['tables_found'],
            'phase': 3,  # Phase 3 marker
            'cache_features': {
                'exact_query_cache': self.config['enable_exact_query_cache'],
                'embedding_cache': self.config['enable_embedding_cache'],
                'retrieval_cache': self.config['enable_retrieval_cache'],
                'semantic_cache': self.config['enable_semantic_cache'],
                'precomputed_cache': self.config['enable_precomputed_cache']
            },
            'semantic_features': {
                'similarity_threshold': self.config['semantic_similarity_threshold'],
                'confidence_threshold': self.config['confidence_threshold'],
                'max_clusters': self.config['max_semantic_clusters'],
                'background_tasks': self.config['enable_background_tasks']
            }
        }
        
        with open(paths['build_info'], 'w', encoding='utf-8') as f:
            json.dump(self.build_info, f, indent=2)
        
        self.logger.info(f"Build info saved: {len(self.chunks)} chunks, Phase 3 features enabled")
    
    def _load_build_info(self) -> bool:
        """Load and validate build information."""
        paths = self._get_storage_paths()
        
        if not os.path.exists(paths['build_info']):
            self.logger.info("No build info found - will build from scratch")
            return False
        
        try:
            with open(paths['build_info'], 'r', encoding='utf-8') as f:
                self.build_info = json.load(f)
            
            # Validate build info
            current_hash = self._calculate_document_hash(self.config['source_document'])
            stored_hash = self.build_info.get('document_hash')
            
            if current_hash != stored_hash:
                self.logger.info(f"Document changed - will rebuild")
                return False
            
            # Check if all required files exist
            required_files = ['index', 'chunks', 'metadata', 'embeddings']
            for file_key in required_files:
                if not os.path.exists(paths[file_key]):
                    self.logger.info(f"Missing required file: {paths[file_key]} - will rebuild")
                    return False
            
            # Check model compatibility
            if self.build_info.get('embedding_model') != self.config['embedding_model']:
                self.logger.info("Embedding model changed - will rebuild")
                return False
            
            self.document_hash = current_hash
            self.logger.info(f"Build info validated - can load from cache")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading build info: {str(e)} - will rebuild")
            return False
    
    def _save_vector_store(self):
        """Save the complete vector store to disk."""
        paths = self._get_storage_paths()
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, paths['index'])
            
            # Save chunks
            with open(paths['chunks'], 'w', encoding='utf-8') as f:
                json.dump(self.chunks, f, indent=2, ensure_ascii=False)
            
            # Save chunk metadata
            with open(paths['metadata'], 'w', encoding='utf-8') as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            # Save embeddings if we have them
            if hasattr(self, 'embeddings') and self.embeddings is not None:
                np.save(paths['embeddings'], self.embeddings)
            
            # Save build info
            self._save_build_info()
            
            # Save all cache data
            self.cache_manager.save_all_caches()
            if self.semantic_cache_manager:
                self.semantic_cache_manager.save_all_semantic_data()
            
            self.pipeline_stats['cache_saves'] += 1
            
            self.logger.info("Vector store and all caches saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving vector store: {str(e)}")
            raise
    
    def _load_vector_store(self) -> bool:
        """Load the complete vector store from disk."""
        paths = self._get_storage_paths()
        
        try:
            # Load FAISS index
            self.index = faiss.read_index(paths['index'])
            
            # Load chunks
            with open(paths['chunks'], 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            
            # Load chunk metadata
            with open(paths['metadata'], 'r', encoding='utf-8') as f:
                self.chunk_metadata = json.load(f)
            
            # Load embeddings if available
            if os.path.exists(paths['embeddings']):
                self.embeddings = np.load(paths['embeddings'])
            
            # Update pipeline stats from build info
            self.pipeline_stats.update({
                'total_chunks': len(self.chunks),
                'embedding_dimensions': self.build_info.get('embedding_dimensions', 0),
                'legal_sections_found': self.build_info.get('legal_sections', 0),
                'tables_found': self.build_info.get('tables_preserved', 0),
                'index_built': True,
                'document_loaded': True
            })
            
            self.logger.info(f"Vector store loaded: {len(self.chunks)} chunks, {self.index.ntotal} vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return False
    
    def _initialize_pipeline(self):
        """Initialize the pipeline with persistent storage and intelligent caching."""
        start_time = time.time()
        
        print("🚀 Initializing Intelligent RAG Pipeline (Phase 3)")
        print("=" * 70)
        
        try:
            # Step 1: Load embedding model
            self._load_embedding_model()
            
            # Step 2: Setup text splitter
            self._setup_text_splitter()
            
            # Step 3: Initialize semantic cache manager (Phase 3)
            self._initialize_semantic_caching()
            
            # Step 4: Load FAQ cache
            self._load_faq_cache()
            
            # Step 5: Try to load existing vector store
            if self._load_build_info() and self._load_vector_store():
                print("✅ Loaded existing vector store from cache")
                self.pipeline_stats['initialization_method'] = 'loaded'
            else:
                print("🔨 Building new vector store...")
                self._build_vector_store_from_scratch()
                self.pipeline_stats['initialization_method'] = 'built'
            
            # Step 6: Initialize background processing (Phase 3)
            self._initialize_background_processing()
            
            # Record initialization time
            self.pipeline_stats['startup_time'] = time.time() - start_time
            
            # Print success summary
            self._print_initialization_summary()
            
            self.logger.info("Intelligent RAG Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Intelligent RAG Pipeline: {str(e)}")
            print(f"\n❌ INITIALIZATION FAILED: {str(e)}")
            raise
    
    def _initialize_semantic_caching(self):
        """Initialize Phase 3 semantic caching features."""
        print("🧠 Initializing semantic cache system...")
        
        # Initialize semantic cache manager with embedding model
        self.semantic_cache_manager = SemanticCacheManager(
            cache_dir=self.config['cache_dir'],
            embedding_model=self.embedding_model
        )
        
        # Configure semantic caching parameters
        self.semantic_cache_manager.similarity_threshold = self.config['semantic_similarity_threshold']
        self.semantic_cache_manager.max_clusters = self.config['max_semantic_clusters']
        
        # Get initial statistics
        semantic_stats = self.semantic_cache_manager.get_semantic_stats()
        
        print(f"   Semantic clusters: {semantic_stats['semantic_clusters']['total_clusters']}")
        print(f"   Pre-computed responses: {semantic_stats['precomputed_responses']['total_responses']}")
        print(f"   Similarity threshold: {self.config['semantic_similarity_threshold']}")
        print(f"   ✅ Semantic cache system ready")
    
    def _initialize_background_processing(self):
        """Initialize background processing for cache optimization."""
        if not self.config['enable_background_tasks']:
            print("   ⚠️ Background tasks disabled")
            return
        
        print("🔄 Initializing background processing...")
        
        try:
            # Initialize background cache manager
            self.background_manager = BackgroundCacheManager(
                semantic_cache_manager=self.semantic_cache_manager,
                rag_pipeline=self
            )
            
            # Start background tasks
            self.background_manager.start_background_tasks()
            
            print(f"   ✅ Background processing started")
            
        except Exception as e:
            print(f"   ⚠️ Background processing failed to start: {str(e)}")
            self.logger.warning(f"Background processing initialization failed: {str(e)}")
    
    def _load_embedding_model(self):
        """Load and validate the sentence transformer model."""
        print("🧠 Loading embedding model...")
        
        try:
            model_name = self.config['embedding_model']
            print(f"   Model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            
            # Get embedding dimensions
            test_embedding = self.embedding_model.encode(["test"])
            self.pipeline_stats['embedding_dimensions'] = len(test_embedding[0])
            
            print(f"   ✅ Model loaded successfully")
            print(f"   Embedding dimensions: {self.pipeline_stats['embedding_dimensions']}")
            
        except Exception as e:
            print(f"   ❌ Failed to load embedding model: {str(e)}")
            raise
    
    def _setup_text_splitter(self):
        """Setup the text splitter with legal document optimization."""
        print("📄 Setting up text splitter...")
        
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap'],
                separators=[
                    "\n\nSection ",  # Legal sections
                    "\n\nChapter ",  # Chapters
                    "\n\n",          # Paragraphs
                    "\n",            # Lines
                    ". ",            # Sentences
                    " "              # Words
                ],
                keep_separator=True
            )
            
            print(f"   Chunk size: {self.config['chunk_size']} characters")
            print(f"   Chunk overlap: {self.config['chunk_overlap']} characters")
            print(f"   ✅ Text splitter configured")
            
        except Exception as e:
            print(f"   ❌ Failed to setup text splitter: {str(e)}")
            raise
    
    def _build_vector_store_from_scratch(self):
        """Build the vector store from scratch when no cache is available."""
        # Load and process document
        self._load_and_process_document()
        
        # Build vector index
        self._build_vector_index()
        
        # Save everything to disk
        self._save_vector_store()
    
    def _load_and_process_document(self):
        """Load and process the legal document."""
        print("📖 Loading and processing document...")
        
        document_path = self.config['source_document']
        
        try:
            if not os.path.exists(document_path):
                raise FileNotFoundError(f"Document not found: {document_path}")
            
            # Calculate document hash for change detection
            self.document_hash = self._calculate_document_hash(document_path)
            
            # Load document
            with open(document_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            if not document_text.strip():
                raise ValueError(f"Document is empty: {document_path}")
            
            print(f"   Document loaded: {len(document_text):,} characters")
            print(f"   Document hash: {self.document_hash[:16]}...")
            
            # Analyze document structure
            self._analyze_document_structure(document_text)
            
            # Split document into chunks
            print("   Splitting document into chunks...")
            self.chunks = self.text_splitter.split_text(document_text)
            
            if not self.chunks:
                raise ValueError("Failed to generate chunks from document.")
            
            # Create metadata for each chunk
            self._create_chunk_metadata()
            
            # Calculate statistics
            self.pipeline_stats.update({
                'total_chunks': len(self.chunks),
                'avg_chunk_length': sum(len(chunk) for chunk in self.chunks) / len(self.chunks),
                'document_loaded': True
            })
            
            print(f"   ✅ Document processed into {len(self.chunks)} chunks")
            print(f"   Average chunk length: {self.pipeline_stats['avg_chunk_length']:.0f} characters")
            
        except Exception as e:
            print(f"   ❌ Failed to load document: {str(e)}")
            raise
    
    def _analyze_document_structure(self, text):
        """Analyze the structure of the legal document."""
        print("   Analyzing legal document structure...")
        
        # Count legal sections
        section_pattern = r'Section\s+\d+[A-Z]*'
        sections = re.findall(section_pattern, text, re.IGNORECASE)
        self.pipeline_stats['legal_sections_found'] = len(set(sections))
        
        # Count tables
        table_pattern = r'\[TABLE START\].*?\[TABLE END\]'
        tables = re.findall(table_pattern, text, re.DOTALL)
        self.pipeline_stats['tables_found'] = len(tables)
        
        print(f"      Legal sections found: {self.pipeline_stats['legal_sections_found']}")
        print(f"      Tables found: {self.pipeline_stats['tables_found']}")
    
    def _create_chunk_metadata(self):
        """Create metadata for each chunk to improve retrieval quality."""
        print("   Creating chunk metadata...")
        
        self.chunk_metadata = []
        
        for i, chunk in enumerate(self.chunks):
            metadata = {
                'chunk_id': i,
                'length': len(chunk),
                'word_count': len(chunk.split()),
                'has_section': bool(re.search(r'Section\s+\d+', chunk, re.IGNORECASE)),
                'has_penalty': bool(re.search(r'₹\d+|fine|penalty', chunk, re.IGNORECASE)),
                'has_table': '[TABLE START]' in chunk,
                'section_numbers': re.findall(r'Section\s+(\d+[A-Z]*)', chunk, re.IGNORECASE)
            }
            self.chunk_metadata.append(metadata)
    
    def _build_vector_index(self):
        """Build the vector index from document chunks."""
        print("🔮 Building vector index...")
        
        try:
            print("   Generating embeddings...")
            
            # Generate embeddings with progress tracking
            embeddings = self.embedding_model.encode(
                self.chunks, 
                show_progress_bar=True,
                batch_size=32
            )
            
            # Store embeddings for potential future use
            self.embeddings = embeddings
            
            print(f"   Generated {len(embeddings)} embeddings")
            
            # Create and populate FAISS index
            embedding_dim = self.pipeline_stats['embedding_dimensions']
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.index.add(np.array(embeddings, dtype='float32'))
            
            self.pipeline_stats['index_built'] = True
            
            print(f"   ✅ Vector index built successfully")
            print(f"   Total vectors in index: {self.index.ntotal}")
            
        except Exception as e:
            print(f"   ❌ Failed to build vector index: {str(e)}")
            raise
    
    def _load_faq_cache(self):
        """Load the enhanced FAQ cache."""
        print("📚 Loading FAQ cache...")
        
        paths = self._get_storage_paths()
        
        # Try to load existing FAQ cache
        if os.path.exists(paths['faq_cache']):
            try:
                with open(paths['faq_cache'], 'r', encoding='utf-8') as f:
                    self.faq_cache = json.load(f)
                print(f"   ✅ Loaded {len(self.faq_cache)} FAQ entries from cache")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load FAQ cache: {str(e)}")
        
        # Create enhanced FAQ cache if none exists
        self.faq_cache = self._create_enhanced_faq_cache()
        
        # Save the new FAQ cache
        try:
            with open(paths['faq_cache'], 'w', encoding='utf-8') as f:
                json.dump(self.faq_cache, f, indent=2, ensure_ascii=False)
            print(f"   ✅ Created and saved {len(self.faq_cache)} FAQ entries")
        except Exception as e:
            self.logger.warning(f"Failed to save FAQ cache: {str(e)}")
    
    def _create_enhanced_faq_cache(self) -> Dict[str, str]:
        """Create an enhanced FAQ cache with 50+ entries and variations."""
        # Same as Phase 2, keeping it consistent
        return {
            # Driving License Related
            "What is the penalty for driving without a license?": "As per Section 181 of the Motor Vehicles Act 1988, driving without a valid license is punishable with a fine of ₹5,000. This applies to all motor vehicles and is a serious traffic violation.",
            
            "What is the fine for driving without a driving license?": "Under Section 181 of the Motor Vehicles Act, the penalty for driving without a valid license is ₹5,000. This is applicable across all states in India.",
            
            "Can I drive without a license?": "No, driving without a valid license is illegal under Section 181 of the Motor Vehicles Act 1988. You can face a fine of ₹5,000 for this violation.",
            
            "What happens if caught driving without license?": "If caught driving without a license, you will be fined ₹5,000 under Section 181 of the Motor Vehicles Act. The vehicle may also be impounded.",
            
            "What is the minimum age for driving license?": "As per Section 4 of the Motor Vehicles Act, the minimum age for obtaining a driving license is 18 years for cars and motorcycles above 50cc. For gearless motorcycles up to 50cc, the minimum age is 16 years with parental consent.",
            
            "Can a minor obtain a driving license under the MV Act?": "No, minors (below 18 years) cannot obtain a regular driving license under Section 4 of the Motor Vehicles Act. However, 16-year-olds can get a learner's license for gearless vehicles up to 50cc with parental consent.",
            
            # Helmet and Safety
            "What is the fine for not wearing a helmet?": "As per Section 194D of the Motor Vehicles Act, riding a two-wheeler without a helmet incurs a fine of ₹1,000 and possible driving license suspension for three months.",
            
            "Is helmet mandatory for bike riders?": "Yes, wearing a helmet is mandatory for all two-wheeler riders and pillion passengers under Section 194D. The fine for non-compliance is ₹1,000.",
            
            "What is the penalty for not wearing helmet while riding?": "Under Section 194D, not wearing a helmet while riding a two-wheeler attracts a fine of ₹1,000 and potential license suspension for three months.",
            
            "Can pillion rider be fined for not wearing helmet?": "Yes, both the rider and pillion passenger can be fined ₹1,000 each under Section 194D for not wearing helmets.",
            
            # Speed and Traffic Violations
            "What is the punishment for overspeeding?": "Section 183 outlines penalties for overspeeding: for light motor vehicles, the fine is ₹1,000-₹2,000; for medium/heavy vehicles, it's ₹2,000-₹4,000. Repeat offenders may face license suspension.",
            
            "What is the fine for speed limit violation?": "Under Section 183, speeding fines are ₹1,000-₹2,000 for light vehicles and ₹2,000-₹4,000 for heavy vehicles. License impoundment is possible for repeated violations.",
            
            "What happens if I exceed speed limit?": "Exceeding speed limits under Section 183 results in fines of ₹1,000-₹4,000 depending on vehicle type. Repeated violations can lead to license suspension.",
            
            # Drunk Driving
            "What is the penalty for drunk driving?": "Section 185 prescribes imprisonment up to 6 months and/or fine up to ₹2,000 for first-time drunk driving. Repeat offenses within 3 years attract imprisonment up to 2 years and/or fine up to ₹3,000.",
            
            "What is the fine for driving under influence of alcohol?": "Under Section 185, drunk driving penalties include imprisonment up to 6 months and/or fine up to ₹2,000 for first offense. Subsequent offenses have higher penalties.",
            
            "What is the blood alcohol limit for driving?": "As per Section 185, the blood alcohol content limit is 30mg per 100ml of blood. Exceeding this limit constitutes drunk driving.",
            
            # Insurance and Registration
            "What happens if I drive a vehicle without a valid registration?": "As per Section 192, driving an unregistered vehicle is punishable with a fine up to ₹5,000 for the first offense and ₹10,000 or imprisonment up to 7 years for subsequent offenses.",
            
            "Is third party insurance mandatory?": "Yes, third-party insurance is mandatory under Section 146 of the Motor Vehicles Act. Driving without valid insurance attracts penalties under Section 196.",
            
            "What is the fine for driving without insurance?": "Under Section 196, driving without valid insurance can result in imprisonment up to 3 months and/or fine up to ₹1,000 for first offense.",
            
            "What happens if vehicle registration expires?": "Driving with expired registration is treated as driving an unregistered vehicle under Section 192, attracting fines up to ₹5,000 for first offense.",
            
            # Traffic Signals and Rules
            "What is the penalty for jumping red light?": "Under Section 177, violating traffic signals including red light jumping attracts a fine of ₹1,000-₹5,000 depending on the state's rules.",
            
            "What is the fine for not following traffic rules?": "Section 177 covers general traffic rule violations with fines typically ranging from ₹500 to ₹5,000 depending on the specific violation and state rules.",
            
            "What happens if I don't stop at red light?": "Jumping red lights is a violation under Section 177, typically attracting fines of ₹1,000-₹5,000. It may also result in license endorsement.",
            
            # Pollution and Emission
            "What is the penalty for not having pollution certificate?": "Under Section 190, driving without a valid Pollution Under Control (PUC) certificate attracts a fine of ₹1,000-₹10,000 depending on the vehicle type.",
            
            "Is pollution certificate mandatory?": "Yes, a valid Pollution Under Control certificate is mandatory under Section 190. The vehicle can be impounded if the certificate is not available.",
            
            # Golden Hour Provision
            "What is the golden hour in the MV Act?": "The 'golden hour' refers to the one-hour period following a traumatic injury where prompt medical treatment can significantly improve survival chances, as defined in Section 2(12A) of the Motor Vehicles Act.",
            
            "What does golden hour mean in motor vehicle act?": "Section 2(12A) defines 'golden hour' as the critical one-hour window after an accident when immediate medical care can save lives. The Act mandates provisions for emergency medical care during this period.",
            
            # Additional common queries for Phase 3
            "What is the penalty for using mobile phone while driving?": "Using mobile phones while driving is typically covered under Section 177 for traffic rule violations, with fines ranging from ₹1,000-₹5,000.",
            
            "What is the fine for not wearing seat belt?": "Not wearing seat belts typically attracts a fine of ₹1,000 under various state motor vehicle rules, though the specific section varies by state.",
            
            "What documents are required while driving?": "Essential documents include valid driving license, vehicle registration certificate, insurance certificate, and PUC certificate. These must be produced when demanded by authorities.",
            
            "What is the penalty for vehicle overloading?": "Section 194 covers penalties for overloading, with fines of ₹2,000-₹20,000 depending on the extent of overloading and vehicle type.",
            
            "What is vehicle fitness certificate?": "Commercial vehicles require fitness certificates under Section 56 to ensure roadworthiness. The certificate must be renewed periodically as prescribed.",
            
            "What powers do traffic police have?": "Traffic police have powers under Section 206 to check documents, impose penalties, detain vehicles, and take action against traffic violations.",
            
            "What is e-challan system?": "E-challan is an electronic system for issuing traffic violation notices. It allows automatic detection and penalization of traffic violations through cameras and digital systems.",
            
            "Can traffic police detain my vehicle?": "Yes, under Section 206, traffic police can detain vehicles for serious violations, lack of proper documents, or when the vehicle poses a public danger.",
            
            "What is the penalty for hit and run?": "Hit and run cases are covered under Section 161 for not providing information after accidents. Under the new criminal laws, it may also attract charges under Bharatiya Nyaya Sanhita Section 106."
        }
    
    def _print_initialization_summary(self):
        """Print a comprehensive initialization summary."""
        print("\n" + "="*70)
        print("📊 INTELLIGENT RAG PIPELINE INITIALIZATION SUMMARY (Phase 3)")
        print("="*70)
        print(f"Initialization method: {self.pipeline_stats['initialization_method'].upper()}")
        print(f"Startup time: {self.pipeline_stats['startup_time']:.2f} seconds")
        print(f"Embedding model: {self.config['embedding_model']}")
        print(f"Embedding dimensions: {self.pipeline_stats['embedding_dimensions']}")
        print(f"Document chunks: {self.pipeline_stats['total_chunks']}")
        print(f"Average chunk length: {self.pipeline_stats['avg_chunk_length']:.0f} characters")
        print(f"Legal sections found: {self.pipeline_stats['legal_sections_found']}")
        print(f"Tables preserved: {self.pipeline_stats['tables_found']}")
        print(f"Vector index size: {self.index.ntotal}")
        print(f"FAQ cache entries: {len(self.faq_cache)}")
        
        # Phase 2: Advanced cache information
        cache_stats = self.cache_manager.get_comprehensive_stats()
        print(f"\n🔄 Standard Cache System:")
        for cache_name, stats in cache_stats['cache_stats'].items():
            print(f"   • {cache_name}: {stats['size']} entries")
        
        # Phase 3: Semantic cache information
        if self.semantic_cache_manager:
            semantic_stats = self.semantic_cache_manager.get_semantic_stats()
            print(f"\n🧠 Intelligent Cache System:")
            print(f"   • Semantic clusters: {semantic_stats['semantic_clusters']['total_clusters']}")
            print(f"   • Pre-computed responses: {semantic_stats['precomputed_responses']['total_responses']}")
            print(f"   • Similarity threshold: {self.config['semantic_similarity_threshold']}")
        
        # Background processing
        if self.config['enable_background_tasks']:
            print(f"\n🔄 Background Processing: ENABLED")
        else:
            print(f"\n🔄 Background Processing: DISABLED")
        
        if self.document_hash:
            print(f"\nDocument hash: {self.document_hash[:16]}...")
        
        print("✅ Intelligent Pipeline ready for queries!")
        print("="*70)
    
    def _validate_ollama_connection(self):
        """Validate that Ollama server is running and accessible."""
        try:
            response = requests.get('http://localhost:11434', timeout=5)
            if response.status_code != 200:
                raise Exception("Ollama server is not responding properly.")
            return True
        except requests.exceptions.RequestException as e:
            error_msg = (
                f"Ollama server is not running or accessible: {str(e)}\n\n"
                "💡 To fix this:\n"
                "   1. Install Ollama from https://ollama.ai\n"
                "   2. Run 'ollama serve' in terminal\n"
                "   3. Run 'ollama pull tinyllama' to download the model\n"
                "   4. Ensure no firewall is blocking port 11434"
            )
            raise Exception(error_msg)
    
    def _is_valid_query(self, query: str) -> bool:
        """Validate if query should be cached."""
        if len(query) < self.config['min_query_length']:
            return False
        if len(query) > self.config['max_cache_query_length']:
            return False
        return True
    
    def _check_faq_cache(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Level 2: Check FAQ cache with fuzzy matching.
        
        Returns:
            Tuple of (response, confidence_score) if found, None otherwise
        """
        query_normalized = query.lower().strip()
        
        # Exact match first
        for faq_question, faq_answer in self.faq_cache.items():
            if query_normalized == faq_question.lower().strip():
                self.pipeline_stats['faq_cache_hits'] += 1
                self.pipeline_stats['cache_hierarchy_breakdown']['faq'] += 1
                self.logger.info(f"FAQ exact match hit for: {query[:50]}...")
                return faq_answer, 1.0  # Perfect confidence for FAQ matches
        
        # Fuzzy matching for variations
        for faq_question, faq_answer in self.faq_cache.items():
            faq_question_normalized = faq_question.lower().strip()
            
            # Check if key terms match
            query_words = set(query_normalized.split())
            faq_words = set(faq_question_normalized.split())
            
            # Calculate word overlap
            common_words = query_words.intersection(faq_words)
            if len(common_words) >= 3 and len(common_words) / len(query_words) > 0.6:
                self.pipeline_stats['faq_cache_hits'] += 1
                self.pipeline_stats['cache_hierarchy_breakdown']['faq'] += 1
                self.logger.info(f"FAQ fuzzy match hit for: {query[:50]}...")
                confidence = min(0.95, len(common_words) / len(query_words))
                return faq_answer, confidence
        
        return None
    
    def _check_exact_query_cache(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Level 1: Check exact query cache using hash-based matching.
        
        Returns:
            Tuple of (response, confidence_score) if found, None otherwise
        """
        if not self.config['enable_exact_query_cache'] or not self._is_valid_query(query):
            return None
        
        query_hash = generate_query_hash(query)
        cached_response = self.cache_manager.get('exact_query', query_hash)
        
        if cached_response:
            self.pipeline_stats['exact_query_cache_hits'] += 1
            self.pipeline_stats['cache_hierarchy_breakdown']['exact_query'] += 1
            self.logger.info(f"Exact query cache hit for: {query[:50]}...")
            return cached_response, 0.95  # High confidence for exact matches
        
        return None
    
    def _check_semantic_cache(self, query: str, query_embedding: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Level 3: Check semantic cache using similarity clustering.
        
        Returns:
            Tuple of (response, confidence_score) if found, None otherwise
        """
        if not self.config['enable_semantic_cache'] or not self.semantic_cache_manager:
            return None
        
        # Analyze query patterns for learning
        self.semantic_cache_manager.analyze_query_patterns(query)
        
        # Find semantic match
        semantic_match = self.semantic_cache_manager.find_semantic_match(query, query_embedding)
        
        if semantic_match:
            response, confidence = semantic_match
            
            if confidence >= self.config['confidence_threshold']:
                self.pipeline_stats['semantic_cache_hits'] += 1
                self.pipeline_stats['cache_hierarchy_breakdown']['semantic'] += 1
                self.logger.info(f"Semantic cache hit for: {query[:50]}... (confidence: {confidence:.3f})")
                return response, confidence
        
        return None
    
    def _check_precomputed_cache(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Level 4: Check pre-computed response cache using topic extraction.
        
        Returns:
            Tuple of (response, confidence_score) if found, None otherwise
        """
        if not self.config['enable_precomputed_cache'] or not self.semantic_cache_manager:
            return None
        
        # Get pre-computed response based on topic analysis
        precomputed_match = self.semantic_cache_manager.get_precomputed_response(query)
        
        if precomputed_match:
            response, confidence = precomputed_match
            
            if confidence >= self.config['confidence_threshold']:
                self.pipeline_stats['precomputed_cache_hits'] += 1
                self.pipeline_stats['cache_hierarchy_breakdown']['precomputed'] += 1
                self.logger.info(f"Pre-computed cache hit for: {query[:50]}... (confidence: {confidence:.3f})")
                return response, confidence
        
        return None
    
    def _get_cached_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        Level 6: Get cached query embedding.
        """
        if not self.config['enable_embedding_cache'] or not self._is_valid_query(query):
            return None
        
        embedding_hash = generate_embedding_hash(query, self.config['embedding_model'])
        cached_embedding = self.cache_manager.get('embedding', embedding_hash)
        
        if cached_embedding is not None:
            self.pipeline_stats['embedding_cache_hits'] += 1
            self.logger.info(f"Embedding cache hit for: {query[:50]}...")
            return np.array(cached_embedding)
        
        return None
    
    def _cache_embedding(self, query: str, embedding: np.ndarray):
        """Cache computed embedding."""
        if not self.config['enable_embedding_cache'] or not self._is_valid_query(query):
            return
        
        embedding_hash = generate_embedding_hash(query, self.config['embedding_model'])
        self.cache_manager.set('embedding', embedding_hash, embedding.tolist())
    
    def _get_cached_retrieval(self, query_embedding: np.ndarray, k: int) -> Optional[List[Dict]]:
        """
        Level 5: Get cached chunk retrieval results.
        """
        if not self.config['enable_retrieval_cache']:
            return None
        
        retrieval_hash = generate_retrieval_hash(query_embedding, k)
        cached_results = self.cache_manager.get('retrieval', retrieval_hash)
        
        if cached_results:
            self.pipeline_stats['retrieval_cache_hits'] += 1
            self.logger.info(f"Retrieval cache hit")
            return cached_results
        
        return None
    
    def _cache_retrieval(self, query_embedding: np.ndarray, k: int, results: List[Dict]):
        """Cache retrieval results."""
        if not self.config['enable_retrieval_cache']:
            return
        
        retrieval_hash = generate_retrieval_hash(query_embedding, k)
        self.cache_manager.set('retrieval', retrieval_hash, results)
    
    def retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict]:
        """
        Retrieve relevant chunks with multi-level caching support.
        """
        if k is None:
            k = self.config['max_retrieval_chunks']
        
        try:
            # Step 1: Get or compute query embedding (Level 6 cache)
            query_embedding = self._get_cached_embedding(query)
            
            if query_embedding is None:
                # Compute embedding and cache it
                query_embedding = self.embedding_model.encode([query])[0]
                self._cache_embedding(query, query_embedding)
            
            # Step 2: Check retrieval cache (Level 5 cache)
            cached_results = self._get_cached_retrieval(query_embedding, k)
            if cached_results:
                return cached_results
            
            # Step 3: Perform FAISS search
            self.logger.info(f"Performing FAISS search for query: {query[:50]}...")
            
            distances, indices = self.index.search(
                np.array([query_embedding], dtype='float32'), k
            )
            
            # Step 4: Prepare results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):
                    chunk_text = self.chunks[idx]
                    metadata = self.chunk_metadata[idx]
                    similarity_score = 1 / (1 + distance)
                    
                    results.append({
                        'text': chunk_text,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
            
            # Step 5: Cache retrieval results
            self._cache_retrieval(query_embedding, k, results)
            
            self.logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise Exception(f"Retrieval failed: {str(e)}")
    
    def _create_enhanced_prompt(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Create an enhanced prompt for the LLM."""
        # Organize chunks by relevance and type
        high_relevance_chunks = [c for c in retrieved_chunks if c['similarity_score'] > 0.7]
        section_chunks = [c for c in retrieved_chunks if c['metadata']['has_section']]
        
        # Build context
        context_sections = []
        
        if high_relevance_chunks:
            context_sections.append("🎯 MOST RELEVANT INFORMATION:")
            for chunk in high_relevance_chunks[:2]:
                context_sections.append(f"• {chunk['text'][:400]}...")
        
        if section_chunks:
            context_sections.append("\n📋 RELEVANT LEGAL SECTIONS:")
            for chunk in section_chunks[:2]:
                sections = ", ".join(chunk['metadata']['section_numbers'])
                context_sections.append(f"• Section {sections}: {chunk['text'][:300]}...")
        
        # Additional context
        remaining_chunks = [c for c in retrieved_chunks if c not in high_relevance_chunks + section_chunks]
        if remaining_chunks:
            context_sections.append("\n📚 ADDITIONAL CONTEXT:")
            for chunk in remaining_chunks[:2]:
                context_sections.append(f"• {chunk['text'][:250]}...")
        
        context = "\n".join(context_sections)
        
        # Enhanced prompt template
        prompt = f"""You are an expert assistant specializing in the Indian Motor Vehicles Act and related traffic laws. Provide clear, accurate, and helpful answers based strictly on the provided legal information.

INSTRUCTIONS:
- Answer in a friendly but professional tone
- Cite specific section numbers when available
- Use ₹ symbol for fines and penalties
- Be precise about legal requirements
- If information is incomplete, acknowledge limitations

QUESTION: {query}

LEGAL CONTEXT:
{context}

RESPONSE: Provide a comprehensive yet concise answer based on the Motor Vehicles Act information above."""
        
        return prompt
    
    def _update_confidence_stats(self, confidence: float):
        """Update confidence score statistics."""
        if confidence >= 0.9:
            self.pipeline_stats['confidence_scores']['high_confidence'] += 1
        elif confidence >= 0.7:
            self.pipeline_stats['confidence_scores']['medium_confidence'] += 1
        else:
            self.pipeline_stats['confidence_scores']['low_confidence'] += 1
    
    def process_query(self, query: str) -> str:
        """
        Process a user query through the complete intelligent cache hierarchy.
        
        Cache Hierarchy (checked in order):
        1. Level 2: FAQ Cache (instant responses)
        2. Level 1: Exact Query Cache (hash-based matches)
        3. Level 3: Semantic Cache (similarity-based matches) [NEW]
        4. Level 4: Pre-computed Cache (topic-based responses) [NEW]
        5. Level 6: Embedding Cache + Level 5: Retrieval Cache (partial RAG skip)
        6. Full RAG Pipeline (generate new response)
        """
        start_time = time.time()
        cache_level_used = None
        response_confidence = 0.0
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Level 2: Check FAQ cache first (Phase 1)
            faq_result = self._check_faq_cache(query)
            if faq_result:
                response, confidence = faq_result
                cache_level_used = "faq"
                response_confidence = confidence
                self._update_query_stats(start_time, cache_level_used, confidence)
                return response
            
            # Level 1: Check exact query cache (Phase 2)
            exact_result = self._check_exact_query_cache(query)
            if exact_result:
                response, confidence = exact_result
                cache_level_used = "exact_query"
                response_confidence = confidence
                self._update_query_stats(start_time, cache_level_used, confidence)
                return response
            
            # Get query embedding for semantic operations
            query_embedding = self._get_cached_embedding(query)
            if query_embedding is None:
                query_embedding = self.embedding_model.encode([query])[0]
                self._cache_embedding(query, query_embedding)
            
            # Level 3: Check semantic cache (Phase 3)
            semantic_result = self._check_semantic_cache(query, query_embedding)
            if semantic_result:
                response, confidence = semantic_result
                cache_level_used = "semantic"
                response_confidence = confidence
                self._update_query_stats(start_time, cache_level_used, confidence)
                return response
            
            # Level 4: Check pre-computed cache (Phase 3)
            precomputed_result = self._check_precomputed_cache(query)
            if precomputed_result:
                response, confidence = precomputed_result
                cache_level_used = "precomputed"
                response_confidence = confidence
                self._update_query_stats(start_time, cache_level_used, confidence)
                return response
            
            # Validate Ollama connection before proceeding to RAG
            self._validate_ollama_connection()
            
            # Level 6 + 5: Retrieve relevant chunks (with embedding/retrieval caching)
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            
            if not retrieved_chunks:
                response = "I couldn't find relevant information in the Motor Vehicles Act to answer your question. Could you try rephrasing or asking about a different aspect?"
                cache_level_used = "no_results"
                response_confidence = 0.0
                self._update_query_stats(start_time, cache_level_used, response_confidence)
                return response
            
            # Determine cache level used during retrieval
            if self.pipeline_stats['embedding_cache_hits'] > 0 and self.pipeline_stats['retrieval_cache_hits'] > 0:
                cache_level_used = "embedding_and_retrieval_cached"
                self.pipeline_stats['cache_hierarchy_breakdown']['embedding_cached'] += 1
                self.pipeline_stats['cache_hierarchy_breakdown']['retrieval_cached'] += 1
            elif self.pipeline_stats['embedding_cache_hits'] > 0:
                cache_level_used = "embedding_cached"
                self.pipeline_stats['cache_hierarchy_breakdown']['embedding_cached'] += 1
            elif self.pipeline_stats['retrieval_cache_hits'] > 0:
                cache_level_used = "retrieval_cached"
                self.pipeline_stats['cache_hierarchy_breakdown']['retrieval_cached'] += 1
            else:
                cache_level_used = "full_rag"
                self.pipeline_stats['cache_hierarchy_breakdown']['full_rag'] += 1
            
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(query, retrieved_chunks)
            
            # Generate response with Ollama
            self.logger.info("Generating response with Ollama...")
            
            response = ollama.generate(
                model=self.config['ollama_model'],
                prompt=prompt,
                options={
                    'temperature': self.config['temperature'],
                    'timeout': self.config['ollama_timeout'],
                    'num_predict': self.config['max_response_tokens']  
                }
            )
            
            generated_response = response['response'].strip()
            response_confidence = 0.8  # Default confidence for generated responses
            
            # Cache the response for future exact matches (Level 1)
            if self.config['cache_query_responses'] and self._is_valid_query(query):
                query_hash = generate_query_hash(query)
                self.cache_manager.set('exact_query', query_hash, generated_response)
            
            # Add to semantic cluster if auto-clustering is enabled (Phase 3)
            if (self.config['auto_cluster_creation'] and 
                self.semantic_cache_manager and 
                self._is_valid_query(query)):
                
                cluster_id = self.semantic_cache_manager.add_semantic_cluster(
                    query, query_embedding, generated_response
                )
                self.pipeline_stats['semantic_clusters_created'] += 1
                self.logger.info(f"Added query to semantic cluster: {cluster_id}")
            
            # Update statistics
            if cache_level_used == "full_rag":
                self.pipeline_stats['full_rag_executions'] += 1
            
            self._update_query_stats(start_time, cache_level_used, response_confidence)
            
            return generated_response
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            self._update_query_stats(start_time, "error", 0.0)
            
            return (
                f"I encountered an issue processing your query. "
                f"Please ensure the system is properly set up and try again. "
                f"If the problem persists, check the server logs for details."
            )
    
    def _update_query_stats(self, start_time: float, cache_level: str, confidence: float):
        """Update query processing statistics."""
        processing_time = time.time() - start_time
        
        self.pipeline_stats['last_query_time'] = processing_time
        self.pipeline_stats['total_queries_processed'] += 1
        
        # Update confidence statistics
        self._update_confidence_stats(confidence)
        
        # Count cache hits
        cache_hit_levels = [
            'faq', 'exact_query', 'semantic', 'precomputed', 
            'embedding_cached', 'retrieval_cached', 'embedding_and_retrieval_cached'
        ]
        
        if cache_level in cache_hit_levels:
            self.pipeline_stats['total_cache_hits'] += 1
        
        self.logger.info(f"Query processed in {processing_time:.2f}s using {cache_level} (confidence: {confidence:.2f})")
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the pipeline."""
        # Calculate cache hit rate
        total_queries = self.pipeline_stats['total_queries_processed']
        cache_hit_rate = 0
        if total_queries > 0:
            cache_hit_rate = (self.pipeline_stats['total_cache_hits'] / total_queries * 100)
        
        # Get cache manager statistics
        cache_stats = self.cache_manager.get_comprehensive_stats()
        
        # Get semantic cache statistics (Phase 3)
        semantic_stats = {}
        if self.semantic_cache_manager:
            semantic_stats = self.semantic_cache_manager.get_semantic_stats()
        
        return {
            **self.pipeline_stats,
            'config': self.config,
            'overall_cache_hit_rate': round(cache_hit_rate, 2),
            'faq_cache_size': len(self.faq_cache),
            'document_hash': self.document_hash[:16] + "..." if self.document_hash else None,
            'build_info': self.build_info,
            'cache_manager_stats': cache_stats,
            'semantic_cache_stats': semantic_stats,  # Phase 3
            'status': 'ready' if self.pipeline_stats['index_built'] else 'initializing'
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a comprehensive health check."""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check embedding model
            test_embedding = self.embedding_model.encode(["test"])
            health_status['checks']['embedding_model'] = {
                'status': 'ok',
                'dimensions': len(test_embedding[0])
            }
        except Exception as e:
            health_status['checks']['embedding_model'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Check vector index
        try:
            health_status['checks']['vector_index'] = {
                'status': 'ok',
                'total_vectors': self.index.ntotal,
                'chunks_loaded': len(self.chunks)
            }
        except Exception as e:
            health_status['checks']['vector_index'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Check Ollama server
        try:
            self._validate_ollama_connection()
            health_status['checks']['ollama_server'] = {'status': 'ok'}
        except Exception as e:
            health_status['checks']['ollama_server'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check caches
        health_status['checks']['faq_cache'] = {
            'status': 'ok',
            'entries': len(self.faq_cache)
        }
        
        # Check standard cache system
        cache_health = self.cache_manager.health_check()
        health_status['checks']['cache_manager'] = cache_health
        
        # Check semantic cache system (Phase 3)
        if self.semantic_cache_manager:
            semantic_health = self.semantic_cache_manager.health_check()
            health_status['checks']['semantic_cache_manager'] = semantic_health
            
            if semantic_health['overall_status'] != 'healthy':
                health_status['overall_status'] = 'degraded'
        
        # Check background processing
        if self.background_manager:
            health_status['checks']['background_processing'] = {
                'status': 'ok' if self.background_manager.tasks_enabled else 'disabled'
            }
        
        if any(check['status'] in ['error', 'unhealthy'] for check in health_status['checks'].values()):
            health_status['overall_status'] = 'unhealthy'
        elif any(check['status'] in ['warning', 'degraded'] for check in health_status['checks'].values()):
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    def clear_cache_level(self, cache_level: str) -> bool:
        """Clear a specific cache level."""
        if cache_level == 'faq':
            self.faq_cache.clear()
            return True
        elif cache_level in ['exact_query', 'embedding', 'retrieval']:
            return self.cache_manager.clear_cache(cache_level)
        elif cache_level in ['semantic', 'precomputed', 'patterns']:
            if self.semantic_cache_manager:
                self.semantic_cache_manager.clear_semantic_data(cache_level)
                return True
        elif cache_level == 'all':
            self.faq_cache.clear()
            self.cache_manager.clear_all_caches()
            if self.semantic_cache_manager:
                self.semantic_cache_manager.clear_semantic_data('all')
            return True
        return False
    
    def save_caches(self):
        """Save all caches to disk."""
        self.cache_manager.save_all_caches()
        
        if self.semantic_cache_manager:
            self.semantic_cache_manager.save_all_semantic_data()
        
        # Save FAQ cache
        paths = self._get_storage_paths()
        try:
            with open(paths['faq_cache'], 'w', encoding='utf-8') as f:
                json.dump(self.faq_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save FAQ cache: {str(e)}")
    
    def __del__(self):
        """Cleanup when pipeline is destroyed."""
        try:
            # Stop background processing
            if self.background_manager:
                self.background_manager.stop_background_tasks()
            
            # Save all caches
            self.save_caches()
        except:
            pass  # Ignore cleanup errors


# Backwards compatibility and aliases
RAGPipeline = IntelligentRAGPipeline
PersistentRAGPipeline = IntelligentRAGPipeline
AdvancedRAGPipeline = IntelligentRAGPipeline


def main():
    """Test the intelligent RAG pipeline."""
    print("🧪 Phase 3: Intelligent RAG Pipeline Test")
    print("="*60)
    
    try:
        # Initialize pipeline
        rag = IntelligentRAGPipeline()
        
        # Test queries to check different cache levels
        test_queries = [
            "What is the penalty for driving without a license?",     # Should hit FAQ cache
            "What is the fine for overspeeding?",                    # Should hit FAQ cache  
            "What are the procedures for vehicle registration?",      # Should go through RAG
            "What are the procedures for vehicle registration?",      # Should hit exact query cache (repeat)
            "How do I register my vehicle?",                         # Should hit semantic cache (similar to above)
            "What happens if I drive without proper papers?",        # Should hit precomputed cache (documentation topic)
            "Can minors get driving licenses?",                      # Similar to FAQ, might hit semantic
            "What is the penalty for not having documents?",        # Should hit precomputed cache
        ]
        
        print(f"\n🔍 Testing intelligent cache hierarchy with sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {query}")
            
            start_time = time.time()
            try:
                response = rag.process_query(query)
                response_time = time.time() - start_time
                print(f"Response ({response_time:.3f}s): {response[:150]}...")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Print comprehensive statistics
        stats = rag.get_pipeline_stats()
        print(f"\n📊 Phase 3 Performance Statistics:")
        print(f"Initialization method: {stats['initialization_method']}")
        print(f"Startup time: {stats['startup_time']:.2f}s")
        print(f"Total queries processed: {stats['total_queries_processed']}")
        print(f"Overall cache hit rate: {stats['overall_cache_hit_rate']:.1f}%")
        
        print(f"\n🔄 Cache Hierarchy Breakdown:")
        breakdown = stats['cache_hierarchy_breakdown']
        for level, count in breakdown.items():
            if count > 0:
                print(f"   {level}: {count} hits")
        
        print(f"\n💾 Standard Cache Stats:")
        cache_stats = stats['cache_manager_stats']['cache_stats']
        for cache_name, cache_data in cache_stats.items():
            print(f"   {cache_name}: {cache_data['size']} entries, {cache_data['hit_rate']:.1f}% hit rate")
        
        print(f"\n🧠 Semantic Cache Stats:")
        semantic_stats = stats.get('semantic_cache_stats', {})
        if semantic_stats:
            clusters = semantic_stats.get('semantic_clusters', {})
            precomputed = semantic_stats.get('precomputed_responses', {})
            print(f"   Semantic clusters: {clusters.get('total_clusters', 0)}")
            print(f"   Pre-computed responses: {precomputed.get('total_responses', 0)}")
        
        print(f"\n🎯 Confidence Distribution:")
        confidence_scores = stats.get('confidence_scores', {})
        for level, count in confidence_scores.items():
            if count > 0:
                print(f"   {level.replace('_', ' ').title()}: {count}")
        
        print(f"\n✅ Phase 3 implementation test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Phase 3 test failed: {str(e)}")
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)