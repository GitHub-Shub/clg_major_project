# backend/rag_pipeline.py
"""
Enhanced RAG Pipeline for Legal Document Processing
==================================================

This script creates a high-quality Retrieval-Augmented Generation (RAG) system
specifically optimized for legal documents like the Motor Vehicles Act.

Features:
- Intelligent legal document chunking
- Advanced embedding and indexing
- Quality validation and statistics
- Comprehensive error handling
- Beginner-friendly progress tracking
- Optimized retrieval strategies
- Professional response generation
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
from pathlib import Path
from datetime import datetime
import re


class EnhancedRAGPipeline:
    """
    A comprehensive RAG pipeline specifically designed for legal documents.
    
    This class handles document loading, intelligent chunking, embedding generation,
    vector indexing, and query processing with a focus on legal document structure
    and high-quality responses.
    """
    
    def __init__(self, config=None):
        """
        Initialize the RAG pipeline with customizable configuration.
        
        Args:
            config (dict): Configuration options for the pipeline
        """
        # Default configuration optimized for legal documents
        self.config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'chunk_size': 600,  # Slightly larger for legal context
            'chunk_overlap': 100,  # More overlap for legal continuity
            'max_retrieval_chunks': 5,  # More context for complex legal queries
            'temperature': 0.3,  # Lower temperature for more factual responses
            'max_response_tokens': 500,
            'ollama_model': 'tinyllama',
            'ollama_timeout': 45,
            'index_type': 'L2'  # L2 distance for semantic similarity
        }
        
        # Update with user configuration
        if config:
            self.config.update(config)
        
        # Initialize components
        self.embedding_model = None
        self.text_splitter = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Statistics and quality metrics
        self.pipeline_stats = {
            'document_loaded': False,
            'total_chunks': 0,
            'avg_chunk_length': 0,
            'embedding_dimensions': 0,
            'index_built': False,
            'legal_sections_found': 0,
            'tables_found': 0,
            'initialization_time': 0,
            'last_query_time': 0,
            'total_queries_processed': 0
        }
        
        # Setup logging with detailed formatting
        self._setup_logging()
        
        # Initialize the pipeline
        self._initialize_pipeline()
    
    def _setup_logging(self):
        """Configure detailed logging for the RAG pipeline."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('RAGPipeline')
    
    def _initialize_pipeline(self):
        """Initialize all components of the RAG pipeline."""
        start_time = time.time()
        
        print("🚀 Initializing Enhanced RAG Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Load embedding model
            self._load_embedding_model()
            
            # Step 2: Setup text splitter
            self._setup_text_splitter()
            
            # Step 3: Initialize vector index
            self._initialize_vector_index()
            
            # Step 4: Load and process document
            self._load_and_process_document()
            
            # Step 5: Build vector index
            self._build_vector_index()
            
            # Record initialization time
            self.pipeline_stats['initialization_time'] = time.time() - start_time
            
            # Print success summary
            self._print_initialization_summary()
            
            self.logger.info("RAG Pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            print(f"\n❌ INITIALIZATION FAILED: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("   - Run 'python data_cleaner.py' first to create mv_act_cleaned.txt")
            print("   - Ensure you have internet connection for model downloads")
            print("   - Check if Ollama is installed and running")
            print("   - Verify you have sufficient memory (>2GB RAM recommended)")
            raise
    
    def _load_embedding_model(self):
        """Load and validate the sentence transformer model."""
        print("🧠 Loading embedding model...")
        
        try:
            model_name = self.config['embedding_model']
            print(f"   Model: {model_name}")
            
            # Load model with progress tracking
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
            # Configure splitter for legal documents
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap'],
                # Prioritize splitting at legal document boundaries
                separators=[
                    "\n\nSection ",  # Legal sections
                    "\n\nChapter ",  # Chapters
                    "\n\n",          # Paragraphs
                    "\n",            # Lines
                    ". ",            # Sentences
                    " "              # Words
                ],
                keep_separator=True  # Keep section headers with content
            )
            
            print(f"   Chunk size: {self.config['chunk_size']} characters")
            print(f"   Chunk overlap: {self.config['chunk_overlap']} characters")
            print(f"   ✅ Text splitter configured for legal documents")
            
        except Exception as e:
            print(f"   ❌ Failed to setup text splitter: {str(e)}")
            raise
    
    def _initialize_vector_index(self):
        """Initialize the FAISS vector index."""
        print("🔍 Initializing vector index...")
        
        try:
            embedding_dim = self.pipeline_stats['embedding_dimensions']
            
            if self.config['index_type'] == 'L2':
                self.index = faiss.IndexFlatL2(embedding_dim)
            else:
                # Could add other index types here (IVF, HNSW, etc.)
                self.index = faiss.IndexFlatL2(embedding_dim)
            
            print(f"   Index type: {self.config['index_type']}")
            print(f"   Dimensions: {embedding_dim}")
            print(f"   ✅ Vector index initialized")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize vector index: {str(e)}")
            raise
    
    def _load_and_process_document(self):
        """Load and process the cleaned legal document."""
        print("📖 Loading and processing document...")
        
        document_path = 'mv_act_cleaned.txt'
        
        try:
            # Validate document exists
            if not os.path.exists(document_path):
                raise FileNotFoundError(
                    f"Document not found: {document_path}\n"
                    "Please run 'python data_cleaner.py' first to create the cleaned document."
                )
            
            # Load document
            with open(document_path, 'r', encoding='utf-8') as f:
                document_text = f.read()
            
            if not document_text.strip():
                raise ValueError(
                    f"Document is empty: {document_path}\n"
                    "Please ensure extract_pdf.py and data_cleaner.py ran successfully."
                )
            
            print(f"   Document loaded: {len(document_text):,} characters")
            
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
            self.pipeline_stats['total_chunks'] = len(self.chunks)
            self.pipeline_stats['avg_chunk_length'] = sum(len(chunk) for chunk in self.chunks) / len(self.chunks)
            self.pipeline_stats['document_loaded'] = True
            
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
            print("   Generating embeddings (this may take a moment)...")
            
            # Generate embeddings with progress tracking
            embeddings = self.embedding_model.encode(
                self.chunks, 
                show_progress_bar=True,
                batch_size=32  # Process in batches for better memory management
            )
            
            print(f"   Generated {len(embeddings)} embeddings")
            
            # Add embeddings to index
            self.index.add(np.array(embeddings, dtype='float32'))
            
            self.pipeline_stats['index_built'] = True
            
            print(f"   ✅ Vector index built successfully")
            print(f"   Total vectors in index: {self.index.ntotal}")
            
        except Exception as e:
            print(f"   ❌ Failed to build vector index: {str(e)}")
            raise
    
    def _print_initialization_summary(self):
        """Print a comprehensive initialization summary."""
        print("\n" + "="*60)
        print("📊 RAG PIPELINE INITIALIZATION SUMMARY")
        print("="*60)
        print(f"Embedding model: {self.config['embedding_model']}")
        print(f"Embedding dimensions: {self.pipeline_stats['embedding_dimensions']}")
        print(f"Document chunks: {self.pipeline_stats['total_chunks']}")
        print(f"Average chunk length: {self.pipeline_stats['avg_chunk_length']:.0f} characters")
        print(f"Legal sections found: {self.pipeline_stats['legal_sections_found']}")
        print(f"Tables preserved: {self.pipeline_stats['tables_found']}")
        print(f"Vector index size: {self.index.ntotal}")
        print(f"Initialization time: {self.pipeline_stats['initialization_time']:.2f} seconds")
        print("✅ Pipeline ready for queries!")
        print("="*60)
    
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
    
    def retrieve_relevant_chunks(self, query, k=None):
        """
        Retrieve the most relevant chunks for a given query.
        
        Args:
            query (str): User query
            k (int): Number of chunks to retrieve (default from config)
            
        Returns:
            list: List of (chunk_text, metadata, score) tuples
        """
        if k is None:
            k = self.config['max_retrieval_chunks']
        
        try:
            self.logger.info(f"Retrieving chunks for query: {query[:50]}...")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Search the index
            distances, indices = self.index.search(
                np.array([query_embedding], dtype='float32'), k
            )
            
            # Prepare results with metadata and scores
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.chunks):  # Ensure valid index
                    chunk_text = self.chunks[idx]
                    metadata = self.chunk_metadata[idx]
                    similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    results.append({
                        'text': chunk_text,
                        'metadata': metadata,
                        'similarity_score': similarity_score,
                        'rank': i + 1
                    })
            
            self.logger.info(f"Retrieved {len(results)} relevant chunks")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve chunks: {str(e)}")
            raise Exception(f"Retrieval failed: {str(e)}")
    
    def _create_enhanced_prompt(self, query, retrieved_chunks):
        """
        Create an enhanced prompt for the LLM with better context organization.
        
        Args:
            query (str): User query
            retrieved_chunks (list): Retrieved chunks with metadata
            
        Returns:
            str: Enhanced prompt for LLM
        """
        # Organize chunks by relevance and type
        high_relevance_chunks = [c for c in retrieved_chunks if c['similarity_score'] > 0.7]
        section_chunks = [c for c in retrieved_chunks if c['metadata']['has_section']]
        penalty_chunks = [c for c in retrieved_chunks if c['metadata']['has_penalty']]
        
        # Build context sections
        context_sections = []
        
        # Primary context (highest relevance)
        if high_relevance_chunks:
            context_sections.append("🎯 MOST RELEVANT INFORMATION:")
            for chunk in high_relevance_chunks[:2]:
                context_sections.append(f"• {chunk['text'][:300]}...")
        
        # Legal sections context
        if section_chunks:
            context_sections.append("\n📋 RELEVANT LEGAL SECTIONS:")
            for chunk in section_chunks[:2]:
                sections = ", ".join(chunk['metadata']['section_numbers'])
                context_sections.append(f"• Section {sections}: {chunk['text'][:200]}...")
        
        # Additional context
        remaining_chunks = [c for c in retrieved_chunks if c not in high_relevance_chunks + section_chunks]
        if remaining_chunks:
            context_sections.append("\n📚 ADDITIONAL CONTEXT:")
            for chunk in remaining_chunks[:2]:
                context_sections.append(f"• {chunk['text'][:200]}...")
        
        context = "\n".join(context_sections)
        
        # Create the enhanced prompt
        prompt = f"""You are an expert assistant specializing in the Indian Motor Vehicles Act. Provide clear, accurate, and helpful answers based strictly on the provided legal information.

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
    
    def process_query(self, query):
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query (str): User question about the Motor Vehicles Act
            
        Returns:
            str: Generated response based on retrieved context
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Step 1: Validate Ollama connection
            self._validate_ollama_connection()
            
            # Step 2: Retrieve relevant chunks
            retrieved_chunks = self.retrieve_relevant_chunks(query)
            
            if not retrieved_chunks:
                return "I couldn't find relevant information in the Motor Vehicles Act to answer your question. Could you try rephrasing or asking about a different aspect?"
            
            # Step 3: Create enhanced prompt
            prompt = self._create_enhanced_prompt(query, retrieved_chunks)
            
            # Step 4: Generate response
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
            
            # Step 5: Post-process response
            generated_response = response['response'].strip()
            
            # Update statistics
            self.pipeline_stats['last_query_time'] = time.time() - start_time
            self.pipeline_stats['total_queries_processed'] += 1
            
            self.logger.info(f"Query processed successfully in {self.pipeline_stats['last_query_time']:.2f}s")
            
            return generated_response
            
        except Exception as e:
            error_msg = f"Query processing failed: {str(e)}"
            self.logger.error(error_msg)
            
            # Return helpful error message to user
            return (
                f"I encountered an issue processing your query. "
                f"Please ensure the system is properly set up and try again. "
                f"If the problem persists, check the server logs for details."
            )
    
    def get_pipeline_stats(self):
        """
        Get comprehensive statistics about the RAG pipeline.
        
        Returns:
            dict: Pipeline statistics and metrics
        """
        return {
            **self.pipeline_stats,
            'config': self.config,
            'status': 'ready' if self.pipeline_stats['index_built'] else 'initializing'
        }
    
    def health_check(self):
        """
        Perform a comprehensive health check of the pipeline.
        
        Returns:
            dict: Health check results
        """
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Check 1: Embedding model
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
        
        # Check 2: Vector index
        try:
            health_status['checks']['vector_index'] = {
                'status': 'ok',
                'total_vectors': self.index.ntotal,
                'index_built': self.pipeline_stats['index_built']
            }
        except Exception as e:
            health_status['checks']['vector_index'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'unhealthy'
        
        # Check 3: Ollama server
        try:
            self._validate_ollama_connection()
            health_status['checks']['ollama_server'] = {'status': 'ok'}
        except Exception as e:
            health_status['checks']['ollama_server'] = {
                'status': 'error',
                'error': str(e)
            }
            health_status['overall_status'] = 'degraded'
        
        # Check 4: Document chunks
        health_status['checks']['document_chunks'] = {
            'status': 'ok' if self.chunks else 'error',
            'total_chunks': len(self.chunks),
            'avg_chunk_length': self.pipeline_stats['avg_chunk_length']
        }
        
        return health_status


def main():
    """
    Main function for testing and demonstrating the RAG pipeline.
    """
    print("🧪 Enhanced RAG Pipeline Test")
    print("="*50)
    
    try:
        # Initialize pipeline
        rag = EnhancedRAGPipeline()
        
        # Test queries
        test_queries = [
            "What is the penalty for driving without a license?",
            "What is the golden hour in the MV Act?",
            "Can a minor obtain a driving license?"
        ]
        
        print(f"\n🔍 Testing with sample queries...")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Test Query {i} ---")
            print(f"Question: {query}")
            
            try:
                response = rag.process_query(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {str(e)}")
        
        # Print final statistics
        stats = rag.get_pipeline_stats()
        print(f"\n📊 Final Statistics:")
        print(f"Total queries processed: {stats['total_queries_processed']}")
        print(f"Average query time: {stats['last_query_time']:.2f}s")
        
        print(f"\n✅ RAG Pipeline test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ RAG Pipeline test failed: {str(e)}")
        return False
    
    return True


# Backwards compatibility - create an alias for the old class name
RAGPipeline = EnhancedRAGPipeline


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)