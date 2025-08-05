# backend/app.py
"""
Phase 4: Flask App with Integrated Explainable AI
=================================================

This application integrates the Phase 3 IntelligentRAGPipeline with the
Phase 4 Explainable AI module, providing a comprehensive and transparent
AI system through a unified API.

New Features (Phase 4 Integration):
- The primary `/query` endpoint now returns a detailed explanation by default.
- Endpoint `/query/explained` remains for compatibility but is now redundant.
- The core RAG pipeline is wrapped by the explainability module at startup.
- All previous Phase 2/3 features (caching, stats, health checks) remain active.
"""

from flask import Flask, request, jsonify, send_from_directory
import json
import os
import subprocess
import time
import sys
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# PHASE 4: Import the explainability wrapper, route-adder, and Enum
from explainable_ai import RAGPipelineWithExplanations, add_explanation_routes, ExplanationLevel

app = Flask(__name__, static_folder="../static", static_url_path="/static")

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO, # Changed to INFO for cleaner production logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def start_ollama_server():
    """Start Ollama server if not already running."""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            app.logger.info("Ollama server is already running.")
            return None
        
        # Start Ollama server based on platform
        if sys.platform.startswith('win'):
            process = subprocess.Popen(
                ['start', '/B', 'ollama', 'serve'], 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        else:
            process = subprocess.Popen(
                ['nohup', 'ollama', 'serve'], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
        
        # Wait for server to start
        time.sleep(5)
        
        # Verify server is running
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            app.logger.info("Ollama server started successfully.")
            return process
        else:
            app.logger.error("Failed to start Ollama server.")
            return None
            
    except Exception as e:
        app.logger.error(f"Error starting Ollama server: {str(e)}")
        return None

def initialize_rag_pipeline():
    """Initialize the RAG pipeline and wrap it with the explainability module."""
    try:
        app.logger.info("Initializing Intelligent RAG Pipeline (Phase 3)...")
        start_time = time.time()
        
        # Import the intelligent pipeline
        from rag_pipeline import IntelligentRAGPipeline
        
        # Initialize with optimized configuration for Phase 3
        config = {
            'ollama_model': 'tinyllama',
            'ollama_timeout': 30,
            'temperature': 0.3,
            'max_response_tokens': 500,
            'chunk_size': 600,
            'chunk_overlap': 100,
            'max_retrieval_chunks': 5,
            
            # Phase 2 Caching
            'enable_exact_query_cache': True,
            'enable_embedding_cache': True,
            'enable_retrieval_cache': True,
            'cache_query_responses': True,
            'min_query_length': 3,
            'max_cache_query_length': 500,

            # Phase 3 Caching
            'enable_semantic_cache': True,
            'enable_precomputed_cache': True,
            'semantic_similarity_threshold': 0.85,
            'confidence_threshold': 0.7,
            'enable_background_tasks': True,
            'auto_cluster_creation': True,
        }
        
        base_rag_pipeline = IntelligentRAGPipeline(config=config)
        
        # PHASE 4: Wrap the base pipeline with the explainability module
        app.logger.info("Wrapping pipeline with Explainable AI module (Phase 4)...")
        explainable_rag = RAGPipelineWithExplanations(base_rag_pipeline)
        
        initialization_time = time.time() - start_time
        app.logger.info(f"Intelligent & Explainable RAG pipeline initialized in {initialization_time:.2f} seconds")
        app.logger.info(f"Initialization method: {explainable_rag.pipeline_stats['initialization_method']}")
        
        return explainable_rag
        
    except Exception as e:
        app.logger.error(f"Failed to initialize RAG pipeline: {str(e)}", exc_info=True)
        return None

def start_background_tasks(rag_pipeline):
    """Start background maintenance tasks for cache management."""
    if not rag_pipeline:
        return
    
    if rag_pipeline.config.get('enable_background_tasks'):
        app.logger.info("Background tasks are managed by the RAG pipeline.")
    else:
        app.logger.info("Background tasks are disabled by pipeline configuration.")


# --- Global variables & Application Startup ---
print("🚀 Starting Motor Vehicles Act Chatbot with Explainable AI (Phase 4)")
print("=" * 70)

# Step 1: Start Ollama server
print("📡 Starting Ollama server...")
ollama_process = start_ollama_server()

# Step 2: Initialize RAG pipeline with explainability
print("🧠 Initializing RAG pipeline with Explainability...")
rag = initialize_rag_pipeline()

if rag is None:
    print("❌ Failed to initialize RAG pipeline. Server will run in limited mode.")
else:
    print("✅ RAG pipeline with Explainable AI initialized successfully.")
    stats = rag.get_pipeline_stats()
    print(f"   Startup time: {stats['startup_time']:.2f}s")
    print(f"   Method: {stats['initialization_method']}")
    print(f"   Chunks: {stats['total_chunks']}")
    
    # Start background tasks if enabled in config
    start_background_tasks(rag)

# PHASE 4: Add the explanation-specific API routes to the Flask app
if rag:
    print("✅ Adding Explainable AI API routes...")
    add_explanation_routes(app, rag)
    print("   Routes /query/explained, /explanation/*, /explanations/* are now available.")
else:
    print("⚠️ Could not add Explainable AI routes because pipeline initialization failed.")

print("=" * 70)


# --- API Endpoints ---

@app.route('/')
def serve_index():
    """Serve the main HTML interface."""
    app.logger.info("Serving index.html")
    return send_from_directory('../static', 'index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handle user queries. This endpoint now ALWAYS returns the response along
    with a detailed explanation, making explainability a default feature.
    """
    try:
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            app.logger.warning("Empty query received")
            return jsonify({'response': 'Please ask a question!', 'source': 'validation_error'}), 400
        
        app.logger.info(f"Processing query via /query endpoint (with default explanation): {query[:100]}...")
        
        if rag is None:
            app.logger.error("RAG pipeline is not initialized")
            return jsonify({'response': "The answering system is currently unavailable.", 'source': 'system_error'}), 503
        
        # Always call the method that includes the explanation.
        # Defaulting to a detailed explanation for maximum transparency.
        result = rag.process_query_with_explanation(
            query, 
            explanation_level=ExplanationLevel.DETAILED,
            include_explanation=True
        )
        
        app.logger.info(f"Query processed in {result.get('processing_time_ms', 0):.1f}ms with explanation.")
        
        # The 'result' dictionary already contains the response, explanation, etc.
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'response': f"An error occurred: {str(e)}", 'source': 'processing_error'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Comprehensive health check. The call is delegated to the underlying pipeline.
    """
    try:
        app.logger.info("Health check requested")
        if rag is None:
            return jsonify({'overall_status': 'degraded', 'error': 'RAG pipeline not initialized'}), 503
        
        rag_health = rag.health_check()
        status_code = 200 if rag_health.get('overall_status') == 'healthy' else 503
        return jsonify(rag_health), status_code
        
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return jsonify({'overall_status': 'error', 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """
    Get detailed pipeline and cache statistics for monitoring.
    """
    try:
        if rag is None:
            return jsonify({'error': 'RAG pipeline not available'}), 503
        
        stats = rag.get_pipeline_stats()
        return jsonify(stats)
        
    except Exception as e:
        app.logger.error(f"Error getting statistics: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    if rag is None:
        return jsonify({
            'error': 'Endpoint not found or Pipeline not initialized',
            'message': 'The requested endpoint does not exist or the RAG pipeline failed to start.'
        }), 404
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.start_time = time.time()
    
    print("\n🌐 Server starting on http://localhost:5001")
    print("📊 Available endpoints:")
    print("   GET  /                  - Web interface")
    print("   POST /query              - Process queries (now WITH default explanation)")
    print("   GET  /health            - Health check")
    print("   GET  /stats             - Detailed statistics")
    print("\n💡 Phase 4: Explainable AI Endpoints:")
    print("   POST /query/explained    - Get response with full explanation (now redundant)")
    print("   GET  /explanation/<id>  - Retrieve a specific explanation")
    print("   GET  /explanations/stats  - Get statistics on explanation generation")
    print("   POST /explanations/export - Export explanation history")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5001)
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
    finally:
        print("\n👋 Server stopped")