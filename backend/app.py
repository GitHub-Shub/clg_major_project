# backend/app.py
"""
Phase 2: Enhanced Flask App with Advanced Multi-Level Caching
============================================================

This updated version integrates with the new AdvancedRAGPipeline to provide:
- Multi-level cache hierarchy (FAQ, Exact Query, Embedding, Retrieval)
- Advanced cache management and monitoring
- Comprehensive performance analytics
- Cache warming and maintenance endpoints

Phase 2 Features:
- 70%+ cache hit rates through intelligent cache hierarchy
- Sub-100ms responses for cached queries
- Advanced cache analytics and monitoring
- Administrative cache management controls
- Memory usage optimization and monitoring
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

app = Flask(__name__, static_folder="../static", static_url_path="/static")

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG, 
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
    """Initialize the advanced RAG pipeline with Phase 2 features."""
    try:
        app.logger.info("Initializing Advanced RAG Pipeline (Phase 2)...")
        start_time = time.time()
        
        # Import the advanced pipeline
        from rag_pipeline import AdvancedRAGPipeline
        
        # Initialize with optimized configuration for Phase 2
        config = {
            'ollama_model': 'tinyllama',
            'ollama_timeout': 30,
            'temperature': 0.3,
            'max_response_tokens': 500,
            'chunk_size': 600,
            'chunk_overlap': 100,
            'max_retrieval_chunks': 5,
            
            # Phase 2: Advanced caching configuration
            'enable_exact_query_cache': True,
            'enable_embedding_cache': True,
            'enable_retrieval_cache': True,
            'cache_query_responses': True,
            'min_query_length': 3,
            'max_cache_query_length': 500,
        }
        
        rag = AdvancedRAGPipeline(config=config)
        
        initialization_time = time.time() - start_time
        app.logger.info(f"Advanced RAG pipeline initialized in {initialization_time:.2f} seconds")
        app.logger.info(f"Initialization method: {rag.pipeline_stats['initialization_method']}")
        app.logger.info(f"FAQ cache size: {len(rag.faq_cache)}")
        
        # Get cache statistics
        stats = rag.get_pipeline_stats()
        cache_stats = stats.get('cache_manager_stats', {}).get('cache_stats', {})
        
        app.logger.info("Advanced cache levels initialized:")
        for cache_name, cache_data in cache_stats.items():
            app.logger.info(f"  {cache_name}: {cache_data['size']} entries")
        
        return rag
        
    except Exception as e:
        app.logger.error(f"Failed to initialize Advanced RAG pipeline: {str(e)}")
        return None

def start_background_tasks(rag_pipeline):
    """Start background maintenance tasks for cache management."""
    if not rag_pipeline:
        return
    
    def cache_maintenance():
        """Periodic cache maintenance task."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                
                # Cleanup expired cache entries
                rag_pipeline.cache_manager.cleanup_expired()
                
                # Save caches periodically
                rag_pipeline.save_caches()
                
                app.logger.debug("Cache maintenance completed")
                
            except Exception as e:
                app.logger.error(f"Cache maintenance error: {str(e)}")
    
    # Start maintenance thread
    maintenance_thread = threading.Thread(target=cache_maintenance, daemon=True)
    maintenance_thread.start()
    app.logger.info("Background cache maintenance started")

# Global variables
ollama_process = None
rag = None

# Application startup
print("🚀 Starting Enhanced Motor Vehicles Act Chatbot (Phase 2)")
print("=" * 70)

# Step 1: Start Ollama server
print("📡 Starting Ollama server...")
ollama_process = start_ollama_server()

# Step 2: Initialize Advanced RAG pipeline
print("🧠 Initializing Advanced RAG pipeline...")
rag = initialize_rag_pipeline()

if rag is None:
    print("❌ Failed to initialize Advanced RAG pipeline")
    print("⚠️  Server will run in limited mode")
else:
    print("✅ Advanced RAG pipeline initialized successfully")
    stats = rag.get_pipeline_stats()
    print(f"   Startup time: {stats['startup_time']:.2f}s")
    print(f"   Method: {stats['initialization_method']}")
    print(f"   Chunks: {stats['total_chunks']}")
    print(f"   FAQ entries: {stats['faq_cache_size']}")
    
    # Display cache information
    cache_stats = stats.get('cache_manager_stats', {}).get('cache_stats', {})
    print(f"   Advanced caches:")
    for cache_name, cache_data in cache_stats.items():
        print(f"     • {cache_name}: {cache_data['size']} entries")
    
    # Start background tasks
    start_background_tasks(rag)

print("=" * 70)

@app.route('/')
def serve_index():
    """Serve the main HTML interface."""
    app.logger.info("Serving index.html")
    return send_from_directory('../static', 'index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    """
    Handle user queries with advanced multi-level caching.
    """
    try:
        # Parse request
        data = request.json
        query = data.get('query', '').strip()
        
        if not query:
            app.logger.warning("Empty query received")
            return jsonify({
                'response': 'Please ask a question about the Motor Vehicles Act!',
                'source': 'validation_error',
                'cache_level': 'none'
            }), 400
        
        app.logger.info(f"Processing query: {query[:100]}...")
        start_time = time.time()
        
        # Check if RAG pipeline is available
        if rag is None:
            app.logger.error("Advanced RAG pipeline is not initialized")
            return jsonify({
                'response': (
                    "The advanced answering system is currently unavailable. "
                    "Please try a simple question or contact the administrator."
                ),
                'source': 'system_error',
                'cache_level': 'none'
            }), 503
        
        # Process query through advanced RAG pipeline
        response = rag.process_query(query)
        processing_time = time.time() - start_time
        
        # Get updated statistics
        stats = rag.get_pipeline_stats()
        
        # Determine which cache level was used (simplified detection)
        cache_level = 'full_rag'  # Default assumption
        if processing_time < 0.1:
            if stats['faq_cache_hits'] > 0:
                cache_level = 'faq'
            else:
                cache_level = 'exact_query'
        elif processing_time < 0.5:
            cache_level = 'cached_retrieval'
        
        app.logger.info(f"Query processed in {processing_time:.3f}s using {cache_level}")
        app.logger.info(f"Overall cache hit rate: {stats.get('overall_cache_hit_rate', 0):.1f}%")
        
        return jsonify({
            'response': response,
            'processing_time': round(processing_time, 3),
            'cache_level': cache_level,
            'cache_hit_rate': round(stats.get('overall_cache_hit_rate', 0), 1),
            'source': 'advanced_rag_pipeline'
        })
        
    except Exception as e:
        app.logger.error(f"Error processing query: {str(e)}")
        return jsonify({
            'response': (
                f"I encountered an issue processing your query: {str(e)}. "
                f"Please try again or contact the administrator if the problem persists."
            ),
            'source': 'processing_error',
            'cache_level': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Comprehensive health check endpoint with advanced cache monitoring.
    """
    try:
        app.logger.info("Health check requested")
        
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'server_status': 'running',
            'uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0,
            'phase': 2
        }
        
        if rag is not None:
            # Get comprehensive RAG pipeline health
            rag_health = rag.health_check()
            health_data['rag_pipeline'] = rag_health
            
            # Get detailed pipeline statistics
            stats = rag.get_pipeline_stats()
            
            # Basic statistics
            health_data['statistics'] = {
                'total_queries': stats['total_queries_processed'],
                'overall_cache_hit_rate': round(stats.get('overall_cache_hit_rate', 0), 1),
                'avg_response_time': round(stats.get('last_query_time', 0), 3),
                'initialization_method': stats['initialization_method'],
                'startup_time': round(stats['startup_time'], 2)
            }
            
            # Advanced cache statistics (Phase 2)
            cache_breakdown = stats.get('cache_hierarchy_breakdown', {})
            health_data['cache_hierarchy'] = cache_breakdown
            
            # Cache manager statistics
            cache_manager_stats = stats.get('cache_manager_stats', {})
            health_data['cache_levels'] = {}
            
            if 'cache_stats' in cache_manager_stats:
                for cache_name, cache_data in cache_manager_stats['cache_stats'].items():
                    health_data['cache_levels'][cache_name] = {
                        'size': cache_data['size'],
                        'hit_rate': cache_data['hit_rate'],
                        'total_requests': cache_data['total_requests']
                    }
            
            # Memory usage information
            if 'memory_usage' in cache_manager_stats:
                health_data['memory_usage'] = cache_manager_stats['memory_usage']
            
            # Determine overall health status
            overall_status = rag_health.get('overall_status', 'unknown')
            health_data['overall_status'] = overall_status
            
            status_code = 200 if overall_status == 'healthy' else 503
            
        else:
            health_data['rag_pipeline'] = {
                'overall_status': 'unavailable',
                'error': 'Advanced RAG pipeline not initialized'
            }
            health_data['overall_status'] = 'degraded'
            status_code = 503
        
        return jsonify(health_data), status_code
        
    except Exception as e:
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'error',
            'error': str(e)
        }), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """
    Get detailed pipeline and cache statistics for monitoring.
    """
    try:
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        stats = rag.get_pipeline_stats()
        
        # Add server-level information
        server_stats = {
            'server_timestamp': datetime.now().isoformat(),
            'server_uptime': time.time() - app.start_time if hasattr(app, 'start_time') else 0,
            'phase': 2,
            'pipeline_stats': stats
        }
        
        return jsonify(server_stats)
        
    except Exception as e:
        app.logger.error(f"Error getting statistics: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """
    Get detailed cache statistics and performance metrics.
    """
    try:
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available'
            }), 503
        
        # Get comprehensive cache statistics
        stats = rag.get_pipeline_stats()
        cache_manager_stats = stats.get('cache_manager_stats', {})
        
        # Format for easy consumption
        cache_statistics = {
            'timestamp': datetime.now().isoformat(),
            'overall_cache_hit_rate': stats.get('overall_cache_hit_rate', 0),
            'total_queries_processed': stats.get('total_queries_processed', 0),
            'cache_hierarchy_breakdown': stats.get('cache_hierarchy_breakdown', {}),
            'cache_levels': {},
            'memory_usage': cache_manager_stats.get('memory_usage', {}),
            'global_stats': cache_manager_stats.get('global_stats', {})
        }
        
        # Detailed cache level statistics
        if 'cache_stats' in cache_manager_stats:
            for cache_name, cache_data in cache_manager_stats['cache_stats'].items():
                cache_statistics['cache_levels'][cache_name] = cache_data
        
        return jsonify(cache_statistics)
        
    except Exception as e:
        app.logger.error(f"Error getting cache statistics: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/cache/clear', methods=['POST'])  
def clear_cache():
    """
    Clear specific cache levels or all caches.
    """
    try:
        data = request.json or {}
        cache_type = data.get('cache_type', 'all')
        
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available'
            }), 503
        
        # Valid cache types for Phase 2
        valid_cache_types = ['all', 'faq', 'exact_query', 'embedding', 'retrieval']
        
        if cache_type not in valid_cache_types:
            return jsonify({
                'error': f'Invalid cache type: {cache_type}',
                'valid_types': valid_cache_types
            }), 400
        
        # Clear the specified cache
        success = rag.clear_cache_level(cache_type)
        
        if success:
            app.logger.info(f"Cache cleared: {cache_type}")
            return jsonify({
                'message': f'Successfully cleared {cache_type} cache',
                'cache_type': cache_type,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': f'Failed to clear {cache_type} cache',
                'cache_type': cache_type
            }), 500
            
    except Exception as e:
        app.logger.error(f"Error clearing cache: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/cache/save', methods=['POST'])
def save_caches():
    """
    Manually trigger cache saving to disk.
    """
    try:
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available'
            }), 503
        
        app.logger.info("Manual cache save requested")
        rag.save_caches()
        
        return jsonify({
            'message': 'All caches saved successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error saving caches: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/cache/warmup', methods=['POST'])
def warm_cache():
    """
    Warm up caches with common queries (for admin use).
    """
    try:
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available'
            }), 503
        
        # Common queries to warm up the cache
        warmup_queries = [
            "What is the penalty for driving without a license?",
            "What is the fine for not wearing a helmet?",
            "What is the punishment for overspeeding?",
            "What is the penalty for drunk driving?",
            "What happens if I drive without registration?",
            "What is the golden hour in the MV Act?",
            "What documents are required while driving?",
            "What is the fine for jumping red light?",
            "What is the penalty for using mobile phone while driving?",
            "What is vehicle fitness certificate?"
        ]
        
        warmed_count = 0
        start_time = time.time()
        
        app.logger.info(f"Starting cache warmup with {len(warmup_queries)} queries")
        
        for query in warmup_queries:
            try:
                rag.process_query(query)
                warmed_count += 1
            except Exception as e:
                app.logger.warning(f"Warmup query failed: {query[:50]}... - {str(e)}")
        
        warmup_time = time.time() - start_time
        
        return jsonify({
            'message': f'Cache warmup completed',
            'queries_processed': warmed_count,
            'total_queries': len(warmup_queries),
            'warmup_time': round(warmup_time, 2),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error during cache warmup: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/cache/maintenance', methods=['POST'])
def run_cache_maintenance():
    """
    Manually trigger cache maintenance (cleanup expired entries).
    """
    try:
        if rag is None:
            return jsonify({
                'error': 'Advanced RAG pipeline not available'
            }), 503
        
        app.logger.info("Manual cache maintenance requested")
        
        # Run cache cleanup
        rag.cache_manager.cleanup_expired()
        
        # Get updated statistics
        stats = rag.get_pipeline_stats()
        cache_stats = stats.get('cache_manager_stats', {}).get('cache_stats', {})
        
        return jsonify({
            'message': 'Cache maintenance completed',
            'cache_sizes': {name: data['size'] for name, data in cache_stats.items()},
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        app.logger.error(f"Error during cache maintenance: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test', methods=['GET'])
def test_endpoint():
    """
    Test endpoint for basic connectivity and system status.
    """
    app.logger.info("Test endpoint accessed")
    
    test_result = {
        'status': 'Backend is running',
        'timestamp': datetime.now().isoformat(),
        'phase': 2,
        'rag_status': 'Initialized' if rag else 'Failed',
        'ollama_status': 'Running' if ollama_process else 'Not managed by app'
    }
    
    if rag:
        stats = rag.get_pipeline_stats()
        test_result['quick_stats'] = {
            'initialization_method': stats['initialization_method'],
            'startup_time': round(stats['startup_time'], 2),
            'total_chunks': stats['total_chunks'],
            'faq_entries': stats['faq_cache_size'],
            'overall_cache_hit_rate': round(stats.get('overall_cache_hit_rate', 0), 1)
        }
        
        # Add cache level information
        cache_stats = stats.get('cache_manager_stats', {}).get('cache_stats', {})
        test_result['cache_levels'] = {
            name: data['size'] for name, data in cache_stats.items()
        }
    
    return jsonify(test_result), 200

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up resources when application context tears down."""
    if ollama_process and sys.platform.startswith('win'):
        try:
            app.logger.info("Terminating Ollama process")
            subprocess.run(['taskkill', '/PID', str(ollama_process.pid), '/F'], shell=True)
        except Exception as e:
            app.logger.error(f"Error terminating Ollama process: {str(e)}")
    
    # Save caches on shutdown
    if rag:
        try:
            app.logger.info("Saving caches on shutdown")
            rag.save_caches()
        except Exception as e:
            app.logger.error(f"Error saving caches on shutdown: {str(e)}")

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Record application start time for uptime calculation
    app.start_time = time.time()
    
    app.logger.info("Starting Flask server on port 5000")
    print("\n🌐 Server starting on http://localhost:5000")
    print("📊 Available endpoints:")
    print("   GET  /           - Web interface")
    print("   POST /query      - Process queries")
    print("   GET  /health     - Health check")
    print("   GET  /stats      - Detailed statistics")
    print("   GET  /test       - Basic connectivity test")
    print("\n🔄 Phase 2: Advanced Cache Management:")
    print("   GET  /cache/stats      - Cache statistics")
    print("   POST /cache/clear      - Clear caches")
    print("   POST /cache/save       - Save caches")
    print("   POST /cache/warmup     - Warm up caches")
    print("   POST /cache/maintenance - Cache maintenance")
    
    print("\n🎯 Phase 2 Features Active:")
    print("   ✅ Multi-level cache hierarchy")
    print("   ✅ Exact query caching (Level 1)")
    print("   ✅ Embedding caching (Level 6)")
    print("   ✅ Retrieval caching (Level 5)")
    print("   ✅ Advanced cache management")
    print("   ✅ Background maintenance tasks")
    print("   ✅ Comprehensive performance monitoring")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server shutdown requested")
        if ollama_process:
            print("🧹 Cleaning up Ollama process...")
        if rag:
            print("💾 Saving caches...")
            try:
                rag.save_caches()
            except:
                pass
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
    finally:
        print("👋 Server stopped")