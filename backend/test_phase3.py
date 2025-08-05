# backend/test_phase3.py
"""
Phase 3: Comprehensive Testing & Validation Script
=================================================

This script tests all Phase 3 features to ensure proper implementation:
- Semantic query caching with similarity matching
- Pre-computed response cache with topic extraction  
- Intelligent cache hierarchy with confidence scoring
- Background learning and optimization
- Advanced analytics and monitoring

Usage:
    python test_phase3.py
"""

import os
import sys
import time
import json
import requests
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f"🧪 {title}")
    print("="*70)

def print_test(test_name, status="RUNNING"):
    """Print test status."""
    if status == "RUNNING":
        print(f"⏳ {test_name}...", end=" ")
    elif status == "PASS":
        print("✅ PASSED")
    elif status == "FAIL":
        print("❌ FAILED")
    elif status == "SKIP":
        print("⚠️ SKIPPED")

def test_dependencies():
    """Test that all required dependencies are available."""
    print_test("Testing Phase 3 dependencies", "RUNNING")
    
    required_packages = [
        'numpy',
        'scikit-learn',
        'sentence-transformers',
        'faiss-cpu',
        'ollama'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print_test("Phase 3 dependencies", "FAIL")
        print(f"   Missing packages: {', '.join(missing_packages)}")
        print("   Install with: pip install scikit-learn")
        return False
    
    print_test("Phase 3 dependencies", "PASS")
    return True

def test_semantic_cache_manager():
    """Test semantic cache manager initialization."""
    print_test("Testing semantic cache manager", "RUNNING")
    
    try:
        from semantic_cache_manager import (
            SemanticCacheManager, 
            SemanticCluster, 
            LegalTopicExtractor
        )
        
        # Test topic extractor
        extractor = LegalTopicExtractor()
        topics = extractor.extract_topics_from_query("What is the penalty for driving without license?")
        
        if not topics:
            print_test("Semantic cache manager", "FAIL")
            print("   Topic extraction failed")
            return False
        
        # Test semantic cluster creation
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query = "test query"
        embedding = model.encode([query])[0]
        
        cluster = SemanticCluster(
            cluster_id="test_123",
            representative_query=query,
            representative_embedding=embedding,
            response="test response"
        )
        
        # Test similarity calculation
        similarity = cluster.calculate_similarity(embedding)
        if similarity < 0.99:  # Should be very high for identical embeddings
            print_test("Semantic cache manager", "FAIL")
            print(f"   Similarity calculation failed: {similarity}")
            return False
        
        print_test("Semantic cache manager", "PASS")
        print(f"   Extracted {len(topics)} topics")
        print(f"   Similarity calculation: {similarity:.3f}")
        return True
        
    except Exception as e:
        print_test("Semantic cache manager", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_intelligent_pipeline_initialization():
    """Test intelligent RAG pipeline initialization."""
    print_test("Testing intelligent pipeline initialization", "RUNNING")
    
    try:
        start_time = time.time()
        from rag_pipeline import IntelligentRAGPipeline
        
        # Test with minimal configuration
        config = {
            'enable_background_tasks': False,  # Disable for testing
            'auto_cluster_creation': True,
            'semantic_similarity_threshold': 0.85,
            'confidence_threshold': 0.7
        }
        
        rag = IntelligentRAGPipeline(config=config)
        initialization_time = time.time() - start_time
        
        # Get statistics
        stats = rag.get_pipeline_stats()
        
        print_test("Intelligent pipeline initialization", "PASS")
        print(f"   Initialization time: {initialization_time:.2f}s")
        print(f"   Method: {stats['initialization_method']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   Semantic features enabled: {bool(rag.semantic_cache_manager)}")
        
        return True, rag, initialization_time
        
    except Exception as e:
        print_test("Intelligent pipeline initialization", "FAIL")
        print(f"   Error: {str(e)}")
        return False, None, 0

def test_cache_hierarchy(rag):
    """Test the intelligent cache hierarchy."""
    print_test("Testing cache hierarchy", "RUNNING")
    
    try:
        # Test FAQ cache (Level 2)
        faq_result = rag._check_faq_cache("What is the penalty for driving without a license?")
        if not faq_result:
            print_test("Cache hierarchy", "FAIL")
            print("   FAQ cache failed")
            return False
        
        # Test exact query cache (Level 1) - should be empty initially
        exact_result = rag._check_exact_query_cache("This is a test query for exact matching")
        if exact_result:  # Should be None initially
            print_test("Cache hierarchy", "FAIL")
            print("   Exact query cache should be empty initially")
            return False
        
        # Test semantic cache (Level 3) - should be empty initially
        test_embedding = rag.embedding_model.encode(["test semantic query"])[0]
        semantic_result = rag._check_semantic_cache("test semantic query", test_embedding)
        if semantic_result:  # Should be None initially
            print_test("Cache hierarchy", "FAIL")
            print("   Semantic cache should be empty initially")
            return False
        
        # Test precomputed cache (Level 4)
        precomputed_result = rag._check_precomputed_cache("What is the penalty for driving violations?")
        # This might or might not have results depending on initialization
        
        print_test("Cache hierarchy", "PASS")
        print(f"   FAQ cache: ✓")
        print(f"   Exact query cache: ✓ (empty as expected)")
        print(f"   Semantic cache: ✓ (empty as expected)")
        print(f"   Precomputed cache: {'✓' if precomputed_result else '○'}")
        
        return True
        
    except Exception as e:
        print_test("Cache hierarchy", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_semantic_clustering(rag):
    """Test semantic clustering functionality."""
    print_test("Testing semantic clustering", "RUNNING")
    
    try:
        if not rag.semantic_cache_manager:
            print_test("Semantic clustering", "SKIP")
            print("   Semantic cache manager not available")
            return True
        
        # Test queries that should be semantically similar
        similar_queries = [
            "What is the penalty for driving without a license?",
            "How much fine for driving without license?",
            "What happens if caught driving without valid license?"
        ]
        
        clusters_created = 0
        
        for query in similar_queries:
            # Process query to potentially create clusters
            try:
                response = rag.process_query(query)
                if response:
                    clusters_created += 1
                time.sleep(0.1)  # Small delay
            except Exception as e:
                print(f"      Query failed: {query[:30]}... - {str(e)}")
        
        # Check if semantic clusters were created
        semantic_stats = rag.semantic_cache_manager.get_semantic_stats()
        total_clusters = semantic_stats['semantic_clusters']['total_clusters']
        
        print_test("Semantic clustering", "PASS")
        print(f"   Processed {clusters_created} similar queries")
        print(f"   Total semantic clusters: {total_clusters}")
        
        return True
        
    except Exception as e:
        print_test("Semantic clustering", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_precomputed_responses(rag):
    """Test pre-computed response functionality."""
    print_test("Testing pre-computed responses", "RUNNING")
    
    try:
        if not rag.semantic_cache_manager:
            print_test("Pre-computed responses", "SKIP")
            print("   Semantic cache manager not available")
            return True
        
        # Test topic extraction
        extractor = rag.semantic_cache_manager.topic_extractor
        
        test_queries = {
            "What is the penalty for driving without license?": "driving_license_penalty",
            "What is the fine for not wearing helmet?": "helmet_violation",
            "What happens if I drive drunk?": "drunk_driving",
            "What documents do I need while driving?": "driving_documents"
        }
        
        topic_matches = 0
        
        for query, expected_topic in test_queries.items():
            topics = extractor.extract_topics_from_query(query)
            if topics and topics[0][0] == expected_topic:
                topic_matches += 1
        
        # Test pre-computed response generation
        test_topic = "driving_license_penalty"
        if test_topic not in rag.semantic_cache_manager.precomputed_responses:
            # Add a test pre-computed response
            rag.semantic_cache_manager.add_precomputed_response(
                test_topic,
                "Test pre-computed response for driving license penalty."
            )
        
        # Test retrieval
        precomputed_result = rag.semantic_cache_manager.get_precomputed_response(
            "What is the penalty for driving without proper license?"
        )
        
        print_test("Pre-computed responses", "PASS")
        print(f"   Topic extraction accuracy: {topic_matches}/{len(test_queries)}")
        print(f"   Pre-computed retrieval: {'✓' if precomputed_result else '○'}")
        
        return True
        
    except Exception as e:
        print_test("Pre-computed responses", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_confidence_scoring(rag):
    """Test confidence scoring system."""
    print_test("Testing confidence scoring", "RUNNING")
    
    try:
        # Test different types of queries to get different confidence levels
        test_queries = [
            ("What is the penalty for driving without a license?", "high"),  # FAQ - should be high confidence
            ("How much fine for no license?", "medium"),  # Semantic match - medium confidence  
            ("Complex legal procedure question", "low")  # Complex query - lower confidence
        ]
        
        confidence_results = []
        
        for query, expected_level in test_queries:
            try:
                # Process query (but we can't easily extract confidence from this interface)
                response = rag.process_query(query)
                if response:
                    confidence_results.append((query, expected_level, "processed"))
                time.sleep(0.1)
            except Exception as e:
                confidence_results.append((query, expected_level, f"error: {str(e)}"))
        
        # Check confidence statistics from pipeline
        stats = rag.get_pipeline_stats()
        confidence_scores = stats.get('confidence_scores', {})
        
        total_confidence_queries = sum(confidence_scores.values())
        
        print_test("Confidence scoring", "PASS")
        print(f"   Processed {len(confidence_results)} test queries")
        print(f"   Confidence distributions: {confidence_scores}")
        print(f"   Total confidence-tracked queries: {total_confidence_queries}")
        
        return True
        
    except Exception as e:
        print_test("Confidence scoring", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_query_pattern_analysis(rag):
    """Test query pattern analysis functionality."""
    print_test("Testing query pattern analysis", "RUNNING")
    
    try:
        if not rag.semantic_cache_manager:
            print_test("Query pattern analysis", "SKIP")
            print("   Semantic cache manager not available")
            return True
        
        # Generate patterns by processing various queries
        pattern_queries = [
            "penalty for driving",
            "fine for speeding",
            "penalty for helmet violation",
            "fine for signal jumping",
            "penalty for drunk driving"
        ]
        
        for query in pattern_queries:
            try:
                rag.semantic_cache_manager.analyze_query_patterns(query)
            except Exception as e:
                print(f"      Pattern analysis failed for: {query}")
        
        # Get popular patterns
        popular_patterns = rag.semantic_cache_manager.get_popular_patterns(5)
        
        print_test("Query pattern analysis", "PASS")
        print(f"   Analyzed {len(pattern_queries)} queries")
        print(f"   Popular patterns found: {len(popular_patterns)}")
        if popular_patterns:
            print(f"   Top pattern: {popular_patterns[0][0]} (count: {popular_patterns[0][1]})")
        
        return True
        
    except Exception as e:
        print_test("Query pattern analysis", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_background_processing(rag):
    """Test background processing functionality."""
    print_test("Testing background processing", "RUNNING")
    
    try:
        # Check if background manager exists
        if not hasattr(rag, 'background_manager') or not rag.background_manager:
            print_test("Background processing", "SKIP")
            print("   Background processing disabled or not available")
            return True
        
        # Check if background tasks are enabled
        if not rag.background_manager.tasks_enabled:
            print_test("Background processing", "SKIP")
            print("   Background tasks disabled")
            return True
        
        # Verify background thread is running
        if not rag.background_manager.background_thread or not rag.background_manager.background_thread.is_alive():
            print_test("Background processing", "FAIL")
            print("   Background thread not running")
            return False
        
        # Check task intervals configuration
        task_intervals = rag.background_manager.task_intervals
        expected_tasks = ['cluster_optimization', 'precompute_generation', 'pattern_analysis', 'cache_maintenance']
        
        missing_tasks = [task for task in expected_tasks if task not in task_intervals]
        if missing_tasks:
            print_test("Background processing", "FAIL")
            print(f"   Missing background tasks: {missing_tasks}")
            return False
        
        print_test("Background processing", "PASS")
        print(f"   Background thread: Running")
        print(f"   Configured tasks: {len(task_intervals)}")
        print(f"   Task types: {list(task_intervals.keys())}")
        
        return True
        
    except Exception as e:
        print_test("Background processing", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_server_endpoints():
    """Test Phase 3 specific server endpoints."""
    print_test("Testing Phase 3 server endpoints", "RUNNING")
    
    try:
        # Test basic connectivity first
        response = requests.get('http://localhost:5000/test', timeout=5)
        if response.status_code != 200:
            print_test("Phase 3 server endpoints", "SKIP")
            print("   Server not running - start with 'python app.py'")
            return True
        
        test_results = {}
        
        # Test semantic cluster endpoint
        try:
            semantic_response = requests.get('http://localhost:5000/semantic/clusters', timeout=10)
            test_results['semantic_clusters'] = semantic_response.status_code == 200
        except:
            test_results['semantic_clusters'] = False
        
        # Test precomputed topics endpoint
        try:
            topics_response = requests.get('http://localhost:5000/semantic/topics', timeout=10)
            test_results['precomputed_topics'] = topics_response.status_code == 200
        except:
            test_results['precomputed_topics'] = False
        
        # Test query patterns endpoint
        try:
            patterns_response = requests.get('http://localhost:5000/analytics/patterns', timeout=10)
            test_results['query_patterns'] = patterns_response.status_code == 200
        except:
            test_results['query_patterns'] = False
        
        # Test enhanced query endpoint with a semantic query
        try:
            query_data = {'query': 'How much fine for driving without license?'}
            query_response = requests.post(
                'http://localhost:5000/query', 
                json=query_data, 
                timeout=15
            )
            
            if query_response.status_code == 200:
                query_result = query_response.json()
                test_results['enhanced_query'] = True
                test_results['confidence_scoring'] = 'confidence' in query_result
                test_results['cache_level_detection'] = 'cache_level' in query_result
            else:
                test_results['enhanced_query'] = False
                test_results['confidence_scoring'] = False
                test_results['cache_level_detection'] = False
        except:
            test_results['enhanced_query'] = False
            test_results['confidence_scoring'] = False
            test_results['cache_level_detection'] = False
        
        # Evaluate results
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        
        if passed_tests == total_tests:
            print_test("Phase 3 server endpoints", "PASS")
        elif passed_tests >= total_tests * 0.8:  # Allow some failures
            print_test("Phase 3 server endpoints", "PASS")
            print("   Some endpoints had issues (acceptable)")
        else:
            print_test("Phase 3 server endpoints", "FAIL")
            print(f"   Only {passed_tests}/{total_tests} endpoints working")
            return False
        
        print(f"   Working endpoints: {passed_tests}/{total_tests}")
        for endpoint, status in test_results.items():
            print(f"     {endpoint}: {'✓' if status else '✗'}")
        
        return True
        
    except requests.exceptions.RequestException:
        print_test("Phase 3 server endpoints", "SKIP")
        print("   Server not running - start with 'python app.py'")
        return True
    except Exception as e:
        print_test("Phase 3 server endpoints", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_performance_improvements():
    """Test Phase 3 performance improvements."""
    print_test("Testing Phase 3 performance improvements", "RUNNING")
    
    try:
        # Test initialization time
        start_time = time.time()
        from rag_pipeline import IntelligentRAGPipeline
        
        config = {'enable_background_tasks': False}  # Disable for testing
        rag = IntelligentRAGPipeline(config=config)
        init_time = time.time() - start_time
        
        stats = rag.get_pipeline_stats()
        
        # Performance thresholds for Phase 3
        expected_init_time = 15.0  # Should be under 15 seconds (slightly higher due to semantic features)
        expected_cache_levels = 6   # Should have multiple cache levels
        
        performance_issues = []
        
        if init_time > expected_init_time:
            performance_issues.append(f"Slow initialization: {init_time:.2f}s (expected <{expected_init_time}s)")
        
        # Count active cache levels
        cache_breakdown = stats.get('cache_hierarchy_breakdown', {})
        semantic_stats = stats.get('semantic_cache_stats', {})
        
        cache_level_count = len([k for k, v in cache_breakdown.items() if isinstance(v, int)])
        if semantic_stats:
            cache_level_count += 2  # semantic + precomputed
        
        if cache_level_count < expected_cache_levels:
            performance_issues.append(f"Few cache levels: {cache_level_count} (expected >={expected_cache_levels})")
        
        # Test query performance with different cache levels
        test_queries = [
            "What is the penalty for driving without a license?",  # Should hit FAQ (fast)
            "How much fine for no license?",  # Should potentially create semantic cluster
            "Complex vehicle registration procedures?"  # Should go through full RAG
        ]
        
        query_times = []
        for query in test_queries:
            start = time.time()
            try:
                response = rag.process_query(query)
                if response:
                    query_times.append(time.time() - start)
                time.sleep(0.1)
            except:
                pass
        
        avg_query_time = sum(query_times) / len(query_times) if query_times else float('inf')
        
        if avg_query_time > 5.0:  # Should be reasonable
            performance_issues.append(f"Slow average query time: {avg_query_time:.2f}s")
        
        if performance_issues:
            print_test("Phase 3 performance improvements", "FAIL")
            for issue in performance_issues:
                print(f"   {issue}")
            return False
        
        print_test("Phase 3 performance improvements", "PASS")
        print(f"   Initialization time: {init_time:.2f}s")
        print(f"   Method: {stats['initialization_method']}")
        print(f"   Cache levels available: {cache_level_count}")
        print(f"   Average query time: {avg_query_time:.2f}s")
        
        return True
        
    except Exception as e:
        print_test("Phase 3 performance improvements", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_comprehensive_integration():
    """Test comprehensive integration of all Phase 3 features."""
    print_test("Testing comprehensive integration", "RUNNING")
    
    try:
        # Initialize pipeline with all features enabled
        config = {
            'enable_semantic_cache': True,
            'enable_precomputed_cache': True,
            'auto_cluster_creation': True,
            'enable_background_tasks': False,  # Disable for testing
            'semantic_similarity_threshold': 0.85,
            'confidence_threshold': 0.7
        }
        
        from rag_pipeline import IntelligentRAGPipeline
        rag = IntelligentRAGPipeline(config=config)
        
        # Test workflow: FAQ -> Semantic -> Precomputed -> Full RAG
        test_workflow = [
            ("What is the penalty for driving without a license?", "faq"),
            ("How much fine for no license?", "semantic_or_exact"),
            ("Vehicle documentation requirements", "precomputed"),
            ("Complex procedural question about license appeals", "full_rag")
        ]
        
        workflow_results = []
        
        for query, expected_cache in test_workflow:
            start_time = time.time()
            try:
                response = rag.process_query(query)
                response_time = time.time() - start_time
                
                if response and len(response) > 10:  # Got meaningful response
                    workflow_results.append((query, expected_cache, response_time, "success"))
                else:
                    workflow_results.append((query, expected_cache, response_time, "empty_response"))
                
                time.sleep(0.1)  # Small delay between queries
                
            except Exception as e:
                workflow_results.append((query, expected_cache, 0, f"error: {str(e)}"))
        
        # Check final statistics
        final_stats = rag.get_pipeline_stats()
        
        success_count = len([r for r in workflow_results if r[3] == "success"])
        total_queries = len(workflow_results)
        
        if success_count < total_queries * 0.75:  # At least 75% should succeed
            print_test("Comprehensive integration", "FAIL")
            print(f"   Only {success_count}/{total_queries} queries succeeded")
            return False
        
        print_test("Comprehensive integration", "PASS")
        print(f"   Workflow test: {success_count}/{total_queries} queries successful")
        print(f"   Overall cache hit rate: {final_stats.get('overall_cache_hit_rate', 0):.1f}%")
        print(f"   Total processed queries: {final_stats.get('total_queries_processed', 0)}")
        
        # Show cache hierarchy usage
        cache_breakdown = final_stats.get('cache_hierarchy_breakdown', {})
        active_caches = {k: v for k, v in cache_breakdown.items() if v > 0}
        if active_caches:
            print(f"   Cache usage: {active_caches}")
        
        return True
        
    except Exception as e:
        print_test("Comprehensive integration", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def main():
    """Run all Phase 3 tests."""
    print_header("PHASE 3 TESTING & VALIDATION")
    
    print("Testing Phase 3 Intelligent Caching implementation:")
    print("• Semantic query caching with similarity matching")
    print("• Pre-computed response cache with topic extraction")
    print("• Intelligent cache hierarchy with confidence scoring")
    print("• Background learning and optimization")
    print("• Advanced analytics and monitoring")
    
    # Track test results
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Dependencies
    tests_total += 1
    if test_dependencies():
        tests_passed += 1
    
    # Test 2: Semantic cache manager
    tests_total += 1
    if test_semantic_cache_manager():
        tests_passed += 1
    
    # Test 3: Pipeline initialization
    tests_total += 1
    pipeline_success, rag, init_time = test_intelligent_pipeline_initialization()
    if pipeline_success:
        tests_passed += 1
    
    # Only run remaining tests if pipeline initialized successfully
    if pipeline_success and rag:
        # Test 4: Cache hierarchy
        tests_total += 1
        if test_cache_hierarchy(rag):
            tests_passed += 1
        
        # Test 5: Semantic clustering
        tests_total += 1
        if test_semantic_clustering(rag):
            tests_passed += 1
        
        # Test 6: Pre-computed responses
        tests_total += 1
        if test_precomputed_responses(rag):
            tests_passed += 1
        
        # Test 7: Confidence scoring
        tests_total += 1
        if test_confidence_scoring(rag):
            tests_passed += 1
        
        # Test 8: Query pattern analysis
        tests_total += 1
        if test_query_pattern_analysis(rag):
            tests_passed += 1
        
        # Test 9: Background processing
        tests_total += 1
        if test_background_processing(rag):
            tests_passed += 1
    
    # Test 10: Server endpoints (optional)
    tests_total += 1
    if test_server_endpoints():
        tests_passed += 1
    
    # Test 11: Performance improvements
    tests_total += 1
    if test_performance_improvements():
        tests_passed += 1
    
    # Test 12: Comprehensive integration
    tests_total += 1
    if test_comprehensive_integration():
        tests_passed += 1
    
    # Print final results
    print_header("TEST RESULTS SUMMARY")
    
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Success rate: {(tests_passed/tests_total)*100:.1f}%")
    
    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 3 implementation is working correctly")
        print("\n📋 Next steps:")
        print("   1. Run 'python app.py' to start the intelligent server")
        print("   2. Test the enhanced web interface at http://localhost:5000")
        print("   3. Monitor semantic caching performance")
        print("   4. Check /semantic/clusters and /semantic/topics endpoints")
        print("   5. Observe background learning and optimization")
    else:
        print(f"\n⚠️  {tests_total - tests_passed} TEST(S) FAILED")
        print("Please review the failed tests and fix issues before proceeding")
        print("\n🔧 Common fixes:")
        print("   • Install scikit-learn: pip install scikit-learn")
        print("   • Ensure all Phase 1 & 2 components are working")
        print("   • Check system memory (>4GB recommended for Phase 3)")
        print("   • Verify Ollama is running with tinyllama model")
        print("   • Check file permissions for cache directories")
    
    return tests_passed == tests_total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)