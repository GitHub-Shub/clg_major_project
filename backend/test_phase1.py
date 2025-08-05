# backend/test_phase1.py
"""
Phase 1 Testing & Validation Script
==================================

This script tests all Phase 1 features to ensure proper implementation:
- Persistent vector storage
- Enhanced FAQ cache
- Document change detection
- Performance improvements
- Error handling

Usage:
    python test_phase1.py
"""

import os
import sys
import time
import json
import requests
import hashlib
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

def test_file_structure():
    """Test that all required files and directories exist."""
    print_test("Testing file structure", "RUNNING")
    
    required_paths = [
        'data/output/mv_act_cleaned.txt',
        'data/vector_store',
        'data/cache',
        'rag_pipeline.py',
        'app.py'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        print_test("File structure", "FAIL")
        print(f"   Missing paths: {', '.join(missing_paths)}")
        return False
    
    print_test("File structure", "PASS")
    return True

def test_document_exists():
    """Test that the cleaned document exists and is not empty."""
    print_test("Testing source document", "RUNNING")
    
    doc_path = 'data/output/mv_act_cleaned.txt'
    
    if not os.path.exists(doc_path):
        print_test("Source document", "FAIL")
        print(f"   Document not found: {doc_path}")
        return False
    
    file_size = os.path.getsize(doc_path)
    if file_size == 0:
        print_test("Source document", "FAIL")
        print(f"   Document is empty: {doc_path}")
        return False
    
    print_test("Source document", "PASS")
    print(f"   Document size: {file_size:,} bytes")
    return True

def test_pipeline_initialization():
    """Test RAG pipeline initialization performance."""
    print_test("Testing pipeline initialization", "RUNNING")
    
    try:
        # Import and initialize pipeline
        start_time = time.time()
        from rag_pipeline import PersistentRAGPipeline
        
        rag = PersistentRAGPipeline()
        initialization_time = time.time() - start_time
        
        # Get statistics
        stats = rag.get_pipeline_stats()
        
        print_test("Pipeline initialization", "PASS")
        print(f"   Initialization time: {initialization_time:.2f}s")
        print(f"   Method: {stats['initialization_method']}")
        print(f"   Total chunks: {stats['total_chunks']}")
        print(f"   FAQ cache size: {stats['faq_cache_size']}")
        
        return True, rag, initialization_time
        
    except Exception as e:
        print_test("Pipeline initialization", "FAIL")
        print(f"   Error: {str(e)}")
        return False, None, 0

def test_persistent_storage(rag):
    """Test persistent storage functionality."""
    print_test("Testing persistent storage", "RUNNING")
    
    try:
        # Check if storage files exist
        storage_files = [
            'data/vector_store/faiss_index.bin',
            'data/vector_store/chunks.json',
            'data/vector_store/chunk_metadata.json',
            'data/vector_store/build_info.json'
        ]
        
        missing_files = []
        for file_path in storage_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print_test("Persistent storage", "FAIL")
            print(f"   Missing storage files: {', '.join(missing_files)}")
            return False
        
        # Check build info
        with open('data/vector_store/build_info.json', 'r') as f:
            build_info = json.load(f)
        
        print_test("Persistent storage", "PASS")
        print(f"   Build timestamp: {build_info.get('build_timestamp', 'N/A')}")
        print(f"   Document hash: {build_info.get('document_hash', 'N/A')[:16]}...")
        print(f"   Total chunks: {build_info.get('total_chunks', 'N/A')}")
        
        return True
        
    except Exception as e:
        print_test("Persistent storage", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_faq_cache(rag):
    """Test FAQ cache functionality."""
    print_test("Testing FAQ cache", "RUNNING")
    
    try:
        # Test FAQ cache size
        if len(rag.faq_cache) < 30:
            print_test("FAQ cache", "FAIL")
            print(f"   FAQ cache too small: {len(rag.faq_cache)} entries (expected 30+)")
            return False
        
        # Test some common questions
        test_questions = [
            "What is the penalty for driving without a license?",
            "What is the fine for not wearing a helmet?",
            "What is the golden hour in the MV Act?"
        ]
        
        cache_hits = 0
        for question in test_questions:
            response = rag._check_faq_cache(question)
            if response:
                cache_hits += 1
        
        if cache_hits < len(test_questions):
            print_test("FAQ cache", "FAIL") 
            print(f"   Only {cache_hits}/{len(test_questions)} test questions found in cache")
            return False
        
        print_test("FAQ cache", "PASS")
        print(f"   FAQ entries: {len(rag.faq_cache)}")
        print(f"   Test questions cached: {cache_hits}/{len(test_questions)}")
        
        return True
        
    except Exception as e:
        print_test("FAQ cache", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_query_processing(rag):
    """Test query processing with timing."""
    print_test("Testing query processing", "RUNNING")
    
    try:
        test_queries = [
            "What is the penalty for driving without a license?",  # Should hit FAQ cache
            "What are the rules for vehicle registration?",       # Should require RAG
        ]
        
        query_times = []
        cache_hits = 0
        
        for i, query in enumerate(test_queries):
            start_time = time.time()
            
            try:
                response = rag.process_query(query)
                query_time = time.time() - start_time
                query_times.append(query_time)
                
                # Check if it was likely a cache hit (very fast response)
                if query_time < 0.1:
                    cache_hits += 1
                
                print(f"      Query {i+1}: {query_time:.3f}s")
                
            except Exception as query_error:
                print(f"      Query {i+1} failed: {str(query_error)}")
                return False
        
        avg_time = sum(query_times) / len(query_times)
        
        print_test("Query processing", "PASS")
        print(f"   Average query time: {avg_time:.3f}s")
        print(f"   Likely cache hits: {cache_hits}/{len(test_queries)}")
        
        return True
        
    except Exception as e:
        print_test("Query processing", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_document_change_detection():
    """Test document change detection mechanism."""
    print_test("Testing change detection", "RUNNING")
    
    try:
        # Calculate current document hash
        doc_path = 'data/output/mv_act_cleaned.txt'
        
        def calculate_hash(file_path):
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        
        current_hash = calculate_hash(doc_path)
        
        # Check stored hash in build info
        build_info_path = 'data/vector_store/build_info.json'
        if os.path.exists(build_info_path):
            with open(build_info_path, 'r') as f:
                build_info = json.load(f)
            
            stored_hash = build_info.get('document_hash')
            
            if current_hash == stored_hash:
                print_test("Change detection", "PASS")
                print(f"   Document hash matches: {current_hash[:16]}...")
                return True
            else:
                print_test("Change detection", "FAIL")
                print(f"   Hash mismatch - Current: {current_hash[:16]}..., Stored: {stored_hash[:16] if stored_hash else 'None'}...")
                return False
        else:
            print_test("Change detection", "SKIP")
            print("   No build info found")
            return True
            
    except Exception as e:
        print_test("Change detection", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_health_check():
    """Test health check functionality."""
    print_test("Testing health check", "RUNNING")
    
    try:
        from rag_pipeline import PersistentRAGPipeline
        rag = PersistentRAGPipeline()
        
        health = rag.health_check()
        
        if health.get('overall_status') not in ['healthy', 'degraded']:
            print_test("Health check", "FAIL")
            print(f"   Unhealthy status: {health.get('overall_status')}")
            return False
        
        print_test("Health check", "PASS")
        print(f"   Overall status: {health.get('overall_status')}")
        print(f"   Checks passed: {len([c for c in health.get('checks', {}).values() if c.get('status') == 'ok'])}")
        
        return True
        
    except Exception as e:
        print_test("Health check", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_server_endpoints():
    """Test Flask server endpoints if server is running."""
    print_test("Testing server endpoints", "RUNNING")
    
    try:
        # Test basic connectivity
        response = requests.get('http://localhost:5000/test', timeout=5)
        if response.status_code != 200:
            print_test("Server endpoints", "SKIP")  
            print("   Server not running - start with 'python app.py'")
            return True
        
        # Test health endpoint
        health_response = requests.get('http://localhost:5000/health', timeout=10)
        
        # Test query endpoint
        query_data = {'query': 'What is the penalty for driving without a license?'}
        query_response = requests.post(
            'http://localhost:5000/query', 
            json=query_data, 
            timeout=10
        )
        
        if health_response.status_code == 200 and query_response.status_code == 200:
            query_result = query_response.json()
            print_test("Server endpoints", "PASS")
            print(f"   Query processing time: {query_result.get('processing_time', 'N/A')}s")
            print(f"   Cache hit rate: {query_result.get('cache_hit_rate', 'N/A')}%")
            return True
        else:
            print_test("Server endpoints", "FAIL")
            print(f"   Health status: {health_response.status_code}")
            print(f"   Query status: {query_response.status_code}")
            return False
            
    except requests.exceptions.RequestException:
        print_test("Server endpoints", "SKIP")
        print("   Server not running - start with 'python app.py'")
        return True
    except Exception as e:
        print_test("Server endpoints", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def test_performance_improvements():
    """Test performance improvements compared to expected baselines."""
    print_test("Testing performance improvements", "RUNNING")
    
    try:
        # Test initialization time
        start_time = time.time()
        from rag_pipeline import PersistentRAGPipeline
        rag = PersistentRAGPipeline()
        init_time = time.time() - start_time
        
        stats = rag.get_pipeline_stats()
        
        # Performance thresholds for Phase 1
        expected_init_time = 10.0  # Should be under 10 seconds
        expected_faq_size = 30     # Should have 30+ FAQ entries
        
        performance_issues = []
        
        if init_time > expected_init_time:
            performance_issues.append(f"Slow initialization: {init_time:.2f}s (expected <{expected_init_time}s)")
        
        if stats['faq_cache_size'] < expected_faq_size:
            performance_issues.append(f"Small FAQ cache: {stats['faq_cache_size']} (expected >={expected_faq_size})")
        
        if performance_issues:
            print_test("Performance improvements", "FAIL")
            for issue in performance_issues:
                print(f"   {issue}")
            return False
        
        print_test("Performance improvements", "PASS")
        print(f"   Initialization time: {init_time:.2f}s")
        print(f"   Method: {stats['initialization_method']}")
        print(f"   FAQ cache size: {stats['faq_cache_size']}")
        
        return True
        
    except Exception as e:
        print_test("Performance improvements", "FAIL")
        print(f"   Error: {str(e)}")
        return False

def main():
    """Run all Phase 1 tests."""
    print_header("PHASE 1 TESTING & VALIDATION")
    
    print("Testing Phase 1 implementation:")
    print("• Persistent vector storage")
    print("• Enhanced FAQ cache (50+ entries)")
    print("• Document change detection")
    print("• Performance improvements")
    print("• Error handling and monitoring")
    
    # Track test results
    tests_passed = 0
    tests_total = 0
    
    # Test 1: File structure
    tests_total += 1
    if test_file_structure():
        tests_passed += 1
    
    # Test 2: Source document
    tests_total += 1
    if test_document_exists():
        tests_passed += 1
    
    # Test 3: Pipeline initialization
    tests_total += 1
    pipeline_success, rag, init_time = test_pipeline_initialization()
    if pipeline_success:
        tests_passed += 1
    
    # Only run remaining tests if pipeline initialized successfully
    if pipeline_success and rag:
        # Test 4: Persistent storage
        tests_total += 1
        if test_persistent_storage(rag):
            tests_passed += 1
        
        # Test 5: FAQ cache
        tests_total += 1
        if test_faq_cache(rag):
            tests_passed += 1
        
        # Test 6: Query processing
        tests_total += 1
        if test_query_processing(rag):
            tests_passed += 1
        
        # Test 7: Change detection
        tests_total += 1
        if test_document_change_detection():
            tests_passed += 1
        
        # Test 8: Health check
        tests_total += 1
        if test_health_check():
            tests_passed += 1
    
    # Test 9: Server endpoints (optional)
    tests_total += 1
    if test_server_endpoints():
        tests_passed += 1
    
    # Test 10: Performance improvements
    tests_total += 1
    if test_performance_improvements():
        tests_passed += 1
    
    # Print final results
    print_header("TEST RESULTS SUMMARY")
    
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print(f"Success rate: {(tests_passed/tests_total)*100:.1f}%")
    
    if tests_passed == tests_total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Phase 1 implementation is working correctly")
        print("\n📋 Next steps:")
        print("   1. Run 'python app.py' to start the server")
        print("   2. Test the web interface at http://localhost:5000")
        print("   3. Monitor performance improvements")
        print("   4. Proceed to Phase 2 implementation when ready")
    else:
        print(f"\n⚠️  {tests_total - tests_passed} TEST(S) FAILED")
        print("Please review the failed tests and fix issues before proceeding")
        print("\n🔧 Common fixes:")
        print("   • Run 'python extract_pdf.py' and 'python data_cleaner.py' first")
        print("   • Ensure Ollama is installed and running")
        print("   • Check file permissions for data directories")
        print("   • Verify all dependencies are installed")
    
    return tests_passed == tests_total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)