#!/usr/bin/env python3
"""
API Connectivity Troubleshooting Script for VectorSmuggle
"""

import os
import sys
import logging
import time
import requests
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """Setup detailed logging for troubleshooting."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_environment_variables(logger):
    """Check and validate environment variables."""
    logger.info("🔍 Checking environment variables...")
    
    # Check for .env file
    env_files = ['.env', '.env.local', '.env.example']
    for env_file in env_files:
        if Path(env_file).exists():
            logger.info(f"Found environment file: {env_file}")
        else:
            logger.debug(f"Environment file not found: {env_file}")
    
    # Check OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        logger.info(f"✅ OPENAI_API_KEY found (length: {len(api_key)})")
        if api_key.startswith("sk-"):
            logger.info("✅ API key format appears correct")
        else:
            logger.warning("⚠️ API key doesn't start with 'sk-' - may be invalid")
    else:
        logger.error("❌ OPENAI_API_KEY not found in environment")
        return False
    
    # Check other relevant environment variables
    other_vars = [
        "OPENAI_EMBEDDING_MODEL",
        "OPENAI_LLM_MODEL", 
        "OPENAI_MAX_RETRIES",
        "OPENAI_TIMEOUT"
    ]
    
    for var in other_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"✅ {var}: {value}")
        else:
            logger.debug(f"🔧 {var}: using default")
    
    return True

def test_network_connectivity(logger):
    """Test basic network connectivity."""
    logger.info("🌐 Testing network connectivity...")
    
    test_urls = [
        "https://api.openai.com",
        "https://google.com",
        "https://httpbin.org/get"
    ]
    
    for url in test_urls:
        try:
            logger.info(f"Testing connection to {url}...")
            response = requests.get(url, timeout=10)
            logger.info(f"✅ {url}: Status {response.status_code}")
        except requests.exceptions.Timeout:
            logger.error(f"❌ {url}: Connection timeout")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"❌ {url}: Connection error")
            return False
        except Exception as e:
            logger.error(f"❌ {url}: {e}")
            return False
    
    return True

def test_openai_api_direct(logger):
    """Test OpenAI API directly with requests."""
    logger.info("🔑 Testing OpenAI API directly...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("❌ No API key available for testing")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Test models endpoint
    try:
        logger.info("Testing models endpoint...")
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            models = response.json()
            logger.info(f"✅ Models endpoint: {len(models.get('data', []))} models available")
            
            # Check for embedding models
            embedding_models = [
                model for model in models.get('data', [])
                if 'embedding' in model.get('id', '').lower()
            ]
            logger.info(f"✅ Embedding models available: {len(embedding_models)}")
            for model in embedding_models[:3]:  # Show first 3
                logger.info(f"  - {model.get('id')}")
                
        elif response.status_code == 401:
            logger.error("❌ API key authentication failed")
            return False
        else:
            logger.error(f"❌ Models endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Models endpoint error: {e}")
        return False
    
    # Test embeddings endpoint
    try:
        logger.info("Testing embeddings endpoint...")
        test_data = {
            "input": "test embedding",
            "model": "text-embedding-ada-002"
        }
        
        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            embedding = result.get('data', [{}])[0].get('embedding', [])
            logger.info(f"✅ Embeddings endpoint: Generated {len(embedding)}-dimensional embedding")
            return True
        else:
            logger.error(f"❌ Embeddings endpoint failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embeddings endpoint error: {e}")
        return False

def test_langchain_integration(logger):
    """Test LangChain OpenAI integration."""
    logger.info("🔗 Testing LangChain integration...")
    
    try:
        from langchain_openai import OpenAIEmbeddings
        
        # Test basic initialization
        logger.info("Testing OpenAIEmbeddings initialization...")
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        logger.info("✅ OpenAIEmbeddings initialized successfully")
        
        # Test embedding generation
        logger.info("Testing embedding generation...")
        test_text = "This is a test for LangChain integration"
        embedding = embeddings.embed_query(test_text)
        
        if embedding and len(embedding) > 0:
            logger.info(f"✅ LangChain embedding: Generated {len(embedding)}-dimensional vector")
            return True
        else:
            logger.error("❌ LangChain embedding: Empty result")
            return False
            
    except ImportError as e:
        logger.error(f"❌ LangChain import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ LangChain integration error: {e}")
        return False

def test_vectorsmuggle_components(logger):
    """Test VectorSmuggle components with API."""
    logger.info("🎯 Testing VectorSmuggle components...")
    
    try:
        from steganography.fragmentation import MultiModelFragmenter
        from steganography.decoys import DecoyGenerator
        from langchain_openai import OpenAIEmbeddings
        
        # Test fragmentation system
        logger.info("Testing fragmentation system...")
        fragmenter = MultiModelFragmenter()
        stats = fragmenter.get_model_statistics()
        
        if stats['total_models'] > 0:
            logger.info(f"✅ Fragmentation: {stats['total_models']} models initialized")
            
            # Test fragmentation
            test_text = "This is a test document for fragmentation testing with multiple models."
            try:
                result = fragmenter.fragment_and_embed(test_text, num_fragments=2)
                logger.info(f"✅ Fragmentation: Generated {len(result['embeddings'])} fragments")
            except Exception as e:
                logger.error(f"❌ Fragmentation embedding failed: {e}")
        else:
            logger.warning("⚠️ Fragmentation: No models initialized")
        
        # Test decoy generation
        logger.info("Testing decoy generation...")
        embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
        decoy_gen = DecoyGenerator(embedding_model=embeddings_model)
        
        try:
            decoy_embeddings, decoy_texts = decoy_gen.generate_decoy_embeddings_from_text(5)
            logger.info(f"✅ Decoy generation: Generated {len(decoy_embeddings)} embeddings")
        except Exception as e:
            logger.error(f"❌ Decoy generation failed: {e}")
            
    except ImportError as e:
        logger.error(f"❌ VectorSmuggle import error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ VectorSmuggle component error: {e}")
        return False
    
    return True

def suggest_fixes(logger, test_results):
    """Suggest fixes based on test results."""
    logger.info("🔧 Suggested fixes based on test results:")
    
    if not test_results.get('env_vars', False):
        logger.info("📝 Environment Variables:")
        logger.info("  1. Set OPENAI_API_KEY environment variable")
        logger.info("  2. Create .env file with: OPENAI_API_KEY=sk-your-key-here")
        logger.info("  3. Verify API key is valid and has embedding permissions")
    
    if not test_results.get('network', False):
        logger.info("🌐 Network Connectivity:")
        logger.info("  1. Check internet connection")
        logger.info("  2. Verify firewall/proxy settings")
        logger.info("  3. Check if OpenAI API is accessible from your network")
    
    if not test_results.get('openai_api', False):
        logger.info("🔑 OpenAI API:")
        logger.info("  1. Verify API key is correct and active")
        logger.info("  2. Check API key permissions (needs embedding access)")
        logger.info("  3. Verify account has sufficient credits/quota")
        logger.info("  4. Check for any API rate limiting")
    
    if not test_results.get('langchain', False):
        logger.info("🔗 LangChain Integration:")
        logger.info("  1. Update langchain-openai: pip install --upgrade langchain-openai")
        logger.info("  2. Check for version compatibility issues")
        logger.info("  3. Verify all dependencies are installed")

def create_env_template(logger):
    """Create .env template file."""
    logger.info("📄 Creating .env template...")
    
    env_template = """# VectorSmuggle Environment Configuration
# Copy this file to .env and fill in your values

# OpenAI API Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002
OPENAI_LLM_MODEL=gpt-3.5-turbo-instruct

# API Reliability Settings
OPENAI_MAX_RETRIES=3
OPENAI_TIMEOUT=30.0
OPENAI_RETRY_DELAY=1.0
OPENAI_BACKOFF_FACTOR=2.0
OPENAI_FALLBACK_ENABLED=true
OPENAI_FALLBACK_MODELS=text-embedding-3-small,text-embedding-ada-002

# Vector Database Configuration
VECTOR_DB=faiss
COLLECTION_NAME=rag-exfil-poc
INDEX_NAME=rag-exfil-poc

# Document Processing
CHUNK_SIZE=512
CHUNK_OVERLAP=50
ENABLE_PREPROCESSING=true

# Steganography Settings
STEGO_ENABLED=true
STEGO_TECHNIQUES=noise,rotation,scaling,offset,fragmentation,interleaving
STEGO_NOISE_LEVEL=0.01
STEGO_DECOY_RATIO=0.4
"""
    
    try:
        with open('.env.template', 'w') as f:
            f.write(env_template)
        logger.info("✅ Created .env.template file")
        logger.info("📝 Copy .env.template to .env and add your API key")
    except Exception as e:
        logger.error(f"❌ Failed to create .env.template: {e}")

def main():
    """Run comprehensive API troubleshooting."""
    logger = setup_logging()
    logger.info("🚀 Starting VectorSmuggle API Connectivity Troubleshooting")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run tests
    tests = [
        ("Environment Variables", check_environment_variables),
        ("Network Connectivity", test_network_connectivity),
        ("OpenAI API Direct", test_openai_api_direct),
        ("LangChain Integration", test_langchain_integration),
        ("VectorSmuggle Components", test_vectorsmuggle_components)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func(logger)
            test_results[test_name.lower().replace(' ', '_')] = result
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"❌ {test_name} crashed: {e}")
            test_results[test_name.lower().replace(' ', '_')] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("📊 TROUBLESHOOTING SUMMARY")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    # Suggest fixes
    if passed < total:
        logger.info(f"\n{'='*60}")
        suggest_fixes(logger, test_results)
        
        # Create env template if env vars failed
        if not test_results.get('environment_variables', False):
            logger.info(f"\n{'='*60}")
            create_env_template(logger)
    
    if passed == total:
        logger.info("🎉 All tests passed! API connectivity is working correctly.")
        return 0
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Follow suggested fixes above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())