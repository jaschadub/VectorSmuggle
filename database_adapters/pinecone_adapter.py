# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Pinecone database adapter for multi-database effectiveness testing."""

import os
import subprocess  # nosec B404
import sys
import tempfile
from typing import Any

from config import Config

from .base_adapter import DatabaseAdapter


class PineconeAdapter(DatabaseAdapter):
    """Adapter for Pinecone vector database."""

    def get_connection_config(self) -> dict[str, Any]:
        """Get Pinecone connection configuration."""
        return {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp'),
            'index_name': os.getenv('INDEX_NAME', 'rag-exfil-poc'),
            'type': 'pinecone'
        }

    def test_connection(self) -> bool:
        """Test Pinecone connection."""
        try:
            config = self.get_connection_config()
            if not config['api_key']:
                self.logger.debug("Pinecone API key not configured")
                return False

            import pinecone
            pinecone.init(
                api_key=config['api_key'],
                environment=config['environment']
            )

            # Test by listing indexes
            pinecone.list_indexes()
            return True
        except ImportError:
            self.logger.debug("Pinecone client not installed")
            return False
        except Exception as e:
            self.logger.debug(f"Pinecone connection test failed: {e}")
            return False

    def run_effectiveness_test(self, config: Config) -> dict[str, Any]:
        """Run effectiveness test against Pinecone."""
        try:
            # Check if API key is configured
            pinecone_config = self.get_connection_config()
            if not pinecone_config['api_key']:
                return {
                    'success': False,
                    'error': 'PINECONE_API_KEY not configured'
                }

            # Set environment variables for Pinecone
            env = os.environ.copy()
            env['VECTOR_DB'] = 'pinecone'
            env['PINECONE_API_KEY'] = pinecone_config['api_key']
            env['PINECONE_ENVIRONMENT'] = pinecone_config['environment']
            env['INDEX_NAME'] = pinecone_config['index_name']

            # Create temporary script to run the effectiveness test
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write("""
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_effectiveness_report import generate_report

try:
    result = generate_report()
    print(json.dumps(result, default=str))
except Exception as e:
    print(json.dumps({'error': str(e), 'success': False}))
""")
                temp_script = f.name

            try:
                # Run the effectiveness test
                result = subprocess.run(  # nosec B603
                    [sys.executable, temp_script],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=os.getcwd()
                )

                if result.returncode == 0:
                    import json
                    return json.loads(result.stdout)
                else:
                    return {
                        'success': False,
                        'error': f"Test failed with return code {result.returncode}",
                        'stderr': result.stderr
                    }
            finally:
                os.unlink(temp_script)

        except Exception as e:
            self.logger.error(f"Pinecone effectiveness test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cleanup(self) -> None:
        """Clean up Pinecone resources."""
        try:
            # Optional: Clean up test indexes
            # Note: Be careful with cleanup in production environments
            pass
        except Exception as e:
            self.logger.warning(f"Pinecone cleanup failed: {e}")

    def get_setup_instructions(self) -> str:
        """Get Pinecone setup instructions."""
        return """
Pinecone Setup Instructions:

1. Install Pinecone client:
   pip install pinecone-client

2. Get API Key:
   - Sign up at https://www.pinecone.io/
   - Create a project and get your API key
   - Note your environment (e.g., us-west1-gcp)

3. Environment Variables:
   PINECONE_API_KEY=your-api-key-here
   PINECONE_ENVIRONMENT=us-west1-gcp
   INDEX_NAME=rag-exfil-poc

4. Verify Setup:
   python -c "import pinecone; print('Pinecone client installed')"

5. No Docker setup required - Pinecone is cloud-based

6. Troubleshooting:
   - Verify API key is correct
   - Check environment name matches your Pinecone project
   - Ensure you have sufficient Pinecone quota
   - Check network connectivity to Pinecone services
"""
