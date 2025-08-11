# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Qdrant database adapter for multi-database effectiveness testing."""

import os
import subprocess  # nosec B404
import sys
import tempfile
from typing import Any

from config import Config

from .base_adapter import DatabaseAdapter


class QdrantAdapter(DatabaseAdapter):
    """Adapter for Qdrant vector database."""

    def get_connection_config(self) -> dict[str, Any]:
        """Get Qdrant connection configuration."""
        return {
            'url': os.getenv('QDRANT_URL', 'http://localhost:6334'),
            'collection_name': os.getenv('COLLECTION_NAME', 'rag-exfil-poc'),
            'type': 'qdrant'
        }

    def test_connection(self) -> bool:
        """Test Qdrant connection."""
        try:
            import requests
            config = self.get_connection_config()
            response = requests.get(f"{config['url']}/health", timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.debug(f"Qdrant connection test failed: {e}")
            return False

    def run_effectiveness_test(self, config: Config) -> dict[str, Any]:
        """Run effectiveness test against Qdrant."""
        try:
            # Set environment variables for Qdrant
            env = os.environ.copy()
            env['VECTOR_DB'] = 'qdrant'
            env['QDRANT_URL'] = self.get_connection_config()['url']
            env['COLLECTION_NAME'] = self.get_connection_config()['collection_name']

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
            self.logger.error(f"Qdrant effectiveness test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cleanup(self) -> None:
        """Clean up Qdrant resources."""
        try:
            # Optional: Clean up test collections
            pass
        except Exception as e:
            self.logger.warning(f"Qdrant cleanup failed: {e}")

    def get_setup_instructions(self) -> str:
        """Get Qdrant setup instructions."""
        return """
Qdrant Setup Instructions:

1. Using Docker Compose (Recommended):
   cd test_vector_dbs_docker/
   docker compose up -d qdrant

2. Verify Qdrant is running:
   curl http://localhost:6334/health
   Expected response: {"status":"ok"}

3. Environment Variables:
   QDRANT_URL=http://localhost:6334
   COLLECTION_NAME=rag-exfil-poc

4. Manual Docker Setup:
   docker run -p 6334:6333 qdrant/qdrant:v1.9.1

5. Troubleshooting:
   - Check if port 6334 is available
   - Verify Docker container is running: docker ps
   - Check logs: docker logs local-qdrant
"""
