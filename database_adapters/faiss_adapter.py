# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Faiss database adapter for multi-database effectiveness testing."""

import os
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path
from typing import Any

from config import Config

from .base_adapter import DatabaseAdapter


class FaissAdapter(DatabaseAdapter):
    """Adapter for Faiss vector database."""

    def get_connection_config(self) -> dict[str, Any]:
        """Get Faiss connection configuration."""
        return {
            'index_path': os.getenv('FAISS_INDEX_PATH', 'faiss_index'),
            'type': 'faiss'
        }

    def test_connection(self) -> bool:
        """Test Faiss availability."""
        try:
            import faiss
            # Test basic Faiss functionality
            faiss.IndexFlatL2(128)
            return True
        except ImportError:
            self.logger.debug("Faiss not installed")
            return False
        except Exception as e:
            self.logger.debug(f"Faiss test failed: {e}")
            return False

    def run_effectiveness_test(self, config: Config) -> dict[str, Any]:
        """Run effectiveness test against Faiss."""
        try:
            # Set environment variables for Faiss
            env = os.environ.copy()
            env['VECTOR_DB'] = 'faiss'
            env['FAISS_INDEX_PATH'] = self.get_connection_config()['index_path']

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
            self.logger.error(f"Faiss effectiveness test failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def cleanup(self) -> None:
        """Clean up Faiss resources."""
        try:
            # Clean up test index files
            index_path = Path(self.get_connection_config()['index_path'])
            if index_path.exists():
                if index_path.is_dir():
                    import shutil
                    shutil.rmtree(index_path)
                else:
                    index_path.unlink()
                self.logger.debug(f"Cleaned up Faiss index: {index_path}")
        except Exception as e:
            self.logger.warning(f"Faiss cleanup failed: {e}")

    def get_setup_instructions(self) -> str:
        """Get Faiss setup instructions."""
        return """
Faiss Setup Instructions:

1. Install Faiss:
   pip install faiss-cpu
   # OR for GPU support:
   pip install faiss-gpu

2. Environment Variables:
   FAISS_INDEX_PATH=faiss_index

3. Verify Installation:
   python -c "import faiss; print('Faiss installed successfully')"

4. No Docker setup required - Faiss runs locally

5. Troubleshooting:
   - Ensure faiss-cpu or faiss-gpu is installed
   - Check that the index path is writable
   - For GPU version, ensure CUDA is properly installed
"""
