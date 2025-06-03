"""Operational security (OpSec) module for artifact cleanup and evidence removal."""

import logging
import os
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import psutil


class OperationalSecurity:
    """Implements operational security measures for evidence removal and process isolation."""

    def __init__(
        self,
        temp_dir: str | None = None,
        log_retention_hours: int = 24,
        auto_cleanup: bool = True,
        secure_delete_passes: int = 3,
        memory_cleanup_interval: int = 300
    ):
        """
        Initialize operational security system.

        Args:
            temp_dir: Custom temporary directory path
            log_retention_hours: Hours to retain logs before cleanup
            auto_cleanup: Whether to automatically clean up artifacts
            secure_delete_passes: Number of passes for secure file deletion
            memory_cleanup_interval: Interval for memory cleanup in seconds
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "vectorsmuggle"
        self.log_retention_hours = log_retention_hours
        self.auto_cleanup = auto_cleanup
        self.secure_delete_passes = secure_delete_passes
        self.memory_cleanup_interval = memory_cleanup_interval

        self.temp_files: set[Path] = set()
        self.log_files: set[Path] = set()
        self.process_artifacts: dict[str, Any] = {}
        self.cleanup_thread: threading.Thread | None = None
        self.cleanup_running = False

        self.logger = logging.getLogger(__name__)

        # Create secure temp directory
        self._setup_temp_directory()

        if self.auto_cleanup:
            self._start_cleanup_thread()

    def _setup_temp_directory(self) -> None:
        """Set up secure temporary directory with restricted permissions."""
        try:
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            # Set restrictive permissions (owner only)
            os.chmod(self.temp_dir, 0o700)

            self.logger.info(f"Initialized secure temp directory: {self.temp_dir}")

        except Exception as e:
            self.logger.error(f"Failed to setup temp directory: {e}")
            # Fallback to system temp
            self.temp_dir = Path(tempfile.gettempdir())

    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            return

        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()

        self.logger.info("Started background cleanup thread")

    def _cleanup_worker(self) -> None:
        """Background worker for periodic cleanup."""
        while self.cleanup_running:
            try:
                self._periodic_cleanup()
                time.sleep(self.memory_cleanup_interval)
            except Exception as e:
                self.logger.error(f"Cleanup worker error: {e}")

    def _periodic_cleanup(self) -> None:
        """Perform periodic cleanup of temporary artifacts."""
        current_time = datetime.utcnow()

        # Clean up old temporary files
        files_to_remove = set()
        for temp_file in self.temp_files:
            if temp_file.exists():
                file_age = current_time - datetime.fromtimestamp(temp_file.stat().st_mtime)
                if file_age > timedelta(hours=1):  # Remove files older than 1 hour
                    files_to_remove.add(temp_file)

        for temp_file in files_to_remove:
            self.secure_delete_file(temp_file)
            self.temp_files.discard(temp_file)

        # Clean up old log files
        logs_to_remove = set()
        for log_file in self.log_files:
            if log_file.exists():
                file_age = current_time - datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_age > timedelta(hours=self.log_retention_hours):
                    logs_to_remove.add(log_file)

        for log_file in logs_to_remove:
            self.secure_delete_file(log_file)
            self.log_files.discard(log_file)

    def create_temp_file(self, suffix: str = "", prefix: str = "vs_") -> Path:
        """
        Create a temporary file with automatic cleanup tracking.

        Args:
            suffix: File suffix
            prefix: File prefix

        Returns:
            Path to temporary file
        """
        temp_file = self.temp_dir / f"{prefix}{int(time.time() * 1000000)}{suffix}"

        # Create the file
        temp_file.touch()
        os.chmod(temp_file, 0o600)  # Owner read/write only

        # Track for cleanup
        self.temp_files.add(temp_file)

        self.logger.debug(f"Created temp file: {temp_file}")
        return temp_file

    def create_temp_directory(self, prefix: str = "vs_dir_") -> Path:
        """
        Create a temporary directory with automatic cleanup tracking.

        Args:
            prefix: Directory prefix

        Returns:
            Path to temporary directory
        """
        temp_dir = self.temp_dir / f"{prefix}{int(time.time() * 1000000)}"

        # Create the directory
        temp_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(temp_dir, 0o700)  # Owner access only

        # Track for cleanup
        self.temp_files.add(temp_dir)

        self.logger.debug(f"Created temp directory: {temp_dir}")
        return temp_dir

    def secure_delete_file(self, file_path: Path | str) -> bool:
        """
        Securely delete a file by overwriting it multiple times.

        Args:
            file_path: Path to file to delete

        Returns:
            True if successful, False otherwise
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return True

        try:
            if file_path.is_file():
                # Get file size
                file_size = file_path.stat().st_size

                # Overwrite with random data multiple times
                with open(file_path, "r+b") as f:
                    for pass_num in range(self.secure_delete_passes):
                        f.seek(0)
                        # Write random data
                        remaining = file_size
                        while remaining > 0:
                            chunk_size = min(8192, remaining)
                            random_data = os.urandom(chunk_size)
                            f.write(random_data)
                            remaining -= chunk_size
                        f.flush()
                        os.fsync(f.fileno())

                # Finally delete the file
                file_path.unlink()

            elif file_path.is_dir():
                # Recursively secure delete directory contents
                for item in file_path.rglob("*"):
                    if item.is_file():
                        self.secure_delete_file(item)

                # Remove empty directory
                shutil.rmtree(file_path, ignore_errors=True)

            self.logger.debug(f"Securely deleted: {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to securely delete {file_path}: {e}")
            return False

    def setup_secure_logging(
        self,
        log_file: Path | str | None = None,
        max_log_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 0
    ) -> logging.Handler:
        """
        Set up secure logging with automatic cleanup.

        Args:
            log_file: Path to log file (None for temp file)
            max_log_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep (0 = no backups)

        Returns:
            Configured log handler
        """
        if log_file is None:
            log_file = self.create_temp_file(suffix=".log", prefix="vs_log_")
        else:
            log_file = Path(log_file)

        # Track log file for cleanup
        self.log_files.add(log_file)

        # Create rotating file handler
        from logging.handlers import RotatingFileHandler

        handler = RotatingFileHandler(
            log_file,
            maxBytes=max_log_size,
            backupCount=backup_count
        )

        # Set secure permissions
        os.chmod(log_file, 0o600)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        self.logger.info(f"Set up secure logging: {log_file}")
        return handler

    def clear_memory_artifacts(self) -> None:
        """Clear sensitive data from memory where possible."""
        try:
            # Force garbage collection
            import gc
            gc.collect()

            # Clear Python's internal caches
            import sys
            if hasattr(sys, 'intern'):
                # Clear interned strings (Python 3)
                pass  # No direct way to clear intern cache

            self.logger.debug("Cleared memory artifacts")

        except Exception as e:
            self.logger.error(f"Failed to clear memory artifacts: {e}")

    def monitor_process_artifacts(self) -> dict[str, Any]:
        """Monitor current process for potential artifacts."""
        try:
            current_process = psutil.Process()

            artifacts = {
                "pid": current_process.pid,
                "name": current_process.name(),
                "cmdline": current_process.cmdline(),
                "cwd": current_process.cwd(),
                "memory_info": current_process.memory_info()._asdict(),
                "open_files": [],
                "connections": [],
                "threads": current_process.num_threads()
            }

            # Get open files (may require elevated permissions)
            try:
                artifacts["open_files"] = [f.path for f in current_process.open_files()]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

            # Get network connections
            try:
                artifacts["connections"] = [
                    {
                        "fd": conn.fd,
                        "family": conn.family,
                        "type": conn.type,
                        "laddr": conn.laddr,
                        "raddr": conn.raddr,
                        "status": conn.status
                    }
                    for conn in current_process.connections()
                ]
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass

            self.process_artifacts = artifacts
            return artifacts

        except Exception as e:
            self.logger.error(f"Failed to monitor process artifacts: {e}")
            return {}

    def create_process_sandbox(self) -> dict[str, Any]:
        """Create a sandboxed environment for sensitive operations."""
        sandbox_info = {
            "temp_dir": self.create_temp_directory(prefix="sandbox_"),
            "original_cwd": Path.cwd(),
            "env_backup": os.environ.copy(),
            "created_at": datetime.utcnow()
        }

        try:
            # Change to sandbox directory
            os.chdir(sandbox_info["temp_dir"])

            # Clear sensitive environment variables
            sensitive_env_vars = [
                "OPENAI_API_KEY", "PINECONE_API_KEY", "QDRANT_URL",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
            ]

            for var in sensitive_env_vars:
                if var in os.environ:
                    del os.environ[var]

            self.logger.info(f"Created process sandbox: {sandbox_info['temp_dir']}")
            return sandbox_info

        except Exception as e:
            self.logger.error(f"Failed to create sandbox: {e}")
            return sandbox_info

    def restore_from_sandbox(self, sandbox_info: dict[str, Any]) -> None:
        """Restore environment from sandbox."""
        try:
            # Restore working directory
            os.chdir(sandbox_info["original_cwd"])

            # Restore environment variables
            os.environ.clear()
            os.environ.update(sandbox_info["env_backup"])

            # Clean up sandbox
            self.secure_delete_file(sandbox_info["temp_dir"])

            self.logger.info("Restored from sandbox")

        except Exception as e:
            self.logger.error(f"Failed to restore from sandbox: {e}")

    def wipe_swap_space(self) -> bool:
        """Attempt to clear swap space (requires elevated permissions)."""
        try:
            # This is a simplified approach - in practice, this would require
            # platform-specific implementations and elevated permissions

            if os.name == 'posix':
                # On Linux, we could use swapoff/swapon but requires root
                self.logger.warning("Swap space wiping requires elevated permissions")
                return False
            else:
                self.logger.warning("Swap space wiping not implemented for this platform")
                return False

        except Exception as e:
            self.logger.error(f"Failed to wipe swap space: {e}")
            return False

    def emergency_cleanup(self) -> None:
        """Perform emergency cleanup of all tracked artifacts."""
        self.logger.warning("Performing emergency cleanup")

        # Stop cleanup thread
        self.cleanup_running = False
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)

        # Delete all temp files
        for temp_file in self.temp_files.copy():
            self.secure_delete_file(temp_file)
        self.temp_files.clear()

        # Delete all log files
        for log_file in self.log_files.copy():
            self.secure_delete_file(log_file)
        self.log_files.clear()

        # Clear memory
        self.clear_memory_artifacts()

        # Remove temp directory
        if self.temp_dir.exists():
            self.secure_delete_file(self.temp_dir)

        self.logger.info("Emergency cleanup completed")

    def get_opsec_statistics(self) -> dict[str, Any]:
        """Get operational security statistics."""
        stats = {
            "temp_files_tracked": len(self.temp_files),
            "log_files_tracked": len(self.log_files),
            "temp_directory": str(self.temp_dir),
            "auto_cleanup_enabled": self.auto_cleanup,
            "cleanup_thread_running": self.cleanup_running,
            "log_retention_hours": self.log_retention_hours,
            "secure_delete_passes": self.secure_delete_passes,
            "memory_cleanup_interval": self.memory_cleanup_interval
        }

        # Add process artifacts if available
        if self.process_artifacts:
            stats["process_artifacts"] = self.process_artifacts

        return stats

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            if self.auto_cleanup:
                self.emergency_cleanup()
        except Exception:
            pass  # Ignore errors during cleanup
