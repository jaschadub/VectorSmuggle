"""Embedding factory with fallback support for Ollama when OpenAI is unavailable."""

import logging
import os
from typing import Any, Optional

try:
    from langchain_openai import OpenAIEmbeddings
except ImportError:
    OpenAIEmbeddings = None

try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None


class EmbeddingFactory:
    """Factory for creating embedding models with automatic fallback support."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the embedding factory.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def create_embeddings(self, config: Any = None, prefer_ollama: bool = False) -> Any:
        """Create embedding model with fallback support.

        Args:
            config: Configuration object with OpenAI settings
            prefer_ollama: If True, try Ollama first instead of OpenAI

        Returns:
            Embedding model instance (OpenAI or Ollama)

        Raises:
            RuntimeError: If no embedding models are available
        """
        if prefer_ollama:
            # Try Ollama first if preferred
            ollama_embeddings = self._try_ollama()
            if ollama_embeddings:
                return ollama_embeddings

            # Fall back to OpenAI
            openai_embeddings = self._try_openai(config)
            if openai_embeddings:
                return openai_embeddings
        else:
            # Try OpenAI first (default behavior)
            openai_embeddings = self._try_openai(config)
            if openai_embeddings:
                return openai_embeddings

            # Fall back to Ollama
            ollama_embeddings = self._try_ollama()
            if ollama_embeddings:
                return ollama_embeddings

        raise RuntimeError(
            "No embedding models available. Please ensure either:\n"
            "1. OpenAI API key is set and valid, or\n"
            "2. Ollama is running locally with nomic-embed-text:latest model"
        )

    def _try_openai(self, config: Any = None) -> Optional[Any]:
        """Try to create OpenAI embeddings.

        Args:
            config: Configuration object with OpenAI settings

        Returns:
            OpenAI embeddings instance or None if failed
        """
        if OpenAIEmbeddings is None:
            self.logger.debug("OpenAI embeddings not available (package not installed)")
            return None

        try:
            # Get API key from config or environment
            api_key = None
            model = "text-embedding-3-large"  # Default model

            if config and hasattr(config, 'openai'):
                api_key = getattr(config.openai, 'api_key', None)
                model = getattr(config.openai, 'model', model)

            if not api_key:
                api_key = os.getenv('OPENAI_API_KEY')

            if not api_key:
                self.logger.debug("OpenAI API key not found")
                return None

            self.logger.info(f"Initializing OpenAI embeddings with model: {model}")
            embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=model
            )

            # Test the embeddings with a simple query
            test_embedding = embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                self.logger.info("OpenAI embeddings initialized successfully")
                return embeddings
            else:
                self.logger.warning("OpenAI embeddings test failed")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
            return None

    def _try_ollama(self) -> Optional[Any]:
        """Try to create Ollama embeddings.

        Returns:
            Ollama embeddings instance or None if failed
        """
        if OllamaEmbeddings is None:
            self.logger.debug("Ollama embeddings not available (package not installed)")
            return None

        try:
            # Default Ollama configuration
            model = "nomic-embed-text:latest"
            base_url = "http://localhost:11434"

            # Check if custom Ollama settings are available in environment
            ollama_model = os.getenv('OLLAMA_EMBEDDING_MODEL', model)
            ollama_url = os.getenv('OLLAMA_BASE_URL', base_url)

            self.logger.info(f"Initializing Ollama embeddings with model: {ollama_model}")
            embeddings = OllamaEmbeddings(
                model=ollama_model,
                base_url=ollama_url
            )

            # Test the embeddings with a simple query
            test_embedding = embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                self.logger.info("Ollama embeddings initialized successfully")
                return embeddings
            else:
                self.logger.warning("Ollama embeddings test failed")
                return None

        except Exception as e:
            self.logger.warning(f"Failed to initialize Ollama embeddings: {e}")
            return None

    def get_available_providers(self) -> list[str]:
        """Get list of available embedding providers.

        Returns:
            List of available provider names
        """
        providers = []

        if OpenAIEmbeddings is not None:
            providers.append("openai")

        if OllamaEmbeddings is not None:
            providers.append("ollama")

        return providers

    def check_provider_availability(self, provider: str) -> bool:
        """Check if a specific provider is available and working.

        Args:
            provider: Provider name ("openai" or "ollama")

        Returns:
            True if provider is available and working
        """
        if provider == "openai":
            return self._try_openai() is not None
        elif provider == "ollama":
            return self._try_ollama() is not None
        else:
            return False


# Convenience function for backward compatibility
def create_embeddings(config: Any = None, logger: Optional[logging.Logger] = None, prefer_ollama: bool = False) -> Any:
    """Create embedding model with automatic fallback.

    Args:
        config: Configuration object with OpenAI settings
        logger: Optional logger instance
        prefer_ollama: If True, try Ollama first instead of OpenAI

    Returns:
        Embedding model instance
    """
    factory = EmbeddingFactory(logger)
    return factory.create_embeddings(config, prefer_ollama)
