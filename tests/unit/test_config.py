"""Unit tests for VectorSmuggle configuration module."""

import os
import pytest
from unittest.mock import patch

from config import Config, get_config


class TestConfig:
    """Test configuration loading and validation."""

    @pytest.mark.unit
    def test_config_defaults(self):
        """Test default configuration values."""
        config = Config()

        assert config.document.chunk_size == 512
        assert config.document.chunk_overlap == 50
        assert config.document.enable_preprocessing is True
        assert config.document.chunking_strategy == "auto"

        assert config.vector_store.type == "faiss"
        assert config.vector_store.collection_name == "rag-exfil-poc"

        assert config.openai.model == "text-embedding-ada-002"
        assert config.openai.max_retries == 3
        assert config.openai.timeout == 30.0

    @pytest.mark.unit
    def test_config_validation_requires_api_key(self):
        """Test that validation requires OPENAI_API_KEY."""
        config = Config()
        config.openai.api_key = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
            config.validate()

    @pytest.mark.unit
    def test_config_validation_chunk_size(self):
        """Test chunk size validation."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env):
            config = Config()
            config.document.chunk_size = -1
            with pytest.raises(ValueError, match="CHUNK_SIZE must be positive"):
                config.validate()

    @pytest.mark.unit
    def test_config_validation_chunk_overlap(self):
        """Test chunk overlap validation."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env):
            config = Config()
            config.document.chunk_overlap = -1
            with pytest.raises(ValueError, match="CHUNK_OVERLAP cannot be negative"):
                config.validate()

    @pytest.mark.unit
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        env_vars = {
            "OPENAI_API_KEY": "sk-test123",
            "CHUNK_SIZE": "1024",
            "VECTOR_DB": "qdrant",
            "QDRANT_URL": "http://test:6333"
        }

        with patch.dict(os.environ, env_vars):
            config = Config()

            assert config.openai.api_key == "sk-test123"
            assert config.document.chunk_size == 1024
            assert config.vector_store.type == "qdrant"
            assert config.vector_store.qdrant_url == "http://test:6333"

    @pytest.mark.unit
    def test_steganography_config(self):
        """Test steganography configuration."""
        config = Config()

        assert hasattr(config, "steganography")
        assert config.steganography.enabled is True
        assert config.steganography.noise_level == 0.01
        assert "noise" in config.steganography.techniques
        assert "rotation" in config.steganography.techniques

    @pytest.mark.unit
    def test_steganography_config_from_env(self):
        """Test steganography configuration from environment variables."""
        env = {
            "STEGO_NOISE_LEVEL": "0.05",
            "STEGO_ROTATION_ANGLE": "0.5",
            "STEGO_ENABLED": "false",
        }
        with patch.dict(os.environ, env):
            config = Config()
            assert config.steganography.noise_level == 0.05
            assert config.steganography.rotation_angle == 0.5
            assert config.steganography.enabled is False

    @pytest.mark.unit
    def test_evasion_config(self):
        """Test evasion configuration."""
        config = Config()

        assert hasattr(config, "evasion")
        assert config.evasion.behavioral_camouflage_enabled is True
        assert config.evasion.traffic_mimicry_enabled is True
        assert 0 <= config.evasion.legitimate_ratio <= 1

    @pytest.mark.unit
    def test_query_config(self):
        """Test query configuration."""
        config = Config()

        assert hasattr(config, "query")
        assert config.query.cache_enabled is True
        assert config.query.similarity_threshold == 0.7
        assert config.query.batch_size == 10

    @pytest.mark.unit
    def test_api_key_handling(self):
        """Test API key loading from environment."""
        config = Config()
        assert config.openai.api_key is None

        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            config = Config()
            assert config.openai.api_key == "sk-test123"

    @pytest.mark.unit
    @pytest.mark.parametrize("vector_store_type", ["faiss", "qdrant", "pinecone"])
    def test_vector_store_type_config(self, vector_store_type):
        """Test different vector store type configurations."""
        env = {"VECTOR_DB": vector_store_type}
        with patch.dict(os.environ, env):
            config = Config()
            assert config.vector_store.type == vector_store_type

        if vector_store_type == "qdrant":
            assert hasattr(config.vector_store, "qdrant_url")

    @pytest.mark.unit
    def test_config_validation_valid_state(self):
        """Test that valid configuration passes validation."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env):
            config = Config()
            config.validate()  # Should not raise

    @pytest.mark.unit
    def test_steganography_validation_noise_level(self):
        """Test steganography noise level validation."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env):
            config = Config()
            config.steganography.noise_level = 1.5
            with pytest.raises(ValueError, match="STEGO_NOISE_LEVEL"):
                config.validate()

    @pytest.mark.unit
    def test_steganography_validation_rotation_angle(self):
        """Test steganography rotation angle validation."""
        env = {"OPENAI_API_KEY": "sk-test123"}
        with patch.dict(os.environ, env):
            config = Config()
            config.steganography.rotation_angle = 4.0
            with pytest.raises(ValueError, match="STEGO_ROTATION_ANGLE"):
                config.validate()

    @pytest.mark.unit
    def test_openai_fallback_models(self):
        """Test OpenAI fallback model configuration."""
        config = Config()
        assert config.openai.fallback_enabled is True
        assert isinstance(config.openai.fallback_models, list)
        assert len(config.openai.fallback_models) > 0

    @pytest.mark.unit
    def test_config_chunk_size_env_override(self):
        """Test chunk size can be overridden via environment."""
        with patch.dict(os.environ, {"CHUNK_SIZE": "2048"}):
            config = Config()
            assert config.document.chunk_size == 2048
