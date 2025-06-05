# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Configuration management for VectorSmuggle."""

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass
class VectorStoreConfig:
    """Configuration for vector store settings."""

    type: str = "faiss"
    collection_name: str = "rag-exfil-poc"
    index_name: str = "rag-exfil-poc"

    # FAISS specific
    faiss_index_path: str = "faiss_index"

    # Qdrant specific
    qdrant_url: str = "http://localhost:6333"

    # Pinecone specific
    pinecone_environment: str = "us-west1-gcp"


@dataclass
class DocumentConfig:
    """Configuration for document processing."""

    chunk_size: int = 512
    chunk_overlap: int = 50
    document_path: str = "../internal_docs/strategic_roadmap.pdf"

    # Multi-format support
    supported_formats: list[str] = None
    batch_processing: bool = True
    max_files_per_batch: int = 10

    # Content preprocessing
    enable_preprocessing: bool = True
    sanitize_content: bool = False
    detect_sensitive_data: bool = True
    chunking_strategy: str = "auto"  # auto, fixed, semantic

    # Format-specific settings
    office_extract_tables: bool = True
    csv_delimiter: str = "auto"
    json_flatten_nested: bool = False
    email_include_attachments: bool = True
    database_query: str = ""

    def __post_init__(self):
        """Set default supported formats if not provided."""
        if self.supported_formats is None:
            self.supported_formats = [
                ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".json",
                ".xml", ".txt", ".md", ".eml", ".msg", ".mbox",
                ".yaml", ".yml", ".html", ".htm", ".db", ".sqlite", ".sqlite3"
            ]


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI settings."""

    api_key: str | None = None
    model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-3.5-turbo-instruct"

    # API reliability settings
    max_retries: int = 3
    timeout: float = 30.0
    retry_delay: float = 1.0
    backoff_factor: float = 2.0

    # Fallback options
    fallback_enabled: bool = True
    fallback_models: list[str] = None

    def __post_init__(self):
        """Set default fallback models if not provided."""
        if self.fallback_models is None:
            self.fallback_models = [
                "text-embedding-3-small",
                "text-embedding-ada-002"
            ]


@dataclass
class SteganographyConfig:
    """Configuration for steganographic techniques."""

    enabled: bool = True
    noise_level: float = 0.01
    rotation_angle: float = 0.1
    scaling_factor: float = 0.95
    offset_range: float = 0.05
    fragment_size: int = 128
    interleave_ratio: float = 0.3
    decoy_ratio: float = 0.4
    base_delay: float = 60.0
    delay_variance: float = 0.3
    batch_size: int = 5
    max_batches_per_hour: int = 10
    business_hours_only: bool = True
    timezone_offset: int = 0
    fragment_strategy: str = "round_robin"
    decoy_category: str = "general"
    techniques: list[str] = None

    def __post_init__(self):
        """Set default techniques if not provided."""
        if self.techniques is None:
            self.techniques = ["noise", "rotation", "scaling", "offset", "fragmentation", "interleaving"]


@dataclass
class EvasionConfig:
    """Configuration for advanced evasion techniques."""

    # Traffic mimicry settings
    traffic_mimicry_enabled: bool = True
    base_query_interval: float = 300.0
    query_variance: float = 0.4
    burst_probability: float = 0.15
    user_profiles: list[str] = None

    # Behavioral camouflage settings
    behavioral_camouflage_enabled: bool = True
    legitimate_ratio: float = 0.8
    activity_mixing_strategy: str = "interleaved"
    cover_story_enabled: bool = True

    # Network evasion settings
    network_evasion_enabled: bool = True
    proxy_rotation_enabled: bool = False
    user_agent_rotation: bool = True
    rate_limit_delay: tuple[float, float] = (1.0, 5.0)
    connection_timeout: float = 30.0
    max_retries: int = 3

    # Operational security settings
    opsec_enabled: bool = True
    auto_cleanup: bool = True
    log_retention_hours: int = 24
    secure_delete_passes: int = 3
    temp_dir_custom: str = ""

    # Detection avoidance settings
    detection_avoidance_enabled: bool = True
    dlp_keyword_avoidance: bool = True
    content_transformation_strength: float = 0.3
    statistical_noise_level: float = 0.1
    signature_obfuscation: bool = True

    def __post_init__(self):
        """Set default user profiles if not provided."""
        if self.user_profiles is None:
            self.user_profiles = ["researcher", "analyst", "developer", "manager"]


@dataclass
class QueryConfig:
    """Configuration for enhanced query capabilities."""

    # Caching settings
    cache_enabled: bool = True
    cache_dir: str = ".query_cache"
    cache_max_size: int = 1000

    # Batch processing
    batch_size: int = 10

    # Similarity and retrieval settings
    similarity_threshold: float = 0.7
    adaptive_retrieval: bool = True
    performance_tracking: bool = True

    # Advanced features
    multi_step_reasoning: bool = True
    context_reconstruction: bool = True
    cross_reference_analysis: bool = True
    data_recovery: bool = True
    semantic_clustering: bool = True
    entity_extraction: bool = True

    # Optimization parameters
    embedding_cache_size: int = 5000
    result_ranking: bool = True
    expansion_enabled: bool = True
    strategy_recommendation: bool = True


@dataclass
class Config:
    """Main configuration class."""

    def __init__(self):
        # Parse fallback models from environment
        fallback_models_str = os.getenv("OPENAI_FALLBACK_MODELS", "text-embedding-3-small,text-embedding-ada-002")
        fallback_models = [m.strip() for m in fallback_models_str.split(",") if m.strip()]

        self.openai = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002"),
            llm_model=os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo-instruct"),
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "3")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "30.0")),
            retry_delay=float(os.getenv("OPENAI_RETRY_DELAY", "1.0")),
            backoff_factor=float(os.getenv("OPENAI_BACKOFF_FACTOR", "2.0")),
            fallback_enabled=os.getenv("OPENAI_FALLBACK_ENABLED", "true").lower() == "true",
            fallback_models=fallback_models
        )

        self.vector_store = VectorStoreConfig(
            type=os.getenv("VECTOR_DB", "faiss"),
            collection_name=os.getenv("COLLECTION_NAME", "rag-exfil-poc"),
            index_name=os.getenv("INDEX_NAME", "rag-exfil-poc"),
            faiss_index_path=os.getenv("FAISS_INDEX_PATH", "faiss_index"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            pinecone_environment=os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
        )

        self.document = DocumentConfig(
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
            document_path=os.getenv("DOCUMENT_PATH", "../internal_docs/strategic_roadmap.pdf"),
            batch_processing=os.getenv("BATCH_PROCESSING", "true").lower() == "true",
            max_files_per_batch=int(os.getenv("MAX_FILES_PER_BATCH", "10")),
            enable_preprocessing=os.getenv("ENABLE_PREPROCESSING", "true").lower() == "true",
            sanitize_content=os.getenv("SANITIZE_CONTENT", "false").lower() == "true",
            detect_sensitive_data=os.getenv("DETECT_SENSITIVE_DATA", "true").lower() == "true",
            chunking_strategy=os.getenv("CHUNKING_STRATEGY", "auto"),
            office_extract_tables=os.getenv("OFFICE_EXTRACT_TABLES", "true").lower() == "true",
            csv_delimiter=os.getenv("CSV_DELIMITER", "auto"),
            json_flatten_nested=os.getenv("JSON_FLATTEN_NESTED", "false").lower() == "true",
            email_include_attachments=os.getenv("EMAIL_INCLUDE_ATTACHMENTS", "true").lower() == "true",
            database_query=os.getenv("DATABASE_QUERY", "")
        )

        # Parse steganography techniques from environment
        techniques_str = os.getenv("STEGO_TECHNIQUES", "noise,rotation,scaling,offset,fragmentation,interleaving")
        techniques = [t.strip() for t in techniques_str.split(",") if t.strip()]

        self.steganography = SteganographyConfig(
            enabled=os.getenv("STEGO_ENABLED", "true").lower() == "true",
            noise_level=float(os.getenv("STEGO_NOISE_LEVEL", "0.01")),
            rotation_angle=float(os.getenv("STEGO_ROTATION_ANGLE", "0.1")),
            scaling_factor=float(os.getenv("STEGO_SCALING_FACTOR", "0.95")),
            offset_range=float(os.getenv("STEGO_OFFSET_RANGE", "0.05")),
            fragment_size=int(os.getenv("STEGO_FRAGMENT_SIZE", "128")),
            interleave_ratio=float(os.getenv("STEGO_INTERLEAVE_RATIO", "0.3")),
            decoy_ratio=float(os.getenv("STEGO_DECOY_RATIO", "0.4")),
            base_delay=float(os.getenv("STEGO_BASE_DELAY", "60.0")),
            delay_variance=float(os.getenv("STEGO_DELAY_VARIANCE", "0.3")),
            batch_size=int(os.getenv("STEGO_BATCH_SIZE", "5")),
            max_batches_per_hour=int(os.getenv("STEGO_MAX_BATCHES_PER_HOUR", "10")),
            business_hours_only=os.getenv("STEGO_BUSINESS_HOURS_ONLY", "true").lower() == "true",
            timezone_offset=int(os.getenv("STEGO_TIMEZONE_OFFSET", "0")),
            fragment_strategy=os.getenv("STEGO_FRAGMENT_STRATEGY", "round_robin"),
            decoy_category=os.getenv("STEGO_DECOY_CATEGORY", "general"),
            techniques=techniques
        )

        # Parse evasion user profiles from environment
        user_profiles_str = os.getenv("EVASION_USER_PROFILES", "researcher,analyst,developer,manager")
        user_profiles = [p.strip() for p in user_profiles_str.split(",") if p.strip()]

        # Parse rate limit delay tuple
        rate_delay_str = os.getenv("EVASION_RATE_LIMIT_DELAY", "1.0,5.0")
        try:
            rate_delay_parts = [float(x.strip()) for x in rate_delay_str.split(",")]
            rate_limit_delay = (rate_delay_parts[0], rate_delay_parts[1]) if len(rate_delay_parts) >= 2 else (1.0, 5.0)
        except (ValueError, IndexError):
            rate_limit_delay = (1.0, 5.0)

        self.evasion = EvasionConfig(
            traffic_mimicry_enabled=os.getenv("EVASION_TRAFFIC_MIMICRY", "true").lower() == "true",
            base_query_interval=float(os.getenv("EVASION_BASE_QUERY_INTERVAL", "300.0")),
            query_variance=float(os.getenv("EVASION_QUERY_VARIANCE", "0.4")),
            burst_probability=float(os.getenv("EVASION_BURST_PROBABILITY", "0.15")),
            user_profiles=user_profiles,
            behavioral_camouflage_enabled=os.getenv("EVASION_BEHAVIORAL_CAMOUFLAGE", "true").lower() == "true",
            legitimate_ratio=float(os.getenv("EVASION_LEGITIMATE_RATIO", "0.8")),
            activity_mixing_strategy=os.getenv("EVASION_MIXING_STRATEGY", "interleaved"),
            cover_story_enabled=os.getenv("EVASION_COVER_STORY", "true").lower() == "true",
            network_evasion_enabled=os.getenv("EVASION_NETWORK", "true").lower() == "true",
            proxy_rotation_enabled=os.getenv("EVASION_PROXY_ROTATION", "false").lower() == "true",
            user_agent_rotation=os.getenv("EVASION_USER_AGENT_ROTATION", "true").lower() == "true",
            rate_limit_delay=rate_limit_delay,
            connection_timeout=float(os.getenv("EVASION_CONNECTION_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("EVASION_MAX_RETRIES", "3")),
            opsec_enabled=os.getenv("EVASION_OPSEC", "true").lower() == "true",
            auto_cleanup=os.getenv("EVASION_AUTO_CLEANUP", "true").lower() == "true",
            log_retention_hours=int(os.getenv("EVASION_LOG_RETENTION_HOURS", "24")),
            secure_delete_passes=int(os.getenv("EVASION_SECURE_DELETE_PASSES", "3")),
            temp_dir_custom=os.getenv("EVASION_TEMP_DIR", ""),
            detection_avoidance_enabled=os.getenv("EVASION_DETECTION_AVOIDANCE", "true").lower() == "true",
            dlp_keyword_avoidance=os.getenv("EVASION_DLP_AVOIDANCE", "true").lower() == "true",
            content_transformation_strength=float(os.getenv("EVASION_TRANSFORMATION_STRENGTH", "0.3")),
            statistical_noise_level=float(os.getenv("EVASION_STATISTICAL_NOISE", "0.1")),
            signature_obfuscation=os.getenv("EVASION_SIGNATURE_OBFUSCATION", "true").lower() == "true"
        )

        self.query = QueryConfig(
            cache_enabled=os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true",
            cache_dir=os.getenv("QUERY_CACHE_DIR", ".query_cache"),
            cache_max_size=int(os.getenv("QUERY_CACHE_MAX_SIZE", "1000")),
            batch_size=int(os.getenv("QUERY_BATCH_SIZE", "10")),
            similarity_threshold=float(os.getenv("QUERY_SIMILARITY_THRESHOLD", "0.7")),
            adaptive_retrieval=os.getenv("QUERY_ADAPTIVE_RETRIEVAL", "true").lower() == "true",
            performance_tracking=os.getenv("QUERY_PERFORMANCE_TRACKING", "true").lower() == "true",
            multi_step_reasoning=os.getenv("QUERY_MULTI_STEP_REASONING", "true").lower() == "true",
            context_reconstruction=os.getenv("QUERY_CONTEXT_RECONSTRUCTION", "true").lower() == "true",
            cross_reference_analysis=os.getenv("QUERY_CROSS_REFERENCE_ANALYSIS", "true").lower() == "true",
            data_recovery=os.getenv("QUERY_DATA_RECOVERY", "true").lower() == "true",
            semantic_clustering=os.getenv("QUERY_SEMANTIC_CLUSTERING", "true").lower() == "true",
            entity_extraction=os.getenv("QUERY_ENTITY_EXTRACTION", "true").lower() == "true",
            embedding_cache_size=int(os.getenv("QUERY_EMBEDDING_CACHE_SIZE", "5000")),
            result_ranking=os.getenv("QUERY_RESULT_RANKING", "true").lower() == "true",
            expansion_enabled=os.getenv("QUERY_EXPANSION_ENABLED", "true").lower() == "true",
            strategy_recommendation=os.getenv("QUERY_STRATEGY_RECOMMENDATION", "true").lower() == "true"
        )

    def _get_random_seed(self) -> int | None:
        """
        Get random seed from environment variable.

        Returns:
            Random seed integer or None if not set
        """
        seed_str = os.getenv("RANDOM_SEED")
        if seed_str:
            try:
                return int(seed_str)
            except ValueError as e:
                raise ValueError(f"RANDOM_SEED must be an integer, got: {seed_str}") from e
        return None

    def _initialize_random_generators(self) -> None:
        """
        Initialize all random number generators with the configured seed.

        This ensures deterministic behavior across all randomness sources:
        - Python's random module
        - NumPy's random number generator
        - Any other seeded operations
        """
        if self.random_seed is not None:
            # Seed Python's random module
            random.seed(self.random_seed)

            # Seed NumPy's random number generator
            np.random.seed(self.random_seed)

            # Set environment variable for child processes
            os.environ["PYTHONHASHSEED"] = str(self.random_seed)

    def get_seeded_random_state(self, additional_entropy: str = "") -> np.random.RandomState:
        """
        Get a seeded RandomState instance for deterministic operations.

        Args:
            additional_entropy: Additional string to mix into the seed

        Returns:
            Seeded RandomState instance
        """
        if self.random_seed is None:
            return np.random.RandomState()

        # Create deterministic seed from base seed and additional entropy
        if additional_entropy:
            import hashlib
            combined = f"{self.random_seed}_{additional_entropy}"
            seed_hash = int(hashlib.md5(combined.encode(), usedforsecurity=False).hexdigest()[:8], 16)
            seed = (self.random_seed + seed_hash) % (2**32)
        else:
            seed = self.random_seed

        return np.random.RandomState(seed)

    def validate(self) -> None:
        """Validate configuration settings."""
        if not self.openai.api_key:
            raise ValueError("OPENAI_API_KEY is required")

        if self.vector_store.type not in ["faiss", "qdrant", "pinecone"]:
            raise ValueError(f"Unsupported VECTOR_DB type: {self.vector_store.type}")

        if self.vector_store.type == "pinecone":
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY is required when using Pinecone")

        if self.document.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")

        if self.document.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")

        # Validate steganography settings
        if self.steganography.enabled:
            if not 0.0 <= self.steganography.noise_level <= 1.0:
                raise ValueError("STEGO_NOISE_LEVEL must be between 0.0 and 1.0")

            if not 0.0 <= self.steganography.rotation_angle <= 3.14159:
                raise ValueError("STEGO_ROTATION_ANGLE must be between 0.0 and π")

            if not 0.1 <= self.steganography.scaling_factor <= 2.0:
                raise ValueError("STEGO_SCALING_FACTOR must be between 0.1 and 2.0")

            if not 0.0 <= self.steganography.offset_range <= 1.0:
                raise ValueError("STEGO_OFFSET_RANGE must be between 0.0 and 1.0")

            if self.steganography.fragment_size <= 0:
                raise ValueError("STEGO_FRAGMENT_SIZE must be positive")

            if not 0.0 <= self.steganography.interleave_ratio <= 1.0:
                raise ValueError("STEGO_INTERLEAVE_RATIO must be between 0.0 and 1.0")

            if not 0.0 <= self.steganography.decoy_ratio <= 1.0:
                raise ValueError("STEGO_DECOY_RATIO must be between 0.0 and 1.0")

            if self.steganography.base_delay < 0:
                raise ValueError("STEGO_BASE_DELAY cannot be negative")

            if not 0.0 <= self.steganography.delay_variance <= 1.0:
                raise ValueError("STEGO_DELAY_VARIANCE must be between 0.0 and 1.0")

            if self.steganography.batch_size <= 0:
                raise ValueError("STEGO_BATCH_SIZE must be positive")

            if self.steganography.max_batches_per_hour <= 0:
                raise ValueError("STEGO_MAX_BATCHES_PER_HOUR must be positive")

            valid_strategies = ["round_robin", "random", "weighted"]
            if self.steganography.fragment_strategy not in valid_strategies:
                raise ValueError(f"STEGO_FRAGMENT_STRATEGY must be one of: {valid_strategies}")

        # Perform cross-configuration validation
        self._validate_cross_dependencies()

    def _validate_cross_dependencies(self) -> None:
        """
        Validate cross-configuration dependencies to prevent runtime errors.

        This method performs comprehensive validation of configuration dependencies:
        - Fragmentation technique requires multiple embedding models
        - Technique dependencies are satisfied
        - Evasion settings are compatible with steganography techniques
        - Required embedding models can be initialized
        """
        if not self.steganography.enabled:
            return

        # 1. Fragmentation validation
        if "fragmentation" in self.steganography.techniques:
            self._validate_fragmentation_requirements()

        # 2. Technique dependency validation
        self._validate_technique_dependencies()

        # 3. Evasion compatibility checks
        self._validate_evasion_compatibility()

        # 4. Model availability validation
        self._validate_model_availability()

    def _validate_fragmentation_requirements(self) -> None:
        """
        Validate that fragmentation technique has required resources.

        Fragmentation requires multiple embedding models to distribute data across.
        This validation ensures the necessary models are configured and available.
        """
        # Check if multiple models are configured via fallback models
        available_models = []

        # Primary model
        if self.openai.model:
            available_models.append(self.openai.model)

        # Fallback models
        if self.openai.fallback_enabled and self.openai.fallback_models:
            available_models.extend(self.openai.fallback_models)

        # Remove duplicates while preserving order
        unique_models = []
        for model in available_models:
            if model not in unique_models:
                unique_models.append(model)

        if len(unique_models) < 2:
            raise ValueError(
                "Fragmentation technique requires at least 2 embedding models. "
                "Configure multiple models using OPENAI_FALLBACK_MODELS environment variable "
                "or disable fragmentation by removing it from STEGO_TECHNIQUES. "
                f"Currently configured models: {unique_models}"
            )

    def _validate_technique_dependencies(self) -> None:
        """
        Validate that all enabled steganography techniques have required dependencies.

        Each technique may require specific configuration parameters or external resources.
        This validation ensures all dependencies are properly configured.
        """
        technique_requirements = {
            "noise": ["noise_level"],
            "rotation": ["rotation_angle"],
            "scaling": ["scaling_factor"],
            "offset": ["offset_range"],
            "fragmentation": ["fragment_size", "fragment_strategy"],
            "interleaving": ["interleave_ratio"],
            "timing": ["base_delay", "delay_variance"],
            "decoys": ["decoy_ratio", "decoy_category"]
        }

        for technique in self.steganography.techniques:
            if technique in technique_requirements:
                required_params = technique_requirements[technique]
                for param in required_params:
                    if not hasattr(self.steganography, param):
                        raise ValueError(
                            f"Technique '{technique}' requires parameter '{param}' but it is not configured. "
                            f"Please set STEGO_{param.upper()} environment variable or disable the technique."
                        )

                    value = getattr(self.steganography, param)
                    if value is None:
                        raise ValueError(
                            f"Technique '{technique}' requires parameter '{param}' but it is None. "
                            f"Please set STEGO_{param.upper()} environment variable."
                        )

    def _validate_evasion_compatibility(self) -> None:
        """
        Validate that evasion settings are compatible with selected steganography techniques.

        Some evasion techniques may conflict with certain steganography methods or
        require specific configurations to work effectively together.
        """
        # Check timing-based conflicts
        if "timing" in self.steganography.techniques and self.evasion.traffic_mimicry_enabled:
            # Ensure timing parameters don't conflict
            if self.steganography.base_delay < self.evasion.base_query_interval * 0.1:
                raise ValueError(
                    "Timing-based steganography delay is too small compared to traffic mimicry interval. "
                    f"STEGO_BASE_DELAY ({self.steganography.base_delay}s) should be at least 10% of "
                    f"EVASION_BASE_QUERY_INTERVAL ({self.evasion.base_query_interval}s) to avoid detection patterns."
                )

        # Check fragmentation and network evasion compatibility
        if "fragmentation" in self.steganography.techniques and self.evasion.network_evasion_enabled:
            if self.evasion.max_retries < 2:
                raise ValueError(
                    "Fragmentation technique with network evasion requires EVASION_MAX_RETRIES >= 2 "
                    "to handle potential failures when distributing fragments across multiple models."
                )

        # Check behavioral camouflage and decoy compatibility
        if "decoys" in self.steganography.techniques and self.evasion.behavioral_camouflage_enabled:
            if self.evasion.legitimate_ratio + self.steganography.decoy_ratio > 1.0:
                raise ValueError(
                    "Combined legitimate traffic ratio and decoy ratio cannot exceed 1.0. "
                    f"EVASION_LEGITIMATE_RATIO ({self.evasion.legitimate_ratio}) + "
                    f"STEGO_DECOY_RATIO ({self.steganography.decoy_ratio}) = "
                    f"{self.evasion.legitimate_ratio + self.steganography.decoy_ratio}. "
                    "Adjust these values to ensure realistic traffic patterns."
                )

    def _validate_model_availability(self) -> None:
        """
        Validate that required embedding models can be initialized.

        This performs a lightweight check to ensure the configured models
        are accessible and can be instantiated without full initialization.
        """
        # Check OpenAI API key validity format
        if self.openai.api_key:
            if not self.openai.api_key.startswith(('sk-', 'sk-proj-')):
                raise ValueError(
                    "OPENAI_API_KEY appears to be invalid. OpenAI API keys should start with 'sk-' or 'sk-proj-'. "
                    "Please verify your API key is correct."
                )

        # Validate model names format
        all_models = [self.openai.model]
        if self.openai.fallback_enabled and self.openai.fallback_models:
            all_models.extend(self.openai.fallback_models)

        valid_model_prefixes = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]

        for model in all_models:
            if model and not any(model.startswith(prefix) for prefix in valid_model_prefixes):
                raise ValueError(
                    f"Model '{model}' is not a recognized OpenAI embedding model. "
                    f"Supported models: {valid_model_prefixes}. "
                    "Please check your OPENAI_EMBEDDING_MODEL and OPENAI_FALLBACK_MODELS configuration."
                )

        # Check Pinecone specific requirements if using fragmentation
        if "fragmentation" in self.steganography.techniques and self.vector_store.type == "pinecone":
            if not self.vector_store.pinecone_environment:
                raise ValueError(
                    "Fragmentation with Pinecone requires PINECONE_ENVIRONMENT to be configured. "
                    "Please set the appropriate Pinecone environment for your index."
                )


def get_config() -> Config:
    """Get validated configuration instance."""
    config = Config()
    config.validate()
    return config
