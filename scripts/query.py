"""
VectorSmuggle Enhanced Query Script

This script demonstrates advanced query reconstruction capabilities including
semantic search optimization, context reconstruction, cross-reference analysis,
and data recovery tools for comprehensive data exfiltration analysis.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

# Updated LangChain imports
from langchain_community.vectorstores import FAISS, Qdrant
from langchain_openai import OpenAI, OpenAIEmbeddings
from utils.embedding_factory import create_embeddings as create_embeddings_with_fallback

from config import Config, get_config
from evasion import BehavioralCamouflage, TrafficMimicry
from query import AdvancedQueryEngine, ContextReconstructor, CrossReferenceAnalyzer, DataRecoveryTools, QueryOptimizer


def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format
    )

    return logging.getLogger(__name__)


def create_llm(config: Config, logger: logging.Logger) -> OpenAI:
    """
    Create OpenAI LLM instance.

    Args:
        config: Configuration object
        logger: Logger instance

    Returns:
        OpenAI LLM instance
    """
    try:
        logger.info(f"Initializing LLM with model: {config.openai.llm_model}")
        return OpenAI(
            openai_api_key=config.openai.api_key,
            model_name=config.openai.llm_model
        )
    except Exception as e:
        logger.error(f"Failed to create LLM: {e}")
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


def load_faiss_store(config: Config, embeddings: OpenAIEmbeddings, logger: logging.Logger) -> FAISS:
    """Load FAISS vector store."""
    try:
        index_path = Path(config.vector_store.faiss_index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at: {index_path}")

        logger.info(f"Loading FAISS index from: {index_path}")
        return FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)

    except Exception as e:
        logger.error(f"Failed to load FAISS store: {e}")
        raise


def load_qdrant_store(config: Config, embeddings: OpenAIEmbeddings, logger: logging.Logger) -> Qdrant:
    """Load Qdrant vector store."""
    try:
        logger.info(f"Connecting to Qdrant at: {config.vector_store.qdrant_url}")

        return Qdrant(
            collection_name=config.vector_store.collection_name,
            embeddings=embeddings,
            url=config.vector_store.qdrant_url
        )

    except Exception as e:
        logger.error(f"Failed to load Qdrant store: {e}")
        raise


def load_pinecone_store(config: Config, embeddings: OpenAIEmbeddings, logger: logging.Logger):
    """Load Pinecone vector store."""
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
        if index_name not in pinecone.list_indexes():
            raise ValueError(f"Pinecone index '{index_name}' does not exist")

        return Pinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )

    except ImportError:
        logger.error("Pinecone client not installed. Install with: pip install pinecone-client")
        raise
    except Exception as e:
        logger.error(f"Failed to load Pinecone store: {e}")
        raise


def load_vector_store(config: Config, embeddings: OpenAIEmbeddings, logger: logging.Logger):
    """
    Load the appropriate vector store based on configuration.

    Args:
        config: Configuration object
        embeddings: OpenAI embeddings instance
        logger: Logger instance

    Returns:
        Vector store instance
    """
    if config.vector_store.type == "faiss":
        return load_faiss_store(config, embeddings, logger)
    elif config.vector_store.type == "qdrant":
        return load_qdrant_store(config, embeddings, logger)
    elif config.vector_store.type == "pinecone":
        return load_pinecone_store(config, embeddings, logger)
    else:
        raise ValueError(f"Unsupported vector store type: {config.vector_store.type}")


def validate_query(query: str) -> bool:
    """
    Validate user query input.

    Args:
        query: User input query

    Returns:
        True if query is valid, False otherwise
    """
    if not query or not query.strip():
        return False

    # Basic length check
    if len(query.strip()) < 3:
        return False

    return True


def run_simple_query(query_engine: AdvancedQueryEngine, query: str, logger: logging.Logger) -> dict[str, Any]:
    """Run a simple query using the advanced engine."""
    try:
        start_time = time.time()

        # Use multi-strategy search for best results
        results = query_engine.multi_strategy_search(query, k=5)

        # Get answer using QA chain
        answer = query_engine.qa_chain.run(query)

        execution_time = time.time() - start_time

        return {
            "query": query,
            "answer": answer,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results],
            "execution_time": execution_time,
            "num_sources": len(results)
        }

    except Exception as e:
        logger.error(f"Simple query failed: {e}")
        return {"error": str(e)}


def run_complex_query(query_engine: AdvancedQueryEngine, query: str, logger: logging.Logger) -> dict[str, Any]:
    """Run a complex query with multi-step reasoning."""
    try:
        logger.info(f"Running complex query: {query[:50]}...")
        return query_engine.multi_step_reasoning(query)

    except Exception as e:
        logger.error(f"Complex query failed: {e}")
        return {"error": str(e)}


def run_context_analysis(context_reconstructor: ContextReconstructor, vector_store: Any, logger: logging.Logger) -> dict[str, Any]:
    """Run context reconstruction analysis."""
    try:
        logger.info("Running context reconstruction analysis...")

        # Get all documents from vector store
        documents = vector_store.similarity_search("", k=1000)  # Get many documents

        # Reconstruct document structures
        doc_structures = context_reconstructor.reconstruct_document_structure(documents)

        # Export analysis
        analysis = context_reconstructor.export_structure_analysis(doc_structures)

        return analysis

    except Exception as e:
        logger.error(f"Context analysis failed: {e}")
        return {"error": str(e)}


def run_cross_reference_analysis(cross_ref_analyzer: CrossReferenceAnalyzer, vector_store: Any, logger: logging.Logger) -> dict[str, Any]:
    """Run cross-reference analysis."""
    try:
        logger.info("Running cross-reference analysis...")

        # Get all documents from vector store
        documents = vector_store.similarity_search("", k=1000)  # Get many documents

        # Perform cross-reference analysis
        analysis = cross_ref_analyzer.analyze_cross_references(documents)

        return analysis

    except Exception as e:
        logger.error(f"Cross-reference analysis failed: {e}")
        return {"error": str(e)}


def run_data_recovery(recovery_tools: DataRecoveryTools, vector_store: Any, logger: logging.Logger) -> dict[str, Any]:
    """Run data recovery analysis."""
    try:
        logger.info("Running data recovery analysis...")

        # Get all documents from vector store
        documents = vector_store.similarity_search("", k=1000)  # Get many documents

        # Perform data recovery
        recovery_results = recovery_tools.recover_data(documents)

        return recovery_results

    except Exception as e:
        logger.error(f"Data recovery failed: {e}")
        return {"error": str(e)}


def run_interactive_query(query_engine: AdvancedQueryEngine, query_optimizer: QueryOptimizer, logger: logging.Logger) -> None:
    """
    Run interactive query loop with enhanced capabilities.

    Args:
        query_engine: Advanced query engine instance
        query_optimizer: Query optimizer instance
        logger: Logger instance
    """
    logger.info("Starting enhanced interactive query session")
    print("\n=== VectorSmuggle Enhanced Query Interface ===")
    print("Available commands:")
    print("  - Regular query: Just type your question")
    print("  - Complex query: /complex <question>")
    print("  - Strategy query: /strategy <strategy_name> <question>")
    print("  - Suggestions: /suggest <partial_query>")
    print("  - Performance: /perf")
    print("  - Help: /help")
    print("  - Exit: /exit, /quit, or /q")
    print()

    while True:
        try:
            query = input("Query> ").strip()

            # Check for exit commands
            if query.lower() in ("/exit", "/quit", "/q"):
                logger.info("Query session ended by user")
                break

            # Handle special commands
            if query.startswith("/"):
                if query.startswith("/help"):
                    print("\nAvailable commands:")
                    print("  /complex <question> - Use multi-step reasoning")
                    print("  /strategy <name> <question> - Use specific strategy (semantic, hybrid, cluster)")
                    print("  /suggest <partial> - Get query suggestions")
                    print("  /perf - Show performance statistics")
                    print("  /help - Show this help")
                    print("  /exit, /quit, /q - Exit")
                    continue

                elif query.startswith("/complex "):
                    complex_query = query[9:].strip()
                    if validate_query(complex_query):
                        result = run_complex_query(query_engine, complex_query, logger)
                        print("\n→ Complex Analysis Result:")
                        print(f"Answer: {result.get('answer', 'No answer generated')}")
                        if 'reasoning_steps' in result:
                            print(f"Reasoning Steps: {', '.join(result['reasoning_steps'])}")
                        print()
                    else:
                        print("Please provide a valid question after /complex")
                    continue

                elif query.startswith("/strategy "):
                    parts = query[10:].split(" ", 1)
                    if len(parts) == 2:
                        strategy_name, strategy_query = parts
                        if validate_query(strategy_query):
                            try:
                                results = query_engine.execute_strategy(strategy_name, strategy_query, k=5)
                                answer = query_engine.qa_chain.run(strategy_query)
                                print(f"\n→ {strategy_name.title()} Strategy Result:")
                                print(f"Answer: {answer}")
                                print(f"Sources found: {len(results)}")
                                print()
                            except ValueError as e:
                                print(f"Error: {e}")
                        else:
                            print("Please provide a valid question")
                    else:
                        print("Usage: /strategy <strategy_name> <question>")
                    continue

                elif query.startswith("/suggest "):
                    partial_query = query[9:].strip()
                    if partial_query:
                        suggestions = query_engine.get_query_suggestions(partial_query)
                        print("\n→ Query Suggestions:")
                        for i, suggestion in enumerate(suggestions, 1):
                            print(f"  {i}. {suggestion}")
                        print()
                    else:
                        print("Please provide a partial query")
                    continue

                elif query.startswith("/perf"):
                    stats = query_optimizer.get_optimization_stats()
                    print("\n→ Performance Statistics:")
                    print(f"Cache entries: {stats.get('cache_stats', {}).get('total_entries', 0)}")
                    if 'query_stats' in stats:
                        for strategy, strategy_stats in stats['query_stats'].items():
                            print(f"  {strategy}: {strategy_stats.get('total_queries', 0)} queries, "
                                  f"avg time: {strategy_stats.get('avg_execution_time', 0):.2f}s")
                    print()
                    continue

                else:
                    print("Unknown command. Type /help for available commands.")
                    continue

            # Validate regular query
            if not validate_query(query):
                print("Please enter a valid question (at least 3 characters).")
                continue

            logger.info(f"Processing query: {query[:50]}...")

            # Get strategy recommendation
            recommended_strategy = query_optimizer.get_strategy_recommendation(query)
            logger.debug(f"Recommended strategy: {recommended_strategy}")

            # Execute optimized query
            results, query_info = query_optimizer.optimize_query(
                query,
                query_engine.vector_store,
                lambda q, vs, k: query_engine.execute_strategy(recommended_strategy, q, k)
            )

            # Get answer
            answer = query_engine.qa_chain.run(query)

            # Display results
            print(f"→ {answer}")

            if query_info.get("cached"):
                print("  (cached result)")
            else:
                print(f"  Strategy: {recommended_strategy}, Time: {query_info.get('execution_time', 0):.2f}s, Sources: {len(results)}")
            print()

        except KeyboardInterrupt:
            logger.info("Query session interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"Error: {e}\n")


def export_analysis_results(results: dict[str, Any], output_path: str, logger: logging.Logger):
    """Export analysis results to file."""
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Analysis results exported to: {output_file}")

    except Exception as e:
        logger.error(f"Failed to export results: {e}")


def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser(description="VectorSmuggle Enhanced Query Tool")
    parser.add_argument("--mode", choices=["interactive", "context", "cross-ref", "recovery"],
                       default="interactive", help="Query mode")
    parser.add_argument("--query", type=str, help="Single query to execute")
    parser.add_argument("--strategy", choices=["semantic", "hybrid", "cluster"],
                       help="Query strategy to use")
    parser.add_argument("--export", type=str, help="Export results to file")
    parser.add_argument("--complex", action="store_true", help="Use complex multi-step reasoning")

    args = parser.parse_args()

    try:
        # Load and validate configuration
        config = get_config()
        logger = setup_logging(config)

        # Initialize behavioral camouflage if enabled
        behavioral_camouflage = None
        traffic_mimicry = None

        if config.evasion.behavioral_camouflage_enabled:
            behavioral_camouflage = BehavioralCamouflage(
                legitimate_ratio=config.evasion.legitimate_ratio,
                activity_mixing_strategy=config.evasion.activity_mixing_strategy
            )
            behavioral_camouflage.generate_cover_story("data analysis and research")
            logger.info("Initialized behavioral camouflage for queries")

        if config.evasion.traffic_mimicry_enabled:
            traffic_mimicry = TrafficMimicry(
                base_query_interval=config.evasion.base_query_interval,
                query_variance=config.evasion.query_variance,
                burst_probability=config.evasion.burst_probability,
                user_profiles=config.evasion.user_profiles
            )
            logger.info("Initialized traffic mimicry for queries")

        logger.info("Starting VectorSmuggle enhanced query process")
        logger.info(f"Vector store type: {config.vector_store.type}")
        logger.info(f"Query mode: {args.mode}")

        # Create LLM and embeddings
        llm = create_llm(config, logger)
        embeddings = create_embeddings(config, logger)

        # Load vector store
        vector_store = load_vector_store(config, embeddings, logger)

        # Initialize enhanced query components
        logger.info("Initializing enhanced query components...")

        query_optimizer = QueryOptimizer(embeddings)
        advanced_engine = AdvancedQueryEngine(vector_store, llm, embeddings)
        context_reconstructor = ContextReconstructor(embeddings)
        cross_ref_analyzer = CrossReferenceAnalyzer(embeddings, llm)
        recovery_tools = DataRecoveryTools(embeddings)

        # Analyze data characteristics for optimization
        logger.info("Analyzing data characteristics...")
        documents = vector_store.similarity_search("", k=100)  # Sample for analysis
        query_optimizer.analyze_data_for_optimization(documents)

        # Execute based on mode
        if args.mode == "interactive":
            if args.query:
                # Single query mode
                if args.complex:
                    result = run_complex_query(advanced_engine, args.query, logger)
                else:
                    strategy = args.strategy or query_optimizer.get_strategy_recommendation(args.query)
                    results, query_info = query_optimizer.optimize_query(
                        args.query,
                        vector_store,
                        lambda q, vs, k: advanced_engine.execute_strategy(strategy, q, k)
                    )
                    answer = advanced_engine.qa_chain.run(args.query)
                    result = {
                        "query": args.query,
                        "answer": answer,
                        "strategy": strategy,
                        "execution_info": query_info,
                        "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
                    }

                print(json.dumps(result, indent=2, default=str))

                if args.export:
                    export_analysis_results(result, args.export, logger)
            else:
                # Interactive mode
                run_interactive_query(advanced_engine, query_optimizer, logger)

        elif args.mode == "context":
            result = run_context_analysis(context_reconstructor, vector_store, logger)
            print(json.dumps(result, indent=2, default=str))

            if args.export:
                export_analysis_results(result, args.export, logger)

        elif args.mode == "cross-ref":
            result = run_cross_reference_analysis(cross_ref_analyzer, vector_store, logger)
            print(json.dumps(result, indent=2, default=str))

            if args.export:
                export_analysis_results(result, args.export, logger)

        elif args.mode == "recovery":
            result = run_data_recovery(recovery_tools, vector_store, logger)
            print(json.dumps(result, indent=2, default=str))

            if args.export:
                export_analysis_results(result, args.export, logger)

        logger.info("Query process completed")

    except Exception as e:
        logger.error(f"Query process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
