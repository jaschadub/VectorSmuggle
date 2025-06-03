"""Query optimization for performance enhancement and caching strategies."""

import hashlib
import logging
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings


class QueryCache:
    """Caching system for query results."""

    def __init__(self, cache_dir: str = ".query_cache", max_size: int = 1000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.cache_index = {}
        self.access_times = {}
        self.logger = logging.getLogger(__name__)

        # Load existing cache index
        self._load_cache_index()

    def _generate_cache_key(self, query: str, strategy: str, k: int) -> str:
        """Generate a cache key for the query."""
        cache_string = f"{query}:{strategy}:{k}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def _load_cache_index(self):
        """Load cache index from disk."""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_index = data.get("index", {})
                    self.access_times = data.get("access_times", {})
            except Exception as e:
                self.logger.warning(f"Failed to load cache index: {e}")

    def _save_cache_index(self):
        """Save cache index to disk."""
        index_file = self.cache_dir / "cache_index.pkl"
        try:
            with open(index_file, 'wb') as f:
                pickle.dump({
                    "index": self.cache_index,
                    "access_times": self.access_times
                }, f)
        except Exception as e:
            self.logger.warning(f"Failed to save cache index: {e}")

    def get(self, query: str, strategy: str, k: int) -> Optional[list[Document]]:
        """Get cached results for a query."""
        cache_key = self._generate_cache_key(query, strategy, k)

        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        results = pickle.load(f)

                    # Update access time
                    self.access_times[cache_key] = time.time()
                    self.logger.debug(f"Cache hit for query: {query[:50]}...")
                    return results
                except Exception as e:
                    self.logger.warning(f"Failed to load cached results: {e}")
                    # Remove invalid cache entry
                    self._remove_cache_entry(cache_key)

        return None

    def put(self, query: str, strategy: str, k: int, results: list[Document]):
        """Cache query results."""
        cache_key = self._generate_cache_key(query, strategy, k)

        # Check cache size and evict if necessary
        if len(self.cache_index) >= self.max_size:
            self._evict_oldest()

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)

            self.cache_index[cache_key] = {
                "query": query[:100],  # Store truncated query for debugging
                "strategy": strategy,
                "k": k,
                "timestamp": time.time()
            }
            self.access_times[cache_key] = time.time()

            self._save_cache_index()
            self.logger.debug(f"Cached results for query: {query[:50]}...")

        except Exception as e:
            self.logger.warning(f"Failed to cache results: {e}")

    def _evict_oldest(self):
        """Evict the oldest cache entry."""
        if not self.access_times:
            return

        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_cache_entry(oldest_key)

    def _remove_cache_entry(self, cache_key: str):
        """Remove a cache entry."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            cache_file.unlink()

        self.cache_index.pop(cache_key, None)
        self.access_times.pop(cache_key, None)

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()

        self.cache_index.clear()
        self.access_times.clear()
        self._save_cache_index()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache_index),
            "max_size": self.max_size,
            "cache_dir": str(self.cache_dir),
            "oldest_entry": min(self.access_times.values()) if self.access_times else None,
            "newest_entry": max(self.access_times.values()) if self.access_times else None
        }


class BatchProcessor:
    """Processes multiple queries in batches for efficiency."""

    def __init__(self, batch_size: int = 10):
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

    def process_batch(self, queries: list[str], vector_store: Any, strategy_func: callable, k: int = 5) -> dict[str, list[Document]]:
        """Process a batch of queries."""
        results = {}

        # Group similar queries for optimization
        query_groups = self._group_similar_queries(queries)

        for group in query_groups:
            # Process each group
            for query in group:
                try:
                    result = strategy_func(query, vector_store, k)
                    results[query] = result
                except Exception as e:
                    self.logger.warning(f"Failed to process query '{query}': {e}")
                    results[query] = []

        return results

    def _group_similar_queries(self, queries: list[str]) -> list[list[str]]:
        """Group similar queries together."""
        # Simple grouping by length and first word
        groups = defaultdict(list)

        for query in queries:
            words = query.split()
            if words:
                key = (len(words), words[0].lower())
                groups[key].append(query)

        return list(groups.values())


class AdaptiveRetriever:
    """Adapts retrieval strategy based on data characteristics."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)
        self.query_performance = {}
        self.data_characteristics = {}

    def analyze_data_characteristics(self, documents: list[Document]) -> dict[str, Any]:
        """Analyze characteristics of the document collection."""
        if not documents:
            return {}

        characteristics = {
            "total_documents": len(documents),
            "avg_document_length": np.mean([len(doc.page_content) for doc in documents]),
            "document_length_std": np.std([len(doc.page_content) for doc in documents]),
            "total_tokens": sum(len(doc.page_content.split()) for doc in documents),
            "avg_tokens_per_doc": np.mean([len(doc.page_content.split()) for doc in documents]),
            "unique_words": len(set(word.lower() for doc in documents for word in doc.page_content.split())),
            "vocabulary_diversity": 0.0
        }

        # Calculate vocabulary diversity
        total_words = sum(len(doc.page_content.split()) for doc in documents)
        if total_words > 0:
            characteristics["vocabulary_diversity"] = characteristics["unique_words"] / total_words

        # Analyze document similarity distribution
        if len(documents) > 1:
            similarities = self._calculate_document_similarities(documents[:50])  # Sample for efficiency
            characteristics["avg_similarity"] = np.mean(similarities)
            characteristics["similarity_std"] = np.std(similarities)

        self.data_characteristics = characteristics
        return characteristics

    def _calculate_document_similarities(self, documents: list[Document]) -> list[float]:
        """Calculate pairwise similarities between documents."""
        similarities = []

        try:
            embeddings_matrix = []
            for doc in documents:
                embedding = self.embeddings.embed_query(doc.page_content[:500])  # Use first 500 chars
                embeddings_matrix.append(embedding)

            embeddings_array = np.array(embeddings_matrix)

            for i in range(len(embeddings_array)):
                for j in range(i + 1, len(embeddings_array)):
                    similarity = np.dot(embeddings_array[i], embeddings_array[j]) / (
                        np.linalg.norm(embeddings_array[i]) * np.linalg.norm(embeddings_array[j])
                    )
                    similarities.append(similarity)

        except Exception as e:
            self.logger.warning(f"Failed to calculate document similarities: {e}")

        return similarities

    def recommend_strategy(self, query: str) -> str:
        """Recommend the best retrieval strategy for a query."""
        query_length = len(query.split())

        # Default strategy
        recommended = "semantic"

        # Strategy selection based on data characteristics and query
        if self.data_characteristics:
            avg_similarity = self.data_characteristics.get("avg_similarity", 0.5)
            vocabulary_diversity = self.data_characteristics.get("vocabulary_diversity", 0.1)

            if query_length <= 3:
                # Short queries - use hybrid approach
                recommended = "hybrid"
            elif avg_similarity < 0.3:
                # Diverse documents - use clustering
                recommended = "cluster"
            elif vocabulary_diversity > 0.2:
                # High vocabulary diversity - use semantic
                recommended = "semantic"
            else:
                # Similar documents - use hybrid
                recommended = "hybrid"

        # Consider historical performance
        if query in self.query_performance:
            best_strategy = max(self.query_performance[query].items(), key=lambda x: x[1]["score"])
            recommended = best_strategy[0]

        return recommended

    def record_performance(self, query: str, strategy: str, execution_time: float, result_count: int, relevance_score: float):
        """Record performance metrics for a query-strategy combination."""
        if query not in self.query_performance:
            self.query_performance[query] = {}

        # Calculate combined performance score
        time_score = max(0, 1 - execution_time / 10)  # Normalize to 0-1, penalize >10s
        count_score = min(result_count / 10, 1)  # Normalize to 0-1, optimal around 10 results
        combined_score = 0.4 * relevance_score + 0.3 * time_score + 0.3 * count_score

        self.query_performance[query][strategy] = {
            "execution_time": execution_time,
            "result_count": result_count,
            "relevance_score": relevance_score,
            "score": combined_score,
            "timestamp": time.time()
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self.query_performance:
            return {}

        all_performances = []
        strategy_performances = defaultdict(list)

        for query_data in self.query_performance.values():
            for strategy, perf in query_data.items():
                all_performances.append(perf["score"])
                strategy_performances[strategy].append(perf["score"])

        stats = {
            "total_queries": len(self.query_performance),
            "avg_performance": np.mean(all_performances) if all_performances else 0,
            "performance_std": np.std(all_performances) if all_performances else 0,
            "strategy_performance": {}
        }

        for strategy, scores in strategy_performances.items():
            stats["strategy_performance"][strategy] = {
                "avg_score": np.mean(scores),
                "count": len(scores)
            }

        return stats


class QueryOptimizer:
    """Main query optimization class."""

    def __init__(self, embeddings: OpenAIEmbeddings, cache_dir: str = ".query_cache"):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.cache = QueryCache(cache_dir)
        self.batch_processor = BatchProcessor()
        self.adaptive_retriever = AdaptiveRetriever(embeddings)

        # Performance tracking
        self.query_stats = defaultdict(list)

    def optimize_query(self, query: str, vector_store: Any, strategy_func: callable, k: int = 5) -> tuple[list[Document], dict[str, Any]]:
        """Optimize a single query execution."""
        start_time = time.time()

        # Try cache first
        strategy_name = getattr(strategy_func, '__name__', 'unknown')
        cached_results = self.cache.get(query, strategy_name, k)

        if cached_results is not None:
            execution_time = time.time() - start_time
            return cached_results, {
                "cached": True,
                "execution_time": execution_time,
                "strategy": strategy_name
            }

        # Execute query
        try:
            results = strategy_func(query, vector_store, k)
            execution_time = time.time() - start_time

            # Cache results
            self.cache.put(query, strategy_name, k, results)

            # Record performance
            relevance_score = self._estimate_relevance(query, results)
            self.adaptive_retriever.record_performance(
                query, strategy_name, execution_time, len(results), relevance_score
            )

            # Update stats
            self.query_stats[strategy_name].append({
                "execution_time": execution_time,
                "result_count": len(results),
                "relevance_score": relevance_score
            })

            return results, {
                "cached": False,
                "execution_time": execution_time,
                "strategy": strategy_name,
                "result_count": len(results),
                "relevance_score": relevance_score
            }

        except Exception as e:
            self.logger.error(f"Query optimization failed: {e}")
            return [], {"error": str(e)}

    def optimize_batch(self, queries: list[str], vector_store: Any, strategy_func: callable, k: int = 5) -> dict[str, tuple[list[Document], dict[str, Any]]]:
        """Optimize batch query execution."""
        results = {}

        # Separate cached and uncached queries
        cached_queries = []
        uncached_queries = []
        strategy_name = getattr(strategy_func, '__name__', 'unknown')

        for query in queries:
            cached_result = self.cache.get(query, strategy_name, k)
            if cached_result is not None:
                results[query] = (cached_result, {"cached": True, "strategy": strategy_name})
                cached_queries.append(query)
            else:
                uncached_queries.append(query)

        # Process uncached queries in batch
        if uncached_queries:
            batch_results = self.batch_processor.process_batch(
                uncached_queries, vector_store, strategy_func, k
            )

            for query, query_results in batch_results.items():
                # Cache results
                self.cache.put(query, strategy_name, k, query_results)

                # Record performance
                relevance_score = self._estimate_relevance(query, query_results)
                results[query] = (query_results, {
                    "cached": False,
                    "strategy": strategy_name,
                    "result_count": len(query_results),
                    "relevance_score": relevance_score
                })

        return results

    def _estimate_relevance(self, query: str, results: list[Document]) -> float:
        """Estimate relevance score for query results."""
        if not results:
            return 0.0

        try:
            # Simple relevance estimation based on keyword overlap
            query_words = set(query.lower().split())
            relevance_scores = []

            for doc in results:
                doc_words = set(doc.page_content.lower().split())
                overlap = len(query_words.intersection(doc_words))
                relevance = overlap / len(query_words) if query_words else 0
                relevance_scores.append(relevance)

            return np.mean(relevance_scores)

        except Exception as e:
            self.logger.warning(f"Relevance estimation failed: {e}")
            return 0.5  # Default neutral score

    def analyze_data_for_optimization(self, documents: list[Document]) -> dict[str, Any]:
        """Analyze document collection for optimization insights."""
        return self.adaptive_retriever.analyze_data_characteristics(documents)

    def get_strategy_recommendation(self, query: str) -> str:
        """Get recommended strategy for a query."""
        return self.adaptive_retriever.recommend_strategy(query)

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {
            "cache_stats": self.cache.get_stats(),
            "performance_stats": self.adaptive_retriever.get_performance_stats(),
            "data_characteristics": self.data_characteristics,
            "query_stats": {}
        }

        # Aggregate query statistics
        for strategy, strategy_stats in self.query_stats.items():
            if strategy_stats:
                stats["query_stats"][strategy] = {
                    "total_queries": len(strategy_stats),
                    "avg_execution_time": np.mean([s["execution_time"] for s in strategy_stats]),
                    "avg_result_count": np.mean([s["result_count"] for s in strategy_stats]),
                    "avg_relevance": np.mean([s["relevance_score"] for s in strategy_stats])
                }

        return stats

    @property
    def data_characteristics(self) -> dict[str, Any]:
        """Get current data characteristics."""
        return self.adaptive_retriever.data_characteristics

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()
        self.logger.info("Query cache cleared")

    def warm_cache(self, common_queries: list[str], vector_store: Any, strategy_func: callable, k: int = 5):
        """Warm up the cache with common queries."""
        self.logger.info(f"Warming cache with {len(common_queries)} queries...")

        for query in common_queries:
            try:
                self.optimize_query(query, vector_store, strategy_func, k)
            except Exception as e:
                self.logger.warning(f"Failed to warm cache for query '{query}': {e}")

        self.logger.info("Cache warming completed")
