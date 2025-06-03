"""Advanced query engine with semantic search optimization and multi-step reasoning."""

import logging
from typing import Any

import numpy as np
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain_openai import OpenAI, OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class QueryStrategy:
    """Base class for query strategies."""

    def __init__(self, name: str):
        self.name = name

    def execute(self, query: str, vector_store: Any, k: int = 5) -> list[Document]:
        """Execute the query strategy."""
        raise NotImplementedError


class SemanticSearchStrategy(QueryStrategy):
    """Semantic search with embedding similarity."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        super().__init__("semantic_search")
        self.embeddings = embeddings

    def execute(self, query: str, vector_store: Any, k: int = 5) -> list[Document]:
        """Execute semantic search."""
        return vector_store.similarity_search(query, k=k)


class HybridSearchStrategy(QueryStrategy):
    """Hybrid search combining semantic and keyword matching."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        super().__init__("hybrid_search")
        self.embeddings = embeddings

    def execute(self, query: str, vector_store: Any, k: int = 5) -> list[Document]:
        """Execute hybrid search."""
        # Get semantic results
        semantic_results = vector_store.similarity_search(query, k=k*2)

        # Filter by keyword relevance
        query_words = set(query.lower().split())
        scored_results = []

        for doc in semantic_results:
            content_words = set(doc.page_content.lower().split())
            keyword_score = len(query_words.intersection(content_words)) / len(query_words)
            scored_results.append((doc, keyword_score))

        # Sort by keyword score and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_results[:k]]


class ClusterSearchStrategy(QueryStrategy):
    """Search within document clusters for focused results."""

    def __init__(self, embeddings: OpenAIEmbeddings, n_clusters: int = 5):
        super().__init__("cluster_search")
        self.embeddings = embeddings
        self.n_clusters = n_clusters
        self.clusters = None
        self.cluster_centers = None

    def _build_clusters(self, documents: list[Document]) -> None:
        """Build document clusters."""
        if not documents:
            return

        # Get embeddings for all documents
        texts = [doc.page_content for doc in documents]
        embeddings_matrix = np.array([
            self.embeddings.embed_query(text) for text in texts
        ])

        # Perform clustering
        kmeans = KMeans(n_clusters=min(self.n_clusters, len(documents)), random_state=42)
        self.clusters = kmeans.fit_predict(embeddings_matrix)
        self.cluster_centers = kmeans.cluster_centers_

    def execute(self, query: str, vector_store: Any, k: int = 5) -> list[Document]:
        """Execute cluster-based search."""
        # Get all documents for clustering
        all_docs = vector_store.similarity_search("", k=1000)  # Get many docs

        if self.clusters is None:
            self._build_clusters(all_docs)

        if self.cluster_centers is None:
            return vector_store.similarity_search(query, k=k)

        # Find most relevant cluster
        query_embedding = np.array(self.embeddings.embed_query(query)).reshape(1, -1)
        cluster_similarities = cosine_similarity(query_embedding, self.cluster_centers)[0]
        best_cluster = np.argmax(cluster_similarities)

        # Get documents from best cluster
        cluster_docs = [doc for i, doc in enumerate(all_docs)
                       if i < len(self.clusters) and self.clusters[i] == best_cluster]

        # Rank within cluster
        if not cluster_docs:
            return vector_store.similarity_search(query, k=k)

        scored_docs = []
        for doc in cluster_docs:
            doc_embedding = np.array(self.embeddings.embed_query(doc.page_content)).reshape(1, -1)
            similarity = cosine_similarity(query_embedding, doc_embedding)[0][0]
            scored_docs.append((doc, similarity))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]


class QueryExpander:
    """Expands queries with related terms and concepts."""

    def __init__(self, llm: OpenAI):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def expand_query(self, query: str) -> list[str]:
        """Expand query with related terms."""
        try:
            expansion_prompt = f"""
            Given the query: "{query}"

            Generate 3-5 related search terms or phrases that would help find relevant information.
            Focus on synonyms, related concepts, and alternative phrasings.

            Return only the expanded terms, one per line:
            """

            response = self.llm.invoke(expansion_prompt)
            expanded_terms = [term.strip() for term in response.split('\n') if term.strip()]

            # Include original query
            all_terms = [query] + expanded_terms
            return all_terms[:6]  # Limit to 6 total terms

        except Exception as e:
            self.logger.warning(f"Query expansion failed: {e}")
            return [query]


class RelevanceScorer:
    """Scores and ranks query results by relevance."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings

    def score_results(self, query: str, documents: list[Document]) -> list[tuple[Document, float]]:
        """Score documents by relevance to query."""
        if not documents:
            return []

        query_embedding = np.array(self.embeddings.embed_query(query))
        scored_docs = []

        for doc in documents:
            # Semantic similarity
            doc_embedding = np.array(self.embeddings.embed_query(doc.page_content))
            semantic_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]

            # Keyword overlap
            query_words = set(query.lower().split())
            doc_words = set(doc.page_content.lower().split())
            keyword_score = len(query_words.intersection(doc_words)) / len(query_words)

            # Length penalty (prefer more substantial content)
            length_score = min(len(doc.page_content) / 1000, 1.0)

            # Combined score
            final_score = (0.6 * semantic_score + 0.3 * keyword_score + 0.1 * length_score)
            scored_docs.append((doc, final_score))

        return sorted(scored_docs, key=lambda x: x[1], reverse=True)


class AdvancedQueryEngine:
    """Advanced query engine with multiple retrieval strategies and reasoning chains."""

    def __init__(self, vector_store: Any, llm: OpenAI, embeddings: OpenAIEmbeddings):
        self.vector_store = vector_store
        self.llm = llm
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize strategies
        self.strategies = {
            "semantic": SemanticSearchStrategy(embeddings),
            "hybrid": HybridSearchStrategy(embeddings),
            "cluster": ClusterSearchStrategy(embeddings)
        }

        # Initialize components
        self.query_expander = QueryExpander(llm)
        self.relevance_scorer = RelevanceScorer(embeddings)

        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever()
        )

    def execute_strategy(self, strategy_name: str, query: str, k: int = 5) -> list[Document]:
        """Execute a specific search strategy."""
        if strategy_name not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        strategy = self.strategies[strategy_name]
        return strategy.execute(query, self.vector_store, k)

    def multi_strategy_search(self, query: str, k: int = 5) -> list[Document]:
        """Execute multiple strategies and combine results."""
        all_results = []

        for strategy_name in self.strategies:
            try:
                results = self.execute_strategy(strategy_name, query, k)
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(f"Strategy {strategy_name} failed: {e}")

        # Remove duplicates and score
        unique_docs = {}
        for doc in all_results:
            doc_key = doc.page_content[:100]  # Use first 100 chars as key
            if doc_key not in unique_docs:
                unique_docs[doc_key] = doc

        # Score and rank
        scored_results = self.relevance_scorer.score_results(query, list(unique_docs.values()))
        return [doc for doc, _ in scored_results[:k]]

    def expanded_search(self, query: str, k: int = 5) -> list[Document]:
        """Search with query expansion."""
        expanded_queries = self.query_expander.expand_query(query)
        all_results = []

        for expanded_query in expanded_queries:
            try:
                results = self.vector_store.similarity_search(expanded_query, k=k//len(expanded_queries) + 1)
                all_results.extend(results)
            except Exception as e:
                self.logger.warning(f"Expanded query '{expanded_query}' failed: {e}")

        # Score and deduplicate
        unique_docs = {}
        for doc in all_results:
            doc_key = doc.page_content[:100]
            if doc_key not in unique_docs:
                unique_docs[doc_key] = doc

        scored_results = self.relevance_scorer.score_results(query, list(unique_docs.values()))
        return [doc for doc, _ in scored_results[:k]]

    def multi_step_reasoning(self, query: str) -> dict[str, Any]:
        """Execute multi-step reasoning chain for complex queries."""
        try:
            # Step 1: Analyze query complexity
            analysis_prompt = f"""
            Analyze this query for complexity and information needs: "{query}"

            Determine:
            1. Is this a simple factual query or complex analytical query?
            2. What types of information are needed?
            3. Should this be broken into sub-queries?

            Respond with: SIMPLE or COMPLEX, followed by analysis.
            """

            analysis = self.llm.invoke(analysis_prompt)
            is_complex = "COMPLEX" in analysis.upper()

            if not is_complex:
                # Simple query - use standard retrieval
                results = self.multi_strategy_search(query)
                answer = self.qa_chain.run(query)
                return {
                    "query": query,
                    "complexity": "simple",
                    "answer": answer,
                    "sources": results,
                    "reasoning_steps": ["Direct retrieval and answer generation"]
                }

            # Step 2: Break down complex query
            breakdown_prompt = f"""
            Break down this complex query into 2-4 simpler sub-queries: "{query}"

            Each sub-query should focus on a specific aspect of the main question.
            Return only the sub-queries, one per line:
            """

            breakdown = self.llm.invoke(breakdown_prompt)
            sub_queries = [q.strip() for q in breakdown.split('\n') if q.strip()]

            # Step 3: Execute sub-queries
            sub_results = {}
            reasoning_steps = ["Query complexity analysis", "Query decomposition"]

            for i, sub_query in enumerate(sub_queries[:4]):  # Limit to 4 sub-queries
                try:
                    sub_docs = self.multi_strategy_search(sub_query, k=3)
                    sub_answer = self.qa_chain.run(sub_query)
                    sub_results[f"sub_query_{i+1}"] = {
                        "query": sub_query,
                        "answer": sub_answer,
                        "sources": sub_docs
                    }
                    reasoning_steps.append(f"Sub-query {i+1}: {sub_query}")
                except Exception as e:
                    self.logger.warning(f"Sub-query failed: {e}")

            # Step 4: Synthesize final answer
            synthesis_prompt = f"""
            Original query: "{query}"

            Sub-query results:
            {chr(10).join([f"{k}: {v['answer']}" for k, v in sub_results.items()])}

            Synthesize a comprehensive answer to the original query using the sub-query results:
            """

            final_answer = self.llm.invoke(synthesis_prompt)
            reasoning_steps.append("Answer synthesis")

            # Collect all sources
            all_sources = []
            for result in sub_results.values():
                all_sources.extend(result["sources"])

            return {
                "query": query,
                "complexity": "complex",
                "answer": final_answer,
                "sub_queries": sub_results,
                "sources": all_sources,
                "reasoning_steps": reasoning_steps
            }

        except Exception as e:
            self.logger.error(f"Multi-step reasoning failed: {e}")
            # Fallback to simple query
            results = self.multi_strategy_search(query)
            answer = self.qa_chain.run(query)
            return {
                "query": query,
                "complexity": "simple",
                "answer": answer,
                "sources": results,
                "reasoning_steps": ["Fallback to simple retrieval"],
                "error": str(e)
            }

    def get_query_suggestions(self, partial_query: str) -> list[str]:
        """Generate query suggestions based on partial input."""
        try:
            suggestion_prompt = f"""
            Given the partial query: "{partial_query}"

            Generate 3-5 complete, relevant queries that a user might be trying to ask.
            Focus on common information needs and document analysis tasks.

            Return only the suggested queries, one per line:
            """

            response = self.llm.invoke(suggestion_prompt)
            suggestions = [s.strip() for s in response.split('\n') if s.strip()]
            return suggestions[:5]

        except Exception as e:
            self.logger.warning(f"Query suggestion failed: {e}")
            return []

    def analyze_query_performance(self, query: str, results: list[Document]) -> dict[str, Any]:
        """Analyze query performance and suggest improvements."""
        try:
            performance_data = {
                "query": query,
                "num_results": len(results),
                "avg_result_length": np.mean([len(doc.page_content) for doc in results]) if results else 0,
                "query_length": len(query),
                "query_words": len(query.split())
            }

            # Analyze result diversity
            if len(results) > 1:
                embeddings_matrix = np.array([
                    self.embeddings.embed_query(doc.page_content) for doc in results
                ])
                similarities = cosine_similarity(embeddings_matrix)
                avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                performance_data["result_diversity"] = 1 - avg_similarity
            else:
                performance_data["result_diversity"] = 0

            # Generate improvement suggestions
            suggestions = []
            if performance_data["num_results"] < 3:
                suggestions.append("Try expanding your query with more specific terms")
            if performance_data["result_diversity"] < 0.3:
                suggestions.append("Results are very similar - try a more specific query")
            if performance_data["query_words"] < 3:
                suggestions.append("Consider adding more descriptive terms to your query")

            performance_data["suggestions"] = suggestions
            return performance_data

        except Exception as e:
            self.logger.warning(f"Performance analysis failed: {e}")
            return {"query": query, "error": str(e)}
