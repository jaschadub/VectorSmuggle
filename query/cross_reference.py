"""Cross-reference analysis for entity extraction and relationship mapping."""

import logging
import re
from collections import Counter, defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAI, OpenAIEmbeddings
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class Entity:
    """Represents an extracted entity with metadata."""

    def __init__(self, text: str, entity_type: str, confidence: float = 1.0):
        self.text = text
        self.entity_type = entity_type
        self.confidence = confidence
        self.mentions = []
        self.relationships = []
        self.attributes = {}

    def add_mention(self, document_id: str, position: int, context: str):
        """Add a mention of this entity."""
        self.mentions.append({
            "document_id": document_id,
            "position": position,
            "context": context
        })

    def add_relationship(self, target_entity: str, relationship_type: str, strength: float):
        """Add a relationship to another entity."""
        self.relationships.append({
            "target": target_entity,
            "type": relationship_type,
            "strength": strength
        })


class EntityExtractor:
    """Extracts entities from document content."""

    def __init__(self, llm: Optional[OpenAI] = None):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

        # Entity patterns
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            "ip_address": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            "url": r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
            "date": r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b',
            "currency": r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            "person_name": r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
            "organization": r'\b[A-Z][A-Za-z\s&]+(?:Inc|LLC|Corp|Ltd|Company|Organization)\b'
        }

    def extract_pattern_entities(self, content: str, document_id: str) -> list[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        for entity_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(0)
                entity = Entity(entity_text, entity_type)

                # Get context around the match
                start = max(0, match.start() - 50)
                end = min(len(content), match.end() + 50)
                context = content[start:end]

                entity.add_mention(document_id, match.start(), context)
                entities.append(entity)

        return entities

    def extract_llm_entities(self, content: str, document_id: str) -> list[Entity]:
        """Extract entities using LLM."""
        if not self.llm:
            return []

        try:
            extraction_prompt = f"""
            Extract important entities from the following text. Focus on:
            - People (names, titles)
            - Organizations (companies, departments)
            - Locations (cities, addresses)
            - Financial information (amounts, accounts)
            - Technical terms (systems, processes)
            - Dates and times

            Text: {content[:2000]}

            Return entities in format: TYPE:ENTITY_TEXT, one per line.
            """

            response = self.llm.invoke(extraction_prompt)
            entities = []

            for line in response.split('\n'):
                line = line.strip()
                if ':' in line:
                    try:
                        entity_type, entity_text = line.split(':', 1)
                        entity = Entity(entity_text.strip(), entity_type.strip().lower())
                        entity.add_mention(document_id, 0, content[:100])
                        entities.append(entity)
                    except ValueError:
                        continue

            return entities

        except Exception as e:
            self.logger.warning(f"LLM entity extraction failed: {e}")
            return []

    def extract_entities(self, documents: list[Document]) -> dict[str, Entity]:
        """Extract all entities from documents."""
        all_entities = {}
        entity_counter = Counter()

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i:04d}"

            # Extract using patterns
            pattern_entities = self.extract_pattern_entities(doc.page_content, doc_id)

            # Extract using LLM if available
            llm_entities = self.extract_llm_entities(doc.page_content, doc_id)

            # Combine and deduplicate
            for entity in pattern_entities + llm_entities:
                entity_key = f"{entity.entity_type}:{entity.text.lower()}"

                if entity_key in all_entities:
                    # Merge mentions
                    all_entities[entity_key].mentions.extend(entity.mentions)
                else:
                    all_entities[entity_key] = entity

                entity_counter[entity_key] += 1

        # Update confidence based on frequency
        for entity_key, entity in all_entities.items():
            frequency = entity_counter[entity_key]
            entity.confidence = min(frequency / 10, 1.0)  # Normalize to 0-1

        return all_entities


class PatternRecognizer:
    """Recognizes patterns across multiple documents."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

    def find_content_patterns(self, documents: list[Document]) -> dict[str, list[dict[str, Any]]]:
        """Find recurring content patterns."""
        patterns = {
            "repeated_phrases": [],
            "similar_structures": [],
            "common_topics": []
        }

        # Find repeated phrases
        phrase_counts = Counter()
        for doc in documents:
            # Extract phrases (3-5 words)
            words = doc.page_content.split()
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3]).lower()
                if len(phrase) > 10:  # Filter short phrases
                    phrase_counts[phrase] += 1

        # Get most common phrases
        for phrase, count in phrase_counts.most_common(20):
            if count > 1:
                patterns["repeated_phrases"].append({
                    "phrase": phrase,
                    "frequency": count,
                    "documents": self._find_phrase_documents(phrase, documents)
                })

        # Find similar document structures
        patterns["similar_structures"] = self._find_similar_structures(documents)

        # Find common topics using clustering
        patterns["common_topics"] = self._find_common_topics(documents)

        return patterns

    def _find_phrase_documents(self, phrase: str, documents: list[Document]) -> list[int]:
        """Find which documents contain a phrase."""
        doc_indices = []
        for i, doc in enumerate(documents):
            if phrase in doc.page_content.lower():
                doc_indices.append(i)
        return doc_indices

    def _find_similar_structures(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Find documents with similar structures."""
        structures = []

        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i+1:], i+1):
                similarity = self._calculate_structural_similarity(doc1, doc2)
                if similarity > 0.7:
                    structures.append({
                        "doc1_index": i,
                        "doc2_index": j,
                        "similarity": similarity,
                        "common_elements": self._find_common_elements(doc1, doc2)
                    })

        return structures

    def _calculate_structural_similarity(self, doc1: Document, doc2: Document) -> float:
        """Calculate structural similarity between documents."""
        # Simple heuristic based on line patterns
        lines1 = doc1.page_content.split('\n')
        lines2 = doc2.page_content.split('\n')

        # Compare line length patterns
        pattern1 = [len(line) for line in lines1 if line.strip()]
        pattern2 = [len(line) for line in lines2 if line.strip()]

        if not pattern1 or not pattern2:
            return 0.0

        # Normalize patterns
        max_len = max(len(pattern1), len(pattern2))
        pattern1.extend([0] * (max_len - len(pattern1)))
        pattern2.extend([0] * (max_len - len(pattern2)))

        # Calculate correlation
        correlation = np.corrcoef(pattern1, pattern2)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0

    def _find_common_elements(self, doc1: Document, doc2: Document) -> list[str]:
        """Find common structural elements."""
        elements1 = set(re.findall(r'^[A-Z][^:]*:', doc1.page_content, re.MULTILINE))
        elements2 = set(re.findall(r'^[A-Z][^:]*:', doc2.page_content, re.MULTILINE))
        return list(elements1.intersection(elements2))

    def _find_common_topics(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Find common topics using clustering."""
        if len(documents) < 2:
            return []

        try:
            # Get embeddings
            embeddings_matrix = []
            for doc in documents:
                embedding = self.embeddings.embed_query(doc.page_content[:1000])
                embeddings_matrix.append(embedding)

            # Perform clustering
            n_clusters = min(5, len(documents))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings_matrix)

            # Group documents by cluster
            topics = []
            for cluster_id in range(n_clusters):
                doc_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                if len(doc_indices) > 1:
                    topics.append({
                        "cluster_id": cluster_id,
                        "document_indices": doc_indices,
                        "size": len(doc_indices)
                    })

            return topics

        except Exception as e:
            self.logger.warning(f"Topic clustering failed: {e}")
            return []


class NetworkAnalyzer:
    """Analyzes networks of connected information."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def build_entity_network(self, entities: dict[str, Entity]) -> nx.Graph:
        """Build a network graph of entity relationships."""
        graph = nx.Graph()

        # Add entity nodes
        for entity_key, entity in entities.items():
            graph.add_node(entity_key,
                          entity_type=entity.entity_type,
                          confidence=entity.confidence,
                          mention_count=len(entity.mentions))

        # Add edges based on co-occurrence
        entity_list = list(entities.items())
        for i, (key1, entity1) in enumerate(entity_list):
            for j, (key2, entity2) in enumerate(entity_list[i+1:], i+1):
                # Check if entities co-occur in documents
                docs1 = set(mention["document_id"] for mention in entity1.mentions)
                docs2 = set(mention["document_id"] for mention in entity2.mentions)

                common_docs = docs1.intersection(docs2)
                if common_docs:
                    # Calculate relationship strength
                    strength = len(common_docs) / len(docs1.union(docs2))
                    graph.add_edge(key1, key2, weight=strength, common_docs=len(common_docs))

        return graph

    def find_central_entities(self, graph: nx.Graph, top_k: int = 10) -> list[tuple[str, float]]:
        """Find most central entities in the network."""
        if not graph.nodes():
            return []

        # Calculate centrality measures
        centrality_measures = {
            "degree": nx.degree_centrality(graph),
            "betweenness": nx.betweenness_centrality(graph),
            "closeness": nx.closeness_centrality(graph),
            "eigenvector": nx.eigenvector_centrality(graph, max_iter=1000)
        }

        # Combine centrality scores
        combined_scores = {}
        for node in graph.nodes():
            score = (
                0.3 * centrality_measures["degree"].get(node, 0) +
                0.3 * centrality_measures["betweenness"].get(node, 0) +
                0.2 * centrality_measures["closeness"].get(node, 0) +
                0.2 * centrality_measures["eigenvector"].get(node, 0)
            )
            combined_scores[node] = score

        # Return top entities
        sorted_entities = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_entities[:top_k]

    def find_communities(self, graph: nx.Graph) -> list[list[str]]:
        """Find communities in the entity network."""
        if not graph.nodes():
            return []

        try:
            # Use Louvain community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.louvain_communities(graph)
            return [list(community) for community in communities]
        except ImportError:
            # Fallback to simple connected components
            return [list(component) for component in nx.connected_components(graph)]

    def analyze_network_properties(self, graph: nx.Graph) -> dict[str, Any]:
        """Analyze network properties."""
        if not graph.nodes():
            return {}

        properties = {
            "num_nodes": graph.number_of_nodes(),
            "num_edges": graph.number_of_edges(),
            "density": nx.density(graph),
            "is_connected": nx.is_connected(graph),
            "num_components": nx.number_connected_components(graph)
        }

        if graph.number_of_edges() > 0:
            properties["average_clustering"] = nx.average_clustering(graph)

            # Diameter of largest component
            largest_cc = max(nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc)
            if subgraph.number_of_nodes() > 1:
                properties["diameter"] = nx.diameter(subgraph)

        return properties


class AnomalyDetector:
    """Detects anomalies and interesting data points."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

    def detect_content_anomalies(self, documents: list[Document]) -> list[dict[str, Any]]:
        """Detect anomalous content in documents."""
        anomalies = []

        if len(documents) < 3:
            return anomalies

        try:
            # Get embeddings for all documents
            embeddings_matrix = []
            for doc in documents:
                embedding = self.embeddings.embed_query(doc.page_content[:1000])
                embeddings_matrix.append(embedding)

            embeddings_array = np.array(embeddings_matrix)

            # Calculate pairwise similarities
            similarities = cosine_similarity(embeddings_array)

            # Find outliers (documents with low average similarity)
            avg_similarities = np.mean(similarities, axis=1)
            threshold = np.mean(avg_similarities) - 2 * np.std(avg_similarities)

            for i, avg_sim in enumerate(avg_similarities):
                if avg_sim < threshold:
                    anomalies.append({
                        "document_index": i,
                        "anomaly_type": "content_outlier",
                        "score": float(avg_sim),
                        "threshold": float(threshold),
                        "description": "Document content significantly different from others"
                    })

            return anomalies

        except Exception as e:
            self.logger.warning(f"Content anomaly detection failed: {e}")
            return []

    def detect_entity_anomalies(self, entities: dict[str, Entity]) -> list[dict[str, Any]]:
        """Detect anomalous entities."""
        anomalies = []

        # Group entities by type
        entity_types = defaultdict(list)
        for entity_key, entity in entities.items():
            entity_types[entity.entity_type].append(entity)

        # Detect anomalies within each type
        for entity_type, type_entities in entity_types.items():
            if len(type_entities) < 3:
                continue

            # Find entities with unusual mention patterns
            mention_counts = [len(entity.mentions) for entity in type_entities]
            mean_mentions = np.mean(mention_counts)
            std_mentions = np.std(mention_counts)

            for entity in type_entities:
                mention_count = len(entity.mentions)
                z_score = abs(mention_count - mean_mentions) / std_mentions if std_mentions > 0 else 0

                if z_score > 2:  # More than 2 standard deviations
                    anomalies.append({
                        "entity": entity.text,
                        "entity_type": entity.entity_type,
                        "anomaly_type": "unusual_mention_frequency",
                        "mention_count": mention_count,
                        "z_score": float(z_score),
                        "description": f"Entity mentioned {mention_count} times (unusual for type {entity_type})"
                    })

        return anomalies

    def detect_pattern_anomalies(self, patterns: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Detect anomalous patterns."""
        anomalies = []

        # Check for unusual phrase frequencies
        if "repeated_phrases" in patterns:
            frequencies = [p["frequency"] for p in patterns["repeated_phrases"]]
            if frequencies:
                mean_freq = np.mean(frequencies)
                std_freq = np.std(frequencies)

                for pattern in patterns["repeated_phrases"]:
                    freq = pattern["frequency"]
                    z_score = abs(freq - mean_freq) / std_freq if std_freq > 0 else 0

                    if z_score > 2:
                        anomalies.append({
                            "pattern": pattern["phrase"],
                            "anomaly_type": "unusual_phrase_frequency",
                            "frequency": freq,
                            "z_score": float(z_score),
                            "description": f"Phrase appears {freq} times (unusual frequency)"
                        })

        return anomalies


class CrossReferenceAnalyzer:
    """Main class for cross-reference analysis and relationship mapping."""

    def __init__(self, embeddings: OpenAIEmbeddings, llm: Optional[OpenAI] = None):
        self.embeddings = embeddings
        self.llm = llm
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.entity_extractor = EntityExtractor(llm)
        self.pattern_recognizer = PatternRecognizer(embeddings)
        self.network_analyzer = NetworkAnalyzer()
        self.anomaly_detector = AnomalyDetector(embeddings)

    def analyze_cross_references(self, documents: list[Document]) -> dict[str, Any]:
        """Perform comprehensive cross-reference analysis."""
        try:
            analysis_results = {
                "entities": {},
                "patterns": {},
                "network": {},
                "anomalies": {},
                "summary": {}
            }

            # Extract entities
            self.logger.info("Extracting entities...")
            entities = self.entity_extractor.extract_entities(documents)
            analysis_results["entities"] = self._format_entities_output(entities)

            # Find patterns
            self.logger.info("Recognizing patterns...")
            patterns = self.pattern_recognizer.find_content_patterns(documents)
            analysis_results["patterns"] = patterns

            # Build and analyze network
            self.logger.info("Building entity network...")
            entity_network = self.network_analyzer.build_entity_network(entities)

            central_entities = self.network_analyzer.find_central_entities(entity_network)
            communities = self.network_analyzer.find_communities(entity_network)
            network_properties = self.network_analyzer.analyze_network_properties(entity_network)

            analysis_results["network"] = {
                "central_entities": central_entities,
                "communities": communities,
                "properties": network_properties
            }

            # Detect anomalies
            self.logger.info("Detecting anomalies...")
            content_anomalies = self.anomaly_detector.detect_content_anomalies(documents)
            entity_anomalies = self.anomaly_detector.detect_entity_anomalies(entities)
            pattern_anomalies = self.anomaly_detector.detect_pattern_anomalies(patterns)

            analysis_results["anomalies"] = {
                "content": content_anomalies,
                "entities": entity_anomalies,
                "patterns": pattern_anomalies
            }

            # Generate summary
            analysis_results["summary"] = self._generate_analysis_summary(analysis_results)

            return analysis_results

        except Exception as e:
            self.logger.error(f"Cross-reference analysis failed: {e}")
            return {"error": str(e)}

    def _format_entities_output(self, entities: dict[str, Entity]) -> dict[str, Any]:
        """Format entities for output."""
        formatted = {
            "total_entities": len(entities),
            "by_type": defaultdict(list),
            "high_confidence": [],
            "frequent_entities": []
        }

        for entity_key, entity in entities.items():
            entity_info = {
                "text": entity.text,
                "type": entity.entity_type,
                "confidence": entity.confidence,
                "mention_count": len(entity.mentions),
                "documents": list(set(m["document_id"] for m in entity.mentions))
            }

            formatted["by_type"][entity.entity_type].append(entity_info)

            if entity.confidence > 0.8:
                formatted["high_confidence"].append(entity_info)

            if len(entity.mentions) > 2:
                formatted["frequent_entities"].append(entity_info)

        # Convert defaultdict to regular dict
        formatted["by_type"] = dict(formatted["by_type"])

        return formatted

    def _generate_analysis_summary(self, analysis_results: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of the analysis results."""
        summary = {
            "total_documents": len(analysis_results.get("patterns", {}).get("common_topics", [])),
            "total_entities": analysis_results.get("entities", {}).get("total_entities", 0),
            "entity_types": len(analysis_results.get("entities", {}).get("by_type", {})),
            "network_density": analysis_results.get("network", {}).get("properties", {}).get("density", 0),
            "anomalies_found": (
                len(analysis_results.get("anomalies", {}).get("content", [])) +
                len(analysis_results.get("anomalies", {}).get("entities", [])) +
                len(analysis_results.get("anomalies", {}).get("patterns", []))
            ),
            "key_findings": []
        }

        # Generate key findings
        if summary["total_entities"] > 50:
            summary["key_findings"].append("High entity density detected")

        if summary["network_density"] > 0.3:
            summary["key_findings"].append("Highly connected entity network")

        if summary["anomalies_found"] > 0:
            summary["key_findings"].append(f"{summary['anomalies_found']} anomalies detected")

        return summary
