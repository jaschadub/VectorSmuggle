"""Context reconstruction for document structure and relationship mapping."""

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional

import networkx as nx
import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class DocumentFragment:
    """Represents a fragment of a document with metadata."""

    def __init__(self, content: str, metadata: dict[str, Any], fragment_id: str):
        self.content = content
        self.metadata = metadata
        self.fragment_id = fragment_id
        self.relationships = []
        self.confidence_score = 0.0

    def add_relationship(self, target_id: str, relationship_type: str, strength: float):
        """Add a relationship to another fragment."""
        self.relationships.append({
            "target": target_id,
            "type": relationship_type,
            "strength": strength
        })


class DocumentStructure:
    """Represents the reconstructed structure of a document."""

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.fragments = {}
        self.structure_graph = nx.DiGraph()
        self.metadata = {}
        self.confidence_score = 0.0

    def add_fragment(self, fragment: DocumentFragment):
        """Add a fragment to the document structure."""
        self.fragments[fragment.fragment_id] = fragment
        self.structure_graph.add_node(fragment.fragment_id, fragment=fragment)

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str, strength: float):
        """Add a relationship between fragments."""
        if source_id in self.fragments and target_id in self.fragments:
            self.structure_graph.add_edge(source_id, target_id,
                                        type=relationship_type, strength=strength)
            self.fragments[source_id].add_relationship(target_id, relationship_type, strength)

    def get_ordered_fragments(self) -> list[DocumentFragment]:
        """Get fragments in logical order."""
        try:
            # Use topological sort for ordering
            ordered_ids = list(nx.topological_sort(self.structure_graph))
            return [self.fragments[fid] for fid in ordered_ids if fid in self.fragments]
        except nx.NetworkXError:
            # Fallback to confidence-based ordering
            return sorted(self.fragments.values(), key=lambda f: f.confidence_score, reverse=True)


class MetadataCorrelator:
    """Correlates metadata across document fragments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_metadata_patterns(self, documents: list[Document]) -> dict[str, Any]:
        """Extract common metadata patterns from documents."""
        patterns = {
            "dates": [],
            "authors": [],
            "file_types": [],
            "sources": [],
            "topics": []
        }

        for doc in documents:
            metadata = doc.metadata

            # Extract dates
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',
                r'\d{2}/\d{2}/\d{4}',
                r'\d{1,2}/\d{1,2}/\d{2,4}'
            ]

            content = doc.page_content
            for pattern in date_patterns:
                dates = re.findall(pattern, content)
                patterns["dates"].extend(dates)

            # Extract from metadata
            if "author" in metadata:
                patterns["authors"].append(metadata["author"])
            if "source" in metadata:
                patterns["sources"].append(metadata["source"])
            if "file_type" in metadata:
                patterns["file_types"].append(metadata["file_type"])

        return patterns

    def correlate_fragments(self, fragments: list[DocumentFragment]) -> dict[str, list[str]]:
        """Find correlations between fragment metadata."""
        correlations = defaultdict(list)

        for i, frag1 in enumerate(fragments):
            for j, frag2 in enumerate(fragments[i+1:], i+1):
                correlation_score = self._calculate_metadata_similarity(frag1, frag2)

                if correlation_score > 0.7:
                    correlations["high"].append(f"{frag1.fragment_id}-{frag2.fragment_id}")
                elif correlation_score > 0.4:
                    correlations["medium"].append(f"{frag1.fragment_id}-{frag2.fragment_id}")

        return dict(correlations)

    def _calculate_metadata_similarity(self, frag1: DocumentFragment, frag2: DocumentFragment) -> float:
        """Calculate similarity between fragment metadata."""
        meta1, meta2 = frag1.metadata, frag2.metadata

        # Check for common fields
        common_fields = set(meta1.keys()).intersection(set(meta2.keys()))
        if not common_fields:
            return 0.0

        matches = 0
        for field in common_fields:
            if meta1[field] == meta2[field]:
                matches += 1

        return matches / len(common_fields)


class TimelineReconstructor:
    """Reconstructs temporal sequences from document fragments."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def extract_temporal_markers(self, content: str) -> list[tuple[str, datetime]]:
        """Extract temporal markers from content."""
        temporal_markers = []

        # Date patterns
        date_patterns = [
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{2}/\d{2}/\d{4})', '%m/%d/%Y'),
            (r'(\d{1,2}/\d{1,2}/\d{2})', '%m/%d/%y')
        ]

        for pattern, date_format in date_patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                try:
                    date_obj = datetime.strptime(match.group(1), date_format)
                    temporal_markers.append((match.group(1), date_obj))
                except ValueError:
                    continue

        # Relative time expressions
        relative_patterns = [
            r'(yesterday|today|tomorrow)',
            r'(last|next)\s+(week|month|year)',
            r'(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|from now)'
        ]

        for pattern in relative_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # For relative dates, we'd need a reference point
                # For now, just mark their presence
                temporal_markers.append((match.group(0), None))

        return temporal_markers

    def build_timeline(self, fragments: list[DocumentFragment]) -> list[tuple[DocumentFragment, Optional[datetime]]]:
        """Build a timeline from document fragments."""
        timeline_entries = []

        for fragment in fragments:
            temporal_markers = self.extract_temporal_markers(fragment.content)

            if temporal_markers:
                # Use the earliest valid date found
                valid_dates = [tm[1] for tm in temporal_markers if tm[1] is not None]
                if valid_dates:
                    earliest_date = min(valid_dates)
                    timeline_entries.append((fragment, earliest_date))
                else:
                    timeline_entries.append((fragment, None))
            else:
                timeline_entries.append((fragment, None))

        # Sort by date, putting None dates at the end
        timeline_entries.sort(key=lambda x: x[1] if x[1] is not None else datetime.max)

        return timeline_entries

    def detect_temporal_relationships(self, timeline: list[tuple[DocumentFragment, Optional[datetime]]]) -> list[dict[str, Any]]:
        """Detect temporal relationships between fragments."""
        relationships = []

        for i, (frag1, date1) in enumerate(timeline):
            for j, (frag2, date2) in enumerate(timeline[i+1:], i+1):
                if date1 and date2:
                    time_diff = abs((date2 - date1).days)

                    if time_diff <= 1:
                        rel_type = "concurrent"
                    elif time_diff <= 7:
                        rel_type = "sequential_short"
                    elif time_diff <= 30:
                        rel_type = "sequential_medium"
                    else:
                        rel_type = "sequential_long"

                    relationships.append({
                        "source": frag1.fragment_id,
                        "target": frag2.fragment_id,
                        "type": rel_type,
                        "time_diff_days": time_diff
                    })

        return relationships


class ReferenceResolver:
    """Resolves cross-document references and citations."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

    def find_references(self, content: str) -> list[dict[str, str]]:
        """Find references in content."""
        references = []

        # Citation patterns
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+\s+\d{4})\)',  # (Author 2023)
            r'see\s+([^.]+)',  # see reference
            r'according\s+to\s+([^.]+)',  # according to source
        ]

        for pattern in citation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                references.append({
                    "text": match.group(0),
                    "reference": match.group(1),
                    "position": match.start()
                })

        return references

    def resolve_cross_references(self, fragments: list[DocumentFragment]) -> dict[str, list[str]]:
        """Resolve cross-references between fragments."""
        cross_refs = defaultdict(list)

        # Build reference index
        ref_index = {}
        for fragment in fragments:
            refs = self.find_references(fragment.content)
            for ref in refs:
                ref_text = ref["reference"].lower()
                if ref_text not in ref_index:
                    ref_index[ref_text] = []
                ref_index[ref_text].append(fragment.fragment_id)

        # Find matching references
        for fragment in fragments:
            refs = self.find_references(fragment.content)
            for ref in refs:
                ref_text = ref["reference"].lower()
                if ref_text in ref_index:
                    for target_id in ref_index[ref_text]:
                        if target_id != fragment.fragment_id:
                            cross_refs[fragment.fragment_id].append(target_id)

        return dict(cross_refs)

    def find_semantic_references(self, fragments: list[DocumentFragment], threshold: float = 0.8) -> dict[str, list[str]]:
        """Find semantic references using embedding similarity."""
        semantic_refs = defaultdict(list)

        if len(fragments) < 2:
            return dict(semantic_refs)

        # Get embeddings for all fragments
        embeddings_matrix = []
        fragment_ids = []

        for fragment in fragments:
            try:
                embedding = self.embeddings.embed_query(fragment.content)
                embeddings_matrix.append(embedding)
                fragment_ids.append(fragment.fragment_id)
            except Exception as e:
                self.logger.warning(f"Failed to embed fragment {fragment.fragment_id}: {e}")

        if len(embeddings_matrix) < 2:
            return dict(semantic_refs)

        # Calculate similarities
        embeddings_array = np.array(embeddings_matrix)
        similarities = cosine_similarity(embeddings_array)

        # Find high-similarity pairs
        for i, frag_id1 in enumerate(fragment_ids):
            for j, frag_id2 in enumerate(fragment_ids):
                if i != j and similarities[i][j] > threshold:
                    semantic_refs[frag_id1].append(frag_id2)

        return dict(semantic_refs)


class ContextReconstructor:
    """Main class for reconstructing document context and relationships."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.metadata_correlator = MetadataCorrelator()
        self.timeline_reconstructor = TimelineReconstructor()
        self.reference_resolver = ReferenceResolver(embeddings)

    def create_fragments_from_documents(self, documents: list[Document]) -> list[DocumentFragment]:
        """Create document fragments from LangChain documents."""
        fragments = []

        for i, doc in enumerate(documents):
            fragment_id = f"frag_{i:04d}"
            fragment = DocumentFragment(
                content=doc.page_content,
                metadata=doc.metadata,
                fragment_id=fragment_id
            )
            fragments.append(fragment)

        return fragments

    def cluster_related_fragments(self, fragments: list[DocumentFragment], eps: float = 0.3) -> dict[int, list[str]]:
        """Cluster related fragments using DBSCAN."""
        if len(fragments) < 2:
            return {0: [f.fragment_id for f in fragments]}

        # Get embeddings
        embeddings_matrix = []
        fragment_ids = []

        for fragment in fragments:
            try:
                embedding = self.embeddings.embed_query(fragment.content)
                embeddings_matrix.append(embedding)
                fragment_ids.append(fragment.fragment_id)
            except Exception as e:
                self.logger.warning(f"Failed to embed fragment {fragment.fragment_id}: {e}")

        if len(embeddings_matrix) < 2:
            return {0: fragment_ids}

        # Perform clustering
        embeddings_array = np.array(embeddings_matrix)
        clustering = DBSCAN(eps=eps, min_samples=2, metric='cosine')
        cluster_labels = clustering.fit_predict(embeddings_array)

        # Group by cluster
        clusters = defaultdict(list)
        for frag_id, label in zip(fragment_ids, cluster_labels, strict=False):
            clusters[label].append(frag_id)

        return dict(clusters)

    def reconstruct_document_structure(self, documents: list[Document]) -> list[DocumentStructure]:
        """Reconstruct document structures from fragments."""
        try:
            # Create fragments
            fragments = self.create_fragments_from_documents(documents)

            if not fragments:
                return []

            # Cluster related fragments
            clusters = self.cluster_related_fragments(fragments)

            # Create document structures
            document_structures = []
            fragment_map = {f.fragment_id: f for f in fragments}

            for cluster_id, fragment_ids in clusters.items():
                doc_structure = DocumentStructure(f"doc_{cluster_id}")

                # Add fragments to structure
                for frag_id in fragment_ids:
                    if frag_id in fragment_map:
                        doc_structure.add_fragment(fragment_map[frag_id])

                # Find relationships within cluster
                cluster_fragments = [fragment_map[fid] for fid in fragment_ids if fid in fragment_map]

                # Metadata correlations
                correlations = self.metadata_correlator.correlate_fragments(cluster_fragments)
                for correlation_type, pairs in correlations.items():
                    strength = 0.8 if correlation_type == "high" else 0.5
                    for pair in pairs:
                        source_id, target_id = pair.split("-")
                        doc_structure.add_relationship(source_id, target_id, "metadata_correlation", strength)

                # Cross-references
                cross_refs = self.reference_resolver.resolve_cross_references(cluster_fragments)
                for source_id, target_ids in cross_refs.items():
                    for target_id in target_ids:
                        doc_structure.add_relationship(source_id, target_id, "cross_reference", 0.9)

                # Semantic references
                semantic_refs = self.reference_resolver.find_semantic_references(cluster_fragments)
                for source_id, target_ids in semantic_refs.items():
                    for target_id in target_ids:
                        doc_structure.add_relationship(source_id, target_id, "semantic_reference", 0.7)

                # Temporal relationships
                timeline = self.timeline_reconstructor.build_timeline(cluster_fragments)
                temporal_rels = self.timeline_reconstructor.detect_temporal_relationships(timeline)
                for rel in temporal_rels:
                    strength = 0.6 if rel["type"].startswith("sequential") else 0.8
                    doc_structure.add_relationship(rel["source"], rel["target"], rel["type"], strength)

                # Calculate confidence score
                doc_structure.confidence_score = self._calculate_structure_confidence(doc_structure)
                document_structures.append(doc_structure)

            return document_structures

        except Exception as e:
            self.logger.error(f"Document structure reconstruction failed: {e}")
            return []

    def _calculate_structure_confidence(self, doc_structure: DocumentStructure) -> float:
        """Calculate confidence score for document structure."""
        if not doc_structure.fragments:
            return 0.0

        # Factors affecting confidence
        num_fragments = len(doc_structure.fragments)
        num_relationships = doc_structure.structure_graph.number_of_edges()

        # Base confidence from fragment count
        fragment_confidence = min(num_fragments / 10, 1.0)

        # Relationship density
        max_relationships = num_fragments * (num_fragments - 1)
        relationship_density = num_relationships / max_relationships if max_relationships > 0 else 0

        # Combined confidence
        confidence = 0.6 * fragment_confidence + 0.4 * relationship_density
        return min(confidence, 1.0)

    def export_structure_analysis(self, doc_structures: list[DocumentStructure]) -> dict[str, Any]:
        """Export structure analysis results."""
        analysis = {
            "num_documents": len(doc_structures),
            "total_fragments": sum(len(ds.fragments) for ds in doc_structures),
            "total_relationships": sum(ds.structure_graph.number_of_edges() for ds in doc_structures),
            "avg_confidence": np.mean([ds.confidence_score for ds in doc_structures]) if doc_structures else 0,
            "documents": []
        }

        for doc_structure in doc_structures:
            doc_info = {
                "doc_id": doc_structure.doc_id,
                "num_fragments": len(doc_structure.fragments),
                "num_relationships": doc_structure.structure_graph.number_of_edges(),
                "confidence_score": doc_structure.confidence_score,
                "relationship_types": {}
            }

            # Count relationship types
            for _, _, data in doc_structure.structure_graph.edges(data=True):
                rel_type = data.get("type", "unknown")
                doc_info["relationship_types"][rel_type] = doc_info["relationship_types"].get(rel_type, 0) + 1

            analysis["documents"].append(doc_info)

        return analysis
