"""Data recovery tools for partial data recovery and forensic analysis."""

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity


class DataFragment:
    """Represents a fragment of recovered data."""

    def __init__(self, content: str, fragment_id: str, confidence: float = 0.0):
        self.content = content
        self.fragment_id = fragment_id
        self.confidence = confidence
        self.metadata = {}
        self.relationships = []
        self.integrity_score = 0.0
        self.recovery_method = ""

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the fragment."""
        self.metadata[key] = value

    def add_relationship(self, target_id: str, relationship_type: str, strength: float):
        """Add a relationship to another fragment."""
        self.relationships.append({
            "target": target_id,
            "type": relationship_type,
            "strength": strength
        })


class IntegrityValidator:
    """Validates data integrity and detects corruption."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_fragment(self, fragment: DataFragment) -> dict[str, Any]:
        """Validate the integrity of a data fragment."""
        validation_result = {
            "fragment_id": fragment.fragment_id,
            "is_valid": True,
            "issues": [],
            "confidence": fragment.confidence,
            "integrity_score": 0.0
        }

        content = fragment.content

        # Check for common corruption indicators
        issues = []

        # Check for encoding issues
        try:
            content.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("encoding_corruption")

        # Check for truncation
        if content.endswith("...") or len(content) < 10:
            issues.append("potential_truncation")

        # Check for repeated patterns (possible corruption)
        if self._has_repeated_patterns(content):
            issues.append("repeated_patterns")

        # Check for missing structure
        if not self._has_logical_structure(content):
            issues.append("missing_structure")

        # Check for character anomalies
        if self._has_character_anomalies(content):
            issues.append("character_anomalies")

        validation_result["issues"] = issues
        validation_result["is_valid"] = len(issues) == 0

        # Calculate integrity score
        integrity_score = 1.0 - (len(issues) * 0.2)
        integrity_score = max(0.0, integrity_score)
        validation_result["integrity_score"] = integrity_score
        fragment.integrity_score = integrity_score

        return validation_result

    def _has_repeated_patterns(self, content: str) -> bool:
        """Check for suspicious repeated patterns."""
        # Look for repeated sequences of 10+ characters
        for i in range(len(content) - 20):
            substring = content[i:i+10]
            if content.count(substring) > 3:
                return True
        return False

    def _has_logical_structure(self, content: str) -> bool:
        """Check if content has logical structure."""
        # Look for common structural elements
        structure_indicators = [
            r'\n',  # Line breaks
            r'\.',  # Sentences
            r':',   # Key-value pairs
            r',',   # Lists
            r'\s{2,}',  # Indentation
        ]

        indicator_count = 0
        for pattern in structure_indicators:
            if re.search(pattern, content):
                indicator_count += 1

        return indicator_count >= 2

    def _has_character_anomalies(self, content: str) -> bool:
        """Check for character anomalies."""
        # Check for excessive special characters
        special_char_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace()) / len(content)
        if special_char_ratio > 0.3:
            return True

        # Check for null bytes or control characters
        if any(ord(c) < 32 and c not in '\n\r\t' for c in content):
            return True

        return False

    def validate_collection(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Validate a collection of fragments."""
        results = {
            "total_fragments": len(fragments),
            "valid_fragments": 0,
            "invalid_fragments": 0,
            "avg_integrity_score": 0.0,
            "common_issues": defaultdict(int),
            "fragment_results": []
        }

        integrity_scores = []

        for fragment in fragments:
            validation = self.validate_fragment(fragment)
            results["fragment_results"].append(validation)

            if validation["is_valid"]:
                results["valid_fragments"] += 1
            else:
                results["invalid_fragments"] += 1

            integrity_scores.append(validation["integrity_score"])

            # Count common issues
            for issue in validation["issues"]:
                results["common_issues"][issue] += 1

        if integrity_scores:
            results["avg_integrity_score"] = np.mean(integrity_scores)

        results["common_issues"] = dict(results["common_issues"])
        return results


class PartialRecovery:
    """Recovers partial data from fragmented embeddings."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

    def recover_from_fragments(self, documents: list[Document], similarity_threshold: float = 0.7) -> list[DataFragment]:
        """Recover data fragments from document embeddings."""
        fragments = []

        # Convert documents to fragments
        for i, doc in enumerate(documents):
            fragment = DataFragment(
                content=doc.page_content,
                fragment_id=f"frag_{i:04d}",
                confidence=1.0  # Start with full confidence
            )
            fragment.recovery_method = "direct_extraction"
            fragment.add_metadata("source_document", i)
            fragment.add_metadata("original_metadata", doc.metadata)
            fragments.append(fragment)

        # Find related fragments using embedding similarity
        if len(fragments) > 1:
            self._find_fragment_relationships(fragments, similarity_threshold)

        # Attempt to reconstruct missing parts
        reconstructed_fragments = self._reconstruct_missing_parts(fragments)
        fragments.extend(reconstructed_fragments)

        return fragments

    def _find_fragment_relationships(self, fragments: list[DataFragment], threshold: float):
        """Find relationships between fragments using embeddings."""
        try:
            # Get embeddings for all fragments
            embeddings_matrix = []
            for fragment in fragments:
                embedding = self.embeddings.embed_query(fragment.content)
                embeddings_matrix.append(embedding)

            embeddings_array = np.array(embeddings_matrix)
            similarities = cosine_similarity(embeddings_array)

            # Find high-similarity pairs
            for i, frag1 in enumerate(fragments):
                for j, frag2 in enumerate(fragments[i+1:], i+1):
                    similarity = similarities[i][j]
                    if similarity > threshold:
                        frag1.add_relationship(frag2.fragment_id, "similar_content", similarity)
                        frag2.add_relationship(frag1.fragment_id, "similar_content", similarity)

        except Exception as e:
            self.logger.warning(f"Failed to find fragment relationships: {e}")

    def _reconstruct_missing_parts(self, fragments: list[DataFragment]) -> list[DataFragment]:
        """Attempt to reconstruct missing parts from existing fragments."""
        reconstructed = []

        # Look for patterns that suggest missing content
        for fragment in fragments:
            content = fragment.content

            # Check for incomplete sentences
            if self._has_incomplete_sentences(content):
                reconstructed_content = self._attempt_sentence_completion(content, fragments)
                if reconstructed_content and reconstructed_content != content:
                    new_fragment = DataFragment(
                        content=reconstructed_content,
                        fragment_id=f"{fragment.fragment_id}_reconstructed",
                        confidence=0.6
                    )
                    new_fragment.recovery_method = "sentence_reconstruction"
                    new_fragment.add_metadata("source_fragment", fragment.fragment_id)
                    reconstructed.append(new_fragment)

            # Check for missing references
            missing_refs = self._find_missing_references(content, fragments)
            for ref_content in missing_refs:
                new_fragment = DataFragment(
                    content=ref_content,
                    fragment_id=f"{fragment.fragment_id}_ref_{len(missing_refs)}",
                    confidence=0.4
                )
                new_fragment.recovery_method = "reference_reconstruction"
                new_fragment.add_metadata("source_fragment", fragment.fragment_id)
                reconstructed.append(new_fragment)

        return reconstructed

    def _has_incomplete_sentences(self, content: str) -> bool:
        """Check if content has incomplete sentences."""
        # Look for sentences that don't end with proper punctuation
        sentences = re.split(r'[.!?]', content)
        for sentence in sentences[:-1]:  # Exclude last sentence
            if sentence.strip() and not sentence.strip().endswith(('.', '!', '?')):
                return True
        return False

    def _attempt_sentence_completion(self, content: str, fragments: list[DataFragment]) -> Optional[str]:
        """Attempt to complete incomplete sentences using other fragments."""
        # Simple approach: look for similar content in other fragments
        incomplete_parts = []

        # Find incomplete sentences
        sentences = content.split('.')
        for sentence in sentences:
            if sentence.strip() and len(sentence.strip()) < 20:  # Likely incomplete
                incomplete_parts.append(sentence.strip())

        if not incomplete_parts:
            return None

        # Try to find completions in other fragments
        completed_content = content
        for incomplete in incomplete_parts:
            for fragment in fragments:
                if incomplete.lower() in fragment.content.lower():
                    # Find the complete sentence containing this part
                    fragment_sentences = fragment.content.split('.')
                    for frag_sentence in fragment_sentences:
                        if incomplete.lower() in frag_sentence.lower():
                            completed_content = completed_content.replace(incomplete, frag_sentence.strip())
                            break

        return completed_content if completed_content != content else None

    def _find_missing_references(self, content: str, fragments: list[DataFragment]) -> list[str]:
        """Find missing references that might be in other fragments."""
        missing_refs = []

        # Look for reference patterns
        ref_patterns = [
            r'see\s+([^.]+)',
            r'according\s+to\s+([^.]+)',
            r'as\s+mentioned\s+in\s+([^.]+)',
            r'refer\s+to\s+([^.]+)'
        ]

        for pattern in ref_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                reference = match.group(1).strip()

                # Try to find this reference in other fragments
                for fragment in fragments:
                    if reference.lower() in fragment.content.lower():
                        # Extract relevant context
                        context_start = max(0, fragment.content.lower().find(reference.lower()) - 100)
                        context_end = min(len(fragment.content),
                                        fragment.content.lower().find(reference.lower()) + len(reference) + 100)
                        context = fragment.content[context_start:context_end]
                        missing_refs.append(context)
                        break

        return missing_refs


class ForensicAnalyzer:
    """Performs forensic analysis on recovered data."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_data_provenance(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Analyze the provenance and history of data fragments."""
        analysis = {
            "total_fragments": len(fragments),
            "recovery_methods": defaultdict(int),
            "confidence_distribution": [],
            "integrity_distribution": [],
            "metadata_analysis": {},
            "timeline_analysis": {},
            "source_analysis": {}
        }

        # Analyze recovery methods
        for fragment in fragments:
            analysis["recovery_methods"][fragment.recovery_method] += 1
            analysis["confidence_distribution"].append(fragment.confidence)
            analysis["integrity_distribution"].append(fragment.integrity_score)

        # Convert to regular dict
        analysis["recovery_methods"] = dict(analysis["recovery_methods"])

        # Analyze metadata patterns
        analysis["metadata_analysis"] = self._analyze_metadata_patterns(fragments)

        # Analyze timeline if available
        analysis["timeline_analysis"] = self._analyze_timeline(fragments)

        # Analyze sources
        analysis["source_analysis"] = self._analyze_sources(fragments)

        return analysis

    def _analyze_metadata_patterns(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Analyze patterns in fragment metadata."""
        metadata_keys = set()
        metadata_values = defaultdict(set)

        for fragment in fragments:
            for key, value in fragment.metadata.items():
                metadata_keys.add(key)
                metadata_values[key].add(str(value))

        return {
            "unique_keys": list(metadata_keys),
            "key_value_counts": {k: len(v) for k, v in metadata_values.items()},
            "most_common_keys": list(metadata_keys)[:10]
        }

    def _analyze_timeline(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Analyze temporal patterns in fragments."""
        timestamps = []

        for fragment in fragments:
            # Look for timestamps in metadata
            for key, value in fragment.metadata.items():
                if 'time' in key.lower() or 'date' in key.lower():
                    try:
                        if isinstance(value, (int, float)):
                            timestamps.append(value)
                        elif isinstance(value, str):
                            # Try to parse as timestamp
                            timestamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            timestamps.append(timestamp.timestamp())
                    except (ValueError, TypeError):
                        continue

            # Look for timestamps in content
            timestamp_patterns = [
                r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
                r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
                r'\d{10,13}'  # Unix timestamps
            ]

            for pattern in timestamp_patterns:
                matches = re.findall(pattern, fragment.content)
                for match in matches:
                    try:
                        if match.isdigit():
                            timestamp = int(match)
                            if timestamp > 1000000000:  # Valid Unix timestamp
                                timestamps.append(timestamp)
                    except ValueError:
                        continue

        if timestamps:
            return {
                "total_timestamps": len(timestamps),
                "earliest": min(timestamps),
                "latest": max(timestamps),
                "span_seconds": max(timestamps) - min(timestamps) if timestamps else 0
            }

        return {"total_timestamps": 0}

    def _analyze_sources(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Analyze source information for fragments."""
        sources = defaultdict(int)
        source_types = defaultdict(int)

        for fragment in fragments:
            # Check metadata for source information
            source = fragment.metadata.get("source", "unknown")
            sources[source] += 1

            # Determine source type
            if "document" in str(source).lower():
                source_types["document"] += 1
            elif "email" in str(source).lower():
                source_types["email"] += 1
            elif "database" in str(source).lower():
                source_types["database"] += 1
            else:
                source_types["other"] += 1

        return {
            "unique_sources": len(sources),
            "source_distribution": dict(sources),
            "source_types": dict(source_types)
        }

    def detect_tampering(self, fragments: list[DataFragment]) -> list[dict[str, Any]]:
        """Detect potential tampering or manipulation."""
        tampering_indicators = []

        for fragment in fragments:
            indicators = []

            # Check for inconsistent metadata
            if self._has_inconsistent_metadata(fragment):
                indicators.append("inconsistent_metadata")

            # Check for suspicious content patterns
            if self._has_suspicious_patterns(fragment.content):
                indicators.append("suspicious_content_patterns")

            # Check for integrity issues
            if fragment.integrity_score < 0.5:
                indicators.append("low_integrity_score")

            # Check for unusual confidence scores
            if fragment.confidence < 0.3:
                indicators.append("low_confidence")

            if indicators:
                tampering_indicators.append({
                    "fragment_id": fragment.fragment_id,
                    "indicators": indicators,
                    "risk_level": len(indicators) / 4.0  # Normalize to 0-1
                })

        return tampering_indicators

    def _has_inconsistent_metadata(self, fragment: DataFragment) -> bool:
        """Check for inconsistent metadata."""
        metadata = fragment.metadata

        # Check for timestamp inconsistencies
        timestamps = []
        for key, value in metadata.items():
            if 'time' in key.lower() or 'date' in key.lower():
                try:
                    if isinstance(value, (int, float)):
                        timestamps.append(value)
                except (ValueError, TypeError):
                    continue

        # If we have multiple timestamps, check if they're reasonable
        if len(timestamps) > 1:
            time_diff = max(timestamps) - min(timestamps)
            if time_diff > 86400 * 365:  # More than a year difference
                return True

        return False

    def _has_suspicious_patterns(self, content: str) -> bool:
        """Check for suspicious content patterns."""
        # Look for patterns that might indicate tampering
        suspicious_patterns = [
            r'[A-Za-z]{50,}',  # Very long words (possible encoding artifacts)
            r'\d{20,}',        # Very long numbers
            r'(.)\1{10,}',     # Repeated characters
            r'[^\x20-\x7E]{5,}' # Non-printable character sequences
        ]

        for pattern in suspicious_patterns:
            if re.search(pattern, content):
                return True

        return False


class DataRecoveryTools:
    """Main class for data recovery and forensic analysis."""

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.integrity_validator = IntegrityValidator()
        self.partial_recovery = PartialRecovery(embeddings)
        self.forensic_analyzer = ForensicAnalyzer()

    def recover_data(self, documents: list[Document], similarity_threshold: float = 0.7) -> dict[str, Any]:
        """Perform comprehensive data recovery."""
        try:
            recovery_results = {
                "fragments": [],
                "validation": {},
                "forensic_analysis": {},
                "export_data": {},
                "summary": {}
            }

            # Step 1: Recover fragments
            self.logger.info("Recovering data fragments...")
            fragments = self.partial_recovery.recover_from_fragments(documents, similarity_threshold)

            # Step 2: Validate integrity
            self.logger.info("Validating data integrity...")
            validation_results = self.integrity_validator.validate_collection(fragments)

            # Step 3: Forensic analysis
            self.logger.info("Performing forensic analysis...")
            forensic_results = self.forensic_analyzer.analyze_data_provenance(fragments)
            tampering_indicators = self.forensic_analyzer.detect_tampering(fragments)

            # Step 4: Prepare export data
            export_data = self._prepare_export_data(fragments)

            # Step 5: Generate summary
            summary = self._generate_recovery_summary(fragments, validation_results, forensic_results)

            recovery_results.update({
                "fragments": [self._fragment_to_dict(f) for f in fragments],
                "validation": validation_results,
                "forensic_analysis": {
                    "provenance": forensic_results,
                    "tampering_indicators": tampering_indicators
                },
                "export_data": export_data,
                "summary": summary
            })

            return recovery_results

        except Exception as e:
            self.logger.error(f"Data recovery failed: {e}")
            return {"error": str(e)}

    def _fragment_to_dict(self, fragment: DataFragment) -> dict[str, Any]:
        """Convert fragment to dictionary for serialization."""
        return {
            "fragment_id": fragment.fragment_id,
            "content": fragment.content,
            "confidence": fragment.confidence,
            "integrity_score": fragment.integrity_score,
            "recovery_method": fragment.recovery_method,
            "metadata": fragment.metadata,
            "relationships": fragment.relationships
        }

    def _prepare_export_data(self, fragments: list[DataFragment]) -> dict[str, Any]:
        """Prepare data for export."""
        export_data = {
            "formats": {
                "json": self._export_as_json(fragments),
                "csv": self._export_as_csv(fragments),
                "text": self._export_as_text(fragments)
            },
            "statistics": {
                "total_fragments": len(fragments),
                "total_content_length": sum(len(f.content) for f in fragments),
                "avg_confidence": np.mean([f.confidence for f in fragments]) if fragments else 0,
                "avg_integrity": np.mean([f.integrity_score for f in fragments]) if fragments else 0
            }
        }

        return export_data

    def _export_as_json(self, fragments: list[DataFragment]) -> str:
        """Export fragments as JSON."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_fragments": len(fragments),
            "fragments": [self._fragment_to_dict(f) for f in fragments]
        }
        return json.dumps(data, indent=2)

    def _export_as_csv(self, fragments: list[DataFragment]) -> str:
        """Export fragments as CSV."""
        lines = ["fragment_id,confidence,integrity_score,recovery_method,content_length,content_preview"]

        for fragment in fragments:
            content_preview = fragment.content[:100].replace('"', '""').replace('\n', ' ')
            line = f'"{fragment.fragment_id}",{fragment.confidence},{fragment.integrity_score},"{fragment.recovery_method}",{len(fragment.content)},"{content_preview}"'
            lines.append(line)

        return '\n'.join(lines)

    def _export_as_text(self, fragments: list[DataFragment]) -> str:
        """Export fragments as plain text."""
        lines = [f"Data Recovery Export - {datetime.now().isoformat()}", "=" * 50, ""]

        for fragment in fragments:
            lines.extend([
                f"Fragment ID: {fragment.fragment_id}",
                f"Confidence: {fragment.confidence:.2f}",
                f"Integrity: {fragment.integrity_score:.2f}",
                f"Recovery Method: {fragment.recovery_method}",
                f"Content Length: {len(fragment.content)}",
                "Content:",
                "-" * 20,
                fragment.content,
                "=" * 50,
                ""
            ])

        return '\n'.join(lines)

    def _generate_recovery_summary(self, fragments: list[DataFragment], validation: dict[str, Any], forensic: dict[str, Any]) -> dict[str, Any]:
        """Generate a summary of the recovery process."""
        summary = {
            "recovery_success": len(fragments) > 0,
            "total_fragments_recovered": len(fragments),
            "data_quality": {
                "avg_confidence": np.mean([f.confidence for f in fragments]) if fragments else 0,
                "avg_integrity": np.mean([f.integrity_score for f in fragments]) if fragments else 0,
                "valid_fragments": validation.get("valid_fragments", 0),
                "invalid_fragments": validation.get("invalid_fragments", 0)
            },
            "recovery_methods": forensic.get("recovery_methods", {}),
            "recommendations": []
        }

        # Generate recommendations
        if summary["data_quality"]["avg_confidence"] < 0.5:
            summary["recommendations"].append("Low confidence scores detected - manual review recommended")

        if summary["data_quality"]["avg_integrity"] < 0.7:
            summary["recommendations"].append("Integrity issues detected - data may be corrupted")

        if validation.get("invalid_fragments", 0) > 0:
            summary["recommendations"].append("Invalid fragments found - forensic analysis recommended")

        return summary

    def export_recovery_report(self, recovery_results: dict[str, Any], output_path: str):
        """Export a comprehensive recovery report."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Create comprehensive report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "tool_version": "VectorSmuggle Data Recovery v1.0",
                    "total_fragments": len(recovery_results.get("fragments", []))
                },
                "recovery_results": recovery_results
            }

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Recovery report exported to: {output_file}")

        except Exception as e:
            self.logger.error(f"Failed to export recovery report: {e}")
            raise
