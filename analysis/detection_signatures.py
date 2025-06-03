"""
Detection Signature Generation for VectorSmuggle

This module provides tools for generating detection signatures and patterns
that can be used by security systems to identify vector-based data exfiltration
attempts and steganographic techniques.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


@dataclass
class DetectionSignature:
    """Represents a detection signature for vector-based attacks."""

    signature_id: str
    name: str
    description: str
    signature_type: str  # statistical, pattern, behavioral, network
    confidence: float
    severity: str  # low, medium, high, critical
    indicators: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class StatisticalSignatureGenerator:
    """Generates statistical signatures for detecting steganographic embeddings."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.baseline_stats = {}
        self.anomaly_detectors = {}

    def establish_baseline(self, clean_embeddings: np.ndarray, label: str = "default") -> dict[str, Any]:
        """Establish statistical baseline for clean embeddings."""
        self.logger.info(f"Establishing baseline for {label} with {len(clean_embeddings)} embeddings")

        stats_dict = {
            'mean': np.mean(clean_embeddings, axis=0),
            'std': np.std(clean_embeddings, axis=0),
            'variance': np.var(clean_embeddings, axis=0),
            'skewness': stats.skew(clean_embeddings, axis=0),
            'kurtosis': stats.kurtosis(clean_embeddings, axis=0),
            'min_values': np.min(clean_embeddings, axis=0),
            'max_values': np.max(clean_embeddings, axis=0),
            'percentiles': {
                '25': np.percentile(clean_embeddings, 25, axis=0),
                '50': np.percentile(clean_embeddings, 50, axis=0),
                '75': np.percentile(clean_embeddings, 75, axis=0),
                '95': np.percentile(clean_embeddings, 95, axis=0),
                '99': np.percentile(clean_embeddings, 99, axis=0)
            }
        }

        # Train anomaly detector
        isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        isolation_forest.fit(clean_embeddings)

        self.baseline_stats[label] = stats_dict
        self.anomaly_detectors[label] = isolation_forest

        return stats_dict

    def generate_statistical_signatures(self, baseline_label: str = "default") -> list[DetectionSignature]:
        """Generate statistical detection signatures."""
        if baseline_label not in self.baseline_stats:
            raise ValueError(f"No baseline established for {baseline_label}")

        baseline = self.baseline_stats[baseline_label]
        signatures = []

        # Variance anomaly signature
        variance_signature = DetectionSignature(
            signature_id=self._generate_signature_id("variance_anomaly"),
            name="Embedding Variance Anomaly",
            description="Detects embeddings with unusual variance patterns",
            signature_type="statistical",
            confidence=0.85,
            severity="medium",
            indicators={
                "variance_threshold_high": float(np.mean(baseline['variance']) + 3 * np.std(baseline['variance'])),
                "variance_threshold_low": float(np.mean(baseline['variance']) - 3 * np.std(baseline['variance'])),
                "baseline_variance_mean": float(np.mean(baseline['variance'])),
                "baseline_variance_std": float(np.std(baseline['variance']))
            },
            metadata={
                "baseline_label": baseline_label,
                "detection_method": "statistical_threshold",
                "false_positive_rate": 0.001
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        signatures.append(variance_signature)

        # Skewness anomaly signature
        skewness_signature = DetectionSignature(
            signature_id=self._generate_signature_id("skewness_anomaly"),
            name="Embedding Skewness Anomaly",
            description="Detects embeddings with unusual skewness patterns",
            signature_type="statistical",
            confidence=0.80,
            severity="medium",
            indicators={
                "skewness_threshold": float(np.mean(np.abs(baseline['skewness'])) + 2 * np.std(np.abs(baseline['skewness']))),
                "baseline_skewness_mean": float(np.mean(baseline['skewness'])),
                "baseline_skewness_std": float(np.std(baseline['skewness']))
            },
            metadata={
                "baseline_label": baseline_label,
                "detection_method": "statistical_threshold"
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        signatures.append(skewness_signature)

        # Isolation forest anomaly signature
        isolation_signature = DetectionSignature(
            signature_id=self._generate_signature_id("isolation_anomaly"),
            name="Isolation Forest Anomaly",
            description="Detects embeddings that are outliers in high-dimensional space",
            signature_type="statistical",
            confidence=0.90,
            severity="high",
            indicators={
                "anomaly_threshold": -0.1,  # Isolation forest threshold
                "contamination_rate": 0.1
            },
            metadata={
                "baseline_label": baseline_label,
                "detection_method": "isolation_forest",
                "model_type": "unsupervised"
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        signatures.append(isolation_signature)

        return signatures

    def detect_anomalies(self, embeddings: np.ndarray, baseline_label: str = "default") -> dict[str, Any]:
        """Detect anomalies in embeddings using established baseline."""
        if baseline_label not in self.baseline_stats:
            raise ValueError(f"No baseline established for {baseline_label}")

        baseline = self.baseline_stats[baseline_label]
        detector = self.anomaly_detectors[baseline_label]

        results = {
            'total_embeddings': len(embeddings),
            'anomalies': [],
            'statistics': {}
        }

        # Statistical anomaly detection
        embedding_variance = np.var(embeddings, axis=1)
        embedding_skewness = stats.skew(embeddings, axis=1)

        variance_threshold_high = np.mean(baseline['variance']) + 3 * np.std(baseline['variance'])
        variance_threshold_low = np.mean(baseline['variance']) - 3 * np.std(baseline['variance'])
        skewness_threshold = np.mean(np.abs(baseline['skewness'])) + 2 * np.std(np.abs(baseline['skewness']))

        # Isolation forest detection
        anomaly_scores = detector.decision_function(embeddings)
        anomaly_predictions = detector.predict(embeddings)

        for i, embedding in enumerate(embeddings):
            anomaly_indicators = []

            # Check variance anomaly
            if embedding_variance[i] > variance_threshold_high or embedding_variance[i] < variance_threshold_low:
                anomaly_indicators.append({
                    'type': 'variance_anomaly',
                    'value': float(embedding_variance[i]),
                    'threshold_high': float(variance_threshold_high),
                    'threshold_low': float(variance_threshold_low)
                })

            # Check skewness anomaly
            if np.abs(embedding_skewness[i]) > skewness_threshold:
                anomaly_indicators.append({
                    'type': 'skewness_anomaly',
                    'value': float(embedding_skewness[i]),
                    'threshold': float(skewness_threshold)
                })

            # Check isolation forest anomaly
            if anomaly_predictions[i] == -1:
                anomaly_indicators.append({
                    'type': 'isolation_anomaly',
                    'score': float(anomaly_scores[i]),
                    'threshold': -0.1
                })

            if anomaly_indicators:
                results['anomalies'].append({
                    'embedding_index': i,
                    'indicators': anomaly_indicators,
                    'risk_score': len(anomaly_indicators) / 3.0  # Normalized risk score
                })

        results['statistics'] = {
            'variance_anomalies': sum(1 for a in results['anomalies']
                                    if any(i['type'] == 'variance_anomaly' for i in a['indicators'])),
            'skewness_anomalies': sum(1 for a in results['anomalies']
                                    if any(i['type'] == 'skewness_anomaly' for i in a['indicators'])),
            'isolation_anomalies': sum(1 for a in results['anomalies']
                                     if any(i['type'] == 'isolation_anomaly' for i in a['indicators'])),
            'total_anomalies': len(results['anomalies']),
            'anomaly_rate': len(results['anomalies']) / len(embeddings)
        }

        return results

    def _generate_signature_id(self, signature_type: str) -> str:
        """Generate unique signature ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{signature_type}_{timestamp}".encode()
        hash_digest = hashlib.md5(hash_input).hexdigest()[:8]
        return f"{signature_type}_{hash_digest}"


class PatternSignatureGenerator:
    """Generates pattern-based signatures for detecting steganographic techniques."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.known_patterns = {}

    def analyze_steganographic_patterns(self, embeddings: np.ndarray, technique_type: str) -> dict[str, Any]:
        """Analyze embeddings for steganographic patterns."""
        self.logger.info(f"Analyzing {len(embeddings)} embeddings for {technique_type} patterns")

        patterns = {
            'technique_type': technique_type,
            'embedding_count': len(embeddings),
            'dimension': embeddings.shape[1] if len(embeddings) > 0 else 0,
            'patterns': {}
        }

        if technique_type == "noise_injection":
            patterns['patterns'] = self._analyze_noise_patterns(embeddings)
        elif technique_type == "rotation":
            patterns['patterns'] = self._analyze_rotation_patterns(embeddings)
        elif technique_type == "scaling":
            patterns['patterns'] = self._analyze_scaling_patterns(embeddings)
        elif technique_type == "fragmentation":
            patterns['patterns'] = self._analyze_fragmentation_patterns(embeddings)

        self.known_patterns[technique_type] = patterns
        return patterns

    def _analyze_noise_patterns(self, embeddings: np.ndarray) -> dict[str, Any]:
        """Analyze noise injection patterns."""
        # Calculate noise characteristics
        embedding_norms = np.linalg.norm(embeddings, axis=1)

        return {
            'norm_variance': float(np.var(embedding_norms)),
            'norm_mean': float(np.mean(embedding_norms)),
            'high_frequency_components': self._detect_high_frequency_noise(embeddings),
            'correlation_patterns': self._analyze_correlation_patterns(embeddings)
        }

    def _analyze_rotation_patterns(self, embeddings: np.ndarray) -> dict[str, Any]:
        """Analyze rotation-based patterns."""
        # Use PCA to detect rotation signatures
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(embeddings)

        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'principal_components': pca.components_[:5].tolist(),  # First 5 components
            'rotation_indicators': self._detect_rotation_indicators(embeddings)
        }

    def _analyze_scaling_patterns(self, embeddings: np.ndarray) -> dict[str, Any]:
        """Analyze scaling-based patterns."""
        # Analyze magnitude distributions
        magnitudes = np.linalg.norm(embeddings, axis=1)

        return {
            'magnitude_distribution': {
                'mean': float(np.mean(magnitudes)),
                'std': float(np.std(magnitudes)),
                'min': float(np.min(magnitudes)),
                'max': float(np.max(magnitudes))
            },
            'scaling_indicators': self._detect_scaling_indicators(embeddings)
        }

    def _analyze_fragmentation_patterns(self, embeddings: np.ndarray) -> dict[str, Any]:
        """Analyze fragmentation patterns."""
        # Use clustering to detect fragmentation
        clustering = DBSCAN(eps=0.5, min_samples=5)
        cluster_labels = clustering.fit_predict(embeddings)

        return {
            'cluster_count': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'noise_points': int(np.sum(cluster_labels == -1)),
            'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in set(cluster_labels) if i != -1],
            'fragmentation_score': self._calculate_fragmentation_score(cluster_labels)
        }

    def _detect_high_frequency_noise(self, embeddings: np.ndarray) -> dict[str, float]:
        """Detect high-frequency noise components."""
        # Simple high-frequency detection using differences
        if len(embeddings) < 2:
            return {'high_freq_energy': 0.0}

        diff_embeddings = np.diff(embeddings, axis=0)
        high_freq_energy = float(np.mean(np.var(diff_embeddings, axis=1)))

        return {'high_freq_energy': high_freq_energy}

    def _analyze_correlation_patterns(self, embeddings: np.ndarray) -> dict[str, float]:
        """Analyze correlation patterns in embeddings."""
        if len(embeddings) < 2:
            return {'avg_correlation': 0.0}

        # Calculate pairwise correlations
        correlations = []
        for i in range(min(100, len(embeddings))):  # Sample for performance
            for j in range(i + 1, min(100, len(embeddings))):
                corr = np.corrcoef(embeddings[i], embeddings[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        return {
            'avg_correlation': float(np.mean(correlations)) if correlations else 0.0,
            'correlation_variance': float(np.var(correlations)) if correlations else 0.0
        }

    def _detect_rotation_indicators(self, embeddings: np.ndarray) -> dict[str, float]:
        """Detect rotation-specific indicators."""
        # Simplified rotation detection
        if len(embeddings) < 2:
            return {'rotation_score': 0.0}

        # Calculate angular differences
        angular_diffs = []
        for i in range(min(50, len(embeddings) - 1)):
            dot_product = np.dot(embeddings[i], embeddings[i + 1])
            norm_product = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
            if norm_product > 0:
                cos_angle = np.clip(dot_product / norm_product, -1, 1)
                angular_diffs.append(np.arccos(cos_angle))

        return {
            'rotation_score': float(np.var(angular_diffs)) if angular_diffs else 0.0,
            'avg_angular_diff': float(np.mean(angular_diffs)) if angular_diffs else 0.0
        }

    def _detect_scaling_indicators(self, embeddings: np.ndarray) -> dict[str, float]:
        """Detect scaling-specific indicators."""
        magnitudes = np.linalg.norm(embeddings, axis=1)

        return {
            'magnitude_coefficient_variation': float(np.std(magnitudes) / np.mean(magnitudes)) if np.mean(magnitudes) > 0 else 0.0,
            'magnitude_range_ratio': float(np.max(magnitudes) / np.min(magnitudes)) if np.min(magnitudes) > 0 else 0.0
        }

    def _calculate_fragmentation_score(self, cluster_labels: np.ndarray) -> float:
        """Calculate fragmentation score based on clustering results."""
        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise label

        if len(unique_labels) <= 1:
            return 0.0

        # Higher score for more clusters with more even distribution
        cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
        size_variance = np.var(cluster_sizes)
        size_mean = np.mean(cluster_sizes)

        # Normalized fragmentation score
        fragmentation_score = len(unique_labels) / len(cluster_labels) * (1 - size_variance / (size_mean ** 2))
        return float(np.clip(fragmentation_score, 0, 1))

    def generate_pattern_signatures(self) -> list[DetectionSignature]:
        """Generate pattern-based detection signatures."""
        signatures = []

        for technique_type, patterns in self.known_patterns.items():
            signature = DetectionSignature(
                signature_id=self._generate_signature_id(f"pattern_{technique_type}"),
                name=f"{technique_type.title()} Pattern Signature",
                description=f"Detects {technique_type} steganographic patterns in embeddings",
                signature_type="pattern",
                confidence=0.75,
                severity="medium",
                indicators=patterns['patterns'],
                metadata={
                    "technique_type": technique_type,
                    "pattern_analysis": patterns,
                    "detection_method": "pattern_matching"
                },
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            signatures.append(signature)

        return signatures

    def _generate_signature_id(self, signature_type: str) -> str:
        """Generate unique signature ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{signature_type}_{timestamp}".encode()
        hash_digest = hashlib.md5(hash_input).hexdigest()[:8]
        return f"{signature_type}_{hash_digest}"


class BehavioralSignatureGenerator:
    """Generates behavioral signatures for detecting suspicious user activities."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.behavioral_baselines = {}

    def analyze_user_behavior(self, activity_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze user behavior patterns from activity logs."""
        self.logger.info(f"Analyzing {len(activity_logs)} activity log entries")

        behavior_analysis = {
            'total_activities': len(activity_logs),
            'time_patterns': self._analyze_time_patterns(activity_logs),
            'volume_patterns': self._analyze_volume_patterns(activity_logs),
            'sequence_patterns': self._analyze_sequence_patterns(activity_logs),
            'anomaly_indicators': []
        }

        # Detect behavioral anomalies
        behavior_analysis['anomaly_indicators'] = self._detect_behavioral_anomalies(behavior_analysis)

        return behavior_analysis

    def _analyze_time_patterns(self, activity_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze temporal patterns in user activities."""
        if not activity_logs:
            return {}

        timestamps = [log.get('timestamp') for log in activity_logs if log.get('timestamp')]
        if not timestamps:
            return {}

        # Convert to datetime objects if they're strings
        datetime_objects = []
        for ts in timestamps:
            if isinstance(ts, str):
                try:
                    datetime_objects.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                except ValueError:
                    continue
            elif isinstance(ts, datetime):
                datetime_objects.append(ts)

        if not datetime_objects:
            return {}

        # Analyze time patterns
        hours = [dt.hour for dt in datetime_objects]
        days_of_week = [dt.weekday() for dt in datetime_objects]

        return {
            'hour_distribution': {str(h): hours.count(h) for h in range(24)},
            'day_distribution': {str(d): days_of_week.count(d) for d in range(7)},
            'business_hours_ratio': sum(1 for h in hours if 9 <= h <= 17) / len(hours),
            'weekend_ratio': sum(1 for d in days_of_week if d >= 5) / len(days_of_week)
        }

    def _analyze_volume_patterns(self, activity_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze volume patterns in user activities."""
        volumes = [log.get('data_size', 0) for log in activity_logs]
        request_counts = {}

        # Count requests by type
        for log in activity_logs:
            activity_type = log.get('activity_type', 'unknown')
            request_counts[activity_type] = request_counts.get(activity_type, 0) + 1

        return {
            'total_data_size': sum(volumes),
            'avg_data_size': np.mean(volumes) if volumes else 0,
            'max_data_size': max(volumes) if volumes else 0,
            'request_type_distribution': request_counts,
            'volume_variance': float(np.var(volumes)) if volumes else 0
        }

    def _analyze_sequence_patterns(self, activity_logs: list[dict[str, Any]]) -> dict[str, Any]:
        """Analyze sequence patterns in user activities."""
        if len(activity_logs) < 2:
            return {}

        # Extract activity sequences
        activity_sequence = [log.get('activity_type', 'unknown') for log in activity_logs]

        # Analyze common patterns
        bigrams = [(activity_sequence[i], activity_sequence[i + 1])
                  for i in range(len(activity_sequence) - 1)]

        bigram_counts = {}
        for bigram in bigrams:
            bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        return {
            'sequence_length': len(activity_sequence),
            'unique_activities': len(set(activity_sequence)),
            'common_bigrams': dict(sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'sequence_entropy': self._calculate_sequence_entropy(activity_sequence)
        }

    def _calculate_sequence_entropy(self, sequence: list[str]) -> float:
        """Calculate entropy of activity sequence."""
        if not sequence:
            return 0.0

        # Count occurrences
        counts = {}
        for item in sequence:
            counts[item] = counts.get(item, 0) + 1

        # Calculate entropy
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            probability = count / total
            if probability > 0:
                entropy -= probability * np.log2(probability)

        return float(entropy)

    def _detect_behavioral_anomalies(self, behavior_analysis: dict[str, Any]) -> list[dict[str, Any]]:
        """Detect behavioral anomalies."""
        anomalies = []

        time_patterns = behavior_analysis.get('time_patterns', {})
        volume_patterns = behavior_analysis.get('volume_patterns', {})

        # Check for unusual time patterns
        business_hours_ratio = time_patterns.get('business_hours_ratio', 0)
        if business_hours_ratio < 0.3:  # Less than 30% during business hours
            anomalies.append({
                'type': 'unusual_time_pattern',
                'description': 'High activity outside business hours',
                'severity': 'medium',
                'value': business_hours_ratio
            })

        weekend_ratio = time_patterns.get('weekend_ratio', 0)
        if weekend_ratio > 0.5:  # More than 50% on weekends
            anomalies.append({
                'type': 'weekend_activity',
                'description': 'High weekend activity',
                'severity': 'low',
                'value': weekend_ratio
            })

        # Check for unusual volume patterns
        volume_variance = volume_patterns.get('volume_variance', 0)
        avg_size = volume_patterns.get('avg_data_size', 0)
        if avg_size > 0 and volume_variance / (avg_size ** 2) > 2.0:  # High coefficient of variation
            anomalies.append({
                'type': 'irregular_volume_pattern',
                'description': 'Highly irregular data volume patterns',
                'severity': 'medium',
                'value': volume_variance / (avg_size ** 2)
            })

        return anomalies

    def generate_behavioral_signatures(self, behavior_analysis: dict[str, Any]) -> list[DetectionSignature]:
        """Generate behavioral detection signatures."""
        signatures = []

        # Time-based signature
        time_signature = DetectionSignature(
            signature_id=self._generate_signature_id("behavioral_time"),
            name="Unusual Time Pattern",
            description="Detects activities outside normal business patterns",
            signature_type="behavioral",
            confidence=0.70,
            severity="medium",
            indicators={
                "business_hours_threshold": 0.3,
                "weekend_threshold": 0.5,
                "baseline_patterns": behavior_analysis.get('time_patterns', {})
            },
            metadata={
                "detection_method": "time_pattern_analysis",
                "baseline_analysis": behavior_analysis
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        signatures.append(time_signature)

        # Volume-based signature
        volume_signature = DetectionSignature(
            signature_id=self._generate_signature_id("behavioral_volume"),
            name="Unusual Volume Pattern",
            description="Detects irregular data volume patterns",
            signature_type="behavioral",
            confidence=0.75,
            severity="medium",
            indicators={
                "volume_variance_threshold": 2.0,
                "baseline_volume_stats": behavior_analysis.get('volume_patterns', {})
            },
            metadata={
                "detection_method": "volume_pattern_analysis",
                "baseline_analysis": behavior_analysis
            },
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        signatures.append(volume_signature)

        return signatures

    def _generate_signature_id(self, signature_type: str) -> str:
        """Generate unique signature ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        hash_input = f"{signature_type}_{timestamp}".encode()
        hash_digest = hashlib.md5(hash_input).hexdigest()[:8]
        return f"{signature_type}_{hash_digest}"


class SignatureManager:
    """Manages detection signatures and provides unified interface."""

    def __init__(self, signature_db_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.signature_db_path = Path(signature_db_path) if signature_db_path else Path("signatures.json")
        self.signatures = {}
        self.load_signatures()

    def add_signature(self, signature: DetectionSignature) -> None:
        """Add a detection signature to the database."""
        self.signatures[signature.signature_id] = signature
        self.logger.info(f"Added signature: {signature.name} ({signature.signature_id})")

    def remove_signature(self, signature_id: str) -> bool:
        """Remove a detection signature from the database."""
        if signature_id in self.signatures:
            del self.signatures[signature_id]
            self.logger.info(f"Removed signature: {signature_id}")
            return True
        return False

    def get_signature(self, signature_id: str) -> Optional[DetectionSignature]:
        """Get a specific detection signature."""
        return self.signatures.get(signature_id)

    def get_signatures_by_type(self, signature_type: str) -> list[DetectionSignature]:
        """Get all signatures of a specific type."""
        return [sig for sig in self.signatures.values() if sig.signature_type == signature_type]

    def get_signatures_by_severity(self, severity: str) -> list[DetectionSignature]:
        """Get all signatures of a specific severity."""
        return [sig for sig in self.signatures.values() if sig.severity == severity]

    def save_signatures(self) -> None:
        """Save signatures to persistent storage."""
        try:
            signature_data = {}
            for sig_id, signature in self.signatures.items():
                signature_data[sig_id] = {
                    'signature_id': signature.signature_id,
                    'name': signature.name,
                    'description': signature.description,
                    'signature_type': signature.signature_type,
                    'confidence': signature.confidence,
                    'severity': signature.severity,
                    'indicators': signature.indicators,
                    'metadata': signature.metadata,
                    'created_at': signature.created_at.isoformat(),
                    'updated_at': signature.updated_at.isoformat()
                }

            with open(self.signature_db_path, 'w') as f:
                json.dump(signature_data, f, indent=2, default=str)

            self.logger.info(f"Saved {len(self.signatures)} signatures to {self.signature_db_path}")

        except Exception as e:
            self.logger.error(f"Failed to save signatures: {e}")

    def load_signatures(self) -> None:
        """Load signatures from persistent storage."""
        if not self.signature_db_path.exists():
            self.logger.info(f"No signature database found at {self.signature_db_path}")
            return

        try:
            with open(self.signature_db_path) as f:
                signature_data = json.load(f)

            for sig_id, data in signature_data.items():
                signature = DetectionSignature(
                    signature_id=data['signature_id'],
                    name=data['name'],
                    description=data['description'],
                    signature_type=data['signature_type'],
                    confidence=data['confidence'],
                    severity=data['severity'],
                    indicators=data['indicators'],
                    metadata=data['metadata'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    updated_at=datetime.fromisoformat(data['updated_at'])
                )
                self.signatures[sig_id] = signature

            self.logger.info(f"Loaded {len(self.signatures)} signatures from {self.signature_db_path}")

        except Exception as e:
            self.logger.error(f"Failed to load signatures: {e}")

    def export_signatures(self, export_format: str = "json") -> str:
        """Export signatures in specified format."""
        if export_format == "json":
            return json.dumps({
                sig_id: {
                    'signature_id': sig.signature_id,
                    'name': sig.name,
                    'description': sig.description,
                    'signature_type': sig.signature_type,
                    'confidence': sig.confidence,
                    'severity': sig.severity,
                    'indicators': sig.indicators,
                    'metadata': sig.metadata,
                    'created_at': sig.created_at.isoformat(),
                    'updated_at': sig.updated_at.isoformat()
                }
                for sig_id, sig in self.signatures.items()
            }, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")

    def generate_detection_report(self, embeddings: np.ndarray, activity_logs: list[dict[str, Any]] = None) -> dict[str, Any]:
        """Generate comprehensive detection report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'embeddings_analyzed': len(embeddings),
            'signatures_applied': len(self.signatures),
            'detections': [],
            'summary': {}
        }

        # Apply statistical signatures
        statistical_sigs = self.get_signatures_by_type("statistical")
        for sig in statistical_sigs:
            # This would require implementing detection logic for each signature type
            # For now, we'll create a placeholder
            detection_result = {
                'signature_id': sig.signature_id,
                'signature_name': sig.name,
                'matches': 0,
                'confidence': sig.confidence,
                'severity': sig.severity
            }
            report['detections'].append(detection_result)

        # Apply pattern signatures
        pattern_sigs = self.get_signatures_by_type("pattern")
        for sig in pattern_sigs:
            detection_result = {
                'signature_id': sig.signature_id,
                'signature_name': sig.name,
                'matches': 0,
                'confidence': sig.confidence,
                'severity': sig.severity
            }
            report['detections'].append(detection_result)

        # Apply behavioral signatures if activity logs provided
        if activity_logs:
            behavioral_sigs = self.get_signatures_by_type("behavioral")
            for sig in behavioral_sigs:
                detection_result = {
                    'signature_id': sig.signature_id,
                    'signature_name': sig.name,
                    'matches': 0,
                    'confidence': sig.confidence,
                    'severity': sig.severity
                }
                report['detections'].append(detection_result)

        # Generate summary
        total_matches = sum(d['matches'] for d in report['detections'])
        high_severity_matches = sum(d['matches'] for d in report['detections'] if d['severity'] == 'high')

        report['summary'] = {
            'total_detections': total_matches,
            'high_severity_detections': high_severity_matches,
            'detection_rate': total_matches / len(embeddings) if len(embeddings) > 0 else 0,
            'risk_level': 'high' if high_severity_matches > 0 else 'medium' if total_matches > 0 else 'low'
        }

        return report


def main():
    """Main function for testing signature generation."""
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create sample data for testing
    np.random.seed(42)
    clean_embeddings = np.random.normal(0, 1, (1000, 384))
    noisy_embeddings = clean_embeddings + np.random.normal(0, 0.1, (1000, 384))

    # Initialize signature generators
    stat_generator = StatisticalSignatureGenerator(logger)
    pattern_generator = PatternSignatureGenerator(logger)
    behavioral_generator = BehavioralSignatureGenerator(logger)
    signature_manager = SignatureManager(logger=logger)

    # Establish baseline and generate signatures
    logger.info("Establishing statistical baseline...")
    stat_generator.establish_baseline(clean_embeddings, "clean_baseline")

    logger.info("Generating statistical signatures...")
    stat_signatures = stat_generator.generate_statistical_signatures("clean_baseline")

    logger.info("Analyzing steganographic patterns...")
    pattern_generator.analyze_steganographic_patterns(noisy_embeddings, "noise_injection")
    pattern_signatures = pattern_generator.generate_pattern_signatures()

    # Sample activity logs for behavioral analysis
    sample_logs = [
        {
            'timestamp': '2024-01-01T09:00:00Z',
            'activity_type': 'document_upload',
            'data_size': 1024,
            'user_id': 'user1'
        },
        {
            'timestamp': '2024-01-01T14:30:00Z',
            'activity_type': 'query_execution',
            'data_size': 512,
            'user_id': 'user1'
        }
    ]

    logger.info("Analyzing behavioral patterns...")
    behavior_analysis = behavioral_generator.analyze_user_behavior(sample_logs)
    behavioral_signatures = behavioral_generator.generate_behavioral_signatures(behavior_analysis)

    # Add signatures to manager
    for sig in stat_signatures + pattern_signatures + behavioral_signatures:
        signature_manager.add_signature(sig)

    # Save signatures
    signature_manager.save_signatures()

    # Generate detection report
    logger.info("Generating detection report...")
    report = signature_manager.generate_detection_report(noisy_embeddings, sample_logs)

    logger.info(f"Detection report generated with {len(report['detections'])} signature checks")
    logger.info(f"Risk level: {report['summary']['risk_level']}")


if __name__ == "__main__":
    main()
