"""
Forensic Analysis Tools for VectorSmuggle

This module provides forensic analysis capabilities for investigating
vector-based data exfiltration incidents and reconstructing attack timelines.
"""

import hashlib
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class ForensicEvidence:
    """Represents a piece of forensic evidence."""
    evidence_id: str
    timestamp: datetime
    evidence_type: str
    source: str
    content: Any
    hash_value: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttackTimeline:
    """Represents a reconstructed attack timeline."""
    timeline_id: str
    start_time: datetime
    end_time: datetime
    attack_phases: List[Dict[str, Any]]
    evidence_items: List[ForensicEvidence]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceCollector:
    """Collects and preserves digital evidence."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.evidence_store = []
        self.chain_of_custody = []
    
    def collect_vector_store_evidence(self, vector_store_data: Dict[str, Any]) -> List[ForensicEvidence]:
        """Collect evidence from vector store."""
        evidence_items = []
        
        # Collect embeddings
        if "embeddings" in vector_store_data:
            embeddings = vector_store_data["embeddings"]
            evidence = self._create_evidence(
                "vector_embeddings",
                "vector_store",
                embeddings,
                {"embedding_count": len(embeddings), "dimension": embeddings.shape[1] if hasattr(embeddings, 'shape') else 0}
            )
            evidence_items.append(evidence)
        
        # Collect metadata
        if "metadata" in vector_store_data:
            metadata = vector_store_data["metadata"]
            evidence = self._create_evidence(
                "vector_metadata",
                "vector_store",
                metadata,
                {"metadata_keys": list(metadata.keys()) if isinstance(metadata, dict) else []}
            )
            evidence_items.append(evidence)
        
        # Collect index information
        if "index_info" in vector_store_data:
            index_info = vector_store_data["index_info"]
            evidence = self._create_evidence(
                "index_information",
                "vector_store",
                index_info,
                {"index_type": index_info.get("type", "unknown")}
            )
            evidence_items.append(evidence)
        
        self.logger.info(f"Collected {len(evidence_items)} evidence items from vector store")
        return evidence_items
    
    def collect_network_evidence(self, network_logs: List[Dict[str, Any]]) -> List[ForensicEvidence]:
        """Collect evidence from network logs."""
        evidence_items = []
        
        # Group logs by connection
        connections = defaultdict(list)
        for log_entry in network_logs:
            connection_id = f"{log_entry.get('src_ip', 'unknown')}:{log_entry.get('dst_ip', 'unknown')}"
            connections[connection_id].append(log_entry)
        
        # Create evidence for each connection
        for connection_id, logs in connections.items():
            evidence = self._create_evidence(
                "network_connection",
                "network_logs",
                logs,
                {
                    "connection_id": connection_id,
                    "packet_count": len(logs),
                    "total_bytes": sum(log.get("bytes", 0) for log in logs),
                    "duration": self._calculate_connection_duration(logs)
                }
            )
            evidence_items.append(evidence)
        
        self.logger.info(f"Collected {len(evidence_items)} network evidence items")
        return evidence_items
    
    def collect_system_evidence(self, system_logs: List[Dict[str, Any]]) -> List[ForensicEvidence]:
        """Collect evidence from system logs."""
        evidence_items = []
        
        # Group by log type
        log_types = defaultdict(list)
        for log_entry in system_logs:
            log_type = log_entry.get("log_type", "unknown")
            log_types[log_type].append(log_entry)
        
        # Create evidence for each log type
        for log_type, logs in log_types.items():
            evidence = self._create_evidence(
                f"system_logs_{log_type}",
                "system_logs",
                logs,
                {
                    "log_type": log_type,
                    "entry_count": len(logs),
                    "time_range": self._calculate_time_range(logs)
                }
            )
            evidence_items.append(evidence)
        
        self.logger.info(f"Collected {len(evidence_items)} system evidence items")
        return evidence_items
    
    def collect_application_evidence(self, app_logs: List[Dict[str, Any]]) -> List[ForensicEvidence]:
        """Collect evidence from application logs."""
        evidence_items = []
        
        # Filter for suspicious activities
        suspicious_patterns = [
            r"embedding.*obfuscat",
            r"steganograph",
            r"fragment.*data",
            r"decoy.*inject",
            r"evasion.*technique"
        ]
        
        suspicious_logs = []
        for log_entry in app_logs:
            log_message = str(log_entry.get("message", ""))
            for pattern in suspicious_patterns:
                if re.search(pattern, log_message, re.IGNORECASE):
                    suspicious_logs.append(log_entry)
                    break
        
        if suspicious_logs:
            evidence = self._create_evidence(
                "suspicious_application_activity",
                "application_logs",
                suspicious_logs,
                {
                    "suspicious_count": len(suspicious_logs),
                    "total_logs": len(app_logs),
                    "suspicion_ratio": len(suspicious_logs) / len(app_logs)
                }
            )
            evidence_items.append(evidence)
        
        # Collect all application logs as evidence
        evidence = self._create_evidence(
            "application_logs",
            "application_logs",
            app_logs,
            {
                "total_entries": len(app_logs),
                "suspicious_entries": len(suspicious_logs)
            }
        )
        evidence_items.append(evidence)
        
        self.logger.info(f"Collected {len(evidence_items)} application evidence items")
        return evidence_items
    
    def _create_evidence(
        self,
        evidence_type: str,
        source: str,
        content: Any,
        metadata: Dict[str, Any] = None
    ) -> ForensicEvidence:
        """Create a forensic evidence item."""
        timestamp = datetime.utcnow()
        evidence_id = f"{evidence_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hash(str(content))}"
        
        # Calculate hash
        content_str = json.dumps(content, sort_keys=True, default=str)
        hash_value = hashlib.sha256(content_str.encode()).hexdigest()
        
        evidence = ForensicEvidence(
            evidence_id=evidence_id,
            timestamp=timestamp,
            evidence_type=evidence_type,
            source=source,
            content=content,
            hash_value=hash_value,
            metadata=metadata or {}
        )
        
        # Add to evidence store
        self.evidence_store.append(evidence)
        
        # Record chain of custody
        self.chain_of_custody.append({
            "evidence_id": evidence_id,
            "action": "collected",
            "timestamp": timestamp.isoformat(),
            "collector": "EvidenceCollector",
            "hash": hash_value
        })
        
        return evidence
    
    def _calculate_connection_duration(self, logs: List[Dict[str, Any]]) -> float:
        """Calculate duration of network connection."""
        if not logs:
            return 0.0
        
        timestamps = []
        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        
        if len(timestamps) < 2:
            return 0.0
        
        return (max(timestamps) - min(timestamps)).total_seconds()
    
    def _calculate_time_range(self, logs: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate time range for log entries."""
        timestamps = []
        for log in logs:
            timestamp_str = log.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        
        if not timestamps:
            return {"start": "unknown", "end": "unknown"}
        
        return {
            "start": min(timestamps).isoformat(),
            "end": max(timestamps).isoformat()
        }


class TimelineReconstructor:
    """Reconstructs attack timelines from evidence."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.attack_phases = [
            "reconnaissance",
            "initial_access",
            "execution",
            "persistence",
            "privilege_escalation",
            "defense_evasion",
            "credential_access",
            "discovery",
            "lateral_movement",
            "collection",
            "command_and_control",
            "exfiltration",
            "impact"
        ]
    
    def reconstruct_timeline(self, evidence_items: List[ForensicEvidence]) -> AttackTimeline:
        """Reconstruct attack timeline from evidence."""
        self.logger.info(f"Reconstructing timeline from {len(evidence_items)} evidence items")
        
        # Sort evidence by timestamp
        sorted_evidence = sorted(evidence_items, key=lambda x: x.timestamp)
        
        if not sorted_evidence:
            raise ValueError("No evidence items provided")
        
        # Identify attack phases
        phases = self._identify_attack_phases(sorted_evidence)
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score(sorted_evidence, phases)
        
        # Create timeline
        timeline = AttackTimeline(
            timeline_id=f"timeline_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            start_time=sorted_evidence[0].timestamp,
            end_time=sorted_evidence[-1].timestamp,
            attack_phases=phases,
            evidence_items=sorted_evidence,
            confidence_score=confidence,
            metadata={
                "total_evidence": len(evidence_items),
                "duration_hours": (sorted_evidence[-1].timestamp - sorted_evidence[0].timestamp).total_seconds() / 3600,
                "phases_identified": len(phases)
            }
        )
        
        self.logger.info(f"Timeline reconstructed with {len(phases)} phases and {confidence:.2f} confidence")
        return timeline
    
    def _identify_attack_phases(self, evidence_items: List[ForensicEvidence]) -> List[Dict[str, Any]]:
        """Identify attack phases from evidence."""
        phases = []
        
        # Define phase indicators
        phase_indicators = {
            "reconnaissance": [
                "network_scan", "port_scan", "service_discovery",
                "dns_lookup", "whois_query"
            ],
            "initial_access": [
                "login_attempt", "authentication", "credential_validation",
                "session_establishment"
            ],
            "execution": [
                "process_creation", "command_execution", "script_execution",
                "embedding_generation", "document_processing"
            ],
            "persistence": [
                "file_creation", "registry_modification", "scheduled_task",
                "vector_store_creation", "index_creation"
            ],
            "defense_evasion": [
                "log_deletion", "process_hiding", "obfuscation",
                "steganography", "embedding_obfuscation", "detection_avoidance"
            ],
            "collection": [
                "file_access", "data_collection", "document_parsing",
                "content_extraction", "sensitive_data_identification"
            ],
            "exfiltration": [
                "data_transfer", "network_upload", "vector_upload",
                "external_communication", "data_fragmentation"
            ]
        }
        
        # Group evidence by time windows
        time_windows = self._create_time_windows(evidence_items, window_minutes=30)
        
        for window_start, window_evidence in time_windows.items():
            # Analyze evidence in this window
            window_phases = set()
            
            for evidence in window_evidence:
                evidence_type = evidence.evidence_type.lower()
                content_str = str(evidence.content).lower()
                
                # Check for phase indicators
                for phase, indicators in phase_indicators.items():
                    for indicator in indicators:
                        if indicator in evidence_type or indicator in content_str:
                            window_phases.add(phase)
                            break
            
            # Create phase entries
            for phase in window_phases:
                phase_entry = {
                    "phase": phase,
                    "start_time": window_start.isoformat(),
                    "end_time": (window_start + timedelta(minutes=30)).isoformat(),
                    "evidence_count": len(window_evidence),
                    "evidence_ids": [e.evidence_id for e in window_evidence],
                    "confidence": self._calculate_phase_confidence(phase, window_evidence)
                }
                phases.append(phase_entry)
        
        # Sort phases by start time
        phases.sort(key=lambda x: x["start_time"])
        return phases
    
    def _create_time_windows(
        self,
        evidence_items: List[ForensicEvidence],
        window_minutes: int = 30
    ) -> Dict[datetime, List[ForensicEvidence]]:
        """Create time windows for evidence analysis."""
        windows = defaultdict(list)
        
        for evidence in evidence_items:
            # Round timestamp to window boundary
            window_start = evidence.timestamp.replace(
                minute=(evidence.timestamp.minute // window_minutes) * window_minutes,
                second=0,
                microsecond=0
            )
            windows[window_start].append(evidence)
        
        return dict(windows)
    
    def _calculate_phase_confidence(
        self,
        phase: str,
        evidence_items: List[ForensicEvidence]
    ) -> float:
        """Calculate confidence score for a phase identification."""
        # Base confidence on evidence quality and quantity
        base_confidence = 0.5
        
        # Increase confidence based on evidence count
        evidence_bonus = min(0.3, len(evidence_items) * 0.05)
        
        # Increase confidence based on evidence types
        evidence_types = set(e.evidence_type for e in evidence_items)
        type_bonus = min(0.2, len(evidence_types) * 0.05)
        
        return min(1.0, base_confidence + evidence_bonus + type_bonus)
    
    def _calculate_confidence_score(
        self,
        evidence_items: List[ForensicEvidence],
        phases: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence score for timeline."""
        if not evidence_items or not phases:
            return 0.0
        
        # Base confidence on evidence quality
        evidence_score = min(1.0, len(evidence_items) / 50)  # Normalize to 50 evidence items
        
        # Phase coverage score
        unique_phases = set(p["phase"] for p in phases)
        phase_score = len(unique_phases) / len(self.attack_phases)
        
        # Time consistency score
        time_score = self._calculate_time_consistency_score(evidence_items)
        
        # Weighted average
        return (evidence_score * 0.4 + phase_score * 0.3 + time_score * 0.3)
    
    def _calculate_time_consistency_score(self, evidence_items: List[ForensicEvidence]) -> float:
        """Calculate time consistency score."""
        if len(evidence_items) < 2:
            return 1.0
        
        # Check for reasonable time gaps
        time_gaps = []
        for i in range(1, len(evidence_items)):
            gap = (evidence_items[i].timestamp - evidence_items[i-1].timestamp).total_seconds()
            time_gaps.append(gap)
        
        # Penalize very large gaps (indicating missing evidence)
        large_gaps = sum(1 for gap in time_gaps if gap > 3600)  # 1 hour gaps
        gap_penalty = large_gaps / len(time_gaps)
        
        return max(0.0, 1.0 - gap_penalty)


class ArtifactAnalyzer:
    """Analyzes digital artifacts for forensic investigation."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_embedding_artifacts(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Analyze embedding artifacts for signs of manipulation."""
        analysis = {
            "total_embeddings": len(embeddings),
            "dimension": embeddings.shape[1] if len(embeddings) > 0 else 0,
            "statistical_analysis": {},
            "anomaly_indicators": [],
            "steganographic_indicators": []
        }
        
        if len(embeddings) == 0:
            return analysis
        
        # Statistical analysis
        analysis["statistical_analysis"] = {
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "mean_values": embeddings.mean(axis=0).tolist()[:10],  # First 10 dimensions
            "variance": float(np.var(embeddings)),
            "skewness": float(self._calculate_skewness(embeddings)),
            "kurtosis": float(self._calculate_kurtosis(embeddings))
        }
        
        # Detect anomalies
        anomalies = self._detect_embedding_anomalies(embeddings)
        analysis["anomaly_indicators"] = anomalies
        
        # Detect steganographic indicators
        stego_indicators = self._detect_steganographic_indicators(embeddings)
        analysis["steganographic_indicators"] = stego_indicators
        
        return analysis
    
    def analyze_network_artifacts(self, network_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network artifacts for suspicious patterns."""
        analysis = {
            "total_connections": len(network_data),
            "traffic_patterns": {},
            "suspicious_indicators": [],
            "timing_analysis": {}
        }
        
        if not network_data:
            return analysis
        
        # Traffic pattern analysis
        destinations = defaultdict(int)
        protocols = defaultdict(int)
        sizes = []
        
        for connection in network_data:
            dest = connection.get("destination", "unknown")
            destinations[dest] += 1
            
            protocol = connection.get("protocol", "unknown")
            protocols[protocol] += 1
            
            size = connection.get("bytes", 0)
            sizes.append(size)
        
        analysis["traffic_patterns"] = {
            "top_destinations": dict(sorted(destinations.items(), key=lambda x: x[1], reverse=True)[:10]),
            "protocol_distribution": dict(protocols),
            "average_size": np.mean(sizes) if sizes else 0,
            "total_bytes": sum(sizes)
        }
        
        # Detect suspicious patterns
        suspicious = self._detect_suspicious_network_patterns(network_data)
        analysis["suspicious_indicators"] = suspicious
        
        # Timing analysis
        timing = self._analyze_network_timing(network_data)
        analysis["timing_analysis"] = timing
        
        return analysis
    
    def analyze_file_artifacts(self, file_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze file artifacts for forensic indicators."""
        analysis = {
            "total_files": len(file_metadata),
            "file_types": {},
            "temporal_analysis": {},
            "suspicious_files": []
        }
        
        if not file_metadata:
            return analysis
        
        # File type analysis
        file_types = defaultdict(int)
        creation_times = []
        modification_times = []
        
        for file_info in file_metadata:
            file_type = file_info.get("type", "unknown")
            file_types[file_type] += 1
            
            created = file_info.get("created")
            if created:
                creation_times.append(created)
            
            modified = file_info.get("modified")
            if modified:
                modification_times.append(modified)
        
        analysis["file_types"] = dict(file_types)
        
        # Temporal analysis
        if creation_times:
            analysis["temporal_analysis"]["creation_timespan"] = {
                "earliest": min(creation_times),
                "latest": max(creation_times)
            }
        
        # Detect suspicious files
        suspicious = self._detect_suspicious_files(file_metadata)
        analysis["suspicious_files"] = suspicious
        
        return analysis
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _detect_embedding_anomalies(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Detect anomalies in embeddings."""
        anomalies = []
        
        # Check for unusual variance
        variances = np.var(embeddings, axis=1)
        mean_variance = np.mean(variances)
        std_variance = np.std(variances)
        
        high_variance_threshold = mean_variance + 3 * std_variance
        low_variance_threshold = mean_variance - 3 * std_variance
        
        high_variance_count = np.sum(variances > high_variance_threshold)
        low_variance_count = np.sum(variances < low_variance_threshold)
        
        if high_variance_count > 0:
            anomalies.append({
                "type": "high_variance",
                "count": int(high_variance_count),
                "threshold": float(high_variance_threshold),
                "severity": "medium"
            })
        
        if low_variance_count > 0:
            anomalies.append({
                "type": "low_variance",
                "count": int(low_variance_count),
                "threshold": float(low_variance_threshold),
                "severity": "medium"
            })
        
        return anomalies
    
    def _detect_steganographic_indicators(self, embeddings: np.ndarray) -> List[Dict[str, Any]]:
        """Detect potential steganographic indicators."""
        indicators = []
        
        # Check for regular patterns that might indicate hidden data
        norms = np.linalg.norm(embeddings, axis=1)
        
        # Check for clustering of norms (might indicate systematic modification)
        unique_norms = len(np.unique(np.round(norms, 3)))
        if unique_norms < len(norms) * 0.8:  # Less than 80% unique values
            indicators.append({
                "type": "norm_clustering",
                "unique_ratio": unique_norms / len(norms),
                "severity": "low"
            })
        
        # Check for unusual distribution patterns
        if len(embeddings) > 10:
            # Simple entropy check
            flattened = embeddings.flatten()
            hist, _ = np.histogram(flattened, bins=50)
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            
            # Low entropy might indicate structured hidden data
            if entropy < 100:  # Arbitrary threshold
                indicators.append({
                    "type": "low_entropy",
                    "entropy": float(entropy),
                    "severity": "medium"
                })
        
        return indicators
    
    def _detect_suspicious_network_patterns(self, network_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious network patterns."""
        suspicious = []
        
        # Check for unusual destinations
        destinations = defaultdict(int)
        for connection in network_data:
            dest = connection.get("destination", "")
            destinations[dest] += 1
        
        # Flag destinations with very few connections (might be exfiltration targets)
        single_connection_dests = [dest for dest, count in destinations.items() if count == 1]
        if len(single_connection_dests) > len(destinations) * 0.5:
            suspicious.append({
                "type": "many_unique_destinations",
                "count": len(single_connection_dests),
                "severity": "medium"
            })
        
        # Check for unusual timing patterns
        timestamps = []
        for connection in network_data:
            timestamp_str = connection.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        
        if len(timestamps) > 1:
            # Check for very regular intervals (might indicate automated exfiltration)
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
            if len(set(np.round(intervals, 0))) < len(intervals) * 0.3:  # Less than 30% unique intervals
                suspicious.append({
                    "type": "regular_intervals",
                    "unique_interval_ratio": len(set(np.round(intervals, 0))) / len(intervals),
                    "severity": "high"
                })
        
        return suspicious
    
    def _analyze_network_timing(self, network_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze network timing patterns."""
        timestamps = []
        for connection in network_data:
            timestamp_str = connection.get("timestamp")
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except ValueError:
                    continue
        
        if len(timestamps) < 2:
            return {"insufficient_data": True}
        
        timestamps.sort()
        intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        
        return {
            "total_duration": (timestamps[-1] - timestamps[0]).total_seconds(),
            "average_interval": np.mean(intervals),
            "interval_variance": np.var(intervals),
            "min_interval": min(intervals),
            "max_interval": max(intervals)
        }
    
    def _detect_suspicious_files(self, file_metadata: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect suspicious files."""
        suspicious = []
        
        for file_info in file_metadata:
            file_path = file_info.get("path", "")
            file_name = file_info.get("name", "")
            file_size = file_info.get("size", 0)
            
            # Check for suspicious file names
            suspicious_patterns = [
                r"temp.*\d+",
                r"tmp.*\d+",
                r"cache.*\d+",
                r"\.tmp$",
                r"\.cache$",
                r"embedding.*\d+",
                r"vector.*\d+"
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, file_name, re.IGNORECASE):
                    suspicious.append({
                        "file": file_path,
                        "reason": f"suspicious_name_pattern: {pattern}",
                        "severity": "low"
                    })
                    break
            
            # Check for unusual file sizes
            if file_size == 0:
                suspicious.append({
                    "file": file_path,
                    "reason": "zero_byte_file",
                    "severity": "low"
                })
            elif file_size > 100 * 1024 * 1024:  # > 100MB
                suspicious.append({
                    "file": file_path,
                    "reason": "unusually_large_file",
                    "size": file_size,
                    "severity": "medium"
                })
        
        return suspicious


def main():
    """Main function for testing forensic tools."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create sample evidence
    collector = EvidenceCollector(logger)
    
    # Sample vector store data
    vector_data = {
        "embeddings": np.random.normal(0, 1, (100, 384)),
        "metadata": {"model": "text-embedding-ada-002", "created": "2024-01-01"},
        "index_info": {"type": "faiss", "dimension": 384}
    }
    
    # Sample network logs
    network_logs = [
        {
            "timestamp": "2024-01-01T10:00:00Z",
            "src_ip": "192.168.1.100",
            "dst_ip": "203.0.113.1",
            "bytes": 1024,
            "protocol": "HTTPS"
        },
        {
            "timestamp": "2024-01-01T10:05:00Z",
            "src_ip": "192.168.1.100",
            "dst_ip": "203.0.113.1",
            "bytes": 2048,
            "protocol": "HTTPS"
        }
    ]
    
    # Collect evidence
    logger.info("Collecting evidence...")
    vector_evidence = collector.collect_vector_store_evidence(vector_data)
    network_evidence = collector.collect_network_evidence(network_logs)
    
    all_evidence = vector_evidence + network_evidence
    
    # Reconstruct timeline
    logger.info("Reconstructing timeline...")
    reconstructor = TimelineReconstructor(logger)
    timeline = reconstructor.reconstruct_timeline(all_evidence)
    
    # Analyze artifacts
    logger.info("Analyzing artifacts...")
    analyzer = ArtifactAnalyzer(logger)
    embedding_analysis = analyzer.analyze_embedding_artifacts(vector_data["embeddings"])
    network_analysis = analyzer.analyze_network_artifacts(network_logs)
    
    # Print results
    print("=== FORENSIC ANALYSIS RESULTS ===")
    print(f"Timeline ID: {timeline.timeline_id}")
    print(f"Duration: {timeline.metadata['duration_hours']:.2f} hours")
    print(f"Confidence: {timeline.confidence_score:.2f}")
    print(f"Phases identified: {len(timeline.attack_phases)}")
    
    print("\n=== EMBEDDING ANALYSIS ===")
    print(f"Total embeddings: {embedding_analysis['total_embeddings']}")
    print(f"Anomaly indicators: {len(embedding_analysis['anomaly_indicators'])}")
    print(f"Steganographic indicators: {len(embedding_analysis['steganographic_indicators'])}")
    
    print("\n=== NETWORK ANALYSIS ===")
    print(f"Total connections: {network_analysis['total_connections']}")
    print(f"Suspicious indicators: {len(network_analysis['suspicious_indicators'])}")
    print(f"Average packet size: {network_analysis['traffic_patterns']['average_size']:.0f} bytes")


if __name__ == "__main__":
    main()