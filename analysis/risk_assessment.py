# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""
Risk Assessment Framework for VectorSmuggle

This module provides comprehensive risk assessment capabilities for evaluating
the security implications of vector-based data exfiltration techniques.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatCategory(Enum):
    """Threat category enumeration."""
    DATA_EXFILTRATION = "data_exfiltration"
    STEGANOGRAPHY = "steganography"
    EVASION = "evasion"
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"


@dataclass
class RiskFactor:
    """Represents a single risk factor."""
    factor_id: str
    name: str
    description: str
    category: ThreatCategory
    weight: float
    current_value: float
    max_value: float
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """Comprehensive risk assessment result."""
    assessment_id: str
    timestamp: datetime
    overall_risk_level: RiskLevel
    overall_risk_score: float
    risk_factors: list[RiskFactor]
    threat_categories: dict[ThreatCategory, float]
    recommendations: list[str]
    mitigation_strategies: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class RiskCalculator:
    """Calculates risk scores based on various factors."""

    def __init__(self):
        self.risk_weights = {
            ThreatCategory.DATA_EXFILTRATION: 0.3,
            ThreatCategory.STEGANOGRAPHY: 0.25,
            ThreatCategory.EVASION: 0.2,
            ThreatCategory.BEHAVIORAL: 0.15,
            ThreatCategory.TECHNICAL: 0.1
        }

    def calculate_factor_risk(self, factor: RiskFactor) -> float:
        """Calculate risk score for a single factor."""
        normalized_value = factor.current_value / factor.max_value
        weighted_score = normalized_value * factor.weight
        return min(weighted_score, 1.0)

    def calculate_category_risk(self, factors: list[RiskFactor], category: ThreatCategory) -> float:
        """Calculate risk score for a threat category."""
        category_factors = [f for f in factors if f.category == category]
        if not category_factors:
            return 0.0

        total_risk = sum(self.calculate_factor_risk(f) for f in category_factors)
        return min(total_risk / len(category_factors), 1.0)

    def calculate_overall_risk(self, category_scores: dict[ThreatCategory, float]) -> float:
        """Calculate overall risk score."""
        weighted_sum = sum(
            score * self.risk_weights.get(category, 0.1)
            for category, score in category_scores.items()
        )
        return min(weighted_sum, 1.0)

    def determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level based on score."""
        if risk_score >= 0.8:
            return RiskLevel.CRITICAL
        elif risk_score >= 0.6:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW


class VectorExfiltrationRiskAssessor:
    """Assesses risks specific to vector-based data exfiltration."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.calculator = RiskCalculator()
        self.risk_factors_registry = self._initialize_risk_factors()

    def _initialize_risk_factors(self) -> dict[str, RiskFactor]:
        """Initialize standard risk factors."""
        factors = {}

        # Data Exfiltration Factors
        factors["data_sensitivity"] = RiskFactor(
            factor_id="data_sensitivity",
            name="Data Sensitivity Level",
            description="Sensitivity of data being processed",
            category=ThreatCategory.DATA_EXFILTRATION,
            weight=0.4,
            current_value=0.0,
            max_value=1.0
        )

        factors["data_volume"] = RiskFactor(
            factor_id="data_volume",
            name="Data Volume",
            description="Volume of data being exfiltrated",
            category=ThreatCategory.DATA_EXFILTRATION,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        factors["external_access"] = RiskFactor(
            factor_id="external_access",
            name="External Vector Store Access",
            description="Use of external vector databases",
            category=ThreatCategory.DATA_EXFILTRATION,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        # Steganography Factors
        factors["embedding_obfuscation"] = RiskFactor(
            factor_id="embedding_obfuscation",
            name="Embedding Obfuscation",
            description="Level of steganographic obfuscation applied",
            category=ThreatCategory.STEGANOGRAPHY,
            weight=0.4,
            current_value=0.0,
            max_value=1.0
        )

        factors["fragmentation_complexity"] = RiskFactor(
            factor_id="fragmentation_complexity",
            name="Fragmentation Complexity",
            description="Complexity of data fragmentation across models",
            category=ThreatCategory.STEGANOGRAPHY,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        factors["decoy_injection"] = RiskFactor(
            factor_id="decoy_injection",
            name="Decoy Data Injection",
            description="Use of decoy data to hide real content",
            category=ThreatCategory.STEGANOGRAPHY,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        # Evasion Factors
        factors["detection_avoidance"] = RiskFactor(
            factor_id="detection_avoidance",
            name="Detection Avoidance",
            description="Sophistication of detection avoidance techniques",
            category=ThreatCategory.EVASION,
            weight=0.4,
            current_value=0.0,
            max_value=1.0
        )

        factors["traffic_mimicry"] = RiskFactor(
            factor_id="traffic_mimicry",
            name="Traffic Mimicry",
            description="Ability to mimic legitimate traffic patterns",
            category=ThreatCategory.EVASION,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        factors["timing_evasion"] = RiskFactor(
            factor_id="timing_evasion",
            name="Timing-Based Evasion",
            description="Use of timing patterns to avoid detection",
            category=ThreatCategory.EVASION,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        # Behavioral Factors
        factors["user_simulation"] = RiskFactor(
            factor_id="user_simulation",
            name="User Behavior Simulation",
            description="Accuracy of legitimate user behavior simulation",
            category=ThreatCategory.BEHAVIORAL,
            weight=0.5,
            current_value=0.0,
            max_value=1.0
        )

        factors["activity_camouflage"] = RiskFactor(
            factor_id="activity_camouflage",
            name="Activity Camouflage",
            description="Effectiveness of activity camouflage techniques",
            category=ThreatCategory.BEHAVIORAL,
            weight=0.5,
            current_value=0.0,
            max_value=1.0
        )

        # Technical Factors
        factors["system_integration"] = RiskFactor(
            factor_id="system_integration",
            name="System Integration",
            description="Level of integration with target systems",
            category=ThreatCategory.TECHNICAL,
            weight=0.4,
            current_value=0.0,
            max_value=1.0
        )

        factors["operational_security"] = RiskFactor(
            factor_id="operational_security",
            name="Operational Security",
            description="Quality of operational security measures",
            category=ThreatCategory.TECHNICAL,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        factors["persistence_capability"] = RiskFactor(
            factor_id="persistence_capability",
            name="Persistence Capability",
            description="Ability to maintain persistent access",
            category=ThreatCategory.TECHNICAL,
            weight=0.3,
            current_value=0.0,
            max_value=1.0
        )

        return factors

    def assess_data_characteristics(self, documents: list[Any], embeddings: np.ndarray) -> dict[str, float]:
        """Assess risk based on data characteristics."""
        assessment = {}

        # Assess data sensitivity
        sensitive_indicators = 0
        total_content = ""

        for doc in documents:
            content = getattr(doc, 'page_content', str(doc))
            total_content += content

            # Check for sensitive patterns
            sensitive_patterns = [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  # Credit card
                r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # Financial amounts
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\b(?:confidential|secret|private|restricted)\b'  # Classification keywords
            ]

            import re
            for pattern in sensitive_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    sensitive_indicators += 1

        # Normalize sensitivity score
        max_possible_indicators = len(documents) * 5  # 5 patterns per document
        sensitivity_score = min(sensitive_indicators / max(max_possible_indicators, 1), 1.0)
        assessment["data_sensitivity"] = sensitivity_score

        # Assess data volume
        total_chars = len(total_content)
        volume_score = min(total_chars / 1000000, 1.0)  # Normalize to 1M chars
        assessment["data_volume"] = volume_score

        # Assess embedding characteristics
        if len(embeddings) > 0:
            embedding_variance = np.var(embeddings)
            variance_score = min(embedding_variance / 0.1, 1.0)  # Normalize to 0.1 variance
            assessment["embedding_complexity"] = variance_score
        else:
            assessment["embedding_complexity"] = 0.0

        return assessment

    def assess_steganographic_techniques(self, config: dict[str, Any]) -> dict[str, float]:
        """Assess risk based on steganographic techniques used."""
        assessment = {}

        stego_config = config.get("steganography", {})

        # Assess obfuscation level
        techniques = stego_config.get("techniques", [])
        obfuscation_score = len(techniques) / 6.0  # Max 6 techniques
        assessment["embedding_obfuscation"] = min(obfuscation_score, 1.0)

        # Assess fragmentation complexity
        fragment_size = stego_config.get("fragment_size", 128)
        fragmentation_score = min((256 - fragment_size) / 256, 1.0)  # Smaller fragments = higher risk
        assessment["fragmentation_complexity"] = fragmentation_score

        # Assess decoy injection
        decoy_ratio = stego_config.get("decoy_ratio", 0.0)
        assessment["decoy_injection"] = min(decoy_ratio, 1.0)

        return assessment

    def assess_evasion_capabilities(self, config: dict[str, Any]) -> dict[str, float]:
        """Assess risk based on evasion capabilities."""
        assessment = {}

        evasion_config = config.get("evasion", {})

        # Assess detection avoidance
        avoidance_features = [
            evasion_config.get("detection_avoidance_enabled", False),
            evasion_config.get("dlp_keyword_avoidance", False),
            evasion_config.get("signature_obfuscation", False)
        ]
        avoidance_score = sum(avoidance_features) / len(avoidance_features)
        assessment["detection_avoidance"] = avoidance_score

        # Assess traffic mimicry
        mimicry_features = [
            evasion_config.get("traffic_mimicry_enabled", False),
            evasion_config.get("user_agent_rotation", False),
            evasion_config.get("proxy_rotation_enabled", False)
        ]
        mimicry_score = sum(mimicry_features) / len(mimicry_features)
        assessment["traffic_mimicry"] = mimicry_score

        # Assess timing evasion
        timing_variance = evasion_config.get("query_variance", 0.0)
        timing_score = min(timing_variance, 1.0)
        assessment["timing_evasion"] = timing_score

        return assessment

    def assess_behavioral_simulation(self, config: dict[str, Any], activity_logs: list[dict[str, Any]] = None) -> dict[str, float]:
        """Assess risk based on behavioral simulation capabilities."""
        assessment = {}

        evasion_config = config.get("evasion", {})

        # Assess user simulation
        behavioral_features = [
            evasion_config.get("behavioral_camouflage_enabled", False),
            evasion_config.get("cover_story_enabled", False),
            len(evasion_config.get("user_profiles", [])) > 0
        ]
        simulation_score = sum(behavioral_features) / len(behavioral_features)
        assessment["user_simulation"] = simulation_score

        # Assess activity camouflage
        legitimate_ratio = evasion_config.get("legitimate_ratio", 0.0)
        assessment["activity_camouflage"] = min(legitimate_ratio, 1.0)

        return assessment

    def assess_technical_capabilities(self, config: dict[str, Any]) -> dict[str, float]:
        """Assess risk based on technical capabilities."""
        assessment = {}

        # Assess system integration
        vector_stores = config.get("vector_store", {})
        integration_features = [
            vector_stores.get("type") in ["qdrant", "pinecone"],  # External stores
            config.get("document", {}).get("batch_processing", False),
            len(config.get("document", {}).get("supported_formats", [])) > 5
        ]
        integration_score = sum(integration_features) / len(integration_features)
        assessment["system_integration"] = integration_score

        # Assess operational security
        opsec_config = config.get("evasion", {})
        opsec_features = [
            opsec_config.get("opsec_enabled", False),
            opsec_config.get("auto_cleanup", False),
            opsec_config.get("secure_delete_passes", 0) > 0
        ]
        opsec_score = sum(opsec_features) / len(opsec_features)
        assessment["operational_security"] = opsec_score

        # Assess persistence capability
        persistence_features = [
            config.get("steganography", {}).get("enabled", False),
            config.get("query", {}).get("cache_enabled", False),
            config.get("query", {}).get("data_recovery", False)
        ]
        persistence_score = sum(persistence_features) / len(persistence_features)
        assessment["persistence_capability"] = persistence_score

        return assessment

    def perform_comprehensive_assessment(
        self,
        documents: list[Any],
        embeddings: np.ndarray,
        config: dict[str, Any],
        activity_logs: list[dict[str, Any]] = None
    ) -> RiskAssessment:
        """Perform comprehensive risk assessment."""
        self.logger.info("Starting comprehensive risk assessment")

        # Collect all risk factor assessments
        data_assessment = self.assess_data_characteristics(documents, embeddings)
        stego_assessment = self.assess_steganographic_techniques(config)
        evasion_assessment = self.assess_evasion_capabilities(config)
        behavioral_assessment = self.assess_behavioral_simulation(config, activity_logs)
        technical_assessment = self.assess_technical_capabilities(config)

        # Update risk factors with current values
        risk_factors = []
        all_assessments = {
            **data_assessment,
            **stego_assessment,
            **evasion_assessment,
            **behavioral_assessment,
            **technical_assessment
        }

        for factor_id, current_value in all_assessments.items():
            if factor_id in self.risk_factors_registry:
                factor = self.risk_factors_registry[factor_id]
                factor.current_value = current_value
                risk_factors.append(factor)

        # Calculate category risks
        category_scores = {}
        for category in ThreatCategory:
            category_scores[category] = self.calculator.calculate_category_risk(risk_factors, category)

        # Calculate overall risk
        overall_score = self.calculator.calculate_overall_risk(category_scores)
        overall_level = self.calculator.determine_risk_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(risk_factors, category_scores)
        mitigation_strategies = self._generate_mitigation_strategies(risk_factors, overall_level)

        # Create assessment
        assessment = RiskAssessment(
            assessment_id=f"risk_assessment_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.utcnow(),
            overall_risk_level=overall_level,
            overall_risk_score=overall_score,
            risk_factors=risk_factors,
            threat_categories=category_scores,
            recommendations=recommendations,
            mitigation_strategies=mitigation_strategies,
            metadata={
                "documents_analyzed": len(documents),
                "embeddings_analyzed": len(embeddings),
                "config_analyzed": bool(config),
                "activity_logs_analyzed": len(activity_logs) if activity_logs else 0
            }
        )

        self.logger.info(f"Risk assessment completed: {overall_level.value} risk level")
        return assessment

    def _generate_recommendations(self, risk_factors: list[RiskFactor], category_scores: dict[ThreatCategory, float]) -> list[str]:
        """Generate security recommendations based on risk assessment."""
        recommendations = []

        # High-risk factors
        high_risk_factors = [f for f in risk_factors if self.calculator.calculate_factor_risk(f) > 0.7]

        for factor in high_risk_factors:
            if factor.factor_id == "data_sensitivity":
                recommendations.append("Implement data classification and handling policies")
                recommendations.append("Deploy data loss prevention (DLP) solutions")
            elif factor.factor_id == "external_access":
                recommendations.append("Restrict access to external vector databases")
                recommendations.append("Implement network egress monitoring")
            elif factor.factor_id == "embedding_obfuscation":
                recommendations.append("Deploy embedding anomaly detection systems")
                recommendations.append("Implement statistical analysis of vector spaces")
            elif factor.factor_id == "detection_avoidance":
                recommendations.append("Enhance behavioral analysis capabilities")
                recommendations.append("Implement multi-layered detection systems")

        # Category-specific recommendations
        if category_scores.get(ThreatCategory.DATA_EXFILTRATION, 0) > 0.6:
            recommendations.append("Implement comprehensive data governance framework")
            recommendations.append("Deploy advanced threat detection for data exfiltration")

        if category_scores.get(ThreatCategory.STEGANOGRAPHY, 0) > 0.6:
            recommendations.append("Implement steganographic detection algorithms")
            recommendations.append("Deploy content analysis and validation systems")

        if category_scores.get(ThreatCategory.EVASION, 0) > 0.6:
            recommendations.append("Enhance security monitoring and alerting")
            recommendations.append("Implement adaptive security controls")

        return list(set(recommendations))  # Remove duplicates

    def _generate_mitigation_strategies(self, risk_factors: list[RiskFactor], risk_level: RiskLevel) -> list[str]:
        """Generate mitigation strategies based on risk level."""
        strategies = []

        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            strategies.extend([
                "Implement immediate access restrictions",
                "Deploy emergency monitoring procedures",
                "Activate incident response protocols",
                "Conduct immediate security assessment"
            ])

        if risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]:
            strategies.extend([
                "Enhance user activity monitoring",
                "Implement additional authentication factors",
                "Deploy advanced threat detection systems",
                "Conduct regular security audits"
            ])

        # Always include basic strategies
        strategies.extend([
            "Maintain updated security policies",
            "Provide security awareness training",
            "Implement regular vulnerability assessments",
            "Maintain incident response capabilities"
        ])

        return strategies


class RiskReportGenerator:
    """Generates comprehensive risk assessment reports."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def generate_json_report(self, assessment: RiskAssessment) -> str:
        """Generate JSON format risk report."""
        report_data = {
            "assessment_id": assessment.assessment_id,
            "timestamp": assessment.timestamp.isoformat(),
            "overall_risk": {
                "level": assessment.overall_risk_level.value,
                "score": assessment.overall_risk_score
            },
            "threat_categories": {
                category.value: score
                for category, score in assessment.threat_categories.items()
            },
            "risk_factors": [
                {
                    "factor_id": factor.factor_id,
                    "name": factor.name,
                    "category": factor.category.value,
                    "current_value": factor.current_value,
                    "risk_score": factor.current_value * factor.weight,
                    "evidence": factor.evidence
                }
                for factor in assessment.risk_factors
            ],
            "recommendations": assessment.recommendations,
            "mitigation_strategies": assessment.mitigation_strategies,
            "metadata": assessment.metadata
        }

        return json.dumps(report_data, indent=2, default=str)

    def generate_executive_summary(self, assessment: RiskAssessment) -> str:
        """Generate executive summary of risk assessment."""
        summary = f"""
EXECUTIVE RISK ASSESSMENT SUMMARY
=================================

Assessment ID: {assessment.assessment_id}
Date: {assessment.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

OVERALL RISK LEVEL: {assessment.overall_risk_level.value.upper()}
Risk Score: {assessment.overall_risk_score:.2f}/1.00

THREAT CATEGORY BREAKDOWN:
"""

        for category, score in assessment.threat_categories.items():
            risk_level = "CRITICAL" if score >= 0.8 else "HIGH" if score >= 0.6 else "MEDIUM" if score >= 0.3 else "LOW"
            summary += f"- {category.value.replace('_', ' ').title()}: {score:.2f} ({risk_level})\n"

        summary += """
TOP RISK FACTORS:
"""

        # Sort factors by risk score
        sorted_factors = sorted(
            assessment.risk_factors,
            key=lambda f: f.current_value * f.weight,
            reverse=True
        )

        for factor in sorted_factors[:5]:  # Top 5 factors
            risk_score = factor.current_value * factor.weight
            summary += f"- {factor.name}: {risk_score:.2f}\n"

        summary += """
IMMEDIATE RECOMMENDATIONS:
"""

        for i, recommendation in enumerate(assessment.recommendations[:5], 1):
            summary += f"{i}. {recommendation}\n"

        if assessment.overall_risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            summary += f"""
⚠️  URGENT ACTION REQUIRED ⚠️
This assessment indicates {assessment.overall_risk_level.value} risk level.
Immediate implementation of mitigation strategies is recommended.
"""

        return summary


def main():
    """Main function for testing risk assessment."""
    import logging

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create sample data
    sample_documents = [
        type('Document', (), {
            'page_content': 'This document contains confidential financial information including SSN 123-45-6789 and credit card 4111-1111-1111-1111'
        })(),
        type('Document', (), {
            'page_content': 'Employee records with email john.doe@company.com and salary $75,000'
        })()
    ]

    sample_embeddings = np.random.normal(0, 1, (100, 384))

    sample_config = {
        "steganography": {
            "enabled": True,
            "techniques": ["noise", "rotation", "fragmentation"],
            "fragment_size": 64,
            "decoy_ratio": 0.4
        },
        "evasion": {
            "detection_avoidance_enabled": True,
            "traffic_mimicry_enabled": True,
            "behavioral_camouflage_enabled": True,
            "legitimate_ratio": 0.8,
            "user_profiles": ["researcher", "analyst"]
        },
        "vector_store": {
            "type": "qdrant"
        },
        "document": {
            "batch_processing": True,
            "supported_formats": [".pdf", ".docx", ".xlsx", ".csv", ".json", ".xml"]
        }
    }

    # Perform risk assessment
    assessor = VectorExfiltrationRiskAssessor(logger)
    assessment = assessor.perform_comprehensive_assessment(
        sample_documents,
        sample_embeddings,
        sample_config
    )

    # Generate reports
    report_generator = RiskReportGenerator(logger)

    logger.info("Generating JSON report...")
    json_report = report_generator.generate_json_report(assessment)

    logger.info("Generating executive summary...")
    executive_summary = report_generator.generate_executive_summary(assessment)

    print("=== EXECUTIVE SUMMARY ===")
    print(executive_summary)

    print("\n=== DETAILED JSON REPORT ===")
    print(json_report)


if __name__ == "__main__":
    main()
