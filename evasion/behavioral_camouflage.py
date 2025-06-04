# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Behavioral camouflage for creating plausible deniability through mixed legitimate/malicious activity."""

import logging
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class BehavioralCamouflage:
    """Creates behavioral patterns that provide plausible deniability for data exfiltration."""

    def __init__(
        self,
        legitimate_ratio: float = 0.8,
        cover_story_templates: list[str] | None = None,
        role_profiles: dict[str, Any] | None = None,
        activity_mixing_strategy: str = "interleaved"
    ):
        """
        Initialize behavioral camouflage system.

        Args:
            legitimate_ratio: Ratio of legitimate to suspicious activities
            cover_story_templates: Templates for generating cover stories
            role_profiles: Different role-based behavior profiles
            activity_mixing_strategy: Strategy for mixing activities (interleaved, batched, random)
        """
        self.legitimate_ratio = legitimate_ratio
        self.activity_mixing_strategy = activity_mixing_strategy

        if cover_story_templates is None:
            cover_story_templates = [
                "Conducting market research for {project}",
                "Preparing documentation for {project}",
                "Analyzing data for {project}",
                "Researching best practices for {project}",
                "Gathering information for {project}",
                "Creating knowledge base for {project}",
                "Performing competitive analysis for {project}",
                "Building training materials for {project}"
            ]
        self.cover_story_templates = cover_story_templates

        if role_profiles is None:
            role_profiles = self._create_default_role_profiles()
        self.role_profiles = role_profiles

        self.current_role = "researcher"
        self.current_cover_story = None
        self.activity_log: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def _create_default_role_profiles(self) -> dict[str, Any]:
        """Create default role-based behavior profiles."""
        return {
            "researcher": {
                "typical_activities": [
                    "literature_search", "data_analysis", "report_writing",
                    "methodology_research", "statistical_analysis"
                ],
                "work_patterns": {
                    "session_duration": (2, 6),  # hours
                    "break_frequency": 0.3,
                    "deep_work_periods": True
                },
                "document_types": [
                    "academic_papers", "research_reports", "datasets",
                    "methodology_docs", "analysis_results"
                ],
                "query_complexity": "high",
                "suspicious_tolerance": 0.1
            },
            "analyst": {
                "typical_activities": [
                    "data_mining", "trend_analysis", "dashboard_creation",
                    "metric_calculation", "performance_review"
                ],
                "work_patterns": {
                    "session_duration": (1, 4),  # hours
                    "break_frequency": 0.4,
                    "deep_work_periods": False
                },
                "document_types": [
                    "financial_reports", "kpi_dashboards", "trend_analysis",
                    "performance_metrics", "business_intelligence"
                ],
                "query_complexity": "medium",
                "suspicious_tolerance": 0.15
            },
            "developer": {
                "typical_activities": [
                    "code_review", "documentation_writing", "api_research",
                    "troubleshooting", "architecture_planning"
                ],
                "work_patterns": {
                    "session_duration": (1, 8),  # hours
                    "break_frequency": 0.2,
                    "deep_work_periods": True
                },
                "document_types": [
                    "technical_specs", "api_docs", "code_examples",
                    "architecture_diagrams", "troubleshooting_guides"
                ],
                "query_complexity": "medium",
                "suspicious_tolerance": 0.05
            },
            "manager": {
                "typical_activities": [
                    "status_review", "planning", "team_coordination",
                    "reporting", "strategic_analysis"
                ],
                "work_patterns": {
                    "session_duration": (0.5, 2),  # hours
                    "break_frequency": 0.6,
                    "deep_work_periods": False
                },
                "document_types": [
                    "status_reports", "project_plans", "team_metrics",
                    "strategic_docs", "executive_summaries"
                ],
                "query_complexity": "low",
                "suspicious_tolerance": 0.2
            }
        }

    def generate_cover_story(self, project_context: str | None = None) -> str:
        """Generate a plausible cover story for current activities."""
        if project_context is None:
            projects = [
                "quarterly planning", "market expansion", "product development",
                "process improvement", "compliance review", "training program",
                "competitive analysis", "customer research", "technology assessment"
            ]
            project_context = random.choice(projects)

        template = random.choice(self.cover_story_templates)
        cover_story = template.format(project=project_context)

        self.current_cover_story = {
            "story": cover_story,
            "project": project_context,
            "created_at": datetime.utcnow(),
            "role": self.current_role
        }

        self.logger.info(f"Generated cover story: {cover_story}")
        return cover_story

    def create_legitimate_document_access_pattern(
        self,
        num_documents: int = 10,
        access_reason: str | None = None
    ) -> list[dict[str, Any]]:
        """Create a pattern of legitimate document access."""
        if access_reason is None:
            access_reason = self.current_cover_story["story"] if self.current_cover_story else "research"

        profile = self.role_profiles[self.current_role]
        doc_types = profile["document_types"]

        access_pattern = []
        base_time = datetime.utcnow()

        for i in range(num_documents):
            # Simulate realistic access timing
            if i == 0:
                access_time = base_time
            else:
                # Vary access intervals based on role
                if profile["work_patterns"]["deep_work_periods"]:
                    interval = random.uniform(30, 300)  # 30s to 5min for focused work
                else:
                    interval = random.uniform(60, 600)  # 1min to 10min for varied work

                access_time = access_pattern[-1]["timestamp"] + timedelta(seconds=interval)

            doc_access = {
                "document_id": f"doc_{i:03d}",
                "document_type": random.choice(doc_types),
                "access_reason": access_reason,
                "timestamp": access_time,
                "role": self.current_role,
                "access_duration": random.uniform(30, 600),  # 30s to 10min
                "legitimate": True
            }

            access_pattern.append(doc_access)

        return access_pattern

    def mix_activities(
        self,
        legitimate_activities: list[dict[str, Any]],
        suspicious_activities: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Mix legitimate and suspicious activities according to strategy."""
        if self.activity_mixing_strategy == "interleaved":
            return self._interleave_activities(legitimate_activities, suspicious_activities)
        elif self.activity_mixing_strategy == "batched":
            return self._batch_activities(legitimate_activities, suspicious_activities)
        elif self.activity_mixing_strategy == "random":
            return self._randomize_activities(legitimate_activities, suspicious_activities)
        else:
            raise ValueError(f"Unknown mixing strategy: {self.activity_mixing_strategy}")

    def _interleave_activities(
        self,
        legitimate: list[dict[str, Any]],
        suspicious: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Interleave suspicious activities with legitimate ones."""
        mixed = []

        # Calculate how many legitimate activities per suspicious one
        if suspicious:
            ratio = len(legitimate) / len(suspicious)
            legitimate_per_suspicious = max(1, int(ratio * self.legitimate_ratio))
        else:
            return legitimate

        legit_idx = 0
        susp_idx = 0

        while legit_idx < len(legitimate) or susp_idx < len(suspicious):
            # Add legitimate activities
            for _ in range(legitimate_per_suspicious):
                if legit_idx < len(legitimate):
                    mixed.append(legitimate[legit_idx])
                    legit_idx += 1

            # Add one suspicious activity
            if susp_idx < len(suspicious):
                mixed.append(suspicious[susp_idx])
                susp_idx += 1

        return mixed

    def _batch_activities(
        self,
        legitimate: list[dict[str, Any]],
        suspicious: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Group activities in batches with legitimate activities surrounding suspicious ones."""
        mixed = []

        if not suspicious:
            return legitimate

        # Split legitimate activities into groups
        legit_per_batch = len(legitimate) // (len(suspicious) + 1)

        legit_idx = 0

        for i, susp_activity in enumerate(suspicious):
            # Add batch of legitimate activities
            batch_size = legit_per_batch + random.randint(-1, 2)  # Add some variance
            for _ in range(batch_size):
                if legit_idx < len(legitimate):
                    mixed.append(legitimate[legit_idx])
                    legit_idx += 1

            # Add suspicious activity
            mixed.append(susp_activity)

        # Add remaining legitimate activities
        while legit_idx < len(legitimate):
            mixed.append(legitimate[legit_idx])
            legit_idx += 1

        return mixed

    def _randomize_activities(
        self,
        legitimate: list[dict[str, Any]],
        suspicious: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Randomly mix all activities while maintaining ratio."""
        all_activities = legitimate + suspicious

        # Shuffle while maintaining timestamp order within each type
        legitimate.sort(key=lambda x: x.get("timestamp", datetime.utcnow()))
        suspicious.sort(key=lambda x: x.get("timestamp", datetime.utcnow()))

        # Create weighted random selection
        weights = [self.legitimate_ratio] * len(legitimate) + [1 - self.legitimate_ratio] * len(suspicious)

        mixed = []
        remaining_legit = legitimate.copy()
        remaining_susp = suspicious.copy()

        while remaining_legit or remaining_susp:
            if remaining_legit and remaining_susp:
                # Choose based on weights
                if random.random() < self.legitimate_ratio:
                    mixed.append(remaining_legit.pop(0))
                else:
                    mixed.append(remaining_susp.pop(0))
            elif remaining_legit:
                mixed.extend(remaining_legit)
                break
            else:
                mixed.extend(remaining_susp)
                break

        return mixed

    def create_plausible_workflow(
        self,
        workflow_type: str,
        duration_hours: float = 4.0,
        include_breaks: bool = True
    ) -> list[dict[str, Any]]:
        """Create a plausible workflow with realistic timing and activities."""
        profile = self.role_profiles[self.current_role]

        workflows = {
            "research_project": [
                "project_planning", "literature_search", "data_collection",
                "analysis", "documentation", "review"
            ],
            "documentation_update": [
                "content_review", "gap_analysis", "writing",
                "formatting", "review_cycle", "publication"
            ],
            "data_analysis": [
                "data_preparation", "exploratory_analysis", "modeling",
                "validation", "interpretation", "reporting"
            ],
            "compliance_review": [
                "requirement_gathering", "current_state_analysis",
                "gap_identification", "remediation_planning", "documentation"
            ]
        }

        if workflow_type not in workflows:
            workflow_type = "research_project"

        workflow_steps = workflows[workflow_type]
        activities = []

        start_time = datetime.utcnow()
        current_time = start_time

        step_duration = (duration_hours * 3600) / len(workflow_steps)

        for i, step in enumerate(workflow_steps):
            # Add activities for this step
            step_activities = random.randint(2, 6)

            for j in range(step_activities):
                activity_time = current_time + timedelta(
                    seconds=j * (step_duration / step_activities) + random.uniform(-300, 300)
                )

                activity = {
                    "workflow_type": workflow_type,
                    "step": step,
                    "step_index": i,
                    "activity_index": j,
                    "timestamp": activity_time,
                    "role": self.current_role,
                    "legitimate": True,
                    "cover_story": self.current_cover_story["story"] if self.current_cover_story else None
                }

                activities.append(activity)

            current_time += timedelta(seconds=step_duration)

            # Add breaks if enabled
            if include_breaks and i < len(workflow_steps) - 1:
                if random.random() < profile["work_patterns"]["break_frequency"]:
                    break_duration = random.uniform(300, 1800)  # 5-30 min break
                    current_time += timedelta(seconds=break_duration)

        self.logger.info(f"Created {workflow_type} workflow with {len(activities)} activities")
        return activities

    def assess_suspicion_level(self, activities: list[dict[str, Any]]) -> dict[str, Any]:
        """Assess the suspicion level of a set of activities."""
        profile = self.role_profiles[self.current_role]

        total_activities = len(activities)
        suspicious_activities = len([a for a in activities if not a.get("legitimate", True)])
        legitimate_activities = total_activities - suspicious_activities

        # Calculate suspicion metrics
        suspicious_ratio = suspicious_activities / total_activities if total_activities > 0 else 0

        # Check timing patterns
        timestamps = [a["timestamp"] for a in activities if "timestamp" in a]
        if len(timestamps) > 1:
            timestamps.sort()
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds()
                        for i in range(len(timestamps)-1)]
            avg_interval = np.mean(intervals)
            interval_variance = np.std(intervals)
        else:
            avg_interval = 0
            interval_variance = 0

        # Assess against role profile
        tolerance = profile["suspicious_tolerance"]

        suspicion_score = 0.0

        # Ratio-based suspicion
        if suspicious_ratio > tolerance:
            suspicion_score += (suspicious_ratio - tolerance) * 2

        # Timing-based suspicion (too regular = suspicious)
        if interval_variance < avg_interval * 0.1 and len(intervals) > 5:
            suspicion_score += 0.3

        # Activity type consistency
        activity_types = set(a.get("step", "unknown") for a in activities)
        if len(activity_types) < 2:  # Too focused
            suspicion_score += 0.2

        # Normalize score
        suspicion_score = min(1.0, suspicion_score)

        assessment = {
            "suspicion_score": suspicion_score,
            "suspicious_ratio": suspicious_ratio,
            "total_activities": total_activities,
            "suspicious_activities": suspicious_activities,
            "legitimate_activities": legitimate_activities,
            "avg_interval": avg_interval,
            "interval_variance": interval_variance,
            "role_tolerance": tolerance,
            "assessment": self._categorize_suspicion(suspicion_score)
        }

        return assessment

    def _categorize_suspicion(self, score: float) -> str:
        """Categorize suspicion level based on score."""
        if score < 0.2:
            return "low"
        elif score < 0.5:
            return "medium"
        elif score < 0.8:
            return "high"
        else:
            return "critical"

    def optimize_activity_mix(
        self,
        suspicious_activities: list[dict[str, Any]],
        target_suspicion: float = 0.1
    ) -> list[dict[str, Any]]:
        """Optimize the mix of activities to achieve target suspicion level."""
        current_suspicion = 1.0  # Start high
        iterations = 0
        max_iterations = 10

        while current_suspicion > target_suspicion and iterations < max_iterations:
            # Calculate required legitimate activities
            num_suspicious = len(suspicious_activities)
            required_legitimate = int(num_suspicious * (1 - target_suspicion) / target_suspicion)

            # Generate legitimate activities
            legitimate_activities = self.create_legitimate_document_access_pattern(
                num_documents=required_legitimate
            )

            # Mix activities
            mixed_activities = self.mix_activities(legitimate_activities, suspicious_activities)

            # Assess suspicion
            assessment = self.assess_suspicion_level(mixed_activities)
            current_suspicion = assessment["suspicion_score"]

            iterations += 1

            # Adjust if still too suspicious
            if current_suspicion > target_suspicion:
                target_suspicion *= 1.1  # Slightly relax target

        self.logger.info(f"Optimized activity mix: {current_suspicion:.3f} suspicion score")
        return mixed_activities

    def switch_role(self, new_role: str) -> None:
        """Switch to a different behavioral role."""
        if new_role in self.role_profiles:
            old_role = self.current_role
            self.current_role = new_role
            self.logger.info(f"Switched role from {old_role} to {new_role}")

            # Generate new cover story for the role
            self.generate_cover_story()
        else:
            raise ValueError(f"Unknown role: {new_role}")

    def get_behavioral_statistics(self) -> dict[str, Any]:
        """Get statistics about behavioral patterns."""
        if not self.activity_log:
            return {"message": "No activity log available"}

        legitimate_count = len([a for a in self.activity_log if a.get("legitimate", True)])
        suspicious_count = len(self.activity_log) - legitimate_count

        role_distribution = {}
        for activity in self.activity_log:
            role = activity.get("role", "unknown")
            role_distribution[role] = role_distribution.get(role, 0) + 1

        stats = {
            "total_activities": len(self.activity_log),
            "legitimate_activities": legitimate_count,
            "suspicious_activities": suspicious_count,
            "legitimate_ratio": legitimate_count / len(self.activity_log),
            "role_distribution": role_distribution,
            "current_role": self.current_role,
            "current_cover_story": self.current_cover_story,
            "mixing_strategy": self.activity_mixing_strategy
        }

        return stats
