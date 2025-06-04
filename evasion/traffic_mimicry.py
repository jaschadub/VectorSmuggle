# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Traffic mimicry system for simulating legitimate RAG usage patterns."""

import logging
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class TrafficMimicry:
    """Simulates legitimate RAG usage patterns to blend malicious traffic."""

    def __init__(
        self,
        base_query_interval: float = 300.0,
        query_variance: float = 0.4,
        burst_probability: float = 0.15,
        burst_size_range: tuple[int, int] = (3, 8),
        business_hours_only: bool = True,
        timezone_offset: int = 0,
        user_profiles: list[str] | None = None
    ):
        """
        Initialize traffic mimicry system.

        Args:
            base_query_interval: Base interval between queries in seconds
            query_variance: Variance factor for randomizing intervals
            burst_probability: Probability of query bursts
            burst_size_range: Range for number of queries in a burst
            business_hours_only: Whether to limit activity to business hours
            timezone_offset: Timezone offset from UTC in hours
            user_profiles: List of user profile types to simulate
        """
        self.base_query_interval = base_query_interval
        self.query_variance = query_variance
        self.burst_probability = burst_probability
        self.burst_size_range = burst_size_range
        self.business_hours_only = business_hours_only
        self.timezone_offset = timezone_offset

        if user_profiles is None:
            user_profiles = ["researcher", "analyst", "developer", "manager"]
        self.user_profiles = user_profiles

        self.activity_history: list[dict[str, Any]] = []
        self.current_profile = random.choice(self.user_profiles)
        self.logger = logging.getLogger(__name__)

        # Legitimate query patterns for different user types
        self.query_patterns = {
            "researcher": {
                "topics": [
                    "market research", "competitive analysis", "industry trends",
                    "academic papers", "literature review", "data analysis",
                    "statistical methods", "research methodology"
                ],
                "query_complexity": "high",
                "session_duration": (1800, 7200),  # 30min - 2hrs
                "queries_per_session": (5, 20)
            },
            "analyst": {
                "topics": [
                    "financial data", "performance metrics", "business intelligence",
                    "quarterly reports", "KPI analysis", "dashboard data",
                    "trend analysis", "forecasting models"
                ],
                "query_complexity": "medium",
                "session_duration": (900, 3600),  # 15min - 1hr
                "queries_per_session": (3, 15)
            },
            "developer": {
                "topics": [
                    "API documentation", "code examples", "technical specs",
                    "architecture patterns", "best practices", "debugging",
                    "performance optimization", "security guidelines"
                ],
                "query_complexity": "medium",
                "session_duration": (600, 2400),  # 10min - 40min
                "queries_per_session": (2, 12)
            },
            "manager": {
                "topics": [
                    "executive summary", "project status", "team performance",
                    "budget analysis", "strategic planning", "risk assessment",
                    "compliance reports", "operational metrics"
                ],
                "query_complexity": "low",
                "session_duration": (300, 1800),  # 5min - 30min
                "queries_per_session": (1, 8)
            }
        }

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        if not self.business_hours_only:
            return True

        now = datetime.utcnow() + timedelta(hours=self.timezone_offset)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Business hours: Monday-Friday, 8 AM - 6 PM
        return weekday < 5 and 8 <= hour < 18

    def _calculate_next_interval(self) -> float:
        """Calculate next query interval with realistic variance."""
        profile_data = self.query_patterns[self.current_profile]

        # Adjust base interval based on user profile
        if profile_data["query_complexity"] == "high":
            base_interval = self.base_query_interval * 1.5
        elif profile_data["query_complexity"] == "low":
            base_interval = self.base_query_interval * 0.7
        else:
            base_interval = self.base_query_interval

        # Add variance
        variance = base_interval * self.query_variance
        interval = np.random.normal(base_interval, variance)

        # Ensure minimum interval
        return max(30.0, interval)

    def _generate_legitimate_query(self) -> dict[str, Any]:
        """Generate a legitimate-looking query based on current user profile."""
        profile_data = self.query_patterns[self.current_profile]
        topic = random.choice(profile_data["topics"])

        # Generate query variations
        query_templates = [
            f"What is {topic}?",
            f"How to analyze {topic}",
            f"Best practices for {topic}",
            f"Examples of {topic}",
            f"Latest trends in {topic}",
            f"Compare {topic} methods",
            f"Implementation of {topic}",
            f"Guidelines for {topic}"
        ]

        query_text = random.choice(query_templates)

        return {
            "query": query_text,
            "topic": topic,
            "profile": self.current_profile,
            "timestamp": datetime.utcnow(),
            "complexity": profile_data["query_complexity"]
        }

    def _should_start_burst(self) -> bool:
        """Determine if a query burst should start."""
        return random.random() < self.burst_probability

    def _generate_query_burst(self) -> list[dict[str, Any]]:
        """Generate a burst of related queries."""
        burst_size = random.randint(*self.burst_size_range)
        queries = []

        # Pick a topic for the burst
        profile_data = self.query_patterns[self.current_profile]
        burst_topic = random.choice(profile_data["topics"])

        for i in range(burst_size):
            # Generate related queries for the same topic
            query_variations = [
                f"Details about {burst_topic}",
                f"Advanced {burst_topic} techniques",
                f"Common issues with {burst_topic}",
                f"Tools for {burst_topic}",
                f"Metrics for {burst_topic}",
                f"Process for {burst_topic}",
                f"Standards for {burst_topic}",
                f"Documentation on {burst_topic}"
            ]

            query_text = random.choice(query_variations)

            query = {
                "query": query_text,
                "topic": burst_topic,
                "profile": self.current_profile,
                "timestamp": datetime.utcnow() + timedelta(seconds=i * random.uniform(5, 30)),
                "complexity": profile_data["query_complexity"],
                "burst_id": len(self.activity_history),
                "burst_position": i
            }
            queries.append(query)

        return queries

    def _simulate_user_session(self) -> list[dict[str, Any]]:
        """Simulate a realistic user session with multiple queries."""
        profile_data = self.query_patterns[self.current_profile]

        # Determine session parameters
        session_duration = random.uniform(*profile_data["session_duration"])
        num_queries = random.randint(*profile_data["queries_per_session"])

        session_queries = []
        session_start = datetime.utcnow()

        for i in range(num_queries):
            # Distribute queries throughout the session
            query_time = session_start + timedelta(
                seconds=(session_duration / num_queries) * i + random.uniform(-60, 60)
            )

            query = self._generate_legitimate_query()
            query["timestamp"] = query_time
            query["session_id"] = len(self.activity_history)
            query["session_position"] = i

            session_queries.append(query)

        return session_queries

    def _switch_user_profile(self) -> None:
        """Randomly switch to a different user profile."""
        if random.random() < 0.1:  # 10% chance to switch profiles
            old_profile = self.current_profile
            self.current_profile = random.choice(self.user_profiles)
            if old_profile != self.current_profile:
                self.logger.debug(f"Switched user profile from {old_profile} to {self.current_profile}")

    async def generate_cover_traffic(
        self,
        duration_hours: float = 8.0,
        malicious_callback: Callable | None = None
    ) -> list[dict[str, Any]]:
        """
        Generate cover traffic to mask malicious activities.

        Args:
            duration_hours: Duration to generate traffic for
            malicious_callback: Optional callback for malicious activities

        Returns:
            List of generated traffic activities
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)

        activities = []
        current_time = start_time

        self.logger.info(f"Starting cover traffic generation for {duration_hours} hours")

        while current_time < end_time:
            # Check if we should operate during business hours
            if self.business_hours_only and not self._is_business_hours():
                # Skip to next business hour
                next_business = self._next_business_hour(current_time)
                current_time = next_business
                continue

            # Switch user profiles occasionally
            self._switch_user_profile()

            # Decide on activity type
            if self._should_start_burst():
                # Generate query burst
                burst_queries = self._generate_query_burst()
                activities.extend(burst_queries)

                # Execute malicious activity during burst if callback provided
                if malicious_callback and random.random() < 0.3:  # 30% chance during burst
                    try:
                        malicious_result = await malicious_callback()
                        activities.append({
                            "type": "malicious",
                            "timestamp": datetime.utcnow(),
                            "result": malicious_result,
                            "hidden_in": "burst"
                        })
                    except Exception as e:
                        self.logger.error(f"Malicious callback failed: {e}")

                # Update current time
                current_time = max(query["timestamp"] for query in burst_queries)
                current_time += timedelta(seconds=random.uniform(60, 300))

            else:
                # Generate single query or session
                if random.random() < 0.4:  # 40% chance for session
                    session_queries = self._simulate_user_session()
                    activities.extend(session_queries)
                    current_time = max(query["timestamp"] for query in session_queries)
                else:
                    # Single query
                    query = self._generate_legitimate_query()
                    activities.append(query)
                    current_time = query["timestamp"]

                # Calculate next interval
                interval = self._calculate_next_interval()
                current_time += timedelta(seconds=interval)

        # Record activities in history
        self.activity_history.extend(activities)

        self.logger.info(f"Generated {len(activities)} cover traffic activities")
        return activities

    def _next_business_hour(self, current_time: datetime) -> datetime:
        """Calculate next business hour from given time."""
        adjusted_time = current_time + timedelta(hours=self.timezone_offset)

        # If weekend, go to Monday 8 AM
        if adjusted_time.weekday() >= 5:
            days_to_monday = 7 - adjusted_time.weekday()
            next_business = adjusted_time.replace(hour=8, minute=0, second=0, microsecond=0)
            next_business += timedelta(days=days_to_monday)
        else:
            # If before 8 AM, go to 8 AM today
            if adjusted_time.hour < 8:
                next_business = adjusted_time.replace(hour=8, minute=0, second=0, microsecond=0)
            # If after 6 PM, go to 8 AM tomorrow
            elif adjusted_time.hour >= 18:
                next_business = adjusted_time.replace(hour=8, minute=0, second=0, microsecond=0)
                next_business += timedelta(days=1)
            else:
                # Currently in business hours
                return current_time

        # Convert back to UTC
        return next_business - timedelta(hours=self.timezone_offset)

    def get_traffic_statistics(self) -> dict[str, Any]:
        """Get statistics about generated traffic patterns."""
        if not self.activity_history:
            return {"message": "No traffic history available"}

        # Analyze patterns
        profile_counts = {}
        topic_counts = {}
        hourly_distribution = {}

        for activity in self.activity_history:
            # Profile distribution
            profile = activity.get("profile", "unknown")
            profile_counts[profile] = profile_counts.get(profile, 0) + 1

            # Topic distribution
            topic = activity.get("topic", "unknown")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

            # Hourly distribution
            hour = activity["timestamp"].hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

        # Calculate intervals between activities
        timestamps = [activity["timestamp"] for activity in self.activity_history]
        timestamps.sort()
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)

        stats = {
            "total_activities": len(self.activity_history),
            "profile_distribution": profile_counts,
            "topic_distribution": topic_counts,
            "hourly_distribution": hourly_distribution,
            "average_interval": np.mean(intervals) if intervals else 0,
            "interval_std": np.std(intervals) if intervals else 0,
            "business_hours_activities": len([
                a for a in self.activity_history
                if 8 <= a["timestamp"].hour < 18 and a["timestamp"].weekday() < 5
            ]),
            "current_profile": self.current_profile,
            "configuration": {
                "base_query_interval": self.base_query_interval,
                "query_variance": self.query_variance,
                "burst_probability": self.burst_probability,
                "business_hours_only": self.business_hours_only
            }
        }

        return stats

    def create_realistic_workflow(self, workflow_type: str = "research") -> list[dict[str, Any]]:
        """
        Create a realistic workflow pattern for specific business activities.

        Args:
            workflow_type: Type of workflow to simulate

        Returns:
            List of workflow activities
        """
        workflows = {
            "research": [
                "literature search", "data collection", "analysis planning",
                "statistical analysis", "result interpretation", "report writing"
            ],
            "documentation": [
                "requirement gathering", "outline creation", "content writing",
                "review process", "revision", "final formatting"
            ],
            "qa_session": [
                "question preparation", "information gathering", "answer formulation",
                "fact checking", "response refinement", "follow-up questions"
            ]
        }

        if workflow_type not in workflows:
            workflow_type = "research"

        workflow_steps = workflows[workflow_type]
        activities = []

        start_time = datetime.utcnow()

        for i, step in enumerate(workflow_steps):
            # Generate queries for each workflow step
            step_queries = []
            num_queries = random.randint(1, 4)

            for j in range(num_queries):
                query_time = start_time + timedelta(
                    minutes=i * random.uniform(10, 30) + j * random.uniform(2, 8)
                )

                query = {
                    "query": f"How to {step}",
                    "topic": step,
                    "profile": self.current_profile,
                    "timestamp": query_time,
                    "workflow_type": workflow_type,
                    "workflow_step": i,
                    "step_name": step
                }
                step_queries.append(query)

            activities.extend(step_queries)

        self.logger.info(f"Created {workflow_type} workflow with {len(activities)} activities")
        return activities
