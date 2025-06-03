"""
Baseline Traffic Generation for VectorSmuggle

This module generates baseline traffic patterns and legitimate activity
to help establish normal behavior patterns for comparison with potentially
malicious vector-based data exfiltration activities.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ActivityPattern:
    """Represents a pattern of user activity."""
    pattern_id: str
    name: str
    description: str
    user_type: str
    time_distribution: Dict[int, float]  # Hour -> probability
    activity_types: List[str]
    volume_range: tuple[int, int]
    frequency_per_day: tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BaselineActivity:
    """Represents a single baseline activity."""
    activity_id: str
    timestamp: datetime
    user_id: str
    user_type: str
    activity_type: str
    data_size: int
    duration: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class LegitimateUserSimulator:
    """Simulates legitimate user behavior patterns."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.user_patterns = self._initialize_user_patterns()
        self.activity_templates = self._initialize_activity_templates()
    
    def _initialize_user_patterns(self) -> Dict[str, ActivityPattern]:
        """Initialize standard user behavior patterns."""
        patterns = {}
        
        # Data Scientist Pattern
        patterns["data_scientist"] = ActivityPattern(
            pattern_id="data_scientist",
            name="Data Scientist",
            description="Researcher working with ML models and datasets",
            user_type="data_scientist",
            time_distribution={
                9: 0.1, 10: 0.15, 11: 0.2, 12: 0.1, 13: 0.05,
                14: 0.15, 15: 0.15, 16: 0.1, 17: 0.05, 18: 0.03, 19: 0.02
            },
            activity_types=[
                "document_upload", "embedding_generation", "similarity_search",
                "model_training", "data_analysis", "query_execution"
            ],
            volume_range=(1024, 10485760),  # 1KB to 10MB
            frequency_per_day=(5, 25)
        )
        
        # Business Analyst Pattern
        patterns["business_analyst"] = ActivityPattern(
            pattern_id="business_analyst",
            name="Business Analyst",
            description="Analyst working with business documents and reports",
            user_type="business_analyst",
            time_distribution={
                8: 0.05, 9: 0.15, 10: 0.2, 11: 0.15, 12: 0.05,
                13: 0.1, 14: 0.15, 15: 0.1, 16: 0.05
            },
            activity_types=[
                "document_search", "report_generation", "data_extraction",
                "content_analysis", "query_execution"
            ],
            volume_range=(512, 5242880),  # 512B to 5MB
            frequency_per_day=(3, 15)
        )
        
        # Developer Pattern
        patterns["developer"] = ActivityPattern(
            pattern_id="developer",
            name="Software Developer",
            description="Developer integrating with vector databases",
            user_type="developer",
            time_distribution={
                9: 0.1, 10: 0.15, 11: 0.2, 12: 0.05, 13: 0.05,
                14: 0.2, 15: 0.15, 16: 0.1
            },
            activity_types=[
                "api_testing", "integration_testing", "performance_testing",
                "embedding_validation", "system_monitoring"
            ],
            volume_range=(256, 1048576),  # 256B to 1MB
            frequency_per_day=(10, 50)
        )
        
        # Manager Pattern
        patterns["manager"] = ActivityPattern(
            pattern_id="manager",
            name="Manager",
            description="Manager accessing reports and summaries",
            user_type="manager",
            time_distribution={
                8: 0.1, 9: 0.2, 10: 0.15, 11: 0.1, 12: 0.05,
                13: 0.1, 14: 0.15, 15: 0.1, 16: 0.05
            },
            activity_types=[
                "report_access", "summary_generation", "dashboard_view",
                "executive_search"
            ],
            volume_range=(1024, 2097152),  # 1KB to 2MB
            frequency_per_day=(2, 8)
        )
        
        return patterns
    
    def _initialize_activity_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize activity templates with realistic parameters."""
        templates = {
            "document_upload": {
                "base_duration": 2.5,
                "duration_variance": 1.0,
                "success_rate": 0.95,
                "typical_size_multiplier": 1.0
            },
            "embedding_generation": {
                "base_duration": 5.0,
                "duration_variance": 2.0,
                "success_rate": 0.92,
                "typical_size_multiplier": 0.1
            },
            "similarity_search": {
                "base_duration": 1.2,
                "duration_variance": 0.5,
                "success_rate": 0.98,
                "typical_size_multiplier": 0.01
            },
            "query_execution": {
                "base_duration": 0.8,
                "duration_variance": 0.3,
                "success_rate": 0.97,
                "typical_size_multiplier": 0.05
            },
            "data_analysis": {
                "base_duration": 8.0,
                "duration_variance": 4.0,
                "success_rate": 0.90,
                "typical_size_multiplier": 2.0
            },
            "report_generation": {
                "base_duration": 3.0,
                "duration_variance": 1.5,
                "success_rate": 0.94,
                "typical_size_multiplier": 0.5
            },
            "api_testing": {
                "base_duration": 0.5,
                "duration_variance": 0.2,
                "success_rate": 0.85,
                "typical_size_multiplier": 0.1
            }
        }
        return templates
    
    def generate_user_activity(
        self,
        user_id: str,
        user_type: str,
        start_time: datetime,
        duration_hours: int = 8
    ) -> List[BaselineActivity]:
        """Generate realistic user activity for a given time period."""
        if user_type not in self.user_patterns:
            raise ValueError(f"Unknown user type: {user_type}")
        
        pattern = self.user_patterns[user_type]
        activities = []
        
        # Determine number of activities for the day
        min_freq, max_freq = pattern.frequency_per_day
        num_activities = random.randint(min_freq, max_freq)
        
        # Generate activities throughout the day
        for _ in range(num_activities):
            # Choose activity time based on distribution
            hour = self._choose_weighted_hour(pattern.time_distribution, start_time, duration_hours)
            if hour is None:
                continue
            
            # Create timestamp
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            activity_time = start_time.replace(hour=hour, minute=minute, second=second)
            
            # Choose activity type
            activity_type = random.choice(pattern.activity_types)
            
            # Generate activity details
            activity = self._generate_activity_details(
                user_id, user_type, activity_type, activity_time, pattern
            )
            activities.append(activity)
        
        # Sort by timestamp
        activities.sort(key=lambda x: x.timestamp)
        return activities
    
    def _choose_weighted_hour(
        self,
        time_distribution: Dict[int, float],
        start_time: datetime,
        duration_hours: int
    ) -> Optional[int]:
        """Choose an hour based on weighted distribution."""
        end_hour = start_time.hour + duration_hours
        valid_hours = []
        weights = []
        
        for hour, weight in time_distribution.items():
            if start_time.hour <= hour < end_hour:
                valid_hours.append(hour)
                weights.append(weight)
        
        if not valid_hours:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(valid_hours)
        
        normalized_weights = [w / total_weight for w in weights]
        return np.random.choice(valid_hours, p=normalized_weights)
    
    def _generate_activity_details(
        self,
        user_id: str,
        user_type: str,
        activity_type: str,
        timestamp: datetime,
        pattern: ActivityPattern
    ) -> BaselineActivity:
        """Generate detailed activity information."""
        template = self.activity_templates.get(activity_type, {})
        
        # Generate data size
        min_size, max_size = pattern.volume_range
        base_size = random.randint(min_size, max_size)
        size_multiplier = template.get("typical_size_multiplier", 1.0)
        data_size = int(base_size * size_multiplier)
        
        # Generate duration
        base_duration = template.get("base_duration", 1.0)
        duration_variance = template.get("duration_variance", 0.5)
        duration = max(0.1, random.normalvariate(base_duration, duration_variance))
        
        # Determine success
        success_rate = template.get("success_rate", 0.95)
        success = random.random() < success_rate
        
        # Generate activity ID
        activity_id = f"{user_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        return BaselineActivity(
            activity_id=activity_id,
            timestamp=timestamp,
            user_id=user_id,
            user_type=user_type,
            activity_type=activity_type,
            data_size=data_size,
            duration=duration,
            success=success,
            metadata={
                "pattern_id": pattern.pattern_id,
                "generated": True,
                "baseline": True
            }
        )


class TrafficPatternGenerator:
    """Generates realistic network traffic patterns."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.network_patterns = self._initialize_network_patterns()
    
    def _initialize_network_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize network traffic patterns."""
        return {
            "api_request": {
                "base_latency": 0.15,
                "latency_variance": 0.05,
                "packet_size_range": (64, 1500),
                "retry_probability": 0.02
            },
            "file_upload": {
                "base_latency": 2.0,
                "latency_variance": 1.0,
                "packet_size_range": (1024, 65536),
                "retry_probability": 0.05
            },
            "query_response": {
                "base_latency": 0.3,
                "latency_variance": 0.1,
                "packet_size_range": (256, 8192),
                "retry_probability": 0.01
            }
        }
    
    def generate_network_activity(
        self,
        activities: List[BaselineActivity]
    ) -> List[Dict[str, Any]]:
        """Generate network traffic data for activities."""
        network_events = []
        
        for activity in activities:
            # Map activity type to network pattern
            if activity.activity_type in ["document_upload", "data_analysis"]:
                pattern_type = "file_upload"
            elif activity.activity_type in ["similarity_search", "query_execution"]:
                pattern_type = "query_response"
            else:
                pattern_type = "api_request"
            
            pattern = self.network_patterns[pattern_type]
            
            # Generate network events for this activity
            events = self._generate_network_events(activity, pattern)
            network_events.extend(events)
        
        return sorted(network_events, key=lambda x: x["timestamp"])
    
    def _generate_network_events(
        self,
        activity: BaselineActivity,
        pattern: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate network events for a single activity."""
        events = []
        
        # Calculate number of packets based on data size
        min_packet_size, max_packet_size = pattern["packet_size_range"]
        avg_packet_size = (min_packet_size + max_packet_size) / 2
        num_packets = max(1, int(activity.data_size / avg_packet_size))
        
        # Generate packets
        current_time = activity.timestamp
        for i in range(num_packets):
            # Calculate packet timing
            base_latency = pattern["base_latency"]
            latency_variance = pattern["latency_variance"]
            packet_delay = max(0.001, random.normalvariate(base_latency / num_packets, latency_variance / num_packets))
            
            current_time += timedelta(seconds=packet_delay)
            
            # Generate packet size
            packet_size = random.randint(*pattern["packet_size_range"])
            
            # Check for retries
            retry_needed = random.random() < pattern["retry_probability"]
            
            event = {
                "timestamp": current_time.isoformat(),
                "activity_id": activity.activity_id,
                "user_id": activity.user_id,
                "packet_sequence": i + 1,
                "packet_size": packet_size,
                "direction": "outbound" if i % 2 == 0 else "inbound",
                "retry": retry_needed,
                "success": activity.success and not retry_needed
            }
            events.append(event)
            
            # Add retry packet if needed
            if retry_needed:
                retry_time = current_time + timedelta(seconds=random.uniform(0.1, 0.5))
                retry_event = event.copy()
                retry_event["timestamp"] = retry_time.isoformat()
                retry_event["retry"] = False
                retry_event["success"] = True
                events.append(retry_event)
        
        return events


class BaselineDatasetGenerator:
    """Generates comprehensive baseline datasets."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.user_simulator = LegitimateUserSimulator(logger)
        self.traffic_generator = TrafficPatternGenerator(logger)
    
    def generate_baseline_dataset(
        self,
        num_users: int = 50,
        days: int = 7,
        start_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive baseline dataset."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=days)
        
        self.logger.info(f"Generating baseline dataset for {num_users} users over {days} days")
        
        # Generate user profiles
        user_profiles = self._generate_user_profiles(num_users)
        
        # Generate activities for each user and day
        all_activities = []
        all_network_events = []
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Skip weekends for most users (simulate business environment)
            if current_date.weekday() >= 5:  # Saturday or Sunday
                weekend_users = random.sample(user_profiles, max(1, num_users // 10))
                day_users = weekend_users
            else:
                day_users = user_profiles
            
            for user in day_users:
                # Generate activities for this user on this day
                user_activities = self.user_simulator.generate_user_activity(
                    user["user_id"],
                    user["user_type"],
                    current_date,
                    duration_hours=10  # 10-hour work day
                )
                all_activities.extend(user_activities)
                
                # Generate network traffic for these activities
                network_events = self.traffic_generator.generate_network_activity(user_activities)
                all_network_events.extend(network_events)
        
        # Generate statistics
        statistics = self._generate_dataset_statistics(all_activities, all_network_events)
        
        # Create dataset
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "num_users": num_users,
                "days": days,
                "start_date": start_date.isoformat(),
                "end_date": (start_date + timedelta(days=days)).isoformat(),
                "total_activities": len(all_activities),
                "total_network_events": len(all_network_events)
            },
            "user_profiles": user_profiles,
            "activities": [self._activity_to_dict(a) for a in all_activities],
            "network_events": all_network_events,
            "statistics": statistics
        }
        
        self.logger.info(f"Generated baseline dataset with {len(all_activities)} activities")
        return dataset
    
    def _generate_user_profiles(self, num_users: int) -> List[Dict[str, Any]]:
        """Generate realistic user profiles."""
        user_types = ["data_scientist", "business_analyst", "developer", "manager"]
        type_distribution = [0.3, 0.3, 0.3, 0.1]  # Distribution of user types
        
        profiles = []
        for i in range(num_users):
            user_type = np.random.choice(user_types, p=type_distribution)
            
            profile = {
                "user_id": f"user_{i+1:03d}",
                "user_type": user_type,
                "department": self._assign_department(user_type),
                "access_level": self._assign_access_level(user_type),
                "created_at": (datetime.now() - timedelta(days=random.randint(30, 365))).isoformat()
            }
            profiles.append(profile)
        
        return profiles
    
    def _assign_department(self, user_type: str) -> str:
        """Assign department based on user type."""
        department_mapping = {
            "data_scientist": random.choice(["Data Science", "Research", "Analytics"]),
            "business_analyst": random.choice(["Business Intelligence", "Strategy", "Operations"]),
            "developer": random.choice(["Engineering", "IT", "Product"]),
            "manager": random.choice(["Management", "Executive", "Operations"])
        }
        return department_mapping.get(user_type, "General")
    
    def _assign_access_level(self, user_type: str) -> str:
        """Assign access level based on user type."""
        access_mapping = {
            "data_scientist": random.choice(["standard", "elevated"]),
            "business_analyst": "standard",
            "developer": random.choice(["standard", "elevated"]),
            "manager": random.choice(["elevated", "admin"])
        }
        return access_mapping.get(user_type, "standard")
    
    def _activity_to_dict(self, activity: BaselineActivity) -> Dict[str, Any]:
        """Convert activity to dictionary format."""
        return {
            "activity_id": activity.activity_id,
            "timestamp": activity.timestamp.isoformat(),
            "user_id": activity.user_id,
            "user_type": activity.user_type,
            "activity_type": activity.activity_type,
            "data_size": activity.data_size,
            "duration": activity.duration,
            "success": activity.success,
            "metadata": activity.metadata
        }
    
    def _generate_dataset_statistics(
        self,
        activities: List[BaselineActivity],
        network_events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate statistics for the dataset."""
        if not activities:
            return {}
        
        # Activity statistics
        activity_types = [a.activity_type for a in activities]
        user_types = [a.user_type for a in activities]
        data_sizes = [a.data_size for a in activities]
        durations = [a.duration for a in activities]
        success_rate = sum(1 for a in activities if a.success) / len(activities)
        
        # Time distribution
        hours = [a.timestamp.hour for a in activities]
        hour_distribution = {str(h): hours.count(h) for h in range(24)}
        
        # Network statistics
        total_network_bytes = sum(event.get("packet_size", 0) for event in network_events)
        network_success_rate = sum(1 for event in network_events if event.get("success", True)) / len(network_events) if network_events else 0
        
        return {
            "activity_statistics": {
                "total_activities": len(activities),
                "activity_type_distribution": {t: activity_types.count(t) for t in set(activity_types)},
                "user_type_distribution": {t: user_types.count(t) for t in set(user_types)},
                "success_rate": success_rate,
                "avg_data_size": np.mean(data_sizes),
                "avg_duration": np.mean(durations),
                "hour_distribution": hour_distribution
            },
            "network_statistics": {
                "total_events": len(network_events),
                "total_bytes": total_network_bytes,
                "success_rate": network_success_rate,
                "avg_packet_size": total_network_bytes / len(network_events) if network_events else 0
            }
        }
    
    def save_dataset(self, dataset: Dict[str, Any], filepath: str) -> None:
        """Save dataset to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
            self.logger.info(f"Baseline dataset saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, filepath: str) -> Dict[str, Any]:
        """Load dataset from file."""
        try:
            with open(filepath, 'r') as f:
                dataset = json.load(f)
            self.logger.info(f"Baseline dataset loaded from {filepath}")
            return dataset
        except Exception as e:
            self.logger.error(f"Failed to load dataset: {e}")
            raise


async def generate_realtime_baseline(
    duration_minutes: int = 60,
    users_per_minute: int = 5,
    output_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """Generate real-time baseline activity."""
    logger = logging.getLogger(__name__)
    simulator = LegitimateUserSimulator(logger)
    
    activities = []
    start_time = datetime.now()
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    logger.info(f"Starting real-time baseline generation for {duration_minutes} minutes")
    
    minute_count = 0
    while datetime.now() < end_time:
        minute_start = datetime.now()
        
        # Generate activities for this minute
        for i in range(users_per_minute):
            user_id = f"realtime_user_{i+1}"
            user_type = random.choice(["data_scientist", "business_analyst", "developer"])
            
            # Generate single activity
            activity_list = simulator.generate_user_activity(
                user_id, user_type, minute_start, duration_hours=1
            )
            
            if activity_list:
                activity = activity_list[0]  # Take first activity
                activity_dict = {
                    "activity_id": activity.activity_id,
                    "timestamp": activity.timestamp.isoformat(),
                    "user_id": activity.user_id,
                    "user_type": activity.user_type,
                    "activity_type": activity.activity_type,
                    "data_size": activity.data_size,
                    "duration": activity.duration,
                    "success": activity.success,
                    "realtime": True
                }
                activities.append(activity_dict)
                
                # Call output callback if provided
                if output_callback:
                    output_callback(activity_dict)
        
        minute_count += 1
        if minute_count % 10 == 0:
            logger.info(f"Generated {len(activities)} activities in {minute_count} minutes")
        
        # Wait until next minute
        elapsed = (datetime.now() - minute_start).total_seconds()
        if elapsed < 60:
            await asyncio.sleep(60 - elapsed)
    
    logger.info(f"Real-time baseline generation completed: {len(activities)} activities")
    return activities


def main():
    """Main function for testing baseline generation."""
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Generate baseline dataset
    generator = BaselineDatasetGenerator(logger)
    
    logger.info("Generating baseline dataset...")
    dataset = generator.generate_baseline_dataset(
        num_users=20,
        days=3,
        start_date=datetime.now() - timedelta(days=3)
    )
    
    # Save dataset
    generator.save_dataset(dataset, "baseline_dataset.json")
    
    # Print statistics
    stats = dataset["statistics"]
    print("\n=== BASELINE DATASET STATISTICS ===")
    print(f"Total Activities: {stats['activity_statistics']['total_activities']}")
    print(f"Success Rate: {stats['activity_statistics']['success_rate']:.2%}")
    print(f"Average Data Size: {stats['activity_statistics']['avg_data_size']:.0f} bytes")
    print(f"Average Duration: {stats['activity_statistics']['avg_duration']:.2f} seconds")
    print(f"Total Network Events: {stats['network_statistics']['total_events']}")
    print(f"Total Network Bytes: {stats['network_statistics']['total_bytes']:,}")
    
    print("\n=== ACTIVITY TYPE DISTRIBUTION ===")
    for activity_type, count in stats['activity_statistics']['activity_type_distribution'].items():
        print(f"{activity_type}: {count}")
    
    print("\n=== USER TYPE DISTRIBUTION ===")
    for user_type, count in stats['activity_statistics']['user_type_distribution'].items():
        print(f"{user_type}: {count}")


if __name__ == "__main__":
    main()