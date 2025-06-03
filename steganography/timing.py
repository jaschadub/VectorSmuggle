"""Time-delayed exfiltration with randomized timing patterns."""

import asyncio
import logging
import random
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import numpy as np


class TimedExfiltrator:
    """Implements time-delayed data exfiltration with randomized patterns."""

    def __init__(
        self,
        base_delay: float = 60.0,
        delay_variance: float = 0.3,
        batch_size: int = 5,
        max_batches_per_hour: int = 10,
        business_hours_only: bool = True,
        timezone_offset: int = 0,
        jitter_factor: float = 0.2
    ):
        """
        Initialize timed exfiltrator.

        Args:
            base_delay: Base delay between uploads in seconds
            delay_variance: Variance factor for randomizing delays (0.0-1.0)
            batch_size: Number of embeddings per batch
            max_batches_per_hour: Maximum batches to upload per hour
            business_hours_only: Whether to only upload during business hours (9-17)
            timezone_offset: Timezone offset from UTC in hours
            jitter_factor: Additional random jitter factor
        """
        self.base_delay = base_delay
        self.delay_variance = delay_variance
        self.batch_size = batch_size
        self.max_batches_per_hour = max_batches_per_hour
        self.business_hours_only = business_hours_only
        self.timezone_offset = timezone_offset
        self.jitter_factor = jitter_factor

        self.upload_history: list[dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)

    def _is_business_hours(self) -> bool:
        """Check if current time is within business hours."""
        if not self.business_hours_only:
            return True

        now = datetime.utcnow() + timedelta(hours=self.timezone_offset)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Business hours: Monday-Friday, 9 AM - 5 PM
        return weekday < 5 and 9 <= hour < 17

    def _calculate_next_delay(self) -> float:
        """Calculate next delay with randomization and jitter."""
        # Base delay with variance
        variance = self.base_delay * self.delay_variance
        delay = np.random.normal(self.base_delay, variance)

        # Add jitter
        jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * delay
        delay += jitter

        # Ensure minimum delay of 10 seconds
        delay = max(10.0, delay)

        return delay

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits for uploads."""
        now = datetime.utcnow()
        one_hour_ago = now - timedelta(hours=1)

        # Count uploads in the last hour
        recent_uploads = [
            upload for upload in self.upload_history
            if upload["timestamp"] > one_hour_ago
        ]

        return len(recent_uploads) < self.max_batches_per_hour

    def _wait_for_business_hours(self) -> float:
        """Calculate wait time until next business hours."""
        if not self.business_hours_only:
            return 0.0

        now = datetime.utcnow() + timedelta(hours=self.timezone_offset)

        # If it's weekend, wait until Monday 9 AM
        if now.weekday() >= 5:  # Saturday or Sunday
            days_until_monday = 7 - now.weekday()
            next_business_day = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=days_until_monday)
        else:
            # If it's a weekday but outside business hours
            if now.hour < 9:
                # Wait until 9 AM today
                next_business_day = now.replace(hour=9, minute=0, second=0, microsecond=0)
            elif now.hour >= 17:
                # Wait until 9 AM tomorrow
                next_business_day = now.replace(hour=9, minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                # Currently in business hours
                return 0.0

        wait_seconds = (next_business_day - now).total_seconds()
        return max(0.0, wait_seconds)

    def create_upload_schedule(
        self,
        total_batches: int,
        start_time: datetime | None = None
    ) -> list[dict[str, Any]]:
        """
        Create a randomized upload schedule for batches.

        Args:
            total_batches: Total number of batches to schedule
            start_time: Start time for scheduling (defaults to now)

        Returns:
            List of scheduled upload times with metadata
        """
        if start_time is None:
            start_time = datetime.utcnow()

        schedule = []
        current_time = start_time

        for batch_id in range(total_batches):
            # Wait for business hours if required
            business_hours_wait = self._wait_for_business_hours()
            if business_hours_wait > 0:
                current_time += timedelta(seconds=business_hours_wait)

            # Calculate delay for this batch
            delay = self._calculate_next_delay()

            # Add some randomization to spread uploads throughout the day
            if self.business_hours_only:
                # Spread uploads randomly within business hours
                business_hours_duration = 8 * 3600  # 8 hours in seconds
                random_offset = random.uniform(0, min(delay, business_hours_duration / total_batches))
                delay += random_offset

            current_time += timedelta(seconds=delay)

            schedule_entry = {
                "batch_id": batch_id,
                "scheduled_time": current_time,
                "delay_seconds": delay,
                "is_business_hours": self._is_business_hours()
            }

            schedule.append(schedule_entry)

        self.logger.info(f"Created upload schedule for {total_batches} batches")
        return schedule

    async def delayed_upload(
        self,
        upload_func: Callable,
        data_batch: Any,
        delay: float | None = None
    ) -> dict[str, Any]:
        """
        Perform delayed upload with specified or calculated delay.

        Args:
            upload_func: Function to call for uploading data
            data_batch: Data to upload
            delay: Optional specific delay. If None, calculates delay automatically

        Returns:
            Upload result with timing metadata
        """
        if delay is None:
            delay = self._calculate_next_delay()

        # Wait for business hours if required
        business_hours_wait = self._wait_for_business_hours()
        if business_hours_wait > 0:
            self.logger.info(f"Waiting {business_hours_wait:.1f}s for business hours")
            await asyncio.sleep(business_hours_wait)

        # Check rate limits
        if not self._check_rate_limit():
            # Calculate wait time until rate limit resets
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            oldest_recent = min(
                upload["timestamp"] for upload in self.upload_history
                if upload["timestamp"] > one_hour_ago
            )
            wait_time = (oldest_recent + timedelta(hours=1) - datetime.utcnow()).total_seconds()

            if wait_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        # Apply the calculated delay
        self.logger.debug(f"Applying delay of {delay:.1f}s before upload")
        await asyncio.sleep(delay)

        # Perform the upload
        start_time = datetime.utcnow()
        try:
            result = await upload_func(data_batch)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
            self.logger.error(f"Upload failed: {e}")

        end_time = datetime.utcnow()

        # Record upload in history
        upload_record = {
            "timestamp": start_time,
            "duration": (end_time - start_time).total_seconds(),
            "success": success,
            "error": error,
            "delay_applied": delay,
            "business_hours": self._is_business_hours()
        }

        self.upload_history.append(upload_record)

        # Keep only last 100 upload records
        if len(self.upload_history) > 100:
            self.upload_history = self.upload_history[-100:]

        return {
            "result": result,
            "metadata": upload_record
        }

    async def batch_upload_with_timing(
        self,
        upload_func: Callable,
        data_batches: list[Any],
        schedule: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """
        Upload multiple batches with timing control.

        Args:
            upload_func: Function to call for uploading each batch
            data_batches: List of data batches to upload
            schedule: Optional pre-computed schedule. If None, creates one automatically

        Returns:
            List of upload results with timing metadata
        """
        if schedule is None:
            schedule = self.create_upload_schedule(len(data_batches))

        if len(schedule) != len(data_batches):
            raise ValueError("Schedule length must match number of data batches")

        results = []

        for i, (batch, schedule_entry) in enumerate(zip(data_batches, schedule, strict=False)):
            # Calculate delay until scheduled time
            now = datetime.utcnow()
            scheduled_time = schedule_entry["scheduled_time"]

            if scheduled_time > now:
                delay = (scheduled_time - now).total_seconds()
            else:
                delay = 0.0

            self.logger.info(f"Uploading batch {i+1}/{len(data_batches)} (delay: {delay:.1f}s)")

            result = await self.delayed_upload(upload_func, batch, delay)
            result["batch_id"] = schedule_entry["batch_id"]
            result["scheduled_time"] = scheduled_time

            results.append(result)

        self.logger.info(f"Completed batch upload of {len(data_batches)} batches")
        return results

    def get_timing_statistics(self) -> dict[str, Any]:
        """Get statistics about upload timing and patterns."""
        if not self.upload_history:
            return {"message": "No upload history available"}

        successful_uploads = [u for u in self.upload_history if u["success"]]
        failed_uploads = [u for u in self.upload_history if not u["success"]]

        delays = [u["delay_applied"] for u in self.upload_history]
        durations = [u["duration"] for u in successful_uploads]

        stats = {
            "total_uploads": len(self.upload_history),
            "successful_uploads": len(successful_uploads),
            "failed_uploads": len(failed_uploads),
            "success_rate": len(successful_uploads) / len(self.upload_history) if self.upload_history else 0,
            "average_delay": np.mean(delays) if delays else 0,
            "delay_std": np.std(delays) if delays else 0,
            "average_duration": np.mean(durations) if durations else 0,
            "business_hours_uploads": len([u for u in self.upload_history if u["business_hours"]]),
            "configuration": {
                "base_delay": self.base_delay,
                "delay_variance": self.delay_variance,
                "batch_size": self.batch_size,
                "max_batches_per_hour": self.max_batches_per_hour,
                "business_hours_only": self.business_hours_only
            }
        }

        return stats

    def simulate_legitimate_pattern(self, duration_hours: int = 24) -> list[dict[str, Any]]:
        """
        Simulate a legitimate usage pattern for the specified duration.

        Args:
            duration_hours: Duration to simulate in hours

        Returns:
            List of simulated activity timestamps
        """
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(hours=duration_hours)

        activities = []
        current_time = start_time

        while current_time < end_time:
            # Simulate legitimate user activity patterns
            if self.business_hours_only:
                # Skip to next business day if outside hours
                business_wait = self._wait_for_business_hours()
                if business_wait > 0:
                    current_time += timedelta(seconds=business_wait)
                    continue

            # Simulate realistic user behavior
            # Users typically have bursts of activity followed by longer pauses
            burst_size = random.randint(1, 5)

            for _ in range(burst_size):
                if current_time >= end_time:
                    break

                activity = {
                    "timestamp": current_time,
                    "activity_type": random.choice(["query", "upload", "browse"]),
                    "is_business_hours": self._is_business_hours()
                }
                activities.append(activity)

                # Short delay between activities in a burst
                current_time += timedelta(seconds=random.uniform(5, 30))

            # Longer pause between bursts
            pause_duration = random.uniform(300, 1800)  # 5-30 minutes
            current_time += timedelta(seconds=pause_duration)

        self.logger.info(f"Simulated {len(activities)} legitimate activities over {duration_hours} hours")
        return activities
