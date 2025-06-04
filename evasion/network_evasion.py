# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Network evasion techniques for traffic shaping, proxy rotation, and connection management."""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class NetworkEvasion:
    """Implements network-level evasion techniques for stealthy data exfiltration."""

    def __init__(
        self,
        proxy_list: list[str] | None = None,
        user_agents: list[str] | None = None,
        rate_limit_delay: tuple[float, float] = (1.0, 5.0),
        connection_timeout: float = 30.0,
        max_retries: int = 3,
        retry_backoff_factor: float = 2.0,
        session_rotation_interval: int = 10
    ):
        """
        Initialize network evasion system.

        Args:
            proxy_list: List of proxy servers to rotate through
            user_agents: List of user agent strings to rotate
            rate_limit_delay: Min/max delay between requests
            connection_timeout: Connection timeout in seconds
            max_retries: Maximum number of retries for failed requests
            retry_backoff_factor: Exponential backoff factor for retries
            session_rotation_interval: Number of requests before rotating session
        """
        self.proxy_list = proxy_list or []
        self.rate_limit_delay = rate_limit_delay
        self.connection_timeout = connection_timeout
        self.max_retries = max_retries
        self.retry_backoff_factor = retry_backoff_factor
        self.session_rotation_interval = session_rotation_interval

        if user_agents is None:
            user_agents = self._get_default_user_agents()
        self.user_agents = user_agents

        self.current_proxy_index = 0
        self.current_user_agent_index = 0
        self.request_count = 0
        self.session_pool: list[requests.Session] = []
        self.connection_stats: dict[str, Any] = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "proxy_rotations": 0,
            "session_rotations": 0,
            "retry_attempts": 0
        }

        self.logger = logging.getLogger(__name__)
        self._initialize_sessions()

    def _get_default_user_agents(self) -> list[str]:
        """Get default user agent strings for common browsers."""
        return [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
        ]

    def _initialize_sessions(self) -> None:
        """Initialize HTTP session pool with different configurations."""
        for i in range(3):  # Create 3 different sessions
            session = requests.Session()

            # Configure retry strategy
            retry_strategy = Retry(
                total=self.max_retries,
                backoff_factor=self.retry_backoff_factor,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            session.mount("http://", adapter)
            session.mount("https://", adapter)

            # Set timeout
            session.timeout = self.connection_timeout

            self.session_pool.append(session)

    def _get_next_proxy(self) -> str | None:
        """Get next proxy from rotation list."""
        if not self.proxy_list:
            return None

        proxy = self.proxy_list[self.current_proxy_index]
        self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxy_list)
        self.connection_stats["proxy_rotations"] += 1

        return proxy

    def _get_next_user_agent(self) -> str:
        """Get next user agent from rotation list."""
        user_agent = self.user_agents[self.current_user_agent_index]
        self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.user_agents)

        return user_agent

    def _get_session(self) -> requests.Session:
        """Get session from pool with rotation."""
        if self.request_count % self.session_rotation_interval == 0:
            self.connection_stats["session_rotations"] += 1

        session_index = (self.request_count // self.session_rotation_interval) % len(self.session_pool)
        return self.session_pool[session_index]

    def _apply_traffic_shaping(self) -> None:
        """Apply traffic shaping delays to mimic human behavior."""
        delay = random.uniform(*self.rate_limit_delay)

        # Add additional jitter for more realistic timing
        jitter = random.uniform(-0.2, 0.2) * delay
        total_delay = max(0.1, delay + jitter)

        time.sleep(total_delay)

    def _configure_session_for_request(self, session: requests.Session) -> None:
        """Configure session with current proxy and user agent."""
        # Set user agent
        session.headers.update({
            'User-Agent': self._get_next_user_agent()
        })

        # Set proxy if available
        proxy = self._get_next_proxy()
        if proxy:
            session.proxies = {
                'http': proxy,
                'https': proxy
            }
        else:
            session.proxies = {}

        # Add common headers to look more legitimate
        session.headers.update({
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })

    async def make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> requests.Response | None:
        """
        Make HTTP request with evasion techniques applied.

        Args:
            method: HTTP method
            url: Target URL
            **kwargs: Additional arguments for requests

        Returns:
            Response object or None if failed
        """
        self.request_count += 1
        self.connection_stats["total_requests"] += 1

        # Apply traffic shaping
        self._apply_traffic_shaping()

        # Get session and configure it
        session = self._get_session()
        self._configure_session_for_request(session)

        try:
            response = session.request(method, url, **kwargs)
            response.raise_for_status()

            self.connection_stats["successful_requests"] += 1
            self.logger.debug(f"Request successful: {method} {url}")

            return response

        except requests.exceptions.RequestException as e:
            self.connection_stats["failed_requests"] += 1
            self.connection_stats["retry_attempts"] += 1
            self.logger.warning(f"Request failed: {method} {url} - {e}")

            return None

    async def batch_requests(
        self,
        requests_data: list[dict[str, Any]],
        max_concurrent: int = 3,
        shuffle_order: bool = True
    ) -> list[requests.Response | None]:
        """
        Execute batch of requests with concurrency control and randomization.

        Args:
            requests_data: List of request dictionaries with method, url, and kwargs
            max_concurrent: Maximum concurrent requests
            shuffle_order: Whether to shuffle request order

        Returns:
            List of response objects
        """
        if shuffle_order:
            requests_data = requests_data.copy()
            random.shuffle(requests_data)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def limited_request(request_data: dict[str, Any]) -> requests.Response | None:
            async with semaphore:
                return await self.make_request(**request_data)

        tasks = [limited_request(req_data) for req_data in requests_data]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        results = []
        for response in responses:
            if isinstance(response, Exception):
                self.logger.error(f"Request failed with exception: {response}")
                results.append(None)
            else:
                results.append(response)

        return results

    def create_connection_pool(
        self,
        target_hosts: list[str],
        pool_size: int = 5
    ) -> dict[str, list[requests.Session]]:
        """
        Create connection pools for specific target hosts.

        Args:
            target_hosts: List of target hostnames
            pool_size: Number of sessions per host

        Returns:
            Dictionary mapping hostnames to session pools
        """
        pools = {}

        for host in target_hosts:
            host_pool = []

            for i in range(pool_size):
                session = requests.Session()

                # Configure session for this host
                retry_strategy = Retry(
                    total=self.max_retries,
                    backoff_factor=self.retry_backoff_factor,
                    status_forcelist=[429, 500, 502, 503, 504]
                )

                adapter = HTTPAdapter(
                    max_retries=retry_strategy,
                    pool_connections=10,
                    pool_maxsize=20
                )

                session.mount("http://", adapter)
                session.mount("https://", adapter)
                session.timeout = self.connection_timeout

                host_pool.append(session)

            pools[host] = host_pool
            self.logger.info(f"Created connection pool for {host} with {pool_size} sessions")

        return pools

    def implement_circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0
    ) -> Callable:
        """
        Implement circuit breaker pattern for failing endpoints.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery

        Returns:
            Circuit breaker decorator function
        """
        circuit_state = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"  # closed, open, half-open
        }

        def circuit_breaker(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                now = datetime.utcnow()

                # Check if circuit should be half-open
                if (circuit_state["state"] == "open" and
                    circuit_state["last_failure"] and
                    (now - circuit_state["last_failure"]).total_seconds() > recovery_timeout):
                    circuit_state["state"] = "half-open"
                    self.logger.info("Circuit breaker entering half-open state")

                # Reject requests if circuit is open
                if circuit_state["state"] == "open":
                    self.logger.warning("Circuit breaker is open, rejecting request")
                    return None

                try:
                    result = await func(*args, **kwargs)

                    # Success - reset circuit if it was half-open
                    if circuit_state["state"] == "half-open":
                        circuit_state["state"] = "closed"
                        circuit_state["failures"] = 0
                        self.logger.info("Circuit breaker closed after successful recovery")

                    return result

                except Exception as e:
                    circuit_state["failures"] += 1
                    circuit_state["last_failure"] = now

                    # Open circuit if threshold reached
                    if circuit_state["failures"] >= failure_threshold:
                        circuit_state["state"] = "open"
                        self.logger.warning(f"Circuit breaker opened after {failure_threshold} failures")

                    raise e

            return wrapper
        return circuit_breaker

    def rotate_proxy_list(self, new_proxies: list[str]) -> None:
        """Update proxy list with new proxies."""
        self.proxy_list = new_proxies
        self.current_proxy_index = 0
        self.logger.info(f"Updated proxy list with {len(new_proxies)} proxies")

    def test_proxy_connectivity(self, test_url: str = "http://httpbin.org/ip") -> dict[str, Any]:
        """Test connectivity through all configured proxies."""
        results = {}

        for i, proxy in enumerate(self.proxy_list):
            try:
                session = requests.Session()
                session.proxies = {'http': proxy, 'https': proxy}
                session.timeout = 10

                start_time = time.time()
                response = session.get(test_url)
                end_time = time.time()

                results[proxy] = {
                    "status": "success",
                    "response_time": end_time - start_time,
                    "status_code": response.status_code,
                    "ip": response.json().get("origin", "unknown") if response.status_code == 200 else None
                }

            except Exception as e:
                results[proxy] = {
                    "status": "failed",
                    "error": str(e)
                }

        working_proxies = len([r for r in results.values() if r["status"] == "success"])
        self.logger.info(f"Proxy test complete: {working_proxies}/{len(self.proxy_list)} working")

        return results

    def implement_adaptive_throttling(
        self,
        target_success_rate: float = 0.95,
        adjustment_factor: float = 0.1
    ) -> None:
        """
        Implement adaptive throttling based on success rate.

        Args:
            target_success_rate: Target success rate for requests
            adjustment_factor: Factor for adjusting delays
        """
        if self.connection_stats["total_requests"] < 10:
            return  # Need minimum requests for meaningful statistics

        current_success_rate = (
            self.connection_stats["successful_requests"] /
            self.connection_stats["total_requests"]
        )

        if current_success_rate < target_success_rate:
            # Increase delays to reduce load
            min_delay, max_delay = self.rate_limit_delay
            new_min = min_delay * (1 + adjustment_factor)
            new_max = max_delay * (1 + adjustment_factor)
            self.rate_limit_delay = (new_min, new_max)

            self.logger.info(f"Increased throttling: {self.rate_limit_delay}")

        elif current_success_rate > target_success_rate + 0.05:
            # Decrease delays to increase throughput
            min_delay, max_delay = self.rate_limit_delay
            new_min = max(0.1, min_delay * (1 - adjustment_factor))
            new_max = max(0.5, max_delay * (1 - adjustment_factor))
            self.rate_limit_delay = (new_min, new_max)

            self.logger.info(f"Decreased throttling: {self.rate_limit_delay}")

    def get_network_statistics(self) -> dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = self.connection_stats.copy()

        if stats["total_requests"] > 0:
            stats["success_rate"] = stats["successful_requests"] / stats["total_requests"]
            stats["failure_rate"] = stats["failed_requests"] / stats["total_requests"]
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        stats.update({
            "current_proxy_index": self.current_proxy_index,
            "total_proxies": len(self.proxy_list),
            "current_user_agent_index": self.current_user_agent_index,
            "total_user_agents": len(self.user_agents),
            "session_pool_size": len(self.session_pool),
            "rate_limit_delay": self.rate_limit_delay,
            "session_rotation_interval": self.session_rotation_interval
        })

        return stats

    def cleanup_sessions(self) -> None:
        """Clean up all active sessions."""
        for session in self.session_pool:
            session.close()

        self.session_pool.clear()
        self.logger.info("Cleaned up all network sessions")
