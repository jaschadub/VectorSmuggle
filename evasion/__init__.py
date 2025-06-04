# Copyright (c) 2025 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: MIT
#
# This file is part of the VectorSmuggle project.
# You may obtain a copy of the license at https://opensource.org/licenses/MIT

"""Advanced evasion techniques for VectorSmuggle."""

from .behavioral_camouflage import BehavioralCamouflage
from .detection_avoidance import DetectionAvoidance
from .network_evasion import NetworkEvasion
from .opsec import OperationalSecurity
from .traffic_mimicry import TrafficMimicry

__all__ = [
    "TrafficMimicry",
    "BehavioralCamouflage",
    "NetworkEvasion",
    "OperationalSecurity",
    "DetectionAvoidance"
]
