# Copyright 2025-2026 Jascha Wanger / Tarnover, LLC
# SPDX-License-Identifier: Apache-2.0
#
# This file is part of the VectorSmuggle project.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0

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
