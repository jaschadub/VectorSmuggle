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
