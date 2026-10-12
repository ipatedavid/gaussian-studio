from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PodInfo:
    pod_id: str
    name: str
    status: str
    desired_status: str = ""
    gpu_name: str = ""
    cost_per_hour: Optional[float] = None
    recovered_from_error: bool = False
