"""Pacing for jobs that wait for an out-of-stock GPU.

RunPod publishes no API rate limits, so waiting jobs ask politely: each job
checks once a minute, every waiting job in the app together at most once
every MIN_SPACING_S, and a 429 or a network error backs off up to
MAX_BACKOFF_S (longer if RunPod's Retry-After says so).
"""
from __future__ import annotations

import random
import threading
import time

POLL_S = 60.0
MIN_SPACING_S = 20.0
MAX_BACKOFF_S = 600.0
JITTER = 0.2
# How often a waiting job writes "still waiting" to its log.
NOTE_EVERY_S = 10 * 60

_lock = threading.Lock()
_next_slot = 0.0


def wait_turn(stop: threading.Event) -> bool:
    """Wait until this process may make its next capacity check; False if stopped meanwhile."""
    global _next_slot
    with _lock:
        now = time.monotonic()
        slot = max(now, _next_slot)
        _next_slot = slot + MIN_SPACING_S
    return not stop.wait(slot - now)


def next_delay(delay: float, *, failed: bool, retry_after: float | None = None) -> float:
    """The pause before the next check: the normal poll, or a doubled one after a failure."""
    if not failed:
        return POLL_S
    backoff = min(max(delay, POLL_S) * 2, MAX_BACKOFF_S)
    return max(backoff, retry_after or 0.0)


def jittered(delay: float) -> float:
    """Spread checks out so several waiting jobs do not ask in lockstep."""
    return delay * random.uniform(1 - JITTER, 1 + JITTER)
