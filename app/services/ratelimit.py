"""In-memory per-visitor rate limiting for the public demo.

Counters live in process memory, so they reset on a new day or when Cloud Run
spins up a fresh instance. That makes the app-level limit "soft" -- the real,
bulletproof cost ceiling is the monthly hard limit set on the OpenAI key itself.
This layer just stops casual abuse and keeps any single visitor bounded.
"""

import threading
from datetime import date

from app.config import settings

_lock = threading.Lock()
_per_ip: dict[str, int] = {}
_global_count = 0
_current_day = ""


def _roll_day() -> None:
    """Reset all counters when the calendar day changes. Caller must hold _lock."""
    global _current_day, _global_count
    today = date.today().isoformat()
    if today != _current_day:
        _current_day = today
        _global_count = 0
        _per_ip.clear()


def consume_quota(ip: str) -> tuple[bool, int]:
    """Try to spend one free question for this IP.

    Returns (allowed, remaining_after). When not allowed, remaining is 0.
    """
    global _global_count
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return False, 0
        used = _per_ip.get(ip, 0)
        if used >= settings.free_queries_per_day:
            return False, 0
        used += 1
        _per_ip[ip] = used
        _global_count += 1
        return True, max(0, settings.free_queries_per_day - used)


def remaining_quota(ip: str) -> int:
    """Free questions this IP has left today, without spending one."""
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return 0
        return max(0, settings.free_queries_per_day - _per_ip.get(ip, 0))
