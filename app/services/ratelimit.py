"""In-memory per-visitor rate limiting for the public demo.

Counters live in process memory, so they reset on a new day or when Cloud Run
spins up a fresh instance. That makes the app-level limit "soft" -- the real,
bulletproof cost ceiling is the monthly hard limit set on the OpenAI key itself.
This layer just stops casual abuse and keeps any single visitor bounded.

Two buckets are tracked per IP: questions and document uploads. Both count
toward a single global daily ceiling (every OpenAI-costing action).
"""

import threading
from datetime import date

from app.config import settings

_lock = threading.Lock()
_queries: dict[str, int] = {}
_uploads: dict[str, int] = {}
_global_count = 0
_current_day = ""


def _roll_day() -> None:
    """Reset all counters when the calendar day changes. Caller must hold _lock."""
    global _current_day, _global_count
    today = date.today().isoformat()
    if today != _current_day:
        _current_day = today
        _global_count = 0
        _queries.clear()
        _uploads.clear()


def _consume(bucket: dict[str, int], ip: str, per_ip_limit: int) -> tuple[bool, int]:
    global _global_count
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return False, 0
        used = bucket.get(ip, 0)
        if used >= per_ip_limit:
            return False, 0
        used += 1
        bucket[ip] = used
        _global_count += 1
        return True, max(0, per_ip_limit - used)


def _remaining(bucket: dict[str, int], ip: str, per_ip_limit: int) -> int:
    with _lock:
        _roll_day()
        if _global_count >= settings.global_daily_cap:
            return 0
        return max(0, per_ip_limit - bucket.get(ip, 0))


def consume_quota(ip: str) -> tuple[bool, int]:
    """Spend one free question for this IP. Returns (allowed, remaining_after)."""
    return _consume(_queries, ip, settings.free_queries_per_day)


def remaining_quota(ip: str) -> int:
    """Free questions this IP has left today, without spending one."""
    return _remaining(_queries, ip, settings.free_queries_per_day)


def consume_upload(ip: str) -> tuple[bool, int]:
    """Spend one free upload for this IP. Returns (allowed, remaining_after)."""
    return _consume(_uploads, ip, settings.free_uploads_per_day)


def remaining_uploads(ip: str) -> int:
    """Free uploads this IP has left today, without spending one."""
    return _remaining(_uploads, ip, settings.free_uploads_per_day)
