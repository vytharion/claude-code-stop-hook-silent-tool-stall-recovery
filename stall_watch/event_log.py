from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

DEFAULT_LOG_FILE_ENV = "STALL_WATCH_LOG_FILE"

EVENT_STALL_DETECTED = "stall_detected"
EVENT_RECOVERY_DISPATCHED = "recovery_dispatched"
EVENT_RETRY_CAP_HIT = "retry_cap_hit"
EVENT_COOLDOWN_SKIPPED = "cooldown_skipped"
EVENT_KILL_SWITCH_ACTIVE = "kill_switch_active"
EVENT_HEALTHY_STOP = "healthy_stop"


def resolve_log_path(
    environ: Mapping[str, str] | None,
    explicit: Path | None = None,
) -> Path | None:
    if explicit is not None:
        return explicit
    env = environ if environ is not None else os.environ
    raw = env.get(DEFAULT_LOG_FILE_ENV)
    if not raw:
        return None
    return Path(raw)


def log_event(
    log_path: Path | None,
    session_id: str,
    event: str,
    now: float,
    **payload: Any,
) -> None:
    if log_path is None:
        return
    record: dict[str, Any] = {
        "ts": now,
        "event": event,
        "session_id": session_id,
    }
    record.update(payload)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_log(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            entries.append(json.loads(stripped))
    return entries
