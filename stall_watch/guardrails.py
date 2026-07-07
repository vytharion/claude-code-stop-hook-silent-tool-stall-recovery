from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

from stall_watch.transcript import StallSignature

DEFAULT_MAX_RETRIES = 3
DEFAULT_COOLDOWN_SECONDS = 0.0
DEFAULT_KILL_SWITCH_ENV = "STALL_WATCH_DISABLED"
DEFAULT_STATE_SUBDIR = (".claude", "stall_watch")

_FALSY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


@dataclass(frozen=True)
class GuardrailConfig:
    state_dir: Path
    max_retries: int = DEFAULT_MAX_RETRIES
    cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS
    kill_switch_env: str = DEFAULT_KILL_SWITCH_ENV
    kill_switch_file: Path | None = None

    @classmethod
    def from_env(
        cls,
        cwd: Path,
        environ: Mapping[str, str] | None = None,
    ) -> "GuardrailConfig":
        env = environ if environ is not None else os.environ
        state_dir = Path(
            env.get("STALL_WATCH_STATE_DIR", str(_default_state_dir(cwd)))
        )
        return cls(
            state_dir=state_dir,
            max_retries=_read_int(env, "STALL_WATCH_MAX_RETRIES", DEFAULT_MAX_RETRIES),
            cooldown_seconds=_read_float(
                env, "STALL_WATCH_COOLDOWN_SECONDS", DEFAULT_COOLDOWN_SECONDS
            ),
            kill_switch_env=env.get(
                "STALL_WATCH_KILL_SWITCH_ENV", DEFAULT_KILL_SWITCH_ENV
            ),
            kill_switch_file=_optional_path(env.get("STALL_WATCH_KILL_SWITCH_FILE")),
        )


@dataclass
class SessionState:
    retries: dict[str, int] = field(default_factory=dict)
    last_recovery_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "retries": dict(self.retries),
            "last_recovery_at": self.last_recovery_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        raw_retries = data.get("retries") or {}
        retries: dict[str, int] = {}
        if isinstance(raw_retries, dict):
            retries = {str(k): int(v) for k, v in raw_retries.items()}
        return cls(
            retries=retries,
            last_recovery_at=float(data.get("last_recovery_at") or 0.0),
        )


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: list[StallSignature]
    capped: list[StallSignature]


def _default_state_dir(cwd: Path) -> Path:
    return cwd.joinpath(*DEFAULT_STATE_SUBDIR)


def _optional_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value)


def _read_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _read_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in _FALSY_ENV_VALUES


def _safe_session_id(session_id: str) -> str:
    if not session_id:
        return "unknown"
    safe = "".join(
        ch if (ch.isalnum() or ch in "._-") else "_" for ch in session_id
    )
    return safe or "unknown"


def state_path(config: GuardrailConfig, session_id: str) -> Path:
    return config.state_dir / f"{_safe_session_id(session_id)}.json"


def load_state(config: GuardrailConfig, session_id: str) -> SessionState:
    path = state_path(config, session_id)
    if not path.exists():
        return SessionState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return SessionState()
    if not isinstance(data, dict):
        return SessionState()
    return SessionState.from_dict(data)


def save_state(
    config: GuardrailConfig, session_id: str, state: SessionState
) -> None:
    path = state_path(config, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict()), encoding="utf-8")


def is_kill_switch_active(
    config: GuardrailConfig, environ: Mapping[str, str] | None = None
) -> bool:
    env = environ if environ is not None else os.environ
    if _env_truthy(env.get(config.kill_switch_env)):
        return True
    kill_file = config.kill_switch_file
    if kill_file is not None and kill_file.exists():
        return True
    return False


def is_in_cooldown(
    state: SessionState, config: GuardrailConfig, now: float
) -> bool:
    if config.cooldown_seconds <= 0:
        return False
    if state.last_recovery_at <= 0:
        return False
    return (now - state.last_recovery_at) < config.cooldown_seconds


def partition_signatures(
    signatures: Iterable[StallSignature],
    state: SessionState,
    config: GuardrailConfig,
) -> GuardrailDecision:
    allowed: list[StallSignature] = []
    capped: list[StallSignature] = []
    for signature in signatures:
        key = signature.tool_use_id or ""
        count = state.retries.get(key, 0)
        if count >= config.max_retries:
            capped.append(signature)
            continue
        allowed.append(signature)
    return GuardrailDecision(allowed=allowed, capped=capped)


def record_recovery(
    state: SessionState,
    allowed: Iterable[StallSignature],
    now: float,
) -> SessionState:
    updated = dict(state.retries)
    for signature in allowed:
        key = signature.tool_use_id or ""
        updated[key] = updated.get(key, 0) + 1
    return SessionState(retries=updated, last_recovery_at=now)
