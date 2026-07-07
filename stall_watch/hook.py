from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping

from stall_watch.guardrails import (
    GuardrailConfig,
    SessionState,
    is_in_cooldown,
    is_kill_switch_active,
    load_state,
    partition_signatures,
    record_recovery,
    save_state,
)
from stall_watch.recovery import build_recovery_prompt
from stall_watch.transcript import (
    KIND_EMPTY_TOOL_RESULT,
    KIND_HUNG_MCP_CALL,
    KIND_MISSING_FOLLOWUP,
    KIND_PENDING_TOOL_USE,
    StallSignature,
    detect_stalls,
)

KIND_LABEL = {
    KIND_PENDING_TOOL_USE: "pending tool_use",
    KIND_EMPTY_TOOL_RESULT: "empty tool_result",
    KIND_HUNG_MCP_CALL: "hung mcp call",
    KIND_MISSING_FOLLOWUP: "missing follow-up",
}


@dataclass(frozen=True)
class StopHookInput:
    session_id: str
    transcript_path: Path
    hook_event_name: str
    stop_hook_active: bool
    cwd: Path


def parse_stop_hook_input(raw: str) -> StopHookInput:
    payload = json.loads(raw)
    return StopHookInput(
        session_id=str(payload.get("session_id", "")),
        transcript_path=Path(payload["transcript_path"]),
        hook_event_name=str(payload.get("hook_event_name", "Stop")),
        stop_hook_active=bool(payload.get("stop_hook_active", False)),
        cwd=Path(payload.get("cwd", ".")),
    )


def _scan_transcript(hook_input: StopHookInput) -> list[StallSignature]:
    if not hook_input.transcript_path.exists():
        return []
    return detect_stalls(hook_input.transcript_path)


def _label(kind: str) -> str:
    return KIND_LABEL.get(kind, kind.replace("_", " "))


def _report_stall(
    hook_input: StopHookInput,
    signatures: list[StallSignature],
    stderr: IO[str],
) -> None:
    counts = Counter(sig.kind for sig in signatures)
    for kind, count in counts.items():
        stderr.write(
            f"stall_watch: {count} {_label(kind)} "
            f"in {hook_input.transcript_path}\n"
        )
    first = signatures[0]
    stderr.write(
        f"stall_watch: first stall = {first.tool_name} "
        f"(id={first.tool_use_id}) at line {first.line_number} "
        f"[{_label(first.kind)}]\n"
    )
    stderr.write(build_recovery_prompt(signatures))


def _report_exhaustion(
    capped: list[StallSignature],
    stderr: IO[str],
    config: GuardrailConfig,
) -> None:
    stderr.write(
        f"stall_watch: retry cap ({config.max_retries}) reached for "
        f"{len(capped)} stall(s); giving up so the agent can stop\n"
    )
    for signature in capped:
        stderr.write(
            f"stall_watch: exhausted {signature.tool_name} "
            f"(id={signature.tool_use_id}) at line {signature.line_number} "
            f"[{_label(signature.kind)}]\n"
        )


def _report_kill_switch(stderr: IO[str], config: GuardrailConfig) -> None:
    stderr.write(
        "stall_watch: kill switch active "
        f"(env {config.kill_switch_env} or file {config.kill_switch_file}); "
        "skipping recovery\n"
    )


def _report_cooldown(
    stderr: IO[str], state: SessionState, config: GuardrailConfig, now: float
) -> None:
    remaining = config.cooldown_seconds - (now - state.last_recovery_at)
    stderr.write(
        f"stall_watch: cooldown active ({remaining:.1f}s left of "
        f"{config.cooldown_seconds:.1f}s); skipping recovery\n"
    )


def _dispatch(
    hook_input: StopHookInput,
    signatures: list[StallSignature],
    config: GuardrailConfig,
    environ: Mapping[str, str] | None,
    now: float,
    stderr: IO[str],
) -> int:
    if is_kill_switch_active(config, environ):
        _report_kill_switch(stderr, config)
        return 0
    state = load_state(config, hook_input.session_id)
    if is_in_cooldown(state, config, now):
        _report_cooldown(stderr, state, config, now)
        return 0
    decision = partition_signatures(signatures, state, config)
    if not decision.allowed:
        _report_exhaustion(decision.capped, stderr, config)
        return 0
    _report_stall(hook_input, decision.allowed, stderr)
    new_state = record_recovery(state, decision.allowed, now)
    save_state(config, hook_input.session_id, new_state)
    return 2


def run(
    stdin: IO[str],
    stderr: IO[str],
    config: GuardrailConfig | None = None,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
) -> int:
    hook_input = parse_stop_hook_input(stdin.read())
    # stop_hook_active means Claude Code is already re-entering after a
    # prior Stop-hook block; returning non-zero again would loop forever.
    if hook_input.stop_hook_active:
        return 0
    signatures = _scan_transcript(hook_input)
    if not signatures:
        return 0
    effective_config = config or GuardrailConfig.from_env(
        hook_input.cwd, environ=environ
    )
    effective_now = now if now is not None else time.time()
    return _dispatch(
        hook_input=hook_input,
        signatures=signatures,
        config=effective_config,
        environ=environ,
        now=effective_now,
        stderr=stderr,
    )


def main() -> int:
    return run(sys.stdin, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
