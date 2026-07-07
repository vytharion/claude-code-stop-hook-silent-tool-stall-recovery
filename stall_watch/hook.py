from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import IO

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


def run(stdin: IO[str], stderr: IO[str]) -> int:
    hook_input = parse_stop_hook_input(stdin.read())
    # stop_hook_active means Claude Code is already re-entering after a
    # prior Stop-hook block; returning non-zero again would loop forever.
    if hook_input.stop_hook_active:
        return 0
    signatures = _scan_transcript(hook_input)
    if not signatures:
        return 0
    _report_stall(hook_input, signatures, stderr)
    return 2


def main() -> int:
    return run(sys.stdin, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
