from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from stall_watch.transcript import PendingToolCall, find_pending_tool_calls


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


def _scan_transcript(hook_input: StopHookInput) -> list[PendingToolCall]:
    if not hook_input.transcript_path.exists():
        return []
    return find_pending_tool_calls(hook_input.transcript_path)


def _report_stall(
    hook_input: StopHookInput,
    pending: list[PendingToolCall],
    stderr: IO[str],
) -> None:
    first = pending[0]
    stderr.write(
        f"stall_watch: {len(pending)} pending tool_use "
        f"in {hook_input.transcript_path}\n"
    )
    stderr.write(
        f"stall_watch: first stall = {first.tool_name} "
        f"(id={first.tool_use_id}) at line {first.line_number}\n"
    )


def run(stdin: IO[str], stderr: IO[str]) -> int:
    hook_input = parse_stop_hook_input(stdin.read())
    # stop_hook_active means Claude Code is already re-entering after a
    # prior Stop-hook block; returning non-zero again would loop forever.
    if hook_input.stop_hook_active:
        return 0
    pending = _scan_transcript(hook_input)
    if not pending:
        return 0
    _report_stall(hook_input, pending, stderr)
    return 2


def main() -> int:
    return run(sys.stdin, sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
