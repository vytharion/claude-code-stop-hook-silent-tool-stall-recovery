from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class PendingToolCall:
    tool_use_id: str
    tool_name: str
    line_number: int


def _iter_events(transcript_path: Path) -> Iterator[tuple[int, dict]]:
    with transcript_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            yield index, json.loads(stripped)


def _content_blocks(event: dict) -> Iterable[dict]:
    payload = event.get("message") if isinstance(event.get("message"), dict) else event
    content = payload.get("content")
    if isinstance(content, list):
        return content
    return ()


def _record_tool_use(pending: dict, block: dict, line_number: int) -> None:
    tool_id = block.get("id")
    if not tool_id:
        return
    pending[tool_id] = PendingToolCall(
        tool_use_id=tool_id,
        tool_name=block.get("name", ""),
        line_number=line_number,
    )


def _clear_tool_result(pending: dict, block: dict) -> None:
    tool_id = block.get("tool_use_id")
    if not tool_id:
        return
    pending.pop(tool_id, None)


def find_pending_tool_calls(transcript_path: Path) -> list[PendingToolCall]:
    pending: dict[str, PendingToolCall] = {}
    for line_number, event in _iter_events(transcript_path):
        for block in _content_blocks(event):
            block_type = block.get("type")
            if block_type == "tool_use":
                _record_tool_use(pending, block, line_number)
            elif block_type == "tool_result":
                _clear_tool_result(pending, block)
    return list(pending.values())


def has_silent_stall(transcript_path: Path) -> bool:
    return bool(find_pending_tool_calls(transcript_path))
