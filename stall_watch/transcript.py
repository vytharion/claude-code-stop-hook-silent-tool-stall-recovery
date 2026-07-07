from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

# Stall categories reported by detect_stalls. The plain-string form is used
# directly in log output — see stall_watch.hook.KIND_LABEL for the human label.
KIND_PENDING_TOOL_USE = "pending_tool_use"
KIND_EMPTY_TOOL_RESULT = "empty_tool_result"
KIND_HUNG_MCP_CALL = "hung_mcp_call"
KIND_MISSING_FOLLOWUP = "missing_followup"


@dataclass(frozen=True)
class PendingToolCall:
    tool_use_id: str
    tool_name: str
    line_number: int


@dataclass(frozen=True)
class StallSignature:
    kind: str
    tool_use_id: str
    tool_name: str
    line_number: int
    detail: str


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


def _is_empty_result_content(content: Any) -> bool:
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        return not any(_result_block_has_text(block) for block in content)
    return False


def _result_block_has_text(block: Any) -> bool:
    if not isinstance(block, dict):
        return False
    text = block.get("text")
    if isinstance(text, str) and text.strip():
        return True
    nested = block.get("content")
    if isinstance(nested, str) and nested.strip():
        return True
    return False


def _classify_unmatched(tool_name: str) -> str:
    if tool_name.startswith("mcp__"):
        return KIND_HUNG_MCP_CALL
    return KIND_PENDING_TOOL_USE


def _scan_tool_events(
    events: list[tuple[int, dict]],
) -> tuple[dict[str, tuple[int, str]], dict[str, tuple[int, Any]]]:
    tool_uses: dict[str, tuple[int, str]] = {}
    tool_results: dict[str, tuple[int, Any]] = {}
    for line_number, event in events:
        for block in _content_blocks(event):
            block_type = block.get("type")
            if block_type == "tool_use":
                _capture_tool_use(tool_uses, block, line_number)
            elif block_type == "tool_result":
                _capture_tool_result(tool_results, block, line_number)
    return tool_uses, tool_results


def _capture_tool_use(
    tool_uses: dict[str, tuple[int, str]], block: dict, line_number: int
) -> None:
    tool_id = block.get("id")
    if not tool_id:
        return
    tool_uses[tool_id] = (line_number, block.get("name", ""))


def _capture_tool_result(
    tool_results: dict[str, tuple[int, Any]], block: dict, line_number: int
) -> None:
    tool_id = block.get("tool_use_id")
    if not tool_id:
        return
    tool_results[tool_id] = (line_number, block.get("content"))


def _signature_for_unmatched(
    tool_id: str, line_number: int, tool_name: str
) -> StallSignature:
    kind = _classify_unmatched(tool_name)
    detail = f"{tool_name or 'tool'} started at line {line_number} without a tool_result"
    return StallSignature(
        kind=kind,
        tool_use_id=tool_id,
        tool_name=tool_name,
        line_number=line_number,
        detail=detail,
    )


def _signature_for_empty_result(
    tool_id: str, tool_name: str, result_line: int
) -> StallSignature:
    detail = (
        f"{tool_name or 'tool'} returned empty tool_result at line {result_line}"
    )
    return StallSignature(
        kind=KIND_EMPTY_TOOL_RESULT,
        tool_use_id=tool_id,
        tool_name=tool_name,
        line_number=result_line,
        detail=detail,
    )


def _event_is_only_tool_result(event: dict) -> tuple[bool, str]:
    blocks = list(_content_blocks(event))
    if not blocks:
        return False, ""
    trailing_id = ""
    for block in blocks:
        if block.get("type") != "tool_result":
            return False, ""
        candidate = block.get("tool_use_id")
        if isinstance(candidate, str) and candidate:
            trailing_id = candidate
    return True, trailing_id


def _missing_followup_signature(
    events: list[tuple[int, dict]], tool_uses: dict[str, tuple[int, str]]
) -> StallSignature | None:
    if not events:
        return None
    last_line, last_event = events[-1]
    is_trailing_result, tool_id = _event_is_only_tool_result(last_event)
    if not is_trailing_result:
        return None
    tool_name = tool_uses.get(tool_id, (0, ""))[1] if tool_id else ""
    detail = (
        f"tool_result for {tool_name or 'tool'} at line {last_line} "
        "had no assistant follow-up"
    )
    return StallSignature(
        kind=KIND_MISSING_FOLLOWUP,
        tool_use_id=tool_id,
        tool_name=tool_name,
        line_number=last_line,
        detail=detail,
    )


def detect_stalls(transcript_path: Path) -> list[StallSignature]:
    events = list(_iter_events(transcript_path))
    tool_uses, tool_results = _scan_tool_events(events)
    signatures: list[StallSignature] = []

    for tool_id, (line_number, tool_name) in tool_uses.items():
        if tool_id not in tool_results:
            signatures.append(
                _signature_for_unmatched(tool_id, line_number, tool_name)
            )
            continue
        result_line, result_content = tool_results[tool_id]
        if _is_empty_result_content(result_content):
            signatures.append(
                _signature_for_empty_result(tool_id, tool_name, result_line)
            )

    trailing = _missing_followup_signature(events, tool_uses)
    if trailing is not None:
        signatures.append(trailing)

    signatures.sort(key=lambda sig: (sig.line_number, sig.kind))
    return signatures


def has_silent_stall(transcript_path: Path) -> bool:
    return bool(detect_stalls(transcript_path))
