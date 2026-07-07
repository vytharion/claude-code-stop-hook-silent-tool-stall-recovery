from __future__ import annotations

import json
import uuid
from pathlib import Path


def _fresh_tool_id() -> str:
    return f"toolu_{uuid.uuid4().hex[:16]}"


def _text_event(role: str, text: str) -> dict:
    return {
        "type": role,
        "message": {
            "role": role,
            "content": [{"type": "text", "text": text}],
        },
    }


def _tool_use_event(tool_name: str, tool_input: dict, tool_id: str) -> dict:
    return {
        "type": "assistant",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": tool_input,
                }
            ],
        },
    }


def _tool_result_event(tool_id: str, output: str) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": output,
                }
            ],
        },
    }


def _write_jsonl(path: Path, events: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")


def write_stalled_transcript(path: Path, tool_name: str = "Read") -> str:
    tool_id = _fresh_tool_id()
    events = [
        _text_event("user", "Open the config file and tell me the port."),
        _text_event("assistant", "Reading the config now."),
        _tool_use_event(tool_name, {"file_path": "/tmp/config.yaml"}, tool_id),
    ]
    _write_jsonl(path, events)
    return tool_id


def write_healthy_transcript(path: Path, tool_name: str = "Read") -> str:
    tool_id = _fresh_tool_id()
    events = [
        _text_event("user", "Open the config file and tell me the port."),
        _text_event("assistant", "Reading the config now."),
        _tool_use_event(tool_name, {"file_path": "/tmp/config.yaml"}, tool_id),
        _tool_result_event(tool_id, "port: 8080\n"),
        _text_event("assistant", "The config uses port 8080."),
    ]
    _write_jsonl(path, events)
    return tool_id


def write_mixed_transcript(path: Path) -> tuple[str, str]:
    completed_id = _fresh_tool_id()
    stalled_id = _fresh_tool_id()
    events = [
        _text_event("user", "List the repo, then grep for TODOs."),
        _tool_use_event("Bash", {"command": "ls"}, completed_id),
        _tool_result_event(completed_id, "README.md\npyproject.toml\n"),
        _text_event("assistant", "Now grepping."),
        _tool_use_event("Grep", {"pattern": "TODO"}, stalled_id),
    ]
    _write_jsonl(path, events)
    return completed_id, stalled_id
