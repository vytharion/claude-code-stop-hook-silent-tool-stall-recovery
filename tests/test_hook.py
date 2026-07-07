import io
import json
from pathlib import Path

from stall_watch.hook import parse_stop_hook_input, run
from stall_watch.simulate import (
    write_healthy_transcript,
    write_stalled_transcript,
)


def _hook_stdin_payload(transcript_path: Path, stop_hook_active: bool = False) -> str:
    return json.dumps(
        {
            "session_id": "test-session",
            "transcript_path": str(transcript_path),
            "hook_event_name": "Stop",
            "stop_hook_active": stop_hook_active,
            "cwd": str(transcript_path.parent),
        }
    )


def test_parse_stop_hook_input_extracts_expected_fields(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    parsed = parse_stop_hook_input(_hook_stdin_payload(transcript))

    assert parsed.session_id == "test-session"
    assert parsed.transcript_path == transcript
    assert parsed.hook_event_name == "Stop"
    assert parsed.stop_hook_active is False
    assert parsed.cwd == tmp_path


def test_run_returns_zero_on_healthy_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "healthy.jsonl"
    write_healthy_transcript(transcript)
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""


def test_run_returns_two_and_reports_when_stalled(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    stalled_id = write_stalled_transcript(transcript, tool_name="Read")
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)

    assert exit_code == 2
    message = stderr.getvalue()
    assert "Read" in message
    assert stalled_id in message
    assert "line 3" in message
    assert "1 pending tool_use" in message


def test_run_returns_zero_when_stop_hook_active_flag_is_true(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    stdin = io.StringIO(_hook_stdin_payload(transcript, stop_hook_active=True))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""


def test_run_treats_missing_transcript_as_no_stall(tmp_path: Path) -> None:
    missing = tmp_path / "not-yet-written.jsonl"
    stdin = io.StringIO(_hook_stdin_payload(missing))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)

    assert exit_code == 0
    assert stderr.getvalue() == ""
