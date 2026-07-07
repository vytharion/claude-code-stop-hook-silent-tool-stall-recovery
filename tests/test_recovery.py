import io
import json
from pathlib import Path

from stall_watch.hook import run
from stall_watch.recovery import (
    RECOVERY_FOOTER,
    RECOVERY_HEADER,
    build_recovery_prompt,
    nudge_for_signature,
)
from stall_watch.simulate import (
    write_empty_result_transcript,
    write_hung_mcp_transcript,
    write_missing_followup_transcript,
    write_stalled_transcript,
)
from stall_watch.transcript import (
    KIND_EMPTY_TOOL_RESULT,
    KIND_HUNG_MCP_CALL,
    KIND_MISSING_FOLLOWUP,
    KIND_PENDING_TOOL_USE,
    StallSignature,
    detect_stalls,
)


def _hook_stdin_payload(transcript_path: Path) -> str:
    return json.dumps(
        {
            "session_id": "recovery-session",
            "transcript_path": str(transcript_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "cwd": str(transcript_path.parent),
        }
    )


def test_nudge_for_pending_tool_use_asks_for_retry() -> None:
    signature = StallSignature(
        kind=KIND_PENDING_TOOL_USE,
        tool_use_id="toolu_pending_1",
        tool_name="Read",
        line_number=3,
        detail="stub",
    )

    text = nudge_for_signature(signature)

    assert "Read" in text
    assert "toolu_pending_1" in text
    assert "line 3" in text
    assert "Retry" in text or "retry" in text


def test_nudge_for_empty_tool_result_asks_for_verbose_rerun() -> None:
    signature = StallSignature(
        kind=KIND_EMPTY_TOOL_RESULT,
        tool_use_id="toolu_empty_1",
        tool_name="Bash",
        line_number=4,
        detail="stub",
    )

    text = nudge_for_signature(signature)

    assert "Bash" in text
    assert "empty" in text
    assert "verbose" in text or "stderr" in text


def test_nudge_for_hung_mcp_call_asks_to_restart_or_fall_back() -> None:
    signature = StallSignature(
        kind=KIND_HUNG_MCP_CALL,
        tool_use_id="toolu_mcp_1",
        tool_name="mcp__example__slow_call",
        line_number=5,
        detail="stub",
    )

    text = nudge_for_signature(signature)

    assert "mcp__example__slow_call" in text
    assert "MCP" in text or "mcp" in text
    assert "Restart" in text or "restart" in text or "fall back" in text


def test_nudge_for_missing_followup_asks_for_summary() -> None:
    signature = StallSignature(
        kind=KIND_MISSING_FOLLOWUP,
        tool_use_id="toolu_missing_1",
        tool_name="Read",
        line_number=6,
        detail="stub",
    )

    text = nudge_for_signature(signature)

    assert "Read" in text
    assert "Summarize" in text or "summarize" in text


def test_nudge_falls_back_to_default_for_unknown_kind() -> None:
    signature = StallSignature(
        kind="totally_new_kind",
        tool_use_id="toolu_x",
        tool_name="Custom",
        line_number=10,
        detail="stub",
    )

    text = nudge_for_signature(signature)

    assert "Custom" in text
    assert "line 10" in text


def test_build_recovery_prompt_is_empty_for_no_signatures() -> None:
    assert build_recovery_prompt([]) == ""


def test_build_recovery_prompt_includes_header_footer_and_one_bullet_per_signature() -> None:
    signatures = [
        StallSignature(
            kind=KIND_PENDING_TOOL_USE,
            tool_use_id="toolu_a",
            tool_name="Read",
            line_number=3,
            detail="",
        ),
        StallSignature(
            kind=KIND_HUNG_MCP_CALL,
            tool_use_id="toolu_b",
            tool_name="mcp__example__slow_call",
            line_number=7,
            detail="",
        ),
    ]

    prompt = build_recovery_prompt(signatures)

    assert prompt.startswith(RECOVERY_HEADER)
    assert prompt.rstrip().endswith(RECOVERY_FOOTER)
    bullet_lines = [line for line in prompt.splitlines() if line.startswith("- [")]
    assert len(bullet_lines) == 2
    assert f"[{KIND_PENDING_TOOL_USE}]" in prompt
    assert f"[{KIND_HUNG_MCP_CALL}]" in prompt


def test_hook_stderr_contains_recovery_prompt_for_pending_tool_use(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript, tool_name="Read")
    signatures = detect_stalls(transcript)
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)
    output = stderr.getvalue()

    assert exit_code == 2
    assert RECOVERY_HEADER in output
    assert RECOVERY_FOOTER in output
    assert f"[{KIND_PENDING_TOOL_USE}]" in output
    assert signatures[0].tool_use_id in output


def test_hook_stderr_contains_recovery_prompt_for_empty_tool_result(tmp_path: Path) -> None:
    transcript = tmp_path / "empty.jsonl"
    write_empty_result_transcript(transcript, tool_name="Bash")
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)
    output = stderr.getvalue()

    assert exit_code == 2
    assert RECOVERY_HEADER in output
    assert f"[{KIND_EMPTY_TOOL_RESULT}]" in output
    assert "verbose" in output or "stderr" in output


def test_hook_stderr_contains_recovery_prompt_for_hung_mcp_call(tmp_path: Path) -> None:
    transcript = tmp_path / "mcp.jsonl"
    write_hung_mcp_transcript(transcript)
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)
    output = stderr.getvalue()

    assert exit_code == 2
    assert RECOVERY_HEADER in output
    assert f"[{KIND_HUNG_MCP_CALL}]" in output
    assert "Restart" in output or "fall back" in output


def test_hook_stderr_contains_recovery_prompt_for_missing_followup(tmp_path: Path) -> None:
    transcript = tmp_path / "followup.jsonl"
    write_missing_followup_transcript(transcript)
    stdin = io.StringIO(_hook_stdin_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr)
    output = stderr.getvalue()

    assert exit_code == 2
    assert RECOVERY_HEADER in output
    assert f"[{KIND_MISSING_FOLLOWUP}]" in output
    assert "Summarize" in output or "summarize" in output
