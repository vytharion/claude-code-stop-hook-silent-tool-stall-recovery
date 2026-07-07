from pathlib import Path

from stall_watch.simulate import (
    write_empty_result_transcript,
    write_healthy_transcript,
    write_hung_mcp_transcript,
    write_missing_followup_transcript,
    write_stalled_transcript,
)
from stall_watch.transcript import (
    KIND_EMPTY_TOOL_RESULT,
    KIND_HUNG_MCP_CALL,
    KIND_MISSING_FOLLOWUP,
    KIND_PENDING_TOOL_USE,
    detect_stalls,
    has_silent_stall,
)


def test_healthy_transcript_has_no_stall_signatures(tmp_path: Path) -> None:
    transcript = tmp_path / "healthy.jsonl"
    write_healthy_transcript(transcript)

    assert detect_stalls(transcript) == []
    assert has_silent_stall(transcript) is False


def test_missing_tool_result_is_classified_as_pending_tool_use(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    stalled_id = write_stalled_transcript(transcript, tool_name="Read")

    signatures = detect_stalls(transcript)

    assert len(signatures) == 1
    sig = signatures[0]
    assert sig.kind == KIND_PENDING_TOOL_USE
    assert sig.tool_use_id == stalled_id
    assert sig.tool_name == "Read"
    assert sig.line_number == 3
    assert "without a tool_result" in sig.detail


def test_empty_tool_result_is_detected(tmp_path: Path) -> None:
    transcript = tmp_path / "empty.jsonl"
    tool_id = write_empty_result_transcript(transcript, tool_name="Bash")

    signatures = detect_stalls(transcript)
    kinds = [sig.kind for sig in signatures]

    assert KIND_EMPTY_TOOL_RESULT in kinds
    empty = next(sig for sig in signatures if sig.kind == KIND_EMPTY_TOOL_RESULT)
    assert empty.tool_use_id == tool_id
    assert empty.tool_name == "Bash"
    assert "empty tool_result" in empty.detail
    assert has_silent_stall(transcript) is True


def test_hung_mcp_call_is_classified_separately_from_generic_pending(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "mcp.jsonl"
    mcp_id = write_hung_mcp_transcript(transcript)

    signatures = detect_stalls(transcript)

    assert len(signatures) == 1
    sig = signatures[0]
    assert sig.kind == KIND_HUNG_MCP_CALL
    assert sig.tool_use_id == mcp_id
    assert sig.tool_name.startswith("mcp__")


def test_missing_followup_flags_trailing_tool_result(tmp_path: Path) -> None:
    transcript = tmp_path / "missing_followup.jsonl"
    tool_id = write_missing_followup_transcript(transcript)

    signatures = detect_stalls(transcript)

    assert len(signatures) == 1
    sig = signatures[0]
    assert sig.kind == KIND_MISSING_FOLLOWUP
    assert sig.tool_use_id == tool_id
    assert sig.tool_name == "Read"
    assert "no assistant follow-up" in sig.detail


def test_detect_stalls_reports_multiple_kinds_in_order(tmp_path: Path) -> None:
    transcript = tmp_path / "multi.jsonl"
    # Reuse the empty-result fixture as a base then append an unmatched MCP call
    empty_id = write_empty_result_transcript(transcript, tool_name="Bash")

    import json

    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_mcp_multi_1",
                                "name": "mcp__example__slow_call",
                                "input": {},
                            }
                        ],
                    },
                }
            )
            + "\n"
        )

    signatures = detect_stalls(transcript)
    kinds = [sig.kind for sig in signatures]

    assert KIND_EMPTY_TOOL_RESULT in kinds
    assert KIND_HUNG_MCP_CALL in kinds
    empty_sig = next(sig for sig in signatures if sig.kind == KIND_EMPTY_TOOL_RESULT)
    mcp_sig = next(sig for sig in signatures if sig.kind == KIND_HUNG_MCP_CALL)
    assert empty_sig.tool_use_id == empty_id
    assert mcp_sig.line_number > empty_sig.line_number
