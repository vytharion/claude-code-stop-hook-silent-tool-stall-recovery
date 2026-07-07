from pathlib import Path

from stall_watch.simulate import (
    write_healthy_transcript,
    write_mixed_transcript,
    write_stalled_transcript,
)
from stall_watch.transcript import find_pending_tool_calls, has_silent_stall


def test_healthy_transcript_has_no_pending_tool_calls(tmp_path: Path) -> None:
    transcript = tmp_path / "healthy.jsonl"
    write_healthy_transcript(transcript)

    assert find_pending_tool_calls(transcript) == []
    assert has_silent_stall(transcript) is False


def test_stalled_transcript_flags_the_unmatched_tool_use(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    stalled_id = write_stalled_transcript(transcript, tool_name="Read")

    pending = find_pending_tool_calls(transcript)

    assert len(pending) == 1
    call = pending[0]
    assert call.tool_use_id == stalled_id
    assert call.tool_name == "Read"
    assert call.line_number == 3
    assert has_silent_stall(transcript) is True


def test_mixed_transcript_only_flags_the_last_tool_use(tmp_path: Path) -> None:
    transcript = tmp_path / "mixed.jsonl"
    completed_id, stalled_id = write_mixed_transcript(transcript)

    pending_ids = {call.tool_use_id for call in find_pending_tool_calls(transcript)}

    assert stalled_id in pending_ids
    assert completed_id not in pending_ids


def test_blank_lines_and_missing_content_do_not_crash(tmp_path: Path) -> None:
    transcript = tmp_path / "noisy.jsonl"
    write_healthy_transcript(transcript)
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write('{"type":"system","message":{"role":"system"}}\n')

    assert has_silent_stall(transcript) is False
