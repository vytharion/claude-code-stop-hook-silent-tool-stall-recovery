import io
import json
import subprocess
import sys
from pathlib import Path

from stall_watch.event_log import (
    EVENT_COOLDOWN_SKIPPED,
    EVENT_HEALTHY_STOP,
    EVENT_KILL_SWITCH_ACTIVE,
    EVENT_RECOVERY_DISPATCHED,
    EVENT_RETRY_CAP_HIT,
    EVENT_STALL_DETECTED,
    read_log,
    resolve_log_path,
)
from stall_watch.hook import run
from stall_watch.simulate import (
    append_recovery,
    write_stalled_transcript,
)


def _payload(transcript_path: Path, session_id: str = "smoke-session") -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": str(transcript_path),
            "hook_event_name": "Stop",
            "stop_hook_active": False,
            "cwd": str(transcript_path.parent),
        }
    )


def _events(log_path: Path) -> list[str]:
    return [entry["event"] for entry in read_log(log_path)]


def test_resolve_log_path_prefers_explicit_over_env(tmp_path: Path) -> None:
    env_path = tmp_path / "env.jsonl"
    explicit = tmp_path / "explicit.jsonl"

    assert (
        resolve_log_path(environ={"STALL_WATCH_LOG_FILE": str(env_path)})
        == env_path
    )
    assert (
        resolve_log_path(
            environ={"STALL_WATCH_LOG_FILE": str(env_path)},
            explicit=explicit,
        )
        == explicit
    )
    assert resolve_log_path(environ={}) is None


def test_log_event_is_a_noop_when_path_is_none() -> None:
    # Just verifying we can call this without crashing when logging is disabled.
    from stall_watch.event_log import log_event

    log_event(None, "sid", "some_event", now=1.0, foo="bar")


def test_smoke_full_stall_and_recover_cycle(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    tool_id = write_stalled_transcript(transcript, tool_name="Read")
    log_path = tmp_path / "stall_watch.log"

    # 1) First stop with a silent stall -> hook fires recovery.
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    first = run(
        stdin,
        stderr,
        environ={"STALL_WATCH_STATE_DIR": str(tmp_path / "state")},
        now=1000.0,
        log_path=log_path,
    )
    first_stderr = stderr.getvalue()

    assert first == 2
    assert "stall_watch: 1 pending tool_use" in first_stderr
    assert tool_id in first_stderr

    # 2) The agent (simulated) writes the missing tool_result + follow-up.
    append_recovery(transcript, tool_id)

    # 3) Stop again -> transcript is now healthy, hook returns 0 quietly.
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    second = run(
        stdin,
        stderr,
        environ={"STALL_WATCH_STATE_DIR": str(tmp_path / "state")},
        now=1001.0,
        log_path=log_path,
    )

    assert second == 0
    assert stderr.getvalue() == ""

    events = _events(log_path)
    assert events == [
        EVENT_STALL_DETECTED,
        EVENT_RECOVERY_DISPATCHED,
        EVENT_HEALTHY_STOP,
    ]

    entries = read_log(log_path)
    dispatched = entries[1]
    assert dispatched["session_id"] == "smoke-session"
    assert dispatched["allowed"][0]["tool_use_id"] == tool_id
    assert dispatched["retries_after"][tool_id] == 1

    healthy = entries[2]
    assert healthy["transcript"] == str(transcript)


def test_smoke_kill_switch_logged_and_skips_recovery(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    write_stalled_transcript(transcript)
    log_path = tmp_path / "stall_watch.log"

    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    code = run(
        stdin,
        stderr,
        environ={
            "STALL_WATCH_STATE_DIR": str(tmp_path / "state"),
            "STALL_WATCH_DISABLED": "1",
        },
        now=100.0,
        log_path=log_path,
    )

    assert code == 0
    assert "kill switch" in stderr.getvalue()
    events = _events(log_path)
    assert events == [EVENT_STALL_DETECTED, EVENT_KILL_SWITCH_ACTIVE]


def test_smoke_retry_cap_and_cooldown_leave_a_trail(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    write_stalled_transcript(transcript)
    log_path = tmp_path / "stall_watch.log"
    env = {
        "STALL_WATCH_STATE_DIR": str(tmp_path / "state"),
        "STALL_WATCH_MAX_RETRIES": "1",
        "STALL_WATCH_COOLDOWN_SECONDS": "60",
    }

    # First stop -> recovery dispatched, retry counter now 1, cooldown starts.
    first = run(
        io.StringIO(_payload(transcript)),
        io.StringIO(),
        environ=env,
        now=1.0,
        log_path=log_path,
    )

    # Second stop within cooldown -> skipped by cooldown branch.
    second_err = io.StringIO()
    second = run(
        io.StringIO(_payload(transcript)),
        second_err,
        environ=env,
        now=10.0,
        log_path=log_path,
    )

    # Third stop after cooldown expires but max_retries=1 already spent -> exhausted.
    third_err = io.StringIO()
    third = run(
        io.StringIO(_payload(transcript)),
        third_err,
        environ=env,
        now=200.0,
        log_path=log_path,
    )

    assert (first, second, third) == (2, 0, 0)
    assert "cooldown" in second_err.getvalue()
    assert "retry cap" in third_err.getvalue()

    events = _events(log_path)
    assert events == [
        EVENT_STALL_DETECTED,
        EVENT_RECOVERY_DISPATCHED,
        EVENT_STALL_DETECTED,
        EVENT_COOLDOWN_SKIPPED,
        EVENT_STALL_DETECTED,
        EVENT_RETRY_CAP_HIT,
    ]


def test_smoke_runs_from_shell_via_python_m(tmp_path: Path) -> None:
    """Exercise the hook exactly like Claude Code will: subprocess + stdin JSON."""
    transcript = tmp_path / "session.jsonl"
    write_stalled_transcript(transcript, tool_name="Read")
    log_path = tmp_path / "stall_watch.log"

    payload = _payload(transcript, session_id="subprocess-smoke")
    env = {
        "PATH": Path(sys.executable).parent.as_posix(),
        "STALL_WATCH_STATE_DIR": str(tmp_path / "state"),
        "STALL_WATCH_LOG_FILE": str(log_path),
        "STALL_WATCH_DISABLED": "",
    }
    repo_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-m", "stall_watch.hook"],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(repo_root),
        timeout=15,
    )

    assert result.returncode == 2, result.stderr
    assert "stall_watch:" in result.stderr
    assert "pending tool_use" in result.stderr

    events = _events(log_path)
    assert events == [EVENT_STALL_DETECTED, EVENT_RECOVERY_DISPATCHED]
