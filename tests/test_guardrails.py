import io
import json
from pathlib import Path

from stall_watch.guardrails import (
    DEFAULT_COOLDOWN_SECONDS,
    DEFAULT_KILL_SWITCH_ENV,
    DEFAULT_MAX_RETRIES,
    GuardrailConfig,
    SessionState,
    is_in_cooldown,
    is_kill_switch_active,
    load_state,
    partition_signatures,
    record_recovery,
    save_state,
    state_path,
)
from stall_watch.hook import run
from stall_watch.simulate import (
    write_healthy_transcript,
    write_stalled_transcript,
)
from stall_watch.transcript import (
    KIND_PENDING_TOOL_USE,
    StallSignature,
)


def _payload(
    transcript_path: Path,
    session_id: str = "session-alpha",
    stop_hook_active: bool = False,
) -> str:
    return json.dumps(
        {
            "session_id": session_id,
            "transcript_path": str(transcript_path),
            "hook_event_name": "Stop",
            "stop_hook_active": stop_hook_active,
            "cwd": str(transcript_path.parent),
        }
    )


def _config(tmp_path: Path, **overrides) -> GuardrailConfig:
    base: dict = {
        "state_dir": tmp_path / "state",
        "max_retries": 3,
        "cooldown_seconds": 0.0,
        "kill_switch_env": "STALL_WATCH_TEST_OFF",
        "kill_switch_file": None,
    }
    base.update(overrides)
    return GuardrailConfig(**base)


def _sig(tool_id: str, tool_name: str = "Read", line: int = 1) -> StallSignature:
    return StallSignature(
        kind=KIND_PENDING_TOOL_USE,
        tool_use_id=tool_id,
        tool_name=tool_name,
        line_number=line,
        detail="",
    )


def test_defaults_match_documented_constants() -> None:
    assert DEFAULT_MAX_RETRIES == 3
    assert DEFAULT_COOLDOWN_SECONDS == 0.0
    assert DEFAULT_KILL_SWITCH_ENV == "STALL_WATCH_DISABLED"


def test_from_env_uses_defaults_when_no_vars_set(tmp_path: Path) -> None:
    config = GuardrailConfig.from_env(tmp_path, environ={})

    assert config.state_dir == tmp_path / ".claude" / "stall_watch"
    assert config.max_retries == DEFAULT_MAX_RETRIES
    assert config.cooldown_seconds == DEFAULT_COOLDOWN_SECONDS
    assert config.kill_switch_env == DEFAULT_KILL_SWITCH_ENV
    assert config.kill_switch_file is None


def test_from_env_parses_overrides(tmp_path: Path) -> None:
    env = {
        "STALL_WATCH_STATE_DIR": str(tmp_path / "custom"),
        "STALL_WATCH_MAX_RETRIES": "5",
        "STALL_WATCH_COOLDOWN_SECONDS": "12.5",
        "STALL_WATCH_KILL_SWITCH_ENV": "MY_OFF",
        "STALL_WATCH_KILL_SWITCH_FILE": str(tmp_path / "OFF"),
    }
    config = GuardrailConfig.from_env(tmp_path, environ=env)

    assert config.state_dir == tmp_path / "custom"
    assert config.max_retries == 5
    assert config.cooldown_seconds == 12.5
    assert config.kill_switch_env == "MY_OFF"
    assert config.kill_switch_file == tmp_path / "OFF"


def test_state_path_sanitizes_unsafe_session_ids(tmp_path: Path) -> None:
    config = _config(tmp_path)

    sanitized = state_path(config, "abc/../etc").name
    assert "/" not in sanitized
    assert sanitized == "abc_.._etc.json"
    assert state_path(config, "").name == "unknown.json"


def test_load_state_returns_empty_for_missing_file(tmp_path: Path) -> None:
    config = _config(tmp_path)

    state = load_state(config, "no-such-session")

    assert state.retries == {}
    assert state.last_recovery_at == 0.0


def test_load_state_recovers_from_corrupt_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = state_path(config, "corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json", encoding="utf-8")

    state = load_state(config, "corrupt")

    assert state.retries == {}
    assert state.last_recovery_at == 0.0


def test_save_and_load_state_round_trip(tmp_path: Path) -> None:
    config = _config(tmp_path)
    original = SessionState(retries={"tool_a": 3}, last_recovery_at=42.5)

    save_state(config, "roundtrip", original)
    loaded = load_state(config, "roundtrip")

    assert loaded.retries == {"tool_a": 3}
    assert loaded.last_recovery_at == 42.5


def test_is_kill_switch_active_reads_env_and_file(tmp_path: Path) -> None:
    kill_file = tmp_path / "off"
    config = _config(tmp_path, kill_switch_env="MY_OFF", kill_switch_file=kill_file)

    assert is_kill_switch_active(config, environ={}) is False
    assert is_kill_switch_active(config, environ={"MY_OFF": "1"}) is True
    assert is_kill_switch_active(config, environ={"MY_OFF": "true"}) is True
    assert is_kill_switch_active(config, environ={"MY_OFF": "0"}) is False
    assert is_kill_switch_active(config, environ={"MY_OFF": "false"}) is False

    kill_file.write_text("off")
    assert is_kill_switch_active(config, environ={}) is True


def test_is_in_cooldown_respects_zero_window(tmp_path: Path) -> None:
    config = _config(tmp_path, cooldown_seconds=0.0)
    state = SessionState(retries={}, last_recovery_at=100.0)

    assert is_in_cooldown(state, config, now=100.0) is False
    assert is_in_cooldown(state, config, now=200.0) is False


def test_is_in_cooldown_true_when_within_window(tmp_path: Path) -> None:
    config = _config(tmp_path, cooldown_seconds=60.0)
    state = SessionState(retries={}, last_recovery_at=100.0)

    assert is_in_cooldown(state, config, now=110.0) is True
    assert is_in_cooldown(state, config, now=160.0) is False
    assert is_in_cooldown(state, config, now=161.0) is False


def test_partition_signatures_splits_capped_and_allowed(tmp_path: Path) -> None:
    config = _config(tmp_path, max_retries=2)
    state = SessionState(retries={"tool_a": 2, "tool_b": 1})
    sigs = [_sig("tool_a"), _sig("tool_b"), _sig("tool_c")]

    decision = partition_signatures(sigs, state, config)

    assert [s.tool_use_id for s in decision.allowed] == ["tool_b", "tool_c"]
    assert [s.tool_use_id for s in decision.capped] == ["tool_a"]


def test_record_recovery_increments_counters_and_stamps_time(tmp_path: Path) -> None:
    state = SessionState(retries={"tool_a": 1})
    sigs = [_sig("tool_a"), _sig("tool_b")]

    updated = record_recovery(state, sigs, now=999.0)

    assert updated.retries == {"tool_a": 2, "tool_b": 1}
    assert updated.last_recovery_at == 999.0


def test_run_kill_switch_env_short_circuits(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    config = _config(tmp_path, kill_switch_env="STALL_WATCH_TEST_OFF")
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(
        stdin,
        stderr,
        config=config,
        environ={"STALL_WATCH_TEST_OFF": "1"},
        now=1.0,
    )

    assert exit_code == 0
    assert "kill switch" in stderr.getvalue()
    assert not state_path(config, "session-alpha").exists()


def test_run_kill_switch_file_short_circuits(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    off_file = tmp_path / "OFF"
    off_file.write_text("disabled")
    config = _config(tmp_path, kill_switch_file=off_file)
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr, config=config, environ={}, now=1.0)

    assert exit_code == 0
    assert "kill switch" in stderr.getvalue()


def test_run_first_recovery_writes_state_and_returns_two(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    tool_id = write_stalled_transcript(transcript)
    config = _config(tmp_path)
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr, config=config, environ={}, now=1000.0)

    assert exit_code == 2
    persisted = load_state(config, "session-alpha")
    assert persisted.retries.get(tool_id) == 1
    assert persisted.last_recovery_at == 1000.0


def test_run_retry_cap_exhausts_after_max_retries(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    tool_id = write_stalled_transcript(transcript)
    config = _config(tmp_path, max_retries=2)

    def _one_call(clock: float) -> tuple[int, str]:
        stdin = io.StringIO(_payload(transcript))
        stderr = io.StringIO()
        code = run(stdin, stderr, config=config, environ={}, now=clock)
        return code, stderr.getvalue()

    first_code, _ = _one_call(1.0)
    second_code, _ = _one_call(2.0)
    third_code, third_msg = _one_call(3.0)

    assert first_code == 2
    assert second_code == 2
    assert third_code == 0
    assert "retry cap" in third_msg
    assert tool_id in third_msg
    assert "exhausted" in third_msg


def test_run_cooldown_window_suppresses_repeat(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    config = _config(tmp_path, cooldown_seconds=60.0)

    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    assert run(stdin, stderr, config=config, environ={}, now=100.0) == 2

    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    exit_code = run(stdin, stderr, config=config, environ={}, now=110.0)

    assert exit_code == 0
    assert "cooldown" in stderr.getvalue()


def test_run_cooldown_lifts_after_window_expires(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    config = _config(tmp_path, cooldown_seconds=60.0, max_retries=5)

    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    assert run(stdin, stderr, config=config, environ={}, now=100.0) == 2

    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()
    exit_code = run(stdin, stderr, config=config, environ={}, now=200.0)

    assert exit_code == 2


def test_run_healthy_transcript_still_returns_zero_under_guardrails(
    tmp_path: Path,
) -> None:
    transcript = tmp_path / "healthy.jsonl"
    write_healthy_transcript(transcript)
    config = _config(tmp_path)
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr, config=config, environ={}, now=0.0)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert not state_path(config, "session-alpha").exists()


def test_run_uses_from_env_when_config_is_none(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    env = {
        "STALL_WATCH_STATE_DIR": str(tmp_path / "envstate"),
        "STALL_WATCH_MAX_RETRIES": "1",
    }
    stdin = io.StringIO(_payload(transcript))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr, environ=env, now=5.0)

    assert exit_code == 2
    persisted = (tmp_path / "envstate" / "session-alpha.json").read_text()
    assert '"last_recovery_at": 5.0' in persisted


def test_run_stop_hook_active_short_circuits_before_guardrails(tmp_path: Path) -> None:
    transcript = tmp_path / "stalled.jsonl"
    write_stalled_transcript(transcript)
    config = _config(tmp_path)
    stdin = io.StringIO(_payload(transcript, stop_hook_active=True))
    stderr = io.StringIO()

    exit_code = run(stdin, stderr, config=config, environ={}, now=1.0)

    assert exit_code == 0
    assert stderr.getvalue() == ""
    assert not state_path(config, "session-alpha").exists()
