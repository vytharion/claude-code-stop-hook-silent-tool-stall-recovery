from __future__ import annotations

from stall_watch.transcript import (
    KIND_EMPTY_TOOL_RESULT,
    KIND_HUNG_MCP_CALL,
    KIND_MISSING_FOLLOWUP,
    KIND_PENDING_TOOL_USE,
    StallSignature,
)

RECOVERY_HEADER = (
    "stall_watch recovery: a silent tool stall was detected before you "
    "tried to stop. Do NOT end the turn yet — act on the guidance below."
)

RECOVERY_FOOTER = (
    "Take the smallest concrete action that unblocks the task, then produce "
    "the assistant follow-up the user was waiting for. Only stop after that "
    "follow-up is on the transcript."
)

KIND_NUDGE = {
    KIND_PENDING_TOOL_USE: (
        "Retry the {tool_name} call (id {tool_use_id}) or fall back to a "
        "safer alternative — it started at line {line_number} and never "
        "produced a tool_result."
    ),
    KIND_EMPTY_TOOL_RESULT: (
        "The {tool_name} call at line {line_number} returned an empty "
        "tool_result. Re-run it with verbose output (or inspect stderr) "
        "and report what actually came back."
    ),
    KIND_HUNG_MCP_CALL: (
        "The MCP tool {tool_name} (id {tool_use_id}) hung at line "
        "{line_number}. Restart the MCP server or fall back to a native "
        "tool before responding."
    ),
    KIND_MISSING_FOLLOWUP: (
        "The tool_result for {tool_name} at line {line_number} was never "
        "interpreted. Summarize what the tool returned and continue the "
        "task the user asked for."
    ),
}

DEFAULT_NUDGE = (
    "A silent stall was detected on {tool_name} at line {line_number}. "
    "Investigate and produce a follow-up before stopping."
)


def nudge_for_signature(signature: StallSignature) -> str:
    template = KIND_NUDGE.get(signature.kind, DEFAULT_NUDGE)
    return template.format(
        tool_name=signature.tool_name or "the pending tool",
        tool_use_id=signature.tool_use_id or "<unknown>",
        line_number=signature.line_number,
    )


def build_recovery_prompt(signatures: list[StallSignature]) -> str:
    if not signatures:
        return ""
    lines = [RECOVERY_HEADER]
    for signature in signatures:
        lines.append(f"- [{signature.kind}] {nudge_for_signature(signature)}")
    lines.append(RECOVERY_FOOTER)
    return "\n".join(lines) + "\n"
