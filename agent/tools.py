"""
Tool definitions for the in-car agent.

These are written as plain Python functions with a `@tool`-style decorator
from LangChain so they can be wired straight into the LangGraph agent, and
mirror how you'd expose them as MCP tools later (same input/output shape,
just a different transport).

Each tool calls into `dashboard_bridge` to actually change dashboard state,
keeping "decide what to do" (agent) separate from "how state gets written"
(bridge) so you can swap the bridge implementation without touching the
agent logic.
"""

from __future__ import annotations

from langchain_core.tools import tool

from agent.vision import VisionAssistant
from dashboard import dashboard_bridge

_vision_assistant = VisionAssistant(mock_mode=False)


# ---------------------------------------------------------------------------
# Safety gate — checked by the agent graph BEFORE any action tool runs.
# Keep this rule-based and easy to extend; this is the part interviewers
# will ask you to walk through, so keep it legible.
# ---------------------------------------------------------------------------

# Each entry is a set of keywords that must ALL appear (in any order) for
# the rule to match. This is more robust than fixed-phrase substring
# matching (e.g. "play a video" still matches {"play", "video"} even
# though it doesn't contain the literal phrase "play video").
UNSAFE_WHILE_DRIVING: list[set[str]] = [
    {"play", "video"},
    {"show", "video"},
    {"watch", "movie"},
    {"show", "movie"},
    {"camera", "feed", "entertainment"},
]

REQUIRES_CONFIRMATION: list[set[str]] = [
    {"unlock", "all", "doors"},
    {"disable", "safety"},
    {"turn", "off", "collision"},
]


def _keywords_match(text_tokens: set[str], rule: set[str]) -> bool:
    return rule.issubset(text_tokens)


def safety_check(user_text: str, is_driving: bool = True) -> dict:
    """
    Returns a dict: {"allowed": bool, "reason": str, "needs_confirmation": bool}
    Called by the agent graph before dispatching to an action tool.

    NOTE: this is intentionally simple/rule-based for the scaffold. In a
    real eval write-up, this is exactly the piece worth replacing with an
    LLM-based safety classifier and then comparing rule-based vs LLM-based
    false-allow / false-refuse rates — that comparison is a much stronger
    "error analysis" story than either approach alone.
    """
    tokens = set(user_text.lower().translate(str.maketrans("", "", ".,!?")).split())

    if is_driving:
        for rule in UNSAFE_WHILE_DRIVING:
            if _keywords_match(tokens, rule):
                return {
                    "allowed": False,
                    "reason": f"Refusing '{user_text}' while driving — visually distracting.",
                    "needs_confirmation": False,
                }

    for rule in REQUIRES_CONFIRMATION:
        if _keywords_match(tokens, rule):
            return {
                "allowed": True,
                "reason": f"'{user_text}' requires explicit confirmation before acting.",
                "needs_confirmation": True,
            }

    return {"allowed": True, "reason": "No safety concerns detected.", "needs_confirmation": False}


# ---------------------------------------------------------------------------
# Action tools
# ---------------------------------------------------------------------------

@tool
def set_climate(zone: str, action: str, value: str = "") -> str:
    """
    Control the climate system.
    zone: e.g. "front", "rear", "all"
    action: "on", "off", "set_temperature"
    value: temperature value if action == "set_temperature"
    """
    result = dashboard_bridge.update_climate(zone=zone, action=action, value=value)
    return f"Climate updated: {result}"


@tool
def set_navigation(destination: str) -> str:
    """Set a navigation destination on the dashboard."""
    result = dashboard_bridge.update_navigation(destination=destination)
    return f"Navigation updated: {result}"


@tool
def query_cabin_image(question: str, image_path: str = "dashboard/sample_cabin.jpg") -> str:
    """
    Answer a question about what the cabin/dashcam camera currently sees.
    Use this for questions like "what's outside", "is the back seat empty",
    "what's that building".
    """
    result = _vision_assistant.answer(image_path=image_path, question=question)
    return result.answer


ALL_TOOLS = [set_climate, set_navigation, query_cabin_image]
