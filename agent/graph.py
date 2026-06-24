"""
LangGraph state machine for the in-car agent.

Flow:
  classify_intent -> safety_check -> (route to climate / navigation / vision / chat) -> respond

The intent classifier below is a simple keyword-based stub so the whole
graph can be built, run, and tested end-to-end without an LLM API key.
Swap `classify_intent_stub` for a real LLM call (see TODO) once you're
ready — the graph shape doesn't need to change.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import END, StateGraph

from agent.tools import query_cabin_image, safety_check, set_climate, set_navigation

Intent = Literal["climate", "navigation", "vision", "chat"]


class AgentState(TypedDict, total=False):
    user_text: str
    is_driving: bool
    intent: Intent
    safety_allowed: bool
    safety_reason: str
    needs_confirmation: bool
    tool_result: str
    response: str


# ---------------------------------------------------------------------------
# Node: classify intent
# ---------------------------------------------------------------------------

def classify_intent_stub(text: str) -> Intent:
    """
    Rule-based stand-in for an LLM intent classifier.

    Uses whole-word token matching for single-word keywords (so "back seat"
    does NOT match "ac" — a real bug found while building this scaffold:
    naive substring matching on "ac" matched inside "back"). Multi-word
    phrases still use substring matching since false positives there are
    much rarer.

    TODO: replace with a real call, e.g.:
        response = llm.invoke(f"Classify this in-car command into "
                               f"climate/navigation/vision/chat: {text}")
        return parse_intent(response)
    """
    t = text.lower()
    tokens = set(t.replace("?", "").replace(",", "").split())

    climate_words = {"ac", "climate", "heat", "cool", "temperature"}
    climate_phrases = ["air condition"]

    nav_words = {"navigate"}
    nav_phrases = ["directions", "drive to", "route to", "take me to"]

    vision_words = {"weather", "raining", "see"}
    vision_phrases = ["what's outside", "what is outside", "back seat", "building"]

    if tokens & climate_words or any(p in t for p in climate_phrases):
        return "climate"
    if tokens & nav_words or any(p in t for p in nav_phrases):
        return "navigation"
    if tokens & vision_words or any(p in t for p in vision_phrases):
        return "vision"
    return "chat"


def classify_intent_node(state: AgentState) -> AgentState:
    intent = classify_intent_stub(state["user_text"])
    return {**state, "intent": intent}


# ---------------------------------------------------------------------------
# Node: safety check
# ---------------------------------------------------------------------------

def safety_node(state: AgentState) -> AgentState:
    result = safety_check(state["user_text"], is_driving=state.get("is_driving", True))
    return {
        **state,
        "safety_allowed": result["allowed"],
        "safety_reason": result["reason"],
        "needs_confirmation": result["needs_confirmation"],
    }


def route_after_safety(state: AgentState) -> str:
    if not state["safety_allowed"]:
        return "refuse"
    return state["intent"]


# ---------------------------------------------------------------------------
# Action nodes (each calls the matching tool directly for the scaffold;
# swap for real LLM-driven tool-calling once an LLM is wired in)
# ---------------------------------------------------------------------------

def climate_node(state: AgentState) -> AgentState:
    text = state["user_text"].lower()
    action = "on" if "on" in text else "off" if "off" in text else "set_temperature"
    result = set_climate.invoke({"zone": "front", "action": action, "value": ""})
    return {**state, "tool_result": result}


def navigation_node(state: AgentState) -> AgentState:
    # naive destination extraction for the scaffold — replace with NER/LLM parsing
    destination = state["user_text"].split("to", 1)[-1].strip() or "unknown destination"
    result = set_navigation.invoke({"destination": destination})
    return {**state, "tool_result": result}


def vision_node(state: AgentState) -> AgentState:
    result = query_cabin_image.invoke({"question": state["user_text"]})
    return {**state, "tool_result": result}


def chat_node(state: AgentState) -> AgentState:
    return {**state, "tool_result": "I'm here — let me know if you need climate, navigation, or to ask about what's around you."}


def refuse_node(state: AgentState) -> AgentState:
    return {**state, "tool_result": f"Sorry, I can't do that right now: {state['safety_reason']}"}


def respond_node(state: AgentState) -> AgentState:
    prefix = ""
    if state.get("needs_confirmation"):
        prefix = "Please confirm: "
    return {**state, "response": f"{prefix}{state['tool_result']}"}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("safety_check", safety_node)
    graph.add_node("climate", climate_node)
    graph.add_node("navigation", navigation_node)
    graph.add_node("vision", vision_node)
    graph.add_node("chat", chat_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify_intent")
    graph.add_edge("classify_intent", "safety_check")

    graph.add_conditional_edges(
        "safety_check",
        route_after_safety,
        {
            "climate": "climate",
            "navigation": "navigation",
            "vision": "vision",
            "chat": "chat",
            "refuse": "refuse",
        },
    )

    for node in ["climate", "navigation", "vision", "chat", "refuse"]:
        graph.add_edge(node, "respond")

    graph.add_edge("respond", END)

    return graph.compile()


AGENT = build_graph()


def run_agent(user_text: str, is_driving: bool = True) -> AgentState:
    """Convenience entrypoint used by the voice pipeline and the eval suite."""
    return AGENT.invoke({"user_text": user_text, "is_driving": is_driving})


if __name__ == "__main__":
    # Quick manual smoke test
    for sample in [
        "turn on the AC",
        "navigate to the airport",
        "what's that building outside",
        "play a video",
    ]:
        out = run_agent(sample)
        print(f"> {sample}\n  intent={out.get('intent')} response={out.get('response')}\n")
