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

import os
import time
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from agent.tools import query_cabin_image, safety_check, set_climate, set_navigation

load_dotenv()

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

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
    latency_ms: float
    cost_usd: float


# ---------------------------------------------------------------------------
# Node: classify intent
# ---------------------------------------------------------------------------

def classify_intent_stub(text: str) -> Intent:
    """Classifies user intent using GPT-4o-mini."""
    prompt = (
        "You are an in-car voice assistant. Classify the user command into exactly "
        "one of these intents: climate, navigation, vision, chat.\n\n"
        "- climate: AC, heat, temperature, fan\n"
        "- navigation: directions, route, destination\n"
        "- vision: questions about what the camera sees (cabin, outside, weather)\n"
        "- chat: anything else\n\n"
        f"Command: {text}\n"
        "Reply with one word only."
    )
    response = _llm.invoke(prompt)
    intent = response.content.strip().lower()
    if intent not in ("climate", "navigation", "vision", "chat"):
        return "chat"
    return intent  # type: ignore[return-value]


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

_WORD_TO_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty one": 21, "twenty two": 22, "twenty three": 23,
    "twenty four": 24, "twenty five": 25, "twenty six": 26, "twenty seven": 27,
    "twenty eight": 28, "twenty nine": 29, "thirty": 30,
}

def _extract_temperature(text: str):
    import re
    # Try digit first
    m = re.search(r"\b(\d{1,2})\b", text)
    if m:
        return m.group(1)
    # Try word numbers (longest match first)
    for phrase in sorted(_WORD_TO_NUM, key=len, reverse=True):
        if phrase in text:
            return str(_WORD_TO_NUM[phrase])
    return None


def climate_node(state: AgentState) -> AgentState:
    text = state["user_text"].lower()
    temp_value = _extract_temperature(text)
    if temp_value:
        action = "set_temperature"
        msg = f"Temperature set to {temp_value} degrees."
    elif "off" in text:
        action, temp_value = "off", ""
        msg = "Air conditioning is now off."
    else:
        action, temp_value = "on", ""
        msg = "Air conditioning is now on."
    set_climate.invoke({"zone": "front", "action": action, "value": temp_value or ""})
    return {**state, "tool_result": msg}


def navigation_node(state: AgentState) -> AgentState:
    destination = state["user_text"].split("to", 1)[-1].strip() or "unknown destination"
    set_navigation.invoke({"destination": destination})
    return {**state, "tool_result": f"Navigating to {destination}."}


def vision_node(state: AgentState) -> AgentState:
    result = query_cabin_image.invoke({"question": state["user_text"]})
    return {**state, "tool_result": result}


def chat_node(state: AgentState) -> AgentState:
    return {**state, "tool_result": "I'm here. Let me know if you need climate control, navigation, or want to know what's around you."}


def refuse_node(state: AgentState) -> AgentState:
    return {**state, "tool_result": f"Sorry, I can't do that while driving."}


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


# ---------------------------------------------------------------------------
# Cost constants (USD per call, approximate)
# ---------------------------------------------------------------------------
_COST_INTENT_CLASSIFICATION = 0.00015   # GPT-4o-mini, ~500 tokens
_COST_VISION_QUERY          = 0.003     # GPT-4o vision, ~1 image + text
_COST_STT_PER_MIN           = 0.0043    # Deepgram Nova-2
_COST_TTS_PER_CHAR          = 0.00002   # Cartesia


def run_agent(user_text: str, is_driving: bool = True) -> AgentState:
    """Run the agent and log per-component latency and estimated cost."""
    t0 = time.perf_counter()
    state = AGENT.invoke({"user_text": user_text, "is_driving": is_driving})
    total_ms = (time.perf_counter() - t0) * 1000

    # Estimate cost
    cost = _COST_INTENT_CLASSIFICATION
    if state.get("intent") == "vision":
        cost += _COST_VISION_QUERY
    response = state.get("response", "")
    cost += len(response) * _COST_TTS_PER_CHAR

    state["latency_ms"] = round(total_ms, 1)
    state["cost_usd"] = round(cost, 6)

    print(
        f"[metrics] latency={total_ms:.0f}ms  "
        f"intent={state.get('intent')}  "
        f"cost=${cost:.5f}"
    )
    return state


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
