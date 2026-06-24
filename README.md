# Co-Pilot: Multimodal Agentic Assistant for an In-Car Dashboard

A voice- and vision-enabled agent that controls a simulated car dashboard.
It listens to spoken commands, looks at a "cabin/dashcam" image when needed,
decides what to do via an LLM agent, calls tools to act on the dashboard,
and speaks a response back — with a safety-scenario evaluation suite.

This is a **scaffold**: working code structure with clear TODOs, not a
finished product. You fill in API keys, swap in a real dashboard fork, and
run the eval suite to get real numbers before putting anything on a resume.

---

## Architecture

```
 [Microphone] --STT (Pipecat/LiveKit)--> "turn on the AC"
                                              |
                                              v
                                   +----------------------+
                                   |  LangGraph Agent     |
                                   |  1. classify_intent  |
                                   |  2. safety_check     |
                                   |  3. route to tool    |
                                   +----------------------+
                                   /        |         \
                                  v         v          v
                          climate_tool  nav_tool   vision_tool
                          (MCP-style)   (MCP-style) (Qwen2-VL on
                                                      cabin image)
                                   \        |         /
                                    v       v        v
                              dashboard_bridge.py
                              (writes dashboard_state.json)
                                              |
                                              v
                                  [Forked dashboard GUI reads
                                   state, updates gauges/AC/nav]
                                              |
                                              v
                              Agent generates spoken reply
                                              |
                                              v
                                  TTS (Pipecat/LiveKit) --> [Speaker]
```

---

## Components & what to plug in

| Folder | What it does | What you need to do |
|---|---|---|
| `agent/graph.py` | LangGraph state machine: intent routing, safety gate, dispatch to tools | Replace the stub classifier with a real LLM call (GPT-4o-mini, Claude, or local Qwen) |
| `agent/tools.py` | MCP-style tool definitions: climate control, navigation, vision query | Wire `dashboard_bridge` calls to your actual forked dashboard's API/state file |
| `agent/vision.py` | Qwen2-VL wrapper for answering questions about a cabin/dashcam image | Set `mock_mode=False` and provide a real image once you have GPU/API access |
| `voice/voice_pipeline.py` | Pipecat pipeline skeleton: mic → STT → agent → TTS → speaker | Add your STT/TTS provider API keys (Deepgram, Cartesia, etc.) |
| `dashboard/dashboard_bridge.py` | Shared-state bridge between agent and dashboard GUI | Point `STATE_FILE` at wherever your forked dashboard repo polls from, or replace with a direct function call if you integrate in-process |
| `eval/scenarios.py` | Safety + functional test scenarios (the part most people skip) | Add more scenarios specific to your final feature set |
| `eval/run_eval.py` | Runs all scenarios against the agent, reports pass rate | Run this for real, save `results.json`, use the real numbers in your resume bullet |

---

## Suggested dashboard to fork

`SihabSahariar/Smart-CAR-Dashboard-GUI-in-Python` — already has speed, fuel,
door status, AC/music controls, nav, weather, and camera feed, and explicitly
lists "virtual assistant for hands-free control" as a planned-but-unbuilt
feature. That's the gap this project fills.

Wire `dashboard_bridge.py` to read/write whatever state format that repo's
`app.py` uses (a shared JSON file is the simplest integration — poll it on
a timer in the dashboard's render loop).

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API keys
```

Run the eval suite (works in mock mode with no API keys, for a dry run):
```bash
python -m eval.run_eval
```

Run the voice pipeline (needs real API keys):
```bash
python -m voice.voice_pipeline
```

---

## Honest resume-bullet template (fill in only after you've actually run it)

```
Built a multimodal (speech, vision, text) agentic assistant for an in-car
dashboard using LangGraph, MCP-style tool calling, and a VLM (Qwen2-VL) for
visual queries, with real-time speech I/O via Pipecat.
Evaluated against [N] test scenarios spanning functional commands, ambiguous
requests, and safety-critical refusals, achieving [X]% task success and
[Y]% correct safety-refusal rate.
```

Do not fill in `[N]`, `[X]`, `[Y]` until `eval/run_eval.py` has actually
produced them. Don't claim "multimodal" if you skip the vision wiring —
keep the bullet to "speech, text" if vision stays unfinished.
