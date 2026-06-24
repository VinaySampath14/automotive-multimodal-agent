# Project Proposal: Co-Pilot — Multimodal Agentic Assistant for an In-Car Dashboard

## 1. Summary

A voice- and vision-enabled agent that controls a simulated car dashboard.
It listens to spoken commands, looks at a cabin/dashcam image when needed,
reasons about what to do via an LLM-driven agent, calls tools to act on the
dashboard, and replies out loud — backed by a structured safety/evaluation
suite.

**Why this project, specifically:** Across multiple BMW Group postings
(Intelligent Personal Assistant, In-Car Voice Assistant, and a dedicated
Master's Thesis on Automated GenAI Evaluation), the same pattern repeats
independently: **audio + vision + text inputs, inside an agentic/MCP
system, evaluated through test scenarios, performance metrics, and error
analysis.** This project is built around that exact pattern — not a
generic multimodal demo, but one shaped by what these specific postings
actually describe wanting someone to build.

---

## 2. Architecture

```
 [Microphone] --STT--> "turn on the AC"
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
          (MCP-style)   (MCP-style) (VLM on
                                     cabin image)
                   \        |         /
                    v       v        v
              dashboard_bridge.py
              (writes shared state)
                              |
                              v
                  [Dashboard GUI reads
                   state, updates gauges/AC/nav]
                              |
                              v
                  Agent generates spoken reply
                              |
                              v
                      TTS --> [Speaker]
```

**Already built and tested** (see `incar-multimodal-agent.zip`): the
LangGraph routing logic, the rule-based safety gate, the MCP-style tool
stubs, the dashboard state bridge, and an 8-scenario eval suite — all
running end-to-end with mocked LLM/VLM/voice layers. Two real bugs were
found and fixed during that build (a fragile safety-phrase match, and a
substring-matching false positive where "back seat" triggered the climate
intent) — both are documented in the code as examples of the kind of
error analysis this project is meant to produce.

---

## 3. Repos and frameworks to use (not build from scratch)

| Piece | Use this | Why |
|---|---|---|
| Dashboard GUI | Fork `SihabSahariar/Smart-CAR-Dashboard-GUI-in-Python` | Already has speed/fuel/AC/nav/camera widgets; explicitly lists "virtual assistant" as a missing feature |
| Voice I/O | `Pipecat` or `LiveKit Agents` | Production-grade STT/TTS pipelines; don't reinvent audio streaming |
| Agent reasoning | LangGraph + MCP-style tools (already built) | Reuses your existing skill from other projects |
| Vision | Qwen2-VL | Open-weight, integrates with `transformers`, good instruction-following for visual Q&A |

---

## 4. Phased plan

### Phase 1 — Make the dashboard real (≈1 day)
- [ ] Fork the dashboard repo, get it running locally as-is
- [ ] Identify its render loop (`app.py`) and current state format
- [ ] Add a polling step that reads `dashboard_state.json` (already written by `dashboard_bridge.py`)
- [ ] Map JSON fields (`climate`, `navigation`) to existing widgets
- [ ] Manually edit the JSON file by hand and confirm the GUI reacts — proves the bridge before the agent is involved

### Phase 2 — Connect the existing agent to the real dashboard (≈half day)
- [ ] Run `python -m agent.graph`, confirm it writes to the same state file path the dashboard now reads
- [ ] Drive it from a simple CLI loop and watch the dashboard react live
- [ ] **Milestone:** text-in → dashboard-reacts fully working — already a legitimate, demoable artifact at this point

### Phase 3 — Replace the stub "brain" with a real LLM (≈1 day)
- [ ] Get an API key (OpenAI or Anthropic)
- [ ] Replace `classify_intent_stub` in `agent/graph.py` with a real LLM call
- [ ] Re-run `eval/run_eval.py`; compare pass rate against the rule-based version (today: 8/8)
- [ ] Improve destination/value extraction using the LLM's structured output instead of naive string slicing
- [ ] **Bonus eval angle:** keep both the rule-based and LLM-based safety checker, and report false-allow/false-refuse rates for each — this side-by-side comparison is exactly the "test scenarios and validation protocols" framing BMW's evaluation thesis describes

### Phase 4 — Make vision real (≈1–2 days)
- [ ] Get GPU access (Colab/cloud) or a hosted Qwen2-VL endpoint
- [ ] Implement `_lazy_load()` and `answer()` in `agent/vision.py`; set `mock_mode=False`
- [ ] Collect 5–10 real cabin/dashcam-style images (stock photos of car interiors are fine)
- [ ] Re-run vision scenarios against the real model; check answer *quality*, not just routing correctness

### Phase 5 — Make voice real (≈1–2 days)
- [ ] Pick one provider set (e.g. Deepgram STT + Cartesia TTS)
- [ ] Fill in `build_pipeline()` in `voice/voice_pipeline.py` per Pipecat's current quickstart
- [ ] Test STT alone first, then wire the full loop: mic → STT → agent → TTS → speaker
- [ ] **Milestone:** full multimodal loop working — speak a command, dashboard reacts, hear a spoken reply

### Phase 6 — Expand and harden the eval suite (≈half day)
- [ ] Add 10–15 more scenarios beyond the current 8 (more safety edge cases, more ambiguous phrasing, more vision questions)
- [ ] Run the full suite against the real LLM + real VLM; record actual numbers
- [ ] Manually review failures; write 2–3 sentences per failure on *why* it failed — this is interview material
- [ ] Save final `results.json` — these numbers go on the resume, not estimates

### Phase 7 — Polish for presentation (≈half day)
- [ ] Record a 30–60s demo video/GIF of the full voice → dashboard → voice loop
- [ ] Update the README with the real architecture diagram and real eval numbers
- [ ] Write the final resume bullet using real `[N]`, `[X]`, `[Y]` values
- [ ] Push to your own GitHub repo, crediting the dashboard fork

**Rough total:** 5–8 days of focused evening/weekend work.

---

## 5. Evaluation suite — design notes

Four categories, deliberately mirroring BMW's posting language:

| Category | What it tests | Example |
|---|---|---|
| Functional | Normal commands succeed | "turn on the AC" |
| Vision | Routes to and answers correctly using the cabin image | "is the back seat empty" |
| Ambiguous | Graceful handling of under-specified input | "turn it down" (AC or volume?) |
| Safety | Refuses or requires confirmation appropriately | "play a video" while driving |

Current state: 8 scenarios, 8/8 passing against the rule-based stub. Target
for Phase 6: 18–23 scenarios, evaluated against the real LLM + real VLM,
with a written error-analysis section.

---

## 6. Resume bullet template (fill in only with real numbers)

```latex
\cventry
  {Python · LangGraph · MCP · Qwen2-VL · Pipecat · FastAPI}
  {Multimodal Agentic Assistant for In-Car Dashboard}
  {\href{https://github.com/YOUR_USERNAME/YOUR_REPO}{GitHub}}
  {}
  {
   \begin{cvitems}
\item {Built a multimodal (speech, vision, text) agentic assistant for an in-car dashboard using LangGraph for intent routing and MCP-style tool calling for climate, navigation, and vision actions.}
\item {Integrated a Vision-Language Model (Qwen2-VL) for visual cabin/dashcam queries and real-time speech I/O via Pipecat.}
\item {Designed a safety-scenario evaluation suite spanning functional, ambiguous, and safety-critical commands, achieving [X]\% task success and [Y]\% correct safety-refusal rate across [N] test scenarios.}
\end{cvitems}
  }
```

---

## 7. Honesty checklist before using any claim from this project

- [ ] Don't write "speech, vision, text" until **both** Phase 4 (vision) and Phase 5 (voice) are done. If only one is real, name only that modality.
- [ ] Don't quote eval numbers from the rule-based stub once a real LLM/VLM is wired in — re-run and use the new numbers.
- [ ] Be able to open `eval/scenarios.py` and `results.json` live in an interview and explain at least one failure case and why it happened.
- [ ] If asked "why a car dashboard," be ready to name the specific BMW postings this maps to, not just "it seemed cool."

---

## 8. What this project does *not* cover

- VLM **fine-tuning/training** (this project only prompts an existing VLM) — a separate project (e.g. fine-tuning Qwen2-VL on ChartQA) would be needed to cover Gini/Fraunhofer/Rosenberger-style postings that explicitly ask for training/fine-tuning experience.
- Pure computer-vision research framing (e.g. BMW's 3D anomaly segmentation thesis) — that posting wants contrastive VLMs fused with 3D bounding boxes, a different research direction entirely.
