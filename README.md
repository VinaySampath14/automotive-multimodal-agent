# In-Car Multimodal Voice Agent

A real-time voice- and vision-enabled AI assistant that controls a car dashboard. Speaks commands → classifies intent → executes tools → updates dashboard gauges and map → speaks response back. Built with LangGraph, Pipecat, Deepgram, Cartesia, and GPT-4o vision.

---

## Architecture

```
[Microphone]
     │
     ▼
Deepgram STT (Pipecat)
     │  "turn on the AC"
     ▼
┌─────────────────────────────┐
│       LangGraph Agent        │
│  1. classify_intent          │  ← GPT-4o-mini
│  2. safety_check             │  ← rule-based gate
│  3. route to tool            │
└─────────────────────────────┘
     │           │          │
     ▼           ▼          ▼
climate_tool  nav_tool  vision_tool
     │           │          │  ← GPT-4o vision
     └─────┬─────┘          │
           ▼                │
   dashboard_bridge.py      │
   (writes JSON state)      │
           │                │
           ▼                │
   PyQt5 Dashboard GUI  ◄───┘
   (polls state, updates
    AC toggle, temp gauge,
    ORS route on map)
           │
           ▼
   Cartesia TTS → [Speaker]
```

---

## Eval Results (real numbers, run 2026-06-27)

| Category | Pass Rate |
|---|---|
| Functional | 7/7 (100%) |
| Vision | 4/4 (100%) |
| Ambiguous | 3/3 (100%) |
| Safety | 4/4 (100%) |
| Degradation | 2/2 (100%) |
| **Overall** | **20/20 (100%)** |

**Performance:** avg latency 1033ms · max 2441ms · ~$0.002/interaction

**Safety ablation (rule-based vs LLM):** 86.7% agreement · rule-based: 0 errors · LLM: 2 false-refuses

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env        # fill in API keys
```

Required keys in `.env`:
```
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
ORS_API_KEY=...
```

Run eval suite:
```bash
python -m eval.run_eval
```

Run voice pipeline:
```bash
python -m voice.voice_pipeline
```

Run dashboard (separate terminal, from dashboard folder):
```bash
python app.py
```

---

## Architecture Decisions

**Why GPT-4o-mini for intent classification?**
Fast (< 1s), cheap ($0.00015/call), and the task is simple 4-class classification. A local model (Qwen2-VL, Llama) would eliminate API cost and latency but requires GPU infrastructure. For a portfolio demo, GPT-4o-mini gives reliable results immediately. In production, this would be the first component to replace with an on-device model.

**Why GPT-4o vision instead of Qwen2-VL locally?**
Qwen2-VL-7B requires ~16GB VRAM and a dedicated GPU. GPT-4o vision gives comparable quality with zero infrastructure cost for the portfolio stage. The `VisionAssistant` class in `agent/vision.py` is designed to swap — set `mock_mode=False` and implement `_lazy_load()` for the local model path.

**Why Deepgram + Cartesia over alternatives?**
Deepgram Nova-2 has the best WER for conversational speech at $0.0043/min. Cartesia produces the most natural-sounding TTS at low latency. Both have Pipecat-native integrations, which kept the voice pipeline code simple. Alternatives (Whisper + ElevenLabs) would work but add more moving parts.

**Why rule-based safety gate instead of LLM-based?**
Explainability and reliability. An LLM-based safety gate introduces hallucination risk on safety-critical decisions. The rule-based gate is deterministic, auditable, and fast (no API call). The ablation study confirms this was the right call — rule-based achieved 0 errors vs. 2 false-refuses for the LLM on the same 15 test cases.

**Why LangGraph instead of a custom agent loop?**
LangGraph provides a proven state machine with built-in visualization and debugging. The graph shape (classify → safety → route → respond) maps naturally to the problem. The main cost is an extra dependency; the benefit is that each node is independently testable and the graph is easy to extend.

---

## Limitations & Future Work

**What this project is not:**

- **Not on-device inference.** All LLM calls go to OpenAI's cloud. Real automotive systems (BMW, Mercedes) require offline capability for reliability and data privacy. The fix is replacing GPT-4o-mini with a quantized local model (Phi-3-mini, Qwen2.5-1.5B) and GPT-4o vision with Qwen2-VL on a local GPU.

- **Driving state is a manual flag.** `is_driving=True/False` is hardcoded, not sensed. A real system would read vehicle CAN bus data (speed > 0, gear not in Park) to determine driving state automatically.

- **Safety rules are a hand-built list.** `UNSAFE_WHILE_DRIVING` covers known patterns but cannot generalize to novel unsafe requests. A trained safety classifier (fine-tuned on automotive safety datasets) would have better recall. The ablation study shows the LLM alternative over-refuses — prompt tuning or fine-tuning would be needed.

- **Static sample image.** Vision queries run against one fixed dashcam image (`dashboard/sample_cabin.jpg`). A production system would capture live frames from a camera stream and send the current frame on each vision query.

- **No conversational memory.** Each command is stateless. "Navigate to the airport" followed by "actually make it the train station" would set two separate destinations rather than updating the previous one. Adding LangGraph's memory module or a conversation buffer would address this.

- **No multi-turn context.** The agent handles single-shot commands only. A real assistant would maintain context across a conversation turn.

---

## Resume Bullet

```
Built a real-time multimodal in-car voice agent (Python, LangGraph, Pipecat) with
Deepgram STT, Cartesia TTS, and GPT-4o vision for dashcam Q&A; evaluated across
20 scenarios (100% pass rate, 1033ms avg latency, ~$0.002/interaction); conducted
rule-based vs. LLM safety gate ablation study (86.7% agreement, rule-based: 0 errors);
wired live ORS map routing and a PyQt5 dashboard that auto-updates on voice command.
```
