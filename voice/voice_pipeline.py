"""
Voice pipeline skeleton using Pipecat (https://github.com/pipecat-ai/pipecat).

This wires: microphone -> speech-to-text -> our LangGraph agent -> text-to-
speech -> speaker. It's left as a skeleton with TODOs because the exact
classes depend on which STT/TTS providers you pick and which Pipecat
version you install — check Pipecat's quickstart for the current API
before filling these in, since voice frameworks move fast.

If you'd rather use LiveKit Agents instead of Pipecat, the same three-step
shape applies: STT -> agent.run_agent(text) -> TTS. Swap the imports below
accordingly; `agent.graph.run_agent` doesn't change either way.
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv

from agent.graph import run_agent

load_dotenv()


async def handle_transcript(text: str, is_driving: bool = True) -> str:
    """
    Called by the voice pipeline whenever STT produces a final transcript.
    Returns the text that should be spoken back via TTS.
    """
    result = run_agent(user_text=text, is_driving=is_driving)
    return result.get("response", "Sorry, I didn't catch that.")


def build_pipeline():
    """
    TODO: Build the actual Pipecat pipeline here. Rough shape (check current
    Pipecat docs for exact class names/imports — this moves fast):

        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.services.deepgram import DeepgramSTTService
        from pipecat.services.cartesia import CartesiaTTSService
        from pipecat.frames.frames import TranscriptionFrame

        stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
        tts = CartesiaTTSService(api_key=os.getenv("CARTESIA_API_KEY"))

        # Custom processor: on each final transcript, call handle_transcript()
        # and push the returned text into the TTS service.

        pipeline = Pipeline([mic_input, stt, agent_processor, tts, speaker_output])
        return pipeline
    """
    raise NotImplementedError(
        "Fill in build_pipeline() with your chosen STT/TTS providers "
        "following Pipecat's current quickstart docs."
    )


async def main():
    print("Voice pipeline scaffold — fill in build_pipeline() before running for real.")
    # Quick text-only smoke test of the agent hookup without any audio:
    for sample_text in ["turn on the AC", "what's outside", "play a video"]:
        reply = await handle_transcript(sample_text)
        print(f"> {sample_text}\n  agent says: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
