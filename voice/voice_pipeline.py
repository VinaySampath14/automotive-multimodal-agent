"""
Voice pipeline using Pipecat 1.4.x.

Flow: microphone -> Deepgram STT -> AgentProcessor -> Cartesia TTS -> speaker
"""

from __future__ import annotations

import asyncio
import os

import logging
import sys

import pyaudio
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import EndFrame, TextFrame, TranscriptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)

from agent.graph import run_agent

# Suppress pipecat internal debug/warning noise — only show ERROR and our own INFO logs
logger.remove()
logger.add(sys.stderr, level="INFO", filter=lambda r: r["name"].startswith("__main__"))
logging.disable(logging.WARNING)

load_dotenv()

# ---------------------------------------------------------------------------
# Custom processor: intercepts transcription frames, runs the agent,
# pushes the response as a text frame for TTS to consume.
# ---------------------------------------------------------------------------

class AgentProcessor(FrameProcessor):
    """Bridge between Pipecat transcriptions and our LangGraph agent."""

    def __init__(self, is_driving: bool = True):
        super().__init__()
        self.is_driving = is_driving

    async def process_frame(self, frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            text = frame.text.strip()
            if not text:
                return
            logger.info(f"[STT] {text}")
            result = run_agent(user_text=text, is_driving=self.is_driving)
            response = result.get("response", "Sorry, I didn't catch that.")
            logger.info(f"[Agent] {response}")
            await self.push_frame(TTSSpeakFrame(text=response))
        else:
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(is_driving: bool = True) -> tuple[Pipeline, pyaudio.PyAudio]:
    pa = pyaudio.PyAudio()

    params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    )

    mic = LocalAudioInputTransport(pa, params)
    speaker = LocalAudioOutputTransport(pa, params)

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY", ""),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # Cartesia: American female
    )

    agent = AgentProcessor(is_driving=is_driving)

    pipeline = Pipeline([mic, stt, agent, tts, speaker])
    return pipeline, pa


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def handle_transcript(text: str, is_driving: bool = True) -> str:
    """Text-only convenience entry point (used by eval / smoke tests)."""
    result = run_agent(user_text=text, is_driving=is_driving)
    return result.get("response", "Sorry, I didn't catch that.")


async def main():
    pipeline, pa = build_pipeline(is_driving=True)
    worker = PipelineWorker(pipeline, idle_timeout_secs=None)
    runner = PipelineRunner()
    logger.info("Voice pipeline running — Ctrl+C to stop.")
    try:
        await runner.run(worker)
    finally:
        pa.terminate()


if __name__ == "__main__":
    asyncio.run(main())
