"""
Thin wrapper around a Vision-Language Model (default: Qwen2-VL) for
answering questions about a single image (e.g. a cabin or dashcam frame).

Runs in `mock_mode` by default so the rest of the pipeline can be built
and tested before you've set up GPU access or downloaded model weights.
Flip `mock_mode=False` and provide a real image once you're ready.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class VisionAnswer:
    answer: str
    model_used: str
    mock: bool


class VisionAssistant:
    """Answers natural-language questions about an image."""

    def __init__(self, model_name: Optional[str] = None, mock_mode: bool = True):
        self.model_name = model_name or os.getenv(
            "VLM_MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct"
        )
        self.mock_mode = mock_mode
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        """Load the real model on first use. Only called when mock_mode=False."""
        if self._model is not None:
            return

        # TODO: uncomment once you have GPU/API access and have installed
        # the right transformers version for Qwen2-VL.
        #
        # from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        # self._model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     self.model_name, torch_dtype="auto", device_map="auto"
        # )
        # self._processor = AutoProcessor.from_pretrained(self.model_name)
        raise NotImplementedError(
            "Real VLM loading is stubbed out. Either set mock_mode=True, "
            "or implement _lazy_load() with your chosen model/runtime."
        )

    def answer(self, image_path: str, question: str) -> VisionAnswer:
        """Answer `question` about the image at `image_path`."""
        if self.mock_mode:
            return self._mock_answer(image_path, question)

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        ext = os.path.splitext(image_path)[-1].lower().lstrip(".")
        mime = f"image/{ext if ext in ('png', 'jpg', 'jpeg', 'webp') else 'jpeg'}"

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_data}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            max_tokens=256,
        )
        text = response.choices[0].message.content.strip()
        return VisionAnswer(answer=text, model_used="gpt-4o", mock=False)

    def _mock_answer(self, image_path: str, question: str) -> VisionAnswer:
        """
        Deterministic fake answers so the agent graph and eval suite can be
        built/tested end-to-end before the real model is wired in.
        """
        q = question.lower()
        if "rain" in q or "weather" in q:
            answer = "It looks clear outside, no rain visible in the image."
        elif "building" in q or "landmark" in q:
            answer = "That appears to be an office building, no distinctive landmark visible."
        elif "back seat" in q or "passenger" in q:
            answer = "The back seat appears empty in this frame."
        else:
            answer = f"[mock] I looked at {image_path} but don't have a specific answer for: {question}"

        return VisionAnswer(answer=answer, model_used="mock-vlm", mock=True)
