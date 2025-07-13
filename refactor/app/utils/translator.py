import logging
from collections import deque
from typing import List
from jinja2 import Template

from app.core.constant import Constants
from app.core.schema import Context
from app.model.llm import get_structured_data
from app.utils.common import split_into_chunks

logger = logging.getLogger("app")

class Translator:
    """
    Handles Japanese to English translation with context memory.
    """
    def __init__(self):
        self.history = deque(maxlen=Constants.CONTEXT_WINDOW)

    def _prepare_prompt(self, japanese_text: str, size: int) -> str:
        template = Template(Constants.TRANSLATE_PROMPT)
        return template.render(
            japanese_text=japanese_text,
            size=size,
            history=list(self.history)
        )

    async def translate_chunk(self, japanese_text: str, size: int) -> str:
        """
        Translates a chunk of Japanese text with a fixed number of lines.
        """
        prompt = self._prepare_prompt(japanese_text, size)
        response_data = await get_structured_data(prompt, Context)
        translated_text = "\n".join(response_data["translated_outputs"])

        self.history.append({
            "japanese": japanese_text,
            "english": translated_text
        })

        return translated_text

    async def translate_full_text(self, full_text: str) -> List[str]:
        """
        Translates the entire input text and returns list of translated lines.
        """
        full_text = full_text.strip()
        if not full_text:
            logger.warning("Empty input text.")
            return []

        chunks = split_into_chunks(full_text)
        translated_lines = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_lines = chunk.splitlines()
            line_count = len(chunk_lines)

            if not any(line.strip() for line in chunk_lines):
                translated_lines.extend([""] * line_count)
                continue

            logger.info(f"→ Translating chunk {idx}/{len(chunks)}...")

            try:
                raw = await self.translate_chunk(chunk, line_count)
                translated_lines.extend(raw.splitlines())
            except Exception as e:
                logger.error(f"✗ Error in chunk {idx}: {e}", exc_info=True)
                translated_lines.extend(["[Translation Error]"] * line_count)

        return translated_lines
