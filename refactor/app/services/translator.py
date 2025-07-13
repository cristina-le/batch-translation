import logging
from collections import deque
from typing import List

from jinja2 import Template

from app.core.constant import Constants
from app.core.schema import Context
from app.model.llm import get_structured_data
from app.utils.common import split_into_chunks
from app.utils.postprocess import post_process

logger = logging.getLogger("app")

class TranslationService:
    """
    Handles Japanese to English translation with context tracking and chunk-based processing.
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

    async def translate_chunk(self, text: str, size: int) -> str:
        """
        Translates a chunk of text with a fixed number of lines.
        """
        prompt = self._prepare_prompt(text, size)
        response_data = await get_structured_data(prompt, Context)

        translated_text = "\n".join(response_data["translated_outputs"])

        self.history.append({
            'japanese': text,
            'english': translated_text
        })

        return translated_text

    async def translate_full_text(self, text: str) -> str:
        """
        Translates an entire block of Japanese text, handling chunking and post-processing.
        """
        full_text = text.strip()
        if not full_text:
            logger.warning("Empty input text. Skipping translation.")
            return ""

        lines = full_text.splitlines()
        logger.info(f"Translating text with {len(lines)} lines...")

        chunks = split_into_chunks(full_text)
        total = len(chunks)
        logger.info(f"Split into {total} chunk(s).")

        translated_chunks: List[str] = []

        for idx, chunk in enumerate(chunks, start=1):
            chunk_lines = chunk.splitlines()
            line_count = len(chunk_lines)

            if not any(line.strip() for line in chunk_lines):
                translated_chunks.append("\n".join([""] * line_count))
                continue

            logger.info(f"→ Translating chunk {idx}/{total} ({line_count} lines)...")

            try:
                raw_translation = await self.translate_chunk(chunk, line_count)
                cleaned = post_process(raw_translation)
                translated_chunks.append(cleaned)
                logger.info(f"✓ Chunk {idx}/{total} translated.")
            except Exception as e:
                logger.error(f"✗ Error in chunk {idx}: {e}", exc_info=True)
                placeholder = "\n".join(["[Translation Error]"] * line_count)
                translated_chunks.append(placeholder)

        logger.info("All chunks processed. Returning full translation.")
        return "\n".join(translated_chunks)
