import logging
import asyncio
from typing import List

from app.core.constant import Constants
from app.services.translation_service import TranslationService
from app.utils.postprocess import post_process
from app.utils.common import split_into_chunks

logger = logging.getLogger("app")

class Translator:
    """
    Main orchestrator for the translation pipeline.
    """
    def __init__(self):
        self.translation_service = TranslationService()

    async def translate_text(self, full_text: str) -> str:
        """
        Translates a full string of Japanese text
        """
        if not full_text.strip():
            logger.warning("Input text is empty. Returning empty string.")
            return ""

        logger.info(f"Starting translation for text with {len(full_text.splitlines())} lines.")
        
        chunks = split_into_chunks(full_text)
        translated_chunks = []
        
        total_chunks = len(chunks)
        logger.info(f"Text split into {total_chunks} chunks")

        for i, chunk in enumerate(chunks):
            chunk_lines = chunk.splitlines()
            if not any(line.strip() for line in chunk_lines):
                # If chunk is just whitespace, preserve it as empty lines
                translated_chunks.append("\n".join([""] * len(chunk_lines)))
                continue

            logger.info(f"Translating chunk {i + 1}/{total_chunks}...")
            try:
                translated_chunk = await self.translation_service.translate(
                    japanese_text=chunk,
                    size=len(chunk_lines)
                )
                
                # Apply post-processing to each chunk
                processed_chunk = post_process(translated_chunk)
                translated_chunks.append(processed_chunk)
                logger.info(f"Chunk {i + 1}/{total_chunks} translated successfully.")

            except Exception as e:
                logger.error(f"Error translating chunk {i + 1}: {e}", exc_info=True)
                # Add a placeholder to maintain line count on error
                error_placeholder = [f"[Translation Error on line]" for _ in chunk_lines]
                translated_chunks.append("\n".join(error_placeholder))

        logger.info("All chunks processed. Combining results.")
        final_translation = "\n".join(translated_chunks)
        
        return final_translation
