import logging
import json
from datetime import datetime
from fastapi import BackgroundTasks

from app.core.constant import Constants
from refactor.app.services.translator import TranslationService
from app.utils.common import parse_file_to_context

logger = logging.getLogger("app")

class TranslateService:
    """
    Main orchestrator for the translation pipeline.
    """
    def __init__(self):
        self.translator = TranslationService()
        self.output_file = Constants.OUTPUT_FILE

    async def _execute_and_save(self, job_id: str, context: str):
        """
        Executes an automation job and saves the payloads to a JSON file.
        """
        translated_context = await self.translator.translate_full_text(context)

        payloads = []
        for i in range(len(context)):
            payloads.append({
                "job_id": job_id,
                "japanese": context[i],
                "english": translated_context[i],
                "completed_at": datetime.now().isoformat()
            })

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(payloads, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(payloads)} payloads to {self.output_file}")

    async def translate_file(self, job_id: str, url: str, background_tasks: BackgroundTasks):
        """
        Executes an automation job in the background.
        """
        logger.info(f"Processing input for job {job_id}")
        context = parse_file_to_context(url)
        background_tasks.add_task(self._execute_and_save, job_id, context)
        
        logger.info(f"Job {job_id} execution started in background.")
        return {"status": "started", "job_id": job_id}

    async def translate_context(self, job_id: str, context: str, background_tasks: BackgroundTasks):
        """
        Executes an automation job in the background.
        """
        logger.info(f"Processing input for job {job_id}")
        background_tasks.add_task(self._execute_and_save, job_id, context)
        
        logger.info(f"Job {job_id} execution started in background.")
        return {"status": "started", "job_id": job_id}
    
    
