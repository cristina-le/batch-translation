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
        Executes an automation job and appends the payload to a JSON file.
        """
        translated_context = await self.translator.translate_full_text(context)

        payload = {
            "job_id": job_id,
            "completed_at": datetime.now().isoformat(),
            "content": [
                {
                    "japanese": context[i],
                    "english": translated_context[i]
                } for i in range(len(context))
            ]
        }

        with open(self.output_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        data.append(payload)

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Job {job_id} execution completed at {payload['completed_at']}")

    async def translate_file(self, job_id: str, url: str, background_tasks: BackgroundTasks):
        """
        Translate provided text file
        """
        logger.info(f"Processing input for job {job_id}")
        context = parse_file_to_context(url)
        background_tasks.add_task(self._execute_and_save, job_id, context)
        
        logger.info(f"Job {job_id} execution started in background.")
        return {"status": "started", "job_id": job_id}

    async def translate_context(self, job_id: str, context: str, background_tasks: BackgroundTasks):
        """
        Translate provided context
        """
        logger.info(f"Processing input for job {job_id}")
        background_tasks.add_task(self._execute_and_save, job_id, context)
        
        logger.info(f"Job {job_id} execution started in background.")
        return {"status": "started", "job_id": job_id}
    
    async def get_job(self, job_id: str):
        """
        Retrieves a specific job by its ID from the output file.
        """
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for job in data:
                    if job.get("job_id") == job_id:
                        return job

            return {"error": f"Job ID {job_id} not found"}
        except json.JSONDecodeError:
            logger.error("Failed to decode output JSON file.")
            return {"error": "Corrupted output file"}