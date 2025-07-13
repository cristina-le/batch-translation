import logging
from datetime import datetime

from app.core.config import Configs
from app.clients.client import post_request
from app.utils.converter import parse_file_to_steps, parse_context_to_steps
from app.services.postgre_service import PostgreService

logger = logging.getLogger("app")

class Translator:
    def __init__(self, postgre_service: PostgreService):
        self.mcp_url = Configs.MCP_URL
        self.postgre_service = postgre_service

    async def _create_job(self, job_id, steps):
        """
        Create a new job entry in the database.
        """
        if not steps:
            raise ValueError("No steps provided to create a job.")

        new_job = await self.postgre_service.create_job(
            job_id=job_id,
            status="pending",
            steps=steps,
            created_at=datetime.now()
        )
        
        logger.info(f"Job {job_id} created in database with {len(steps)} steps.")
        return new_job

    async def process_url(self, job_id, file_url):
        """
        Parses a case file, creates a job, and saves it to the database.
        """
        logger.info(f"Processing case upload for job {job_id}")
        steps = await parse_file_to_steps(file_url)
        job = await self._create_job(job_id, steps)
        await self.postgre_service.create_automation_history(job_id, file_url, job.get("status"), job.get("created_at"))
        return job

    async def process_string(self, job_id, context):
        """
        Parses a string of test cases, creates a job, and saves it to the database.
        """
        logger.info(f"Processing string input for job {job_id}")
        if not context:
            raise ValueError("No context found in the provided input")

        user_input = context.strip()
        steps = await parse_context_to_steps(user_input)
        job = await self._create_job(job_id, steps)
        await self.postgre_service.create_automation_history(job_id, user_input, job.get("status"), job.get("created_at"))
        return job

    async def execute_automation_job(self, job_id):
        """
        Fetches a job from the DB and triggers its execution.
        """
        logger.info(f"Starting execution for job {job_id}")
        job = await self.get_job_status(job_id)
        if not job:
            raise ValueError(f"Job with ID {job_id} not found.")

        if job.get("status") != "pending":
            raise ValueError(f"Job {job_id} is not pending. Current status: {job.get('status')}")

        payload = {
            "task": "execute_automation_job",
            "role": "web",
            "id": job_id,
            "context": job.get("steps", []),
        }

        try:
            await post_request(payload, self.mcp_url)
            update_fields = {
                "started_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            await self.postgre_service.update_job_status(job_id, "running", update_fields)
            await self.postgre_service.update_automation_history(job_id, "running")
            logger.info(f"Job {job_id} execution started and status updated to 'running'.")
            return {"message": "Job execution started", "job_id": job_id}
        except Exception as e:
            logger.error(f"Failed to post request for job {job_id}: {e}", exc_info=True)
            update_fields = {"updated_at": datetime.now()}
            await self.postgre_service.update_job_status(job_id, "failed", update_fields)
            await self.postgre_service.update_automation_history(job_id, "failed")
            raise

    async def get_job_status(self, job_id):
        """
        Gets the status of a specific job from the database.
        """
        return await self.postgre_service.get_job_by_id(job_id)


    async def process_job_callback(self, callback_data):
        """
        Processes the callback data from a worker, updating the job status and logging.
        """
        job_id = callback_data.get("job_id")
        status = callback_data.get("status")
        logs = callback_data.get("logs", [])

        if not job_id or not status:
            raise ValueError("Callback data must include 'job_id' and 'status'.")

        logger.info(f"Processing callback for job {job_id} with status {status}")
        
        await self.postgre_service.update_log(job_id, logs)

        update_fields = {"updated_at": datetime.now()}
        if status == "completed":
            update_fields["completed_at"] = datetime.now()

        await self.postgre_service.update_job_status(job_id, status, extra_fields=update_fields)
        await self.postgre_service.update_automation_history(job_id, status)
        logger.info(f"Successfully updated job {job_id} to status {status} in database.")
