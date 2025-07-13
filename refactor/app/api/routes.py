import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.schema import TaskRequest
from app.services import get_translate_service
from app.services.translate_service import TranslateService

router = APIRouter()
logger = logging.getLogger("app")


@router.post("/task-execution")
async def task_execution(body: TaskRequest, background_tasks: BackgroundTasks, service: TranslateService = Depends(get_translate_service)):
    """
    Execute automation task based on the provided context.
    """
    try:
        task = body.task.lower()
        context = body.context
        task_id = str(uuid.uuid4())

        match task:
            case "translate_file":
                data = await service.translate_file(task_id, context, background_tasks)
            case "translate_context":
                data = await service.translate_context(task_id, context, background_tasks)
            case "get_job":
                data = await service.get_job(context)
            case _:
                raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")
        
        return JSONResponse(content={"status": "Success", "data": [data]})

    except Exception as e:
        logger.error(f"Error in task_execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))