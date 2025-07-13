from typing import Any, List
from pydantic import BaseModel

class TaskRequest(BaseModel):
    task: str
    context: Any

class Context(BaseModel):
    translated_outputs: List[str]

class QualityScore(BaseModel):
    score: float
    reasoning: str