import json
import logging
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

from app.core.config import Configs
from app.core.constant import Constants

logger = logging.getLogger("app")

client = AsyncOpenAI(
    base_url=Constants.LLM_BASE_URL, 
    api_key=Configs.OPENROUTER_API_KEY
)

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
async def get_structured_data(text, schema):
    """
    Uses LLM to parse a natural language string into a list of structured objects.
    """
    completion = await client.responses.parse(
        model=Constants.MODEL,
        input=[
            {"role": "system", "content": Constants.LLM_PROMPT},
            {"role": "user", "content": text},
        ],
        text_format=schema,
        temperature=Constants.TEMPERATURE,
    )

    message = completion.output_text
    return json.loads(message)