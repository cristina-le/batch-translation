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
    logger.info(f"MESSAGEEEE: {text}")

    completion = await client.beta.chat.completions.parse(
        model=Constants.MODEL,
        messages=[
            {"role": "system", "content": Constants.SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format=schema,
        temperature=Constants.TEMPERATURE,
    )

    message = completion.choices[0].message.content
    return json.loads(message)