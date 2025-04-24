import os
import time
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
from pydantic import BaseModel
import json

load_dotenv()

class Context(BaseModel):
    refined_outputs: List[str]

class TranslationRefiner:
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.2, model="google/gemini-2.5-pro-preview-03-25"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def create_refinement_prompt(self, japanese_text: str, current_translation: str, size: int) -> str:
        prompt = f"""
Refine each line of the following English translation of a Japanese text.

Requirements:
- Maximize BLEU score by improving translation quality, accuracy, and naturalness.
- Ensure the translation closely follows the Japanese meaning and nuance.
- Use natural, high-quality English localization.
- Preserve character speech, personality, honorifics, cultural references, and emotional nuance.
- Do NOT add or remove lines. No empty lines.

CRITICAL:
- Output MUST have EXACTLY {size} lines.
- Return ONLY valid JSON in this format:

"refined_outputs": [
    "English line 1",
    "English line 2",
    "...",
    "English line {size}"
]

- Do not return anything except the JSON object above.

Japanese text:
{japanese_text}

Current translation:
{current_translation}
"""
        return prompt

    def refine(self, japanese_text: str, current_translation: str, size: int) -> str:
        prompt = self.create_refinement_prompt(japanese_text, current_translation, size)
        
        output_lines = 0
        while (output_lines != size):
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a professional Japanese to English translation refiner. Your goal is to maximize BLEU score by improving translation quality and accuracy."},
                    {"role": "user", "content": prompt}
                ],
                response_format=Context
            )
            translation = response.choices[0].message.content
            translation = json.loads(translation)
            translation = "\n".join(translation["translated_outputs"])
            output_lines = len(translation.splitlines())
            
        return translation
