import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

class Context(BaseModel):
    """
    Pydantic model for translated outputs.
    """
    translated_outputs: List[str]

class JapaneseToEnglishTranslator:
    """
    Translates Japanese text to English in batch mode with context preservation.
    """
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.2, model = "openai/gpt-4o-mini"):
        """
        Initialize the translator with API key, temperature, and model.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.prev_context = None

    def create_prompt(self, japanese_text: str, size) -> str:
        """
        Create a prompt for translation, optionally including previous context.
        """
        base_prompt = f"""
Translate each line of the following Japanese text to English.

Requirements:
- Preserve character speech, personality, honorifics, cultural references, and emotional nuance.
- Use natural, high-quality English localization.
- Do NOT add or remove lines. No empty lines.

CRITICAL:
- Output MUST have EXACTLY {size} lines.
- Return ONLY valid JSON in this format:

"translated_outputs": [
    "English line 1",
    "English line 2",
    "...",
    "English line {size}"
]

- Do not return anything except the JSON object above.

Japanese text to translate:
{japanese_text}
"""

        if self.prev_context:
            context_prompt = f"""
Previous context:
Japanese: {self.prev_context['japanese']}
English: {self.prev_context['english']}

Now translate the following with the same style and consistency:

{base_prompt}
"""
            return context_prompt
        return base_prompt

    def translate(self, japanese_text: str, size: int) -> str:
        """
        Translate Japanese text to English, ensuring output matches required line count.
        """
        prompt = self.create_prompt(japanese_text, size)
        
        output_lines = 0
        while (output_lines != size):
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": "You are a professional Japanese to English translator specializing in visual novels. Your goal is to produce translations that maximize BLEU score when compared to professional human translations. CRITICAL: You must follow all instructions exactly, especially regarding the number of lines in the output. The number of lines in your translation must match exactly what is requested."},
                    {"role": "user", "content": prompt}
                ],
                response_format=Context,
            )
            translation = response.choices[0].message.content
            translation = json.loads(translation)
            translation = "\n".join(translation["translated_outputs"])
            output_lines = len(translation.splitlines())

        self.prev_context = {
            'japanese': japanese_text,
            'english': translation
        }
        return translation
