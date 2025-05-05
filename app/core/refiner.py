import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

class Context(BaseModel):
    """
    Pydantic model for refined translation outputs.
    """
    refined_outputs: List[str]

class TranslationRefiner:
    """
    Refines Japanese-to-English translations to maximize BLEU score and naturalness.
    """
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.2, model="google/gemini-2.5-pro-preview-03-25"):
        """
        Initialize the TranslationRefiner with API key, temperature, and model.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def create_refinement_prompt(self, japanese_text: str, current_translation: str, size: int) -> str:
        """
        Create a prompt for refining translations with strict output requirements.
        """
        prompt = f"""
Refine the following English translation of a Japanese text to maximize BLEU score.

IMPORTANT BLEU SCORE GUIDELINES:
- Make MINIMAL changes to preserve existing n-gram matches with reference translations
- Focus on correcting obvious translation errors only
- Maintain exact terminology and phrasing where possible
- Avoid unnecessary paraphrasing or rewording that could reduce n-gram matches
- Only make changes that are likely to improve BLEU score

Requirements:
- Preserve all Japanese names, terms, and honorifics exactly as translated
- Keep sentence structure similar to the original translation when possible
- Maintain character speech patterns and personality
- Ensure the translation follows the Japanese meaning and nuance
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
        """
        Refine the current translation to match the Japanese text and required line count.
        """
        prompt = self.create_refinement_prompt(japanese_text, current_translation, size)
        
        output_lines = 0
        max_attempts = 3
        attempt = 0
        
        while (output_lines != size and attempt < max_attempts):
            attempt += 1
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a professional Japanese to English translation refiner specializing in BLEU score optimization. Your primary goal is to maximize BLEU score by making minimal, strategic changes to preserve n-gram matches with reference translations. Be extremely conservative with changes, only fixing clear errors."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=Context
                )
                translation = response.choices[0].message.content
                translation = json.loads(translation)
                translation = "\n".join(translation["refined_outputs"])
                output_lines = len(translation.splitlines())
            except Exception as e:
                print(f"Error during refinement attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    # If all attempts fail, return the original translation
                    return current_translation
                continue
            
        return translation
