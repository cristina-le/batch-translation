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
        TASK: Refine each line of the following English translation of a Japanese text.
        
        REQUIREMENT:
        - The goal is to maximize its BLEU score by improving translation quality.
        - Focus on accuracy, naturalness, and contextual appropriateness.
        - Ensure the translation closely follows the original Japanese meaning and nuances.
        - Use natural, flowing English suitable for high-quality localization.

        IMPORTANT:
        1. Do not create empty lines in your translation.
        2. Keep the same number of lines as the current translation.
        3. Double-check each line for accuracy against the Japanese original.
        4. Make sure the translation is contextually appropriate and coherent.
        5. Maintain character speech patterns and personality if present.
        6. Preserve Japanese honorifics where appropriate.
        7. Keep cultural references intact.
        8. Ensure emotional nuances are conveyed.

        CRITICAL: 
        - Your refinement MUST have EXACTLY {size} lines, no more and no less.
        - The "refined_outputs" array MUST contain EXACTLY {size} elements.
        - Count the number of elements in your "refined_outputs" array before submitting to ensure it's exactly {size}.
        - Return the result as JSON: "refined_outputs": ["English line 1", "English line 2", ..., "English line {size}"]

        CONTEXT: 
        - Japanese text to translate:
        {japanese_text}

        - Current translation:
        {current_translation}

        RESULT: Refined translation:
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
