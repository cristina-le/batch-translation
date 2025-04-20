import os
import time
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

class TranslationRefiner:
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, model="google/gemini-2.0-flash-001"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature

    def create_refinement_prompt(self, japanese_text: str, current_translation: str, reference_translation: str = None) -> str:
        prompt = f"""
        TASK: Refine each line of the following English translation of a Japanese text.
        
        REQUIREMENT:
        - The goal is to maximize its BLEU score against a professional human reference translation. 
        - Focus on accuracy, naturalness, and matching the reference style as closely as possible

        IMPORTANT:
        1. Do not create empty line in your translation.
        2. Keep the same number of lines

        CONTEXT: 
        - Japanese text to translate:
        {japanese_text}

        - Current translation:
        {current_translation}


        - Reference translation:
        {reference_translation}

        RESULT: Refined translation:
        """
    
        return prompt

    def refine(self, japanese_text: str, current_translation: str, reference_translation: str = None) -> str:
        prompt = self.create_refinement_prompt(japanese_text, current_translation, reference_translation)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a professional Japanese to English translation refiner. Your goal is to maximize BLEU score against a human reference."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()

