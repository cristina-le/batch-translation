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
        prompt = f"""
        Refine an existing English translation of a Japanese text to maximize the BLEU score.
        
        BLEU OPTIMIZATION GUIDELINES:
        - Make ONLY MINIMAL edits that clearly improve BLEU score.
        - Preserve existing n-gram matches wherever possible.
        - Correct clear translation errors ONLY.
        - Avoid paraphrasing or stylistic changes that may reduce BLEU overlap.
        - Retain terminology, phrasing, and structure unless incorrect.

        ADDITIONAL CONSTRAINTS:
        - Preserve all Japanese names, terms, and honorifics exactly as they appear.
        - Maintain sentence structure and tone aligned with the original.
        - Keep characters’ speech style and personality intact.
        - Do NOT add or remove lines. Do NOT leave empty lines.
        - Ensure every English line corresponds clearly to its Japanese counterpart.

        IMPORTANT:
        - The final output MUST contain EXACTLY {size} lines.
        - Output ONLY valid JSON in the following format:

        "refined_outputs": [
            "English line 1",
            "English line 2",
            ...
            "English line {size}"
        ]

        DO NOT include any explanations, markdown, or comments — ONLY return the JSON object as shown.

        Japanese Text:
        {japanese_text}

        Current Translation:
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
                        {"role": "system", "content": "You are an expert English-Japanese translation assistant. Your task is to refine an existing English translation of a Japanese text to maximize the BLEU score."},
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
