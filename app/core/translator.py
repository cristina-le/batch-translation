import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

class Context(BaseModel):
    translated_outputs: List[str]

class JapaneseToEnglishTranslator:
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, model = "google/gemini-2.5-flash-preview"):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.prev_context = None

    def create_prompt(self, japanese_text: str, size) -> str:       
        base_prompt = f"""
        TASK: For each line of the following Japanese text, translate it to English.

        REQUIREMENTS:
        - Maintain character speech patterns and personality.
        - Preserve Japanese honorifics where appropriate.
        - Keep cultural references intact.
        - Ensure emotional nuances are conveyed.
        - Use natural, flowing English suitable for high-quality localization.
        - Do not create empty lines in your translation.

        CRITICAL: 
        - Your translation MUST have EXACTLY {size} lines, no more and no less.
        - The "translated_outputs" array MUST contain EXACTLY {size} elements.
        - Count the number of elements in your "translated_outputs" array before submitting to ensure it's exactly {size}.
        - Return the result as JSON: "translated_outputs": ["English line 1", "English line 2", ..., "English line {size}"]

        CONTEXT: Japanese text to translate:
        {japanese_text} 

        RESULT: Translation into English:
        """

        if self.prev_context:
            context_prompt = f""""
            For context, here is the previous text and its translation:
            Previous Japanese: {self.prev_context['japanese']}

            Previous English: {self.prev_context['english']}

            Now based on the previous translation:
            {base_prompt}
            """
            return context_prompt
        return base_prompt

    def translate(self, japanese_text: str, size: int) -> str:
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
