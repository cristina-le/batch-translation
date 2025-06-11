import os
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import re
from app.utils.postprocess import post_process_translation

load_dotenv()

class Context(BaseModel):
    """Pydantic model for translated outputs."""
    translated_outputs: List[str]

class QualityScore(BaseModel):
    """Pydantic model for translation quality assessment."""
    score: float
    reasoning: str

class JapaneseToEnglishTranslator:
    """Ultra-optimized translator chỉ tập trung vào translation."""
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, 
                 model = "openai/gpt-4o", context_window: int = 5, 
                 quality_threshold: float = 8.0):
        """Initialize the translator."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.quality_threshold = quality_threshold
        self.context_history = []

    def create_ultra_optimized_prompt(self, japanese_text: str, size: int) -> str:
        """Create optimized prompt for translation."""
        base_prompt = f"""
        TASK: Translate Japanese visual novel text to English with MAXIMUM BLEU SCORE optimization.

        CRITICAL REQUIREMENTS:
        - Use natural, fluent English matching professional translations
        - Maintain consistent character voices and terminology
        - Preserve emotional nuances and narrative flow
        - Use common English expressions over literal translations

        TECHNICAL REQUIREMENTS:
        - Return EXACTLY {size} lines as JSON: {{"translated_outputs": ["line1", "line2", ...]}}
        - Each line should be complete and natural-sounding

        TEXT TO TRANSLATE:
        {japanese_text}
        """
    
        if self.context_history:
            context_info = "\nPREVIOUS CONTEXT:\n"
            for i, ctx in enumerate(self.context_history[-self.context_window:]):
                context_info += f"Segment {i+1}:\nJP: {ctx['japanese']}\nEN: {ctx['english']}\n\n"
            
            return f"{context_info}Maintain consistency with previous segments.\n\n{base_prompt}"
            
        return base_prompt

    def _assess_quality(self, japanese_text: str, english_text: str) -> float:
        """Assess translation quality using AI."""
        try:
            prompt = f"""
            Evaluate this Japanese to English translation on a scale of 1-10.
            
            Consider: natural flow, accuracy, character consistency, cultural adaptation.
            
            Japanese: {japanese_text}
            English: {english_text}
            """
            
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are a translation quality assessor."},
                    {"role": "user", "content": prompt}
                ],
                response_format=QualityScore,
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["score"]
        except:
            return 7.0
        
    def translate(self, japanese_text: str, size: int) -> str:
        """Translate with multiple attempts and quality assessment."""
        best_translation = None
        best_score = 0
        
        for attempt in range(3):
            try:
                prompt = self.create_ultra_optimized_prompt(japanese_text, size)
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature + (attempt * 0.05),
                    messages=[
                        {"role": "system", "content": "You are a world-class Japanese to English translator specializing in visual novels."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=Context,
                )
                
                translation_data = json.loads(response.choices[0].message.content)
                translation = "\n".join(translation_data["translated_outputs"])
                
                # Verify line count
                if len(translation.splitlines()) != size:
                    continue
                
                # Post-process and assess
                translation = post_process_translation(translation)
                quality_score = self._assess_quality(japanese_text, translation)

                if quality_score > best_score:
                    best_score = quality_score
                    best_translation = translation
                
                if quality_score >= self.quality_threshold:
                    break
                    
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback if needed
        if best_translation is None:
            try:
                prompt = self.create_ultra_optimized_prompt(japanese_text, size)
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": "You are a professional Japanese translator."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=Context,
                )
                translation_data = json.loads(response.choices[0].message.content)
                best_translation = "\n".join(translation_data["translated_outputs"])
                best_translation = self._post_process(best_translation)
            except:
                best_translation = "Translation failed."

        # Update context history
        self.context_history.append({
            'japanese': japanese_text,
            'english': best_translation,
            'quality_score': best_score
        })
        
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
            
        return best_translation
