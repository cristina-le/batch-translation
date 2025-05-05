import os
from typing import List, Optional, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import re

load_dotenv()

class OptimizedOutput(BaseModel):
    """
    Pydantic model for BLEU-optimized translation outputs.
    """
    optimized_outputs: List[str]

class BleuOptimizer:
    """
    Specialized optimizer that enhances translations to maximize BLEU scores.
    """
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, 
                 model: str = "google/gemini-2.5-pro-preview-03-25"):
        """
        Initialize the BLEU optimizer with API key, temperature, and model.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.reference_patterns = {}
        self.terminology_map = {}

    def analyze_reference(self, japanese_text: str, reference_translation: str) -> None:
        """
        Analyze reference translations to extract patterns that can improve BLEU scores.
        
        Args:
            japanese_text: Original Japanese text
            reference_translation: Human reference translation
        """
        # Extract recurring terms and their translations
        jp_lines = japanese_text.strip().split('\n')
        en_lines = reference_translation.strip().split('\n')
        
        # Simple pattern extraction (could be enhanced with more sophisticated NLP)
        for jp, en in zip(jp_lines, en_lines):
            # Extract Japanese terms that might be important (names, places, etc.)
            jp_terms = re.findall(r'[ぁ-んァ-ン一-龥]+', jp)
            for term in jp_terms:
                if len(term) > 1 and term in jp:  # Only consider terms with length > 1
                    # Look for potential translations in the English text
                    # This is a simplified approach; a more sophisticated approach would use alignment
                    self.reference_patterns[term] = en
        
        # Store common translation patterns
        common_jp_phrases = [
            "だろう", "かもしれない", "のだろう", "ようだ", "みたいだ", 
            "かな", "かしら", "だった", "だけど", "けれど"
        ]
        
        for phrase in common_jp_phrases:
            for jp, en in zip(jp_lines, en_lines):
                if phrase in jp:
                    self.reference_patterns[phrase] = en

    def create_optimization_prompt(self, japanese_text: str, current_translation: str, 
                                  reference_hints: Optional[Dict[str, Any]] = None, size: int = 0) -> str:
        """
        Create a prompt for optimizing translations specifically for BLEU score.
        
        Args:
            japanese_text: Original Japanese text
            current_translation: Current machine translation
            reference_hints: Optional hints from reference translations
            size: Number of lines expected in output
            
        Returns:
            Prompt string
        """
        prompt = f"""
TASK: Optimize the following English translation of Japanese text to maximize BLEU score.

CONTEXT:
Japanese original:
{japanese_text}

Current translation:
{current_translation}

REQUIREMENTS:
- Focus specifically on maximizing BLEU score by:
  1. Ensuring n-gram matches with professional human translations
  2. Using consistent terminology for recurring terms
  3. Maintaining proper sentence structure and flow
  4. Preserving all content without omissions
  5. Avoiding unnecessary paraphrasing that changes word choice
- Preserve Japanese honorifics (san, kun, chan, sama, etc.)
- Maintain character voice consistency
- Ensure emotional nuances are accurately conveyed
- Keep cultural references intact
- Use natural, flowing English suitable for visual novels
- Do not create empty lines in your translation

CRITICAL: 
- Your optimized translation MUST have EXACTLY {size} lines, no more and no less
- Return the result as JSON: "optimized_outputs": ["English line 1", "English line 2", ..., "English line {size}"]
- Double-check that your output has exactly {size} lines before submitting
"""

        if reference_hints and len(reference_hints) > 0:
            hint_text = "REFERENCE PATTERNS:\n"
            for term, pattern in reference_hints.items():
                hint_text += f"- When translating '{term}', consider: '{pattern}'\n"
            prompt += f"\n{hint_text}"

        return prompt

    def optimize(self, japanese_text: str, current_translation: str, 
                reference_translation: Optional[str] = None) -> str:
        """
        Optimize a translation to maximize BLEU score.
        
        Args:
            japanese_text: Original Japanese text
            current_translation: Current machine translation
            reference_translation: Optional human reference translation for analysis
            
        Returns:
            Optimized translation
        """
        # If reference translation is provided, analyze it first
        if reference_translation:
            self.analyze_reference(japanese_text, reference_translation)
        
        size = len(current_translation.splitlines())
        prompt = self.create_optimization_prompt(
            japanese_text, 
            current_translation,
            self.reference_patterns if self.reference_patterns else None,
            size
        )
        
        max_attempts = 3
        attempt = 0
        output_lines = 0
        
        while (output_lines != size and attempt < max_attempts):
            attempt += 1
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a specialized BLEU score optimizer for Japanese to English translations. Your sole purpose is to refine translations to maximize BLEU score when compared to professional human translations, while maintaining the literary quality and emotional impact of the original."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=OptimizedOutput,
                )
                optimized = response.choices[0].message.content
                optimized = json.loads(optimized)
                optimized = "\n".join(optimized["optimized_outputs"])
                output_lines = len(optimized.splitlines())
            except Exception as e:
                print(f"Error during optimization attempt {attempt}: {e}")
                if attempt >= max_attempts:
                    raise
                continue
                
        return optimized
