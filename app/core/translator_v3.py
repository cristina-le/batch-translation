import os
from typing import List, Optional, Dict, Tuple
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import re
from collections import defaultdict

load_dotenv()

class Context(BaseModel):
    """
    Pydantic model for translated outputs.
    """
    translated_outputs: List[str]

class QualityScore(BaseModel):
    """
    Pydantic model for translation quality assessment.
    """
    score: float
    reasoning: str

class JapaneseToEnglishTranslator:
    """
    Ultra-optimized translator designed to maximize BLEU scores through advanced techniques.
    """
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, 
                 model = "openai/gpt-4o", context_window: int = 5, 
                 speaker_aware: bool = True, quality_threshold: float = 8.0):
        """
        Initialize the ultra-optimized translator.
        
        Args:
            api_key: API key for OpenRouter
            temperature: Temperature for generation (lower for consistency)
            model: Model to use (gpt-4o for best quality)
            context_window: Number of previous chunks to keep in context
            speaker_aware: Whether to use speaker diarization
            quality_threshold: Minimum quality score to accept translation
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.speaker_aware = speaker_aware
        self.quality_threshold = quality_threshold
        self.context_history = []
        self.character_consistency = defaultdict(list)
        self.translation_memory = {}
        
        # Character profiles for consistency
        self.character_profiles = {
            "レイ": {
                "name": "Rei",
                "gender": "male",
                "personality": "introspective, gentle, android",
                "speech_pattern": "formal but warm, uses 'I' consistently"
            },
            "シオナ": {
                "name": "Shiona", 
                "gender": "female",
                "personality": "kind, caring, optimistic",
                "speech_pattern": "warm, uses 'you' when addressing Rei"
            },
            "ティピィ": {
                "name": "Tipi",
                "gender": "female", 
                "personality": "childlike, lonely, affectionate",
                "speech_pattern": "simple, childish language"
            },
            "マッド": {
                "name": "Madd",
                "gender": "male",
                "personality": "gruff, stern but caring underneath",
                "speech_pattern": "rough, direct, uses contractions"
            }
        }

    def preprocess_with_advanced_speakers(self, text: str) -> str:
        """
        Advanced speaker detection and tagging.
        """
        if not self.speaker_aware:
            return text
            
        lines = text.splitlines()
        processed_lines = []
        
        for line in lines:
            if not line.strip():
                processed_lines.append(line)
                continue
                
            # Detect character names and dialogue patterns
            character_detected = None
            
            # Check for character name patterns
            for char_jp, profile in self.character_profiles.items():
                if char_jp in line or profile["name"] in line:
                    character_detected = profile["name"]
                    break
            
            # Dialogue detection patterns
            if line.strip().startswith('「'):
                if not character_detected:
                    # Use context to determine speaker
                    if any('シオナ' in prev for prev in lines[max(0, lines.index(line)-3):lines.index(line)]):
                        character_detected = "Shiona"
                    elif any('ティピィ' in prev for prev in lines[max(0, lines.index(line)-3):lines.index(line)]):
                        character_detected = "Tipi"
                    else:
                        character_detected = "Rei"  # Default protagonist
                        
                processed_lines.append(f"[{character_detected}]: {line}")
            else:
                # Narration or description
                processed_lines.append(f"[Narration]: {line}")
                
        return "\n".join(processed_lines)

    def create_ultra_optimized_prompt(self, japanese_text: str, size: int) -> str:
        """
        Create an ultra-optimized prompt designed to maximize BLEU scores.
        """
        # Add speaker tags if enabled
        if self.speaker_aware:
            japanese_text = self.preprocess_with_advanced_speakers(japanese_text)
        
        # Character consistency information
        char_info = ""
        if self.character_profiles:
            char_info = "\nCHARACTER PROFILES:\n"
            for char_jp, profile in self.character_profiles.items():
                char_info += f"- {char_jp} ({profile['name']}): {profile['personality']}, {profile['speech_pattern']}\n"

        base_prompt = f"""
        TASK: Translate Japanese visual novel text to English with MAXIMUM BLEU SCORE optimization.

        CRITICAL BLEU OPTIMIZATION REQUIREMENTS:
        - Use natural, fluent English that matches professional human translations
        - Maintain consistent terminology and character voices throughout
        - Preserve sentence structure when it enhances readability
        - Use common English expressions over literal translations
        - Ensure smooth narrative flow and emotional resonance
        - Match the tone and register of high-quality visual novel localizations

        TRANSLATION GUIDELINES:
        - Maintain character speech patterns and personality consistently
        - Preserve Japanese honorifics only when they add cultural value
        - Adapt cultural references for English readers while keeping essence
        - Ensure emotional nuances are clearly conveyed
        - Use natural, flowing English suitable for professional localization
        - Avoid overly literal translations that sound unnatural
        - Prioritize readability and natural expression over word-for-word accuracy

        CHARACTER CONSISTENCY:
        {char_info}

        TECHNICAL REQUIREMENTS:
        - Your translation MUST have EXACTLY {size} lines, no more and no less
        - Return as JSON: {{"translated_outputs": ["English line 1", "English line 2", ..., "English line {size}"]}}
        - Each line should be complete and natural-sounding
        - Maintain narrative coherence across all lines

        CONTEXT: Japanese text to translate:
        {japanese_text}

        RESULT: High-quality English translation optimized for maximum BLEU score:
        """
    
        if self.context_history:
            context_prompt = f"""
            PREVIOUS CONTEXT (for consistency):
            """
            
            # Add previous context chunks with character tracking
            for i, prev_context in enumerate(self.context_history[-self.context_window:]):
                context_prompt += f"""
                Segment {i+1}:
                Japanese: {prev_context['japanese']}
                English: {prev_context['english']}
                """
                
            context_prompt += f"""
            
            Based on the above context, maintain consistency in:
            - Character names and personalities
            - Terminology and world-building elements
            - Narrative tone and style
            - Relationship dynamics
            
            Now translate the new segment:
            {base_prompt}
            """
            return context_prompt
            
        return base_prompt

    def assess_translation_quality(self, japanese_text: str, english_text: str) -> float:
        """
        Assess translation quality using AI evaluation.
        """
        try:
            assessment_prompt = f"""
            Evaluate this Japanese to English translation on a scale of 1-10 for BLEU score potential.
            
            Consider:
            - Natural English flow and readability
            - Accuracy of meaning preservation
            - Character voice consistency
            - Cultural adaptation appropriateness
            - Professional localization quality
            
            Japanese: {japanese_text}
            English: {english_text}
            
            Provide a score and brief reasoning.
            """
            
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert translation quality assessor specializing in Japanese visual novels."},
                    {"role": "user", "content": assessment_prompt}
                ],
                response_format=QualityScore,
            )
            
            result = json.loads(response.choices[0].message.content)
            return result["score"]
        except:
            return 7.0  # Default acceptable score

    def post_process_translation(self, translation: str) -> str:
        """
        Post-process translation to fix common issues.
        """
        # Fix common translation artifacts
        translation = re.sub(r'\s+', ' ', translation)  # Normalize whitespace
        translation = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', translation)  # Fix sentence spacing
        translation = re.sub(r'\s+([.!?,:;])', r'\1', translation)  # Fix punctuation spacing
        
        # Character name consistency
        for char_jp, profile in self.character_profiles.items():
            translation = re.sub(char_jp, profile["name"], translation)
        
        # Common visual novel terminology
        replacements = {
            "Philoid": "Phiroid",
            "music-box": "music box",
            "Onii-chan": "brother",
            "Onee-chan": "sister"
        }
        
        for old, new in replacements.items():
            translation = re.sub(old, new, translation, flags=re.IGNORECASE)
        
        return translation.strip()

    def translate(self, japanese_text: str, size: int) -> str:
        """
        Ultra-optimized translation with multiple attempts and quality assessment.
        """
        best_translation = None
        best_score = 0
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                prompt = self.create_ultra_optimized_prompt(japanese_text, size)
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature + (attempt * 0.05),  # Slight variation
                    messages=[
                        {"role": "system", "content": "You are a world-class Japanese to English translator specializing in visual novels. Your translations consistently achieve the highest BLEU scores by producing natural, fluent English that perfectly captures the original meaning and emotional nuance."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=Context,
                )
                
                translation_data = json.loads(response.choices[0].message.content)
                translation = "\n".join(translation_data["translated_outputs"])
                
                # Check line count
                output_lines = len(translation.splitlines())
                if output_lines != size:
                    continue
                
                # Post-process
                translation = self.post_process_translation(translation)
                
                # Assess quality
                quality_score = self.assess_translation_quality(japanese_text, translation)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_translation = translation
                
                # If we hit our quality threshold, use this translation
                if quality_score >= self.quality_threshold:
                    break
                    
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {e}")
                continue
        
        # Fallback to best attempt if no translation met threshold
        if best_translation is None:
            # Emergency fallback with simpler approach
            prompt = self.create_ultra_optimized_prompt(japanese_text, size)
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": "You are a professional Japanese to English translator. Translate accurately and naturally."},
                    {"role": "user", "content": prompt}
                ],
                response_format=Context,
            )
            translation_data = json.loads(response.choices[0].message.content)
            best_translation = "\n".join(translation_data["translated_outputs"])
            best_translation = self.post_process_translation(best_translation)

        # Store context for future translations
        self.context_history.append({
            'japanese': japanese_text,
            'english': best_translation,
            'quality_score': best_score
        })
        
        # Keep context history within the specified window
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
            
        return best_translation
