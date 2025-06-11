import os
from typing import List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json
import re
import hashlib
from datetime import datetime

load_dotenv()

class Context(BaseModel):
    """Pydantic model for translated outputs."""
    translated_outputs: List[str]

class QualityScore(BaseModel):
    """Pydantic model for translation quality assessment."""
    score: float
    reasoning: str
class CharacterProfile(BaseModel):
    """
    Pydantic model for discovered character profiles.
    """
    name: str
    gender: str
    personality: str
    speech_patterns: List[str]
    relationships: Dict[str, str]
class CharacterProfiles(BaseModel):
    """Pydantic model for all discovered character profiles."""
    characters: Dict[str, CharacterProfile]

class SpeakerTagging(BaseModel):
    """Pydantic model for speaker tagging results."""
    tagged_lines: List[Dict[str, str]]

class JapaneseToEnglishTranslator:
    """Ultra-optimized translator with LLM-based character detection and caching."""
    
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.1, 
                 model = "openai/gpt-4o", context_window: int = 5, 
                 speaker_aware: bool = True, quality_threshold: float = 8.0):
        """Initialize the translator with caching and LLM-based speaker detection."""
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.speaker_aware = speaker_aware
        self.quality_threshold = quality_threshold
        self.context_history = []
        
        # Cache setup
        self.cache_dir = "app/data/cache"
        self.tagged_texts_dir = os.path.join(self.cache_dir, "tagged_texts")
        self.profiles_cache_file = os.path.join(self.cache_dir, "character_profiles.json")
        os.makedirs(self.tagged_texts_dir, exist_ok=True)
        self.character_profiles_cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load character profiles cache."""
        try:
            if os.path.exists(self.profiles_cache_file):
                with open(self.profiles_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
        return {}

    def _save_profiles_cache(self, profiles: Dict, text_hash: str):
        """Save character profiles to cache."""
        try:
            self.character_profiles_cache[text_hash] = profiles
            with open(self.profiles_cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.character_profiles_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving profiles cache: {e}")

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _load_tagged_text(self, text_hash: str) -> Optional[str]:
        """Load tagged text from cache."""
        try:
            tagged_file = os.path.join(self.tagged_texts_dir, f"{text_hash}.txt")
            if os.path.exists(tagged_file):
                with open(tagged_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception:
            pass
        return None

    def _save_tagged_text(self, text_hash: str, tagged_text: str):
        """Save tagged text to cache."""
        try:
            tagged_file = os.path.join(self.tagged_texts_dir, f"{text_hash}.txt")
            with open(tagged_file, 'w', encoding='utf-8') as f:
                f.write(tagged_text)
            
            # Save metadata
            meta_file = os.path.join(self.tagged_texts_dir, f"{text_hash}_meta.json")
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "lines_count": len(tagged_text.splitlines())
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving tagged text: {e}")

    def _build_character_profiles(self, full_text: str) -> Dict:
        """Use LLM to analyze text and create character profiles."""
        text_hash = self._get_text_hash(full_text)
        
        if text_hash in self.character_profiles_cache:
            print(f"Using cached profiles: {text_hash[:8]}...")
            return self.character_profiles_cache[text_hash]
        
        print("Analyzing character profiles with LLM...")
        try:
            prompt = f"""
            Analyze this Japanese visual novel text and identify ALL characters.
            
            For each character, determine:
            1. Character name and gender
            2. Personality and speech patterns
            3. Relationships with other characters
            
            Text: {full_text}
            
            Return character profiles in JSON format.
            """
            
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing Japanese visual novel characters."},
                    {"role": "user", "content": prompt}
                ],
                response_format=CharacterProfiles,
            )
            
            result = json.loads(response.choices[0].message.content)
            profiles = result["characters"]
            self._save_profiles_cache(profiles, text_hash)
            return profiles
            
        except Exception as e:
            print(f"Error analyzing profiles: {e}")
            return {}

    def _llm_tag_speakers(self, text: str, profiles: Dict) -> str:
        """Use LLM to tag speakers based on profiles."""
        try:
            prompt = f"""
            Tag speakers for each line in this Japanese text using the character profiles.
            
            CHARACTER PROFILES:
            {json.dumps(profiles, indent=2, ensure_ascii=False)}
            
            TEXT TO TAG:
            {text}
            
            Tag each line with [Speaker]: or [Narration]: format.
            Use English character names from profiles.
            """
            
            response = self.client.beta.chat.completions.parse(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "You are an expert at identifying speakers in Japanese visual novel text."},
                    {"role": "user", "content": prompt}
                ],
                response_format=SpeakerTagging,
            )
            
            result = json.loads(response.choices[0].message.content)
            tagged_lines = []
            
            for line_data in result["tagged_lines"]:
                line = line_data["line"]
                speaker = line_data.get("speaker", "Narration")
                tagged_lines.append(f"[{speaker}]: {line}")
            
            return "\n".join(tagged_lines)
            
        except Exception as e:
            print(f"Error in speaker tagging: {e}")
            # Simple fallback
            lines = text.splitlines()
            tagged_lines = []
            for line in lines:
                if line.strip().startswith('「'):
                    tagged_lines.append(f"[Speaker]: {line}")
                else:
                    tagged_lines.append(f"[Narration]: {line}")
            return "\n".join(tagged_lines)

    def _preprocess_speakers(self, text: str, full_text: str = None) -> str:
        """LLM-based speaker detection with caching."""
        if not self.speaker_aware:
            return text
        
        text_hash = self._get_text_hash(text)
        cached_result = self._load_tagged_text(text_hash)
        if cached_result:
            print(f"Using cached tags: {text_hash[:8]}...")
            return cached_result
        
        # Get or build profiles
        profiles = self._build_character_profiles(full_text) if full_text else {}
        
        print(f"Processing with LLM: {text_hash[:8]}...")
        tagged_result = self._llm_tag_speakers(text, profiles)
        self._save_tagged_text(text_hash, tagged_result)
        
        return tagged_result

    def create_ultra_optimized_prompt(self, japanese_text: str, size: int, full_text: str = None) -> str:
        """Create optimized prompt with speaker tagging."""
        if self.speaker_aware:
            japanese_text = self._preprocess_speakers(japanese_text, full_text)

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

    def _post_process(self, translation: str) -> str:
        """Post-process translation to fix common issues."""
        # Normalize whitespace and punctuation
        translation = re.sub(r'\s+', ' ', translation)
        translation = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', translation)
        translation = re.sub(r'\s+([.!?,:;])', r'\1', translation)
        
        # Common replacements
        replacements = {
            "Philoid": "Phiroid",
            "music-box": "music box"
        }
        
        for old, new in replacements.items():
            translation = re.sub(old, new, translation, flags=re.IGNORECASE)
        
        return translation.strip()

    def translate(self, japanese_text: str, size: int, full_text: str = None) -> str:
        """Translate with multiple attempts and quality assessment."""
        best_translation = None
        best_score = 0
        
        for attempt in range(3):
            try:
                prompt = self.create_ultra_optimized_prompt(japanese_text, size, full_text)
                
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
                translation = self._post_process(translation)
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
                prompt = self.create_ultra_optimized_prompt(japanese_text, size, full_text)
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
