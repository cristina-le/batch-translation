from typing import List, Optional, Dict
import os
import json
import hashlib
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel

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

class SpeakerAwareness:
    """Class để xử lý speaker awareness với LLM-based character detection và caching."""
    
    def __init__(self, model: str = "openai/gpt-4o"):
        """Initialize speaker awareness với OpenAI client."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = model
        
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

    def preprocess_speakers(self, text: str, full_text: str = None) -> str:
        """LLM-based speaker detection với caching."""
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

def reader(file_path: str, size: int = 50) -> List[str]:
    """
    Read text from file and split into chunks of lines.

    Args:
        file_path (str): Path to the text file
        size (int): Number of lines per chunk (default 50)

    Returns:
        list: List of text segments, each containing up to 'size' lines
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.readlines()

            segments = [
                "".join(text[i:i+size]) 
                for i in range(0, len(text), size)
            ]

        print(f"Split text into {len(segments)} line-based chunks")
        return segments

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

def writer(file_path: str, data: List[str]) -> bool:
    """
    Write segments to a file, each chunk followed by a newline.

    Args:
        file_path (str): Path to write the data to
        data (list): List of text segments

    Returns:
        bool: Success status
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write("\n".join(data))
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False
