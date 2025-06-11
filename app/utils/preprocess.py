from typing import List, Optional, Dict, Set
import os
import json
import hashlib
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel

class CharacterDiscovery(BaseModel):
    """Pydantic model for character discovery results."""
    characters: List[str]

class CharacterValidation(BaseModel):
    """Pydantic model for character validation results."""
    missing_characters: List[str]

class SpeakerTagging(BaseModel):
    """Pydantic model for speaker tagging results."""
    tagged_lines: List[str]

class ChunkedSpeakerAwareness:
    """Chunk-based speaker awareness with 3-phase processing."""
    
    def __init__(self, model: str = "google/gemini-2.0-flash-exp", chunk_size: int = 50):
        """Initialize with OpenAI client and chunk settings."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = "google/gemini-2.5-flash-preview-05-20"
        self.chunk_size = chunk_size
        
        # Cache setup
        self.cache_dir = "app/data/cache"
        self.chunk_cache_dir = os.path.join(self.cache_dir, "chunks")
        self.result_cache_dir = os.path.join(self.cache_dir, "results")
        os.makedirs(self.chunk_cache_dir, exist_ok=True)
        os.makedirs(self.result_cache_dir, exist_ok=True)

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into non-overlapping chunks."""
        lines = text.splitlines()
        chunks = []
        
        i = 0
        while i < len(lines):
            # Get chunk with size
            chunk_end = min(i + self.chunk_size, len(lines))
            chunk_lines = lines[i:chunk_end]
            
            chunks.append("\n".join(chunk_lines))
            i += self.chunk_size
        
        print(f"Split text into {len(chunks)} chunks (size: {self.chunk_size})")
        return chunks

    def _initial_discovery(self, chunks: List[str]) -> List[str]:
        """Phase 1: Initial character discovery across all chunks."""
        print("\n=== Phase 1: Initial Character Discovery ===")
        all_characters = set()
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                prompt = (
                    "Analyze this Japanese text chunk and identify ALL character names that appear.\n"
                    "IMPORTANT: Only extract PROPER NAMES of characters (e.g., レイ, シオナ, マッド).\n"
                    "DO NOT include personal pronouns like ボク, 私, 俺, あたし, わたし, 僕, etc.\n"
                    "Focus on finding actual character names only.\n"
                    f"Text chunk:\n{chunk}"
                )
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying character names in Japanese text. You understand the difference between proper names and personal pronouns."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=CharacterDiscovery,
                )
                
                characters = response.choices[0].message.parsed.characters
                all_characters.update(characters)
                print(f"  Found {len(characters)} characters in chunk {i+1}")
                
            except Exception as e:
                print(f"  Error in chunk {i+1}: {e}")
        
        initial_list = sorted(list(all_characters))
        print(f"\nPhase 1 complete: Found {len(initial_list)} unique characters")
        return initial_list

    def _validate_and_complete(self, chunks: List[str], initial_characters: List[str]) -> List[str]:
        """Phase 2: Validate and find missing characters."""
        print("\n=== Phase 2: Character Validation & Completion ===")
        complete_characters = set(initial_characters)
        
        for i, chunk in enumerate(chunks):
            print(f"Validating chunk {i+1}/{len(chunks)}...")
            try:
                prompt = f"""
                Current character list: {json.dumps(sorted(list(complete_characters)), ensure_ascii=False)}
                
                Read this text chunk carefully and check:
                1. Are there any PROPER CHARACTER NAMES that appear but are NOT in the current list?
                2. Are there alternative names/nicknames for existing characters?
                
                IMPORTANT: Only add PROPER NAMES (e.g., レイ, シオナ, マッド).
                DO NOT add personal pronouns (ボク, 私, 俺, etc.) to the character list.
                
                Text chunk:
                {chunk}
                
                Return only missing proper character names.
                """
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "You are an expert at validating character names in Japanese text. You understand the difference between proper names and personal pronouns."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=CharacterValidation,
                )
                
                result = response.choices[0].message.parsed
                if result.missing_characters:
                    complete_characters.update(result.missing_characters)
                    print(f"  Added {len(result.missing_characters)} missing characters")
                else:
                    print(f"  No missing characters found")
                    
            except Exception as e:
                print(f"  Error validating chunk {i+1}: {e}")
        
        final_list = sorted(list(complete_characters))
        print(f"\nPhase 2 complete: Total {len(final_list)} characters after validation")
        return final_list

    def _tag_speakers(self, chunks: List[str], complete_characters: List[str]) -> List[str]:
        """Phase 3: Tag speakers in all chunks with complete character list."""
        print("\n=== Phase 3: Speaker Tagging ===")
        tagged_chunks = []
        
        for i, chunk in enumerate(chunks):
            print(f"Tagging chunk {i+1}/{len(chunks)}...")
            try:
                prompt = f"""
                Character list: {json.dumps(complete_characters, ensure_ascii=False)}
                
                Tag each line in this Japanese text with the appropriate speaker.
                
                IMPORTANT RULES:
                1. Use [Character Name] for dialogue lines. Example: [シオナ]「ねぇ、レイ……手、つないでもいい？」
                2. Use [ナレーション] for narrative text. Example: [ナレーション]──空気が動いた。
                3. When you see personal pronouns (ボク, 私, 俺, わたし, あたし, etc.) in dialogue:
                   - Identify WHO is speaking based on context clues
                   - Use the actual character name from the provided list, NOT the pronoun
                   - Look at surrounding dialogue, actions, and who others are addressing
                4. NEVER use pronouns as character names - always map to proper names
                
                Example:
                - If someone says 「ボクは大丈夫」 and context shows it's レイ speaking
                - Tag it as: [レイ]「ボクは大丈夫」
                - NOT as: [ボク]「ボクは大丈夫」
                
                Use exact character names from the provided list.
                
                Text to tag:
                {chunk}
                """
                
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": "You are an expert at identifying speakers in Japanese visual novel text. You can analyze context to determine which character is speaking, even when they use personal pronouns."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=SpeakerTagging,
                )
                
                tagged_lines = response.choices[0].message.parsed.tagged_lines
                tagged_chunks.append("\n".join(tagged_lines))
                print(f"  Tagged {len(tagged_lines)} lines")
                
            except Exception as e:
                print(f"  Error tagging chunk {i+1}: {e}")
                # Fallback tagging
                lines = chunk.splitlines()
                tagged_lines = []
                for line in lines:
                    if line.strip():
                        if line.strip().startswith('「'):
                            tagged_lines.append(f"[Speaker]: {line}")
                        else:
                            tagged_lines.append(f"[ナレーション]: {line}")
                tagged_chunks.append("\n".join(tagged_lines))
        
        print(f"\nPhase 3 complete: Tagged {len(tagged_chunks)} chunks")
        return tagged_chunks

    def _merge_tagged_chunks(self, tagged_chunks: List[str]) -> str:
        """Merge tagged chunks, removing overlap duplicates."""
        if not tagged_chunks:
            return ""
        
        # For now, simple merge. Can be enhanced to handle overlaps intelligently
        merged_lines = []
        for chunk in tagged_chunks:
            lines = chunk.splitlines()
            merged_lines.extend(lines)
        
        # Remove exact duplicates while preserving order
        seen = set()
        unique_lines = []
        for line in merged_lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)
        
        return "\n".join(unique_lines)

    def _load_cached_result(self, text_hash: str) -> Optional[str]:
        """Load cached result if exists."""
        cache_file = os.path.join(self.result_cache_dir, f"{text_hash}.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _save_cached_result(self, text_hash: str, result: str, characters: List[str]):
        """Save result to cache."""
        # Save tagged text
        cache_file = os.path.join(self.result_cache_dir, f"{text_hash}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(result)
        
        # Save metadata
        meta_file = os.path.join(self.result_cache_dir, f"{text_hash}_meta.json")
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "characters": characters,
            "lines_count": len(result.splitlines()),
            "chunk_size": self.chunk_size
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def preprocess_full_text(self, full_text: str) -> str:
        """Main entry point: Process text through 3 phases."""
        text_hash = self._get_text_hash(full_text)
        
        # Check cache
        cached_result = self._load_cached_result(text_hash)
        if cached_result:
            print(f"Using cached result: {text_hash[:8]}...")
            return cached_result
        
        print(f"Processing new text: {text_hash[:8]}...")
        
        # Split into chunks
        chunks = self._split_into_chunks(full_text)
        
        # Phase 1: Initial character discovery
        initial_characters = self._initial_discovery(chunks)
        
        # Phase 2: Validate and complete character list
        complete_characters = self._validate_and_complete(chunks, initial_characters)
        
        # Phase 3: Tag speakers with complete list
        tagged_chunks = self._tag_speakers(chunks, complete_characters)
        
        # Merge results
        final_result = self._merge_tagged_chunks(tagged_chunks)
        
        # Save to cache
        self._save_cached_result(text_hash, final_result, complete_characters)
        
        print(f"\nProcessing complete!")
        print(f"Total characters: {len(complete_characters)}")
        print(f"Total lines: {len(final_result.splitlines())}")
        
        return final_result


# Utility functions remain the same
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
