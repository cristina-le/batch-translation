import os
from typing import List, Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()

class Context(BaseModel):
    """
    Pydantic model for translated outputs.
    """
    translated_outputs: List[str]

class EnhancedJapaneseToEnglishTranslator:
    """
    Enhanced translator with improved context preservation and speaker awareness.
    """
    def __init__(self, api_key: Optional[str] = None, temperature: float = 0.2, 
                 model = "openai/gpt-4o-mini", context_window: int = 3, 
                 speaker_aware: bool = True):
        """
        Initialize the translator with API key, temperature, model and context settings.
        
        Args:
            api_key: API key for OpenRouter
            temperature: Temperature for generation
            model: Model to use for translation
            context_window: Number of previous chunks to keep in context
            speaker_aware: Whether to use speaker diarization
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self.speaker_aware = speaker_aware
        self.context_history = []
        self.character_roles = {
            "speaker": "female",  # Main speaker is female
            "listener": "male"    # Listener/protagonist is male
        }

    def preprocess_with_speakers(self, text: str) -> str:
        """
        Add speaker tags to dialogue lines.
        This is a simple implementation that assumes lines starting with specific
        patterns are dialogue.
        
        Args:
            text: Input Japanese text
            
        Returns:
            Text with speaker tags added
        """
        if not self.speaker_aware:
            return text
            
        lines = text.splitlines()
        processed_lines = []
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                processed_lines.append(line)
                continue
                
            # Simple heuristic: lines that start with quotes or specific patterns
            # are likely dialogue from the main female character
            if (line.strip().startswith('「') or 
                line.strip().startswith('ねぇ') or
                line.strip().startswith('はい') or
                line.strip().startswith('ほら') or
                line.strip().startswith('あ') or
                line.strip().startswith('え') or
                line.strip().startswith('ん')):
                processed_lines.append(f"Female-Speaker: {line}")
            else:
                # Narration or description
                processed_lines.append(line)
                
        return "\n".join(processed_lines)

    def create_prompt(self, japanese_text: str, size: int) -> str:
        """
        Create a translation prompt with enhanced context and character awareness.
        
        Args:
            japanese_text: Japanese text to translate
            size: Expected number of lines in output
            
        Returns:
            Prompt for the translation model
        """
        # Add speaker tags if enabled
        if self.speaker_aware:
            japanese_text = self.preprocess_with_speakers(japanese_text)
        
        base_prompt = f"""
        TASK: For each line of the following Japanese text, translate it to English.

        REQUIREMENTS:
        - Maintain character speech patterns and personality.
        - Preserve Japanese honorifics where appropriate.
        - Keep cultural references intact.
        - Ensure emotional nuances are conveyed.
        - Use natural, flowing English suitable for high-quality localization.
        - Do not create empty lines in your translation.
        
        CHARACTER ROLES:
        - The main speaker is a {self.character_roles["speaker"]} character.
        - The listener/protagonist is a {self.character_roles["listener"]} character.
        - CRITICAL: Maintain the correct subject-object relationships in dialogue. 
          When the speaker refers to "you", she means the male listener.
          When actions are described, be clear about who is performing them.

        CRITICAL: 
        - Your translation MUST have EXACTLY {size} lines, no more and no less.
        - Return the result as JSON: "translated_outputs": ["English line 1", "English line 2", ..., "English line {size}"]
        - Pay special attention to pronouns and subject-object relationships.
        - Ensure that when the female speaker says "you", she is referring to the male listener.
        - When translating actions, be clear about who is performing the action.

        CONTEXT: Japanese text to translate:
        {japanese_text} 

        RESULT: Translation into English:
        """
    
        if self.context_history:
            context_prompt = f"""
            For context, here are the previous segments and their translations:
            """
            
            # Add previous context chunks
            for prev_context in self.context_history[-self.context_window:]:
                context_prompt += f"""
                Previous Japanese:
                {prev_context['japanese']}

                Previous English:
                {prev_context['english']}
                """
                
            context_prompt += f"""
            Now based on the previous translations:
            {base_prompt}
            """
            return context_prompt
            
        return base_prompt
    
    def translate(self, japanese_text: str, size: int) -> str:
        """
        Translate Japanese text to English, ensuring output matches required line count.
        
        Args:
            japanese_text: Japanese text to translate
            size: Expected number of lines in output
            
        Returns:
            Translated English text
        """
        prompt = self.create_prompt(japanese_text, size)
        
        output_lines = 0
        max_attempts = 3
        attempts = 0
        
        while output_lines != size and attempts < max_attempts:
            try:
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {"role": "system", "content": "You are a professional Japanese to English translator specializing in visual novels. Your goal is to produce translations that maximize BLEU score when compared to professional human translations. CRITICAL: You must follow all instructions exactly, especially regarding subject-object relationships and character roles."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format=Context,
                )
                translation = response.choices[0].message.content
                translation = json.loads(translation)
                translation = "\n".join(translation["translated_outputs"])
                output_lines = len(translation.splitlines())
                attempts += 1
            except Exception as e:
                print(f"Error during translation: {e}")
                attempts += 1
                if attempts >= max_attempts:
                    # Return best effort translation or error message
                    return f"Translation error after {max_attempts} attempts: {e}"

        # Store context for future translations
        self.context_history.append({
            'japanese': japanese_text,
            'english': translation
        })
        
        # Keep context history within the specified window
        if len(self.context_history) > self.context_window:
            self.context_history = self.context_history[-self.context_window:]
            
        return translation
