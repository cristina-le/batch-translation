from typing import Literal

class Constants:
    # File Path Constants
    DATA_DIR = "app/data"
    OUTPUT_FILE = f"{DATA_DIR}/output.json"

    # File type constants
    EXPECTED_TYPE = Literal["text"]

    # Supported MIME types for text files (including CSV)
    SUPPORTED_TEXT_MIME_TYPES = [
        "text/plain",
        "text/txt",
        "application/txt"
    ]

    # Supported file extensions for text files
    VALIDATE_TEXT_EXT = ["txt"]

    LLM_BASE_URL = "https://openrouter.ai/api/v1"
    MODEL = "google/gemini-2.0-flash-001"
    TEMPERATURE = 0
    CHUNK_SIZE = 15
    CONTEXT_WINDOW = 6

    # Prompts
    SYSTEM_PROMPT = "You are a professional Japanese to English translator. Your goal is to produce translations that maximize BLEU score when compared to professional human translations."
    
    TRANSLATE_PROMPT = """
{%- if history %}
For context, here are the previous segments and their translations:
{%- for item in history %}
Previous Japanese:
{{ item.japanese }}

Previous English:
{{ item.english }}
{%- endfor %}
Now based on the previous translations:
{%- endif %}

For each line of the following Japanese text, translate it to English.

REQUIREMENTS:
- Maintain character speech patterns and personality.
- Preserve Japanese honorifics where appropriate.
- Keep cultural references intact.
- Ensure emotional nuances are conveyed.
- Use natural, flowing English suitable for high-quality localization.
- Do not create empty lines in your translation.

CRITICAL: 
- Your translation MUST have EXACTLY {{ size }} lines, no more and no less.
- Return the result as JSON: {{"translated_outputs": ["English line 1", "English line 2", ..., "English line {{ size }}"]}}
- Pay special attention to pronouns and subject-object relationships.
- When translating actions, be clear about who is performing the action.

CONTEXT: Japanese text to translate:
{{ japanese_text }}

RESULT: Translation into English:
"""

