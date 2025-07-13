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
    SPEAKER_AWARE = False
    QUALITY_THRESHOLD = 9.0
    CONTEXT_WINDOW = 6