import re
from typing import Dict

def post_process_translation(translation: str) -> str:
    """
    Post-process translation để fix các vấn đề thường gặp.
    
    Args:
        translation: Raw translation text
        
    Returns:
        Processed translation text
    """
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
