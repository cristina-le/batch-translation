import re

def post_process(translation: str) -> str:
    """
    Post-process translation to fix common issues.
    """
    # Normalize whitespace and punctuation
    translation = re.sub(r'\s+', ' ', translation)
    translation = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', translation)
    translation = re.sub(r'\s+([.!?,:;])', r'\1', translation)
    
    return translation.strip()
