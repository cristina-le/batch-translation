from functools import lru_cache
from refactor.app.services.translator import Translator

@lru_cache(maxsize=None)
def get_translate_service() -> Translator:
    return Translator()
