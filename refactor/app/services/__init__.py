from functools import lru_cache
from app.services.translate_service import TranslateService

@lru_cache(maxsize=None)
def get_translate_service() -> TranslateService:
    return TranslateService()
