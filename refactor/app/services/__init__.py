from functools import lru_cache
from refactor.app.services.translate_service import TranslateService

@lru_cache(maxsize=None)
def get_translate_service() -> TranslateService:
    return TranslateService()
