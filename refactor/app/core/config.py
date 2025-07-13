import os
from dotenv import load_dotenv

load_dotenv()

class Configs:
    # App settings
    APP_NAME = os.getenv("APP_NAME", "Secret Box Core")
    VERSION = os.getenv("VERSION", "0.5.0")

    # Azure OpenAI API settings
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
