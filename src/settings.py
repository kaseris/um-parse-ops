import os

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    PORT: int = 8000
    DOTS_OCR_ENDPOINT_ID: str = os.getenv("DOTS_OCR_ENDPOINT_ID")
    DEEPSEEK_OCR_ENDPOINT_ID = os.getenv("DEEPSEEK_OCR_ENDPOINT_ID")
    RUNPOD_API_KEY: str = os.getenv("RUNPOD_API_KEY")

settings = Settings()