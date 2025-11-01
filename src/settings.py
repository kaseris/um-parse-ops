import os

from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Settings:
    PORT: int = 8000
    RUNPOD_ID: str = os.getenv("VLLM_ENDPOINT_ID")
    RUNPOD_API_KEY: str = os.getenv("DOTS_OCR_TEST_KEY") # TODO: change it

settings = Settings()