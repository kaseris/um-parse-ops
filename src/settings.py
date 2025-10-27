from dataclasses import dataclass

@dataclass
class Settings:
    PORT: int = 8000

settings = Settings()