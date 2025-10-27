import os
import sys

from fastapi import FastAPI
import uvicorn
sys.path.insert(0, os.path.abspath(os.getcwd()))

from src.settings import settings

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)