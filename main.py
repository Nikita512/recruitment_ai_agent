from fastapi import FastAPI
import os
from dotenv import load_dotenv

# Load .env located in same folder as this file
base_dir = os.path.dirname(__file__)
load_dotenv(os.path.join(base_dir, ".env"))

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "openai_present": bool(os.getenv("OPENAI_API_KEY"))}
