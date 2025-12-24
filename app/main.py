from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import router
from app.utils.logger import get_logger

logger = get_logger(__name__)
load_dotenv()

app = FastAPI(title="IntelliFraud Copilot (Agentic RAG MVP)")
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}
