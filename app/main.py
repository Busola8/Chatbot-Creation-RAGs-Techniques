from fastapi import FastAPI
from pydantic import BaseModel
from loguru import logger
from app.rag_pipeline import RAGPipeline
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title='Full RAG Chatbot')

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

pipeline = RAGPipeline()

class Ask(BaseModel):
    question: str
    top_k: int = 5

@app.get('/healthz')
def health():
    return {'status': 'ok'}

@app.post('/ask')
def ask(payload: Ask):
    logger.info(f"Question: {payload.question}")
    res = pipeline.answer(payload.question, top_k=payload.top_k)
    return res
