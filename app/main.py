from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import Agent
from app.catalog import Catalog
from app.llm import get_backend
from app.retriever import get_retriever
from app.schemas import ChatRequest, ChatResponse, HealthResponse

app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_catalog = Catalog.load()
_retriever = get_retriever(_catalog)
_backend = get_backend()
_agent = Agent(_catalog, _retriever, _backend)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    # API is stateless: every call carries the full history, nothing is
    # persisted server-side between calls.
    return _agent.handle(request.messages)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
