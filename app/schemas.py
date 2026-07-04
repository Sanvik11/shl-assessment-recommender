"""
Pydantic models for the /chat API.

The response schema is non-negotiable per the assignment brief:
{
  "reply": str,
  "recommendations": [{"name": str, "url": str, "test_type": str}, ...]  # 0, or 1-10 items
  "end_of_conversation": bool
}
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
