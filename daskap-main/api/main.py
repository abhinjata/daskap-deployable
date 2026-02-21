import os
from typing import Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

app = FastAPI(title="Daskap API", version="1.0.0")

allowed_origins = [
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://*.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    system_prompt: Optional[str] = (
        "You are Daskap, a helpful community insights assistant. Keep responses concise and empathetic."
    )


class ChatResponse(BaseModel):
    reply: str


@app.get("/health")
def health_check():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is missing on the server. Add it to .env or deployment environment variables.",
        )

    user_message = payload.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": payload.system_prompt},
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.4,
    }

    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=body,
            )
        response.raise_for_status()
        data = response.json()
        reply = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=reply.strip())
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        raise HTTPException(status_code=502, detail=f"Groq API error: {detail}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc
