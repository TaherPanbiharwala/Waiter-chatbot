# main.py (text-only with POST /chat and /ws/chat)
from dotenv import load_dotenv
load_dotenv(override=True)  # <- make .env win over shell env

import os, pathlib
from typing import cast, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langsmith import traceable
import requests  
import redis     
from utils.nlu import classify
from utils.config import SYSTEM_PROMPT
from graph import build_graph
from state import ChatStateModel, MessageModel, ChatStateTD, to_graph_state, from_graph_state
from utils import db

# ---- startup / env -----------------------------------------
print("[CFG] USE_WAITER_LLM =", os.getenv("USE_WAITER_LLM"))
print("[CFG] WAITER_LLM_BACKEND =", os.getenv("WAITER_LLM_BACKEND"))

DATA_DIR = pathlib.Path("./data_audio")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def _pack_state(session_id: str):
    model = db.load_state(session_id)
    if not model:
        return {"cart": [], "total": 0, "stage": None, "last_order": {}}
    md = model.metadata or {}
    cart = md.get("cart", [])
    total = sum(int(it.get("price", 0)) * int(it.get("qty", 1)) for it in cart)
    return {"cart": cart, "total": total, "stage": md.get("stage"), "last_order": md.get("last_order") or {}}

def ensure_system_prompt(model: ChatStateModel) -> None:
    if not any(m.role == "system" for m in model.messages):
        model.messages.insert(0, MessageModel(role="system", content=SYSTEM_PROMPT))

def _ping_redis(timeout=0.5) -> bool:
    try:
        # reuse your existing client if exposed; else create a tiny throwaway
        r = getattr(db, "_redis", None)
        if r is None:
            r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), socket_timeout=timeout)
        return bool(r.ping())
    except Exception:
        return False

def _ping_ollama(timeout=0.8) -> bool:
    # only run if you actually use Ollama
    if os.getenv("WAITER_LLM_BACKEND", "disabled").lower() != "ollama":
        return True
    base = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=timeout)
        return resp.ok
    except Exception:
        return False

# ---- app / graph -------------------------------------------
app = FastAPI()
graph_app = build_graph()

origins = os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if os.getenv("CORS_ALLOW_ORIGINS") else ["http://localhost:8501"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    user_message: str

@traceable(name="turn", tags=["chat"])
def _handle_turn(model: ChatStateModel, user_text: str) -> ChatStateModel:
    user_text = (user_text or "").strip()
    if not user_text:
        return model
    ensure_system_prompt(model)
    stage = model.metadata.get("stage")
    _ = classify(user_text, stage=stage, history=[m.model_dump() for m in model.messages][-4:])
    model.messages.append(MessageModel(role="user", content=user_text))
    model.metadata["_awaiting_worker"] = True
    out_dict = graph_app.invoke(to_graph_state(model))
    return from_graph_state(cast(ChatStateTD, out_dict))

# ---- Plain REST endpoint for Streamlit ----
@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    model = db.load_state(req.session_id) or ChatStateModel(session_id=req.session_id)
    ensure_system_prompt(model)
    model = _handle_turn(model, req.user_message)
    db.save_state(req.session_id, model.model_dump())

    # ✅ Return the latest assistant message only
    reply = next((m.content for m in reversed(model.messages) if m.role == "assistant"),
                 "Hi! I’m your waiter — would you like the menu?")
    return {"response": reply}
# ---- WebSocket chat (optional if you want streaming) ----
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    session_id = None
    try:
        hello = await ws.receive_json()
        session_id = (hello.get("session_id") or "web") if isinstance(hello, dict) else "web"
        model = db.load_state(session_id) or ChatStateModel(session_id=session_id)
        ensure_system_prompt(model)
        await ws.send_json({"type": "hello", "ok": True, "state": _pack_state(session_id)})

        while True:
            msg = await ws.receive_json()
            text = (msg.get("text") or "").strip()
            if not text:
                continue

            model = _handle_turn(model, text)
            db.save_state(session_id, model.model_dump())
            reply = next((m.content for m in reversed(model.messages) if m.role == "assistant"), "Okay.")

            await ws.send_json({
                "type": "reply",
                "text": reply,
                "state": _pack_state(session_id)
            })

    except WebSocketDisconnect:
        pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass

@app.get("/healthz")
def healthz():
    # lightweight “liveness”: process is up and we can talk to Redis
    ok_redis = _ping_redis()
    status = 200 if ok_redis else 503
    return {"ok": ok_redis}, status

@app.get("/readyz")
def readyz():
    # “readiness”: dependencies (Redis + Ollama if enabled) look good
    ok_redis = _ping_redis()
    ok_ollama = _ping_ollama()
    ready = ok_redis and ok_ollama
    status = 200 if ready else 503
    return {"redis": ok_redis, "ollama": ok_ollama, "ready": ready}, status

@app.get("/state")
def get_state(session_id: str = Query(...)):
    return _pack_state(session_id)

# ---- CLI helpers -------------------------------------------
def _read_user() -> Optional[str]:
    try:
        return input("You: ").strip()
    except EOFError:
        return None

@traceable(name="terminal_turn", tags=["cli"])
def _cli_turn(m: ChatStateModel, text: str) -> ChatStateModel:
    return _handle_turn(m, text)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)