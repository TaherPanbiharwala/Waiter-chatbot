# streamlit_app.py — process input first, then render
import os, json
from pathlib import Path
import requests
import streamlit as st
import uuid

API_BASE   = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
URL_CHAT   = f"{API_BASE}/chat"
URL_STATE  = f"{API_BASE}/state"
ss = st.session_state
if "SESSION_ID" not in ss:
    ss["SESSION_ID"] = f"streamlit-{uuid.uuid4().hex[:8]}"
SESSION_ID = ss["SESSION_ID"]



st.set_page_config(page_title="AI Waiter", layout="wide")
ss = st.session_state
ss.setdefault("chat", [])
ss.setdefault("total", 0)
ss.setdefault("last_order_total", 0)
ss.setdefault("stage", None)

def refresh_state():
    try:
        r = requests.get(URL_STATE, params={"session_id": SESSION_ID}, timeout=10)
        if r.ok:
            stt = r.json() or {}
            ss["total"] = int(stt.get("total", 0) or 0)
            ss["stage"] = stt.get("stage")
            lo = stt.get("last_order") or {}
            ss["last_order_total"] = int(lo.get("total", 0) or 0)
    except Exception:
        pass

def send_message(text: str):
    try:
        with st.spinner("Waiter is typing…"):
            r = requests.post(
                URL_CHAT,
                json={"session_id": SESSION_ID, "user_message": text},
                timeout=60,
            )
        if r.ok:
            payload = r.json() or {}
            reply = payload.get("response") or "(empty response)"
            ss["chat"].append(("ai", reply))
            # prefer state from same response; fallback to GET
            stt = payload.get("state")
            if stt:
                ss["total"] = int(stt.get("total", 0) or 0)
                ss["stage"] = stt.get("stage")
                lo = stt.get("last_order") or {}
                ss["last_order_total"] = int(lo.get("total", 0) or 0)
            else:
                refresh_state()
        else:
            ss["chat"].append(("ai", f"(backend error {r.status_code})"))
    except Exception as e:
        ss["chat"].append(("ai", f"(connection error: {e})"))

# ---------- initial fetch once ----------
if not ss.get("_did_init"):
    refresh_state()
    ss["_did_init"] = True

# ---------- handle input FIRST ----------
user_text = st.chat_input("Message the waiter…")
if user_text:
    msg = user_text.strip()
    if msg:
        ss["chat"].append(("user", msg))
        send_message(msg)

# ---------- then render chat ----------
st.subheader("Chat with your AI Waiter")
for role, msg in ss["chat"]:
    who = "👤 **You:**" if role == "user" else "☻ **Waiter:**"
    st.markdown(f"{who} {msg}")

# ---------- total only after order is placed ----------
st.markdown("---")
st.subheader("💰 Total Order")
total_to_show = ss.get("last_order_total") or 0   # only show confirmed order total
st.write(f"₹{total_to_show}")