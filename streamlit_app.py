# streamlit_app.py — chat + visible menu
import os
import json
from pathlib import Path
import uuid

import requests
import streamlit as st

# -------------------------
# Backend URLs
# -------------------------
API_BASE  = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
URL_CHAT  = f"{API_BASE}/chat"
URL_STATE = f"{API_BASE}/state"

# Where to load the menu from (same default as your rag.py, but you can override)
MENU_JSON_PATH = os.environ.get("MENU_JSON_PATH", "menu.json")

st.set_page_config(page_title="AI Waiter", layout="wide")

ss = st.session_state
if "SESSION_ID" not in ss:
    ss["SESSION_ID"] = f"streamlit-{uuid.uuid4().hex[:8]}"
SESSION_ID = ss["SESSION_ID"]

ss.setdefault("chat", [])
ss.setdefault("total", 0)
ss.setdefault("last_order_total", 0)
ss.setdefault("stage", None)


# -------------------------
# Helpers
# -------------------------
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


@st.cache_data
def load_menu(path: str):
    """Load menu.json and normalise it to a simple list of dicts."""
    p = Path(path)
    if not p.exists():
        return []

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "items" in data:
        items = data["items"]
    elif isinstance(data, list):
        items = data
    else:
        return []

    out = []
    for it in items:
        out.append(
            {
                "name": str(it.get("name") or it.get("title") or "Unknown item"),
                "description": str(it.get("description") or it.get("desc") or ""),
                "category": str(it.get("category") or it.get("type") or "Other"),
                "price": float(it.get("price") or it.get("cost") or 0),
                "tags": it.get("tags") or it.get("labels") or [],
            }
        )
    return out


# -------------------------
# One-time initial state
# -------------------------
if not ss.get("_did_init"):
    refresh_state()
    ss["_did_init"] = True

# -------------------------
# Layout: chat (left) + menu (right)
# -------------------------
col_chat, col_menu = st.columns([2, 1])

# ---------- handle input FIRST (in the chat column) ----------
with col_chat:
    user_text = st.chat_input("Message the waiter…")
    if user_text:
        msg = user_text.strip()
        if msg:
            ss["chat"].append(("user", msg))
            send_message(msg)

# ---------- render chat & total ----------
with col_chat:
    st.subheader("Chat with your AI Waiter")
    for role, msg in ss["chat"]:
        who = "👤 **You:**" if role == "user" else "☻ **Waiter:**"
        st.markdown(f"{who} {msg}")

    st.markdown("---")
    st.subheader("💰 Total Order")
    total_to_show = ss.get("last_order_total") or 0  # only show confirmed order total
    st.write(f"₹{total_to_show}")

# ---------- render menu ----------
with col_menu:
    st.subheader("📋 Menu")

    menu_items = load_menu(MENU_JSON_PATH)
    if not menu_items:
        st.info("Menu not found. Make sure `menu.json` exists or set `MENU_JSON_PATH` in your .env.")
    else:
        # Filters
        all_categories = sorted({m["category"] for m in menu_items if m["category"]})
        category = st.selectbox("Category", ["All"] + all_categories, index=0)

        diet = st.selectbox("Diet", ["All", "Veg", "Non-veg"], index=0)
        search = st.text_input("Search dishes")

        def is_veg_item(tags):
            t = " ".join(str(x).lower() for x in (tags or []))
            if "non veg" in t or "non-veg" in t:
                return False
            return "veg" in t

        filtered = []
        for item in menu_items:
            if category != "All" and item["category"] != category:
                continue
            if diet == "Veg" and not is_veg_item(item["tags"]):
                continue
            if diet == "Non-veg" and is_veg_item(item["tags"]):
                continue
            if search:
                s = search.lower()
                if s not in item["name"].lower() and s not in item["description"].lower():
                    continue
            filtered.append(item)

        if not filtered:
            st.write("No dishes match your filters.")
        else:
            for it in filtered:
                with st.container(border=True):
                    st.markdown(f"**{it['name']}**  \n₹{int(it['price'])}")
                    if it["description"]:
                        st.caption(it["description"])
                    if it["tags"]:
                        st.caption(" • ".join(map(str, it["tags"])))