# utils/db.py
import os, json
from typing import Any, Optional, Union, Dict, cast
import redis

from state import ChatStateModel, ChatStateTD, from_graph_state

# 🔹 Read REDIS_URL from env (Render/Compose can override this)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# 🔹 Create a single Redis client, reusable everywhere
_redis: redis.Redis = redis.Redis.from_url(REDIS_URL, decode_responses=False)

def save_state(session_id: str, state_dict: Dict[str, Any]) -> None:
    try:
        _redis.set(session_id, json.dumps(state_dict))
    except Exception as e:
        print(f"[DB Error] save_state failed: {e}")

def load_state(session_id: str) -> Optional[ChatStateModel]:
    try:
        raw: Optional[Union[bytes, bytearray, memoryview, str]] = _redis.get(session_id)
        if raw is None:
            return None

        # Redis usually returns bytes
        if isinstance(raw, (bytes, bytearray, memoryview)):
            s = bytes(raw).decode("utf-8")
        else:
            s = str(raw)

        as_dict: Dict[str, Any] = json.loads(s)
        return from_graph_state(cast(ChatStateTD, as_dict))
    except Exception as e:
        print(f"[DB Error] load_state failed: {e}")
        return None