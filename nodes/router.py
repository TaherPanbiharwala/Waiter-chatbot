# router.py
from typing import List, Dict, Any, Optional
import os, re, json
from pydantic import BaseModel, Field

from state import ChatStateTD, ChatStateModel, MessageModel
from utils.validation import validate_node
from utils.llm import generate_json


class RouterInput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]] = Field(default_factory=list)

class RouterOutput(BaseModel):
    session_id: str
    messages: List[Dict[str, str]]

#regex helpers
_ADD_VERBS = re.compile(
    r"\b(add|get\s+me|give\s+me|(?:i'll|i will)\s+have|i\s+want|(?:we(?:'ll| will))\s+have|order|take)\b",
    re.I
)
_REMOVE_VERBS = re.compile(r"\b(remove|delete|cancel|drop|take\s*out)\b|(?:\bno\s+more\b)|\bwithout\b", re.I)
_NUM_LIST = re.compile(r"^\s*\d+(?:\s*(?:,|and)\s*\d+)*\s*$", re.I)
_QTY_NAME = re.compile(r"\b\d+\s*x?\s+[a-z]", re.I)
_QUIT_WORDS = re.compile(r"\b(quit|exit|goodbye|bye|done|finished|thanks)\b", re.I)
_EXIT_WORDS = re.compile(r"\b(quit|exit|bye|goodbye)\b", re.I)

# Payment regex
_BILL_WORDS = re.compile(r"\bbill|check|receipt|total\b", re.I)
_SPLIT_WORDS = re.compile(r"\bsplit\b", re.I)
_PAY_WORDS = re.compile(r"\bpay|payment|checkout|settle\b", re.I)
_FEEDBACK_WORDS = re.compile(r"\bfeedback|review|rating|comment\b", re.I)
# Add near the other top-level regexes so it’s available:
_DONE_WORDS = re.compile(r"\b(done|submitted|finished)\b", re.I)


def _facet_from_text(text: str) -> Dict[str, Any]:
    t = (text or "").lower()
    cat = None
    if "starter" in t: cat = "starter"
    elif "main course" in t or "main" in t: cat = "main"
    elif "dessert" in t or "sweet" in t: cat = "dessert"
    elif "drink" in t or "beverage" in t: cat = "beverage"
    return {
        "veg": ("veg" in t and "non veg" not in t and "non-veg" not in t),
        "nonveg": ("non veg" in t or "non-veg" in t),
        "category": cat,
    }

_MENU_WORDS = re.compile(
    r"\b(menu|show.*menu|see.*menu|starters?|mains?|main course|desserts?|drinks?|beverages?)\b",
    re.I
)

@validate_node(name="NLU_Router", tags=["router"], input_model=RouterInput, output_model=RouterOutput)
def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    msgs = state.get("messages", [])
    md: Dict[str, Any] = state.setdefault("metadata", {})

    last_user = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
    lu = (last_user or "").strip().lower()
    stage: Optional[str] = (md.get("stage") or "").lower()

    # ---------- 0. Hard exit ----------
    if _EXIT_WORDS.search(lu):
        md["last_intent"] = "app.exit"
        md["route"] = None
        md["stage"] = None
        return state

    # ---------- 1. Facets from text (for menu filters) ----------
    md["facets"] = _facet_from_text(last_user)

    # ---------- 2. LLM classification (primary) ----------
    schema_hint = '{"intent": "chitchat", "slots": {}}'
    system = f"""
    You are an NLU classifier for a restaurant assistant.
    Current stage: {stage or "none"}.

    Valid intents:
      - ordering.lookup   (see menu items)
      - ordering.more     (ask for more options)
      - ordering.take     (add or remove items to/from cart)
      - confirm.yes       (confirm order)
      - confirm.no        (reject order)
      - payment.bill      (ask to show/generate bill)
      - payment.split     (ask to split bill)
      - payment.pay       (proceed to payment)
      - payment.feedback  (give feedback)
      - chitchat          (small talk, greetings, unrelated chat)

    Rules:
    - Use ordering.take only when the user wants to add/remove dishes.
    - Use payment.split when the user asks to split the bill, or when they reply
      with just a number while we are asking "How many people should I split between?"
    - Use payment.pay when the user wants to settle/pay.
    - Use payment.bill when they ask for the bill/total/check.
    - Use ordering.lookup when they ask to see the menu, items, starters/mains/etc.
    Output ONLY a JSON object with fields: intent (string), slots (object).
    """
    parsed = generate_json(system=system, user=last_user, schema_hint=schema_hint, max_tokens=64)
    label, slots = parsed.get("intent", "chitchat"), parsed.get("slots", {})

    # ---------- 3. Minimal deterministic overrides ----------

    # 3a. Feedback "done" detector
    if md.get("awaiting_feedback"):
        if _DONE_WORDS.search(lu) or re.search(r"\b[1-5]\b", lu) or re.search(r"\b(thanks|thank you|great|nice)\b", lu):
            md["awaiting_feedback"] = False
            label = "chitchat"

    # 3b. Strong menu words → ordering.lookup
    if _MENU_WORDS.search(lu):
        label = "ordering.lookup"

    # 3c. Payment keyword overrides (only the important ones)
    if _BILL_WORDS.search(lu):
        label = "payment.bill"
    if _SPLIT_WORDS.search(lu):
        label = "payment.split"
    if _PAY_WORDS.search(lu):
        label = "payment.pay"
    if _FEEDBACK_WORDS.search(lu):
        label = "payment.feedback"

    # 3d. Stage-aware numeric answers
    if _NUM_LIST.fullmatch(lu):
        if stage in {"payment.split", "split_pending"}:
            # numeric reply to "How many people should I split between?"
            label = "payment.split"
        elif stage in {"take", "menu"} and not label.startswith("payment."):
            # picking menu options by number during ordering
            label = "ordering.take"
            slots = {**slots, "has_numbers": "True"}

    # 3e. Context-aware yes/no (tiny rule, still useful)
    YES_SET = {"yes","y","ok","okay","yeah","yep","confirm","place","proceed"}
    NO_SET  = {"no","n","nope","nah","cancel","change","not now"}

    if lu in YES_SET:
        # If we're around payment, treat as pay; otherwise confirmation
        if stage in {"split", "payment.split", "payment.gateway", "payment.bill", "bill"} or label.startswith("payment."):
            label = "payment.pay"
        else:
            label = "confirm.yes"
    elif lu in NO_SET:
        if stage == "confirm" or label.startswith("confirm."):
            label = "confirm.no"

    # ---------- 4. Update metadata / routes ----------
    md["last_intent"] = label
    md["last_slots"] = slots

    if label.startswith("ordering.") or label.startswith("confirm."):
        md["route"] = "order"
    elif label.startswith("payment."):
        md["route"] = "payment"
    else:
        md["route"] = None

    # Stage transitions (same as before, but using final label)
    if label == "quit":
        md["stage"] = None
        return state
    if label in {"ordering.lookup","ordering.more"}:
        md["stage"] = "menu"
        if label == "ordering.more":
            md["page"] = int(md.get("page", 0)) + 1
    elif label == "ordering.take":
        md["stage"] = "take"
    elif label.startswith("confirm."):
        md["stage"] = "confirm"; md["confirm_signal"] = label
    elif label == "payment.bill":
        md["stage"] = "payment.bill"
    elif label == "payment.split":
        md["stage"] = "payment.split"
    elif label == "payment.pay":
        md["stage"] = "payment.gateway"
    elif label == "payment.feedback":
        md["stage"] = "payment.feedback"
    else:
        md["stage"] = None

    return state