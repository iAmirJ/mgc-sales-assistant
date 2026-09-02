"""Live end-to-end test of the Streamlit chat flow using AppTest.

Requires GEMINI_API_KEY in .env (the app calls the real Gemini API to answer).
Verifies:
- the app boots and renders the title,
- a user's question produces an assistant answer,
- session chat history is retained (both messages appended),
- a second turn continues the same conversation.

Usage: python tests/smoke_app_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def main() -> int:
    from rag.config import config

    if not config.gemini_api_key:
        print("SKIP: GEMINI_API_KEY not set (this test needs the live API).")
        return 2

    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()

    excs = [e.value for e in at.exception]
    if excs:
        print("SCRIPT EXCEPTIONS:", excs)
        return 1

    if not any("MGC Sales Assistant" in t.value for t in at.title):
        print("TITLE missing")
        return 1

    # Turn 1: ask about the base price.
    at.chat_input[0].set_value("What is the base price of a 2-bed in Block B?").run()
    excs = [e.value for e in at.exception]
    if excs:
        print("TURN 1 EXCEPTIONS:", excs)
        return 1

    for a in at.chat_message:
        role = a.get("role")
        txt = " ".join(str(a.get("value")))
        print(f"[{role}] {txt[:90]}")

    messages = at.session_state["messages"]
    print("SESSION MESSAGES:", messages)
    if len(messages) != 2:
        print("FAIL: expected 2 messages after one turn, got", len(messages))
        return 1
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        print("FAIL: roles not user -> assistant")
        return 1
    if "22,425,000" not in messages[1]["content"]:
        print("FAIL: assistant did not answer with the Block B 2-bed base price")
        print("  got:", messages[1]["content"][:200])
        return 1
    print("TURN 1 OK: user question stored, grounded answer stored in session history.")

    # Source isolation: turn 1 assistant message must carry ONLY Block B sources.
    m1_sources = [(s["source_file"], s["section"]) for s in messages[1].get("sources", [])]
    print("TURN 1 SOURCES:", m1_sources)
    if m1_sources != [("02_price_list_payment_plan.md", "Base Prices (Block B)")]:
        print("FAIL: turn 1 sources not isolated to Base Prices (Block B):", m1_sources)
        return 1
    print("SOURCE ISOLATION OK: turn 1 carries only its own Block B sources.")

    # Turn 2: follow-up, same session -> history should grow to 4 messages.
    at.chat_input[0].set_value("What is the transfer fee?").run()
    excs = [e.value for e in at.exception]
    if excs:
        print("TURN 2 EXCEPTIONS:", excs)
        return 1

    messages = at.session_state["messages"]
    print("SESSION MESSAGES AFTER TURN 2:", len(messages))
    if len(messages) != 4:
        print("FAIL: expected 4 messages after two turns, got", len(messages))
        return 1
    # The conversation should preserve the first exchange.
    if "What is the base price of a 2-bed in Block B?" in messages[0]["content"]:
        print("HISTORY RETENTION OK: first user question retained across turns.")
    else:
        print("FAIL: first turn lost")
        return 1
    # Transfer fee conflict should be visible.
    both = messages[3]["content"]
    if ("2%" in both and "2.5%" in both) or ("2.5%" in both and "2%" in both):
        print("CONFLICT SURFACED in turn 2 answer.")
    else:
        print("NOTE: conflict wording not detected in stored answer:", both[:200])

    # Source isolation across turns: the earlier turn's sources must remain
    # its own, and turn 2 must NOT inherit Block B pricing sources.
    m3_sources = [(s["source_file"], s["section"]) for s in messages[3].get("sources", [])]
    print("TURN 2 SOURCES:", m3_sources)
    expected2 = {
        ("03_booking_policy_faq.md", "Transfers"),
        ("02_price_list_payment_plan.md", "Other Charges"),
    }
    if set(m3_sources) != expected2:
        print("FAIL: turn 2 sources should be exactly Transfers + Other Charges:", m3_sources)
        return 1
    if ("02_price_list_payment_plan.md", "Base Prices (Block B)") in set(m3_sources):
        print("FAIL: turn 2 inherited Block B pricing sources")
        return 1
    # And turn 1 must have stayed untouched by turn 2.
    if messages[1].get("sources") != [{"source_file": "02_price_list_payment_plan.md",
                                       "section": "Base Prices (Block B)"}]:
        print("FAIL: turn 1 sources mutated by later turn:", messages[1].get("sources"))
        return 1
    print("SOURCE ISOLATION OK: each turn keeps exactly its own sources.")

    print("\nLIVE STREAMLIT TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())