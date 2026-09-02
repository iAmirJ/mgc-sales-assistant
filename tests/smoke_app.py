"""Smoke test for the Streamlit app using streamlit's AppTest.

Verifies the app script:
- boots without crashing (even with no GEMINI_API_KEY),
- renders the expected title / notice,
- that session_state chat history mechanism works (unsubmitted run).

This requires only Streamlit - no API key. It cannot execute the full Gemini
turn (that needs GEMINI_API_KEY and is covered by the live run).
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


def main() -> None:
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=30)
    at.run()

    # Collect any uncaught exceptions raised by the script.
    exceptions = [e.value for e in at.exception]
    if exceptions:
        for exc in exceptions:
            print("SCRIPT EXCEPTION:", exc)
        return 1

    titles = " | ".join(t.value for t in at.title)
    if "MGC Sales Assistant" not in titles:
        print("TITLE NOT FOUND; titles:", repr(titles))
        return 1
    print("TITLE OK ->", titles)

    # Without a key the app should show the setup notice; with a key it should
    # show the chat input. Either is a correct boot.
    from rag.config import config
    has_key = bool(config.gemini_api_key)
    if not has_key:
        errors = [e.value for e in at.error]
        if not any("GEMINI_API_KEY" in e for e in errors):
            print("EXPECTED API-KEY ERROR NOT SHOWN; errors:", errors)
            return 1
        print("API-KEY NOTICE OK")
    else:
        if not at.chat_input:
            print("CHAT INPUT NOT RENDERED despite API key being set")
            return 1
        print("CHAT INPUT OK (API key set)")

    # Session state exists (messages list initialised).
    has = "messages" in at.session_state
    messages = at.session_state["messages"] if has else None
    if messages is None:
        print("SESSION STATE messages NOT initialised")
        return 1
    print("SESSION STATE OK:", messages)

    # Request-state flags initialised and idle.
    try:
        generating = at.session_state["is_generating"]
    except KeyError:
        generating = None
    try:
        pending = at.session_state["pending_prompt"]
    except KeyError:
        pending = "sentinel-missing"
    if generating is not False:
        print(f"FAIL: is_generating should be False on first boot (got {generating!r})")
        return 1
    if pending not in (None,):
        print(f"FAIL: pending_prompt should be None on first boot (got {pending!r})")
        return 1
    print("REQUEST STATE OK: is_generating=False, pending_prompt=None")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())