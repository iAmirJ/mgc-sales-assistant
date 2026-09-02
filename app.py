"""MGC Sales Assistant - Streamlit UI.

This module ONLY handles the user interface. All retrieval, grounding,
calculation and generation logic lives in the ``rag`` package; the UI simply
calls the Assistant service and renders the result.

Chat history is stored in Streamlit session state (per browser session) and is
isolated PER MESSAGE: every assistant message carries its own ``sources``,
``breakdown``, ``conflicts`` and (optionally) ``evidence``. Nothing is shared
or accumulated across questions.

Request state is tracked entirely inside ``st.session_state``:

- ``messages``      : ordered list of {role, content, ...} dicts (chat history)
- ``is_generating`` : True while a RAG/LLM request is running
- ``pending_prompt``: the submitted question waiting to be processed

Flow: the user submits -> we store the user message, set ``is_generating`` and
``st.rerun()``. In the rerun the input is DISABLED and a spinner is shown while
``Assistant.answer()`` runs synchronously. On success/error we clear
``is_generating`` and ``st.rerun()`` to re-enable the controls. Duplicate
submits are impossible because the input is disabled during generation and the
pending prompt is consumed exactly once.
"""

from __future__ import annotations

import logging

import streamlit as st

from rag.assistant import Assistant, AssistantResponse
from rag.config import config
from rag.embeddings import build_query_embeddings
from rag.ingestion import load_and_chunk_documents
from rag.vectorstore import build_index, load_index

logger = logging.getLogger(__name__)

st.set_page_config(page_title="MGC Sales Assistant", page_icon="🏗️", layout="centered")


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "is_generating" not in st.session_state:
        st.session_state["is_generating"] = False
    if "pending_prompt" not in st.session_state:
        st.session_state["pending_prompt"] = None
    if "last_error" not in st.session_state:
        st.session_state["last_error"] = None


@st.cache_resource(show_spinner=False)
def _get_assistant() -> Assistant | None:
    """Load (or build) the index and return a ready Assistant.

    Cached so re-runs during a Streamlit session reuse the index instead of
    re-embedding the documents on every interaction.
    """
    if not config.gemini_api_key:
        return None

    query_embeddings = build_query_embeddings()

    vectorstore = load_index(query_embeddings)
    if vectorstore is None:
        documents = load_and_chunk_documents(config.docs_dir)
        if not documents:
            raise FileNotFoundError("No MGC Markdown documents found under docs/.")
        vectorstore = build_index(documents, query_embeddings)

    return Assistant(vectorstore=vectorstore, embeddings_query=query_embeddings)


def _assistant_message(prompt: str, response: AssistantResponse) -> dict:
    """Build the per-message assistant dict with its OWN isolated metadata."""
    return {
        "role": "assistant",
        "content": response.answer,
        "question": prompt,
        "sources": response.sources,
        "breakdown": response.breakdown.to_dict() if response.breakdown else None,
        "conflicts": [
            {"values": c.values, "explanation": c.explanation}
            for c in response.conflicts
        ],
        "evidence": [
            {
                "rank": doc.metadata.get("selection_rank"),
                "source_file": doc.metadata.get("source_file", ""),
                "section": doc.metadata.get("section", ""),
                "document_type": doc.metadata.get("document_type", ""),
                "score": doc.metadata.get("score"),
                "grounded_score": doc.metadata.get("grounded_score"),
                "preview": " ".join(doc.page_content.split())[:200],
            }
            for doc in response.evidence
        ],
    }


def _render_breakdown(bd: dict) -> None:
    """Render a stored per-message calculation breakdown."""
    st.markdown("**Calculation breakdown**")
    rows = [f"Base price: PKR {bd['base_price']:,.2f}"]
    for premium in bd["premiums"]:
        percent = premium["percent"]
        amount = bd["base_price"] * percent / 100.0
        rows.append(f"{premium['label']} (+{percent:g}%): PKR {amount:,.2f}")
    rows.append(f"**Final total: PKR {bd['final_total']:,.2f}**")
    st.markdown("\n\n".join(rows))


def _render_conflicts(conflicts: list[dict]) -> None:
    for conflict in conflicts:
        st.warning("⚠️ **Document conflict detected**")
        for source, value in conflict["values"].items():
            st.markdown(f"- `{source}`: `{value}`")
        st.markdown(conflict["explanation"])


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for source in sources:
        st.markdown(f"- `{source['source_file']}` — Section: {source['section']}")


def _render_debug_evidence(evidence: list[dict]) -> None:
    with st.expander("Developer: retrieved evidence"):
        for e in evidence:
            score = f"{e['score']:+.3f}" if e.get("score") is not None else "n/a"
            grounded = f"{e['grounded_score']:+.3f}" if e.get("grounded_score") is not None else "n/a"
            st.markdown(
                f"- **rank {e['rank']}** — `{e['source_file']}` | {e['section']} "
                f"| cos={score} | grounded={grounded}\n"
                f"  `{e['preview']}`"
            )


def _render_history(debug_mode: bool) -> None:
    """Render stored messages; each assistant message shows ITS OWN metadata."""
    for message in st.session_state["messages"]:
        role = message["role"]
        if role == "user":
            with st.chat_message("user"):
                st.write(message["content"])
            continue

        with st.chat_message("assistant"):
            st.write(message["content"])
            if message.get("breakdown"):
                _render_breakdown(message["breakdown"])
            if message.get("conflicts"):
                _render_conflicts(message["conflicts"])
            _render_sources(message.get("sources", []))
            if debug_mode and message.get("evidence"):
                _render_debug_evidence(message["evidence"])


def main() -> None:
    _init_state()
    st.title("MGC Sales Assistant")
    st.caption(
        "Answers are grounded only in the supplied MGC documents. "
        "Price breakdowns are computed deterministically."
    )

    debug_mode = st.sidebar.checkbox(
        "Developer mode: show retrieved evidence",
        value=False,
        help="Hides nothing from the normal chat; adds a per-message "
        "evidence inspector (retrieval scores/sources) for debugging.",
    )

    st.sidebar.divider()
    st.sidebar.header("Lead Scoring")
    
    with st.sidebar.form("lead_scoring_form"):
        source = st.selectbox("Source", ["Facebook Ads", "Property Portal", "Google Search", "Instagram", "Referral", "Walk-in", "WhatsApp Campaign", "Expo Stall", "Billboard"])
        city = st.selectbox("City", ["Islamabad", "Rawalpindi", "Lahore", "Karachi", "Peshawar", "Faisalabad", "Multan", "Gujranwala", "Abbottabad"])
        area = st.selectbox("Area", ["Blue World City", "Gulberg Greens", "Top City", "B-17", "Park View City", "GT Road Corridor", "Bahria Town", "Bani Gala", "Chakri Road", "DHA", "Unknown"])
        property_type = st.selectbox("Property Type", ["Apartment", "Plot", "Villa", "Commercial Shop", "Penthouse", "Farmhouse"])
        budget_pkr_lac = st.number_input("Budget (PKR lac)", min_value=0, value=100)
        bedrooms = st.number_input("Bedrooms", min_value=0, value=2)
        is_overseas = st.checkbox("Overseas?")
        referred_by_existing_client = st.checkbox("Referred by Existing Client?")
        has_financing_approved = st.checkbox("Financing Approved?")
        
        predict_submitted = st.form_submit_button("Predict Conversion")
        
        if predict_submitted:
            # We don't want to break the chat state, so we handle it synchronously here 
            # since prediction is very fast.
            from ml.predict import predict_lead
            
            lead_data = {
                "source": source,
                "city": city,
                "area": area,
                "property_type": property_type,
                "budget_pkr_lac": budget_pkr_lac,
                "bedrooms": bedrooms,
                "is_overseas": int(is_overseas),
                "referred_by_existing_client": int(referred_by_existing_client),
                "has_financing_approved": int(has_financing_approved)
            }
            
            try:
                result = predict_lead(lead_data)
                
                st.sidebar.markdown(f"**Prediction:** {result['label']}")
                st.sidebar.markdown(f"**Probability:** {result['probability']*100:.1f}%")
                st.sidebar.caption("Model: GradientBoostingClassifier")
            except Exception as e:
                st.sidebar.error(f"Prediction failed: {e}")

    if not config.gemini_api_key:
        st.error(
            "GEMINI_API_KEY is not set. Copy .env.example to `.env`, add your "
            "Google AI Studio key, and restart the app."
        )
        return

    try:
        assistant = _get_assistant()
    except Exception as exc:  # surface setup errors clearly, don't swallow
        st.error(f"Failed to initialise the assistant: {exc}")
        return

    _render_history(debug_mode)

    # A persisted, user-friendly error from a failed generation attempt stays
    # visible across the rerun that re-enables the controls.
    if st.session_state["last_error"]:
        st.error(st.session_state["last_error"])

    # ------------------------------------------------------------------
    # Request-state gate. While a request is running the input is disabled
    # and the pending prompt is processed exactly once, synchronously, with
    # a visible spinner. Duplicate submits are structurally impossible.
    # ------------------------------------------------------------------
    if st.session_state["is_generating"]:
        st.chat_input("Ask about MGC Aurora Heights...", disabled=True)

        pending = st.session_state.pop("pending_prompt", None)
        if pending is not None:
            try:
                with st.spinner("Searching MGC documents and generating answer..."):
                    response = assistant.answer(pending)
            except Exception as exc:
                logger.exception("Generation failed for: %s", pending)
                st.session_state["last_error"] = (
                    f"Sorry, I couldn't generate an answer. This is usually a "
                    f"temporary Google API issue (e.g. rate limit). Please try "
                    f"again. Details: {exc}"
                )
            else:
                st.session_state["messages"].append(
                    _assistant_message(pending, response)
                )
                st.session_state["last_error"] = None
            finally:
                st.session_state["is_generating"] = False
                st.rerun()
        else:
            # Stale flag (e.g. app interrupted mid-request): self-heal.
            st.session_state["is_generating"] = False
            st.rerun()
        return

    prompt = st.chat_input(
        "Ask about MGC Aurora Heights (price, payment plan, booking policy)..."
    )
    if not prompt or not prompt.strip():
        return

    st.session_state["messages"].append({"role": "user", "content": prompt.strip()})
    st.session_state["pending_prompt"] = prompt.strip()
    st.session_state["is_generating"] = True
    st.rerun()


if __name__ == "__main__":
    main()