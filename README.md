# MGC Sales Assistant — Grounded Document AI

A production-minded, interview-defensible implementation of **Part 1**
(Grounded Document Assistant) of the MGC Developments technical build task.

A salesperson can ask natural-language questions about the MGC Aurora Heights
project and receive answers **grounded strictly in the supplied MGC documents**,
with sources shown, price calculations done deterministically in Python, and
safe abstention when the documents don't contain the answer.

---

## 1. Problem statement

MGC sales staff answer the same questions about price, payment plans and booking
policy over and over, usually by flipping through PDFs, and get it wrong often
enough to matter. This assistant answers those questions from the three supplied
MGC Markdown documents, showing the source, refusing to guess when the answer
isn't in the documents, and detecting when two documents disagree.

**Five hard cases the system must handle:**

| Question | Expected behaviour |
|---|---|
| "What's the base price of a 2-bed in Block B?" | Straight lookup, cite source |
| "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?" | Base price + stacked premiums, deterministic calculation |
| "What's the transfer fee?" | **Documents disagree (2% vs 2.5%). Surface the conflict.** |
| "What's the rental yield on a 1-bed?" | Not in the documents — abstain, don't invent |
| "Who is the anchor tenant?" | Explicitly "no anchor tenant confirmed" — say so |

---

## 2. Architecture

Two-step, retrieval-first RAG with a deterministic calculation layer. No agents,
no multi-agent orchestration, no LangGraph.

```
MGC Markdown documents
    ↓  Document loader (structure-aware)
Chunks with metadata (source_file, section, document_type, chunk_id, ...)
    ↓  Gemini text embeddings (RETRIEVAL_DOCUMENT)
FAISS vector index (persisted under vectorstore/)
    ↓  Query: embed (RETRIEVAL_QUERY) → similarity search (top-k)
Retrieved evidence + scores
    ↓  Evidence validation (relevance gate)   ← abstain if weak
    ↓  Conflict detection                     ← surface disagreements
    ↓  Deterministic calculator (Python)      ← pricing, transparent breakdown
Grounded Gemini response (constraint: answer ONLY from retrieved evidence)
    ↓
Streamlit UI + session chat history
```

### Component responsibilities

| Module | Responsibility |
|---|---|
| `rag/config.py` | Environment config via python-dotenv (API key, model names, top-k, paths) |
| `rag/ingestion.py` | Load Markdown, split on headings, attach metadata |
| `rag/embeddings.py` | Gemini text embeddings, document vs query task_type |
| `rag/vectorstore.py` | FAISS build / persist / load |
| `rag/retriever.py` | Intent-aware evidence selection (vector + metadata affinity); similarity scores |
| `rag/guardrails.py` | Relevance validation + cross-document conflict detection |
| `rag/calculator.py` | Deterministic price breakdown (no LLM arithmetic) |
| `rag/assistant.py` | Orchestration: retriever → guardrails → calculator → Gemini |
| `app.py` | Streamlit UI; per-message chat history + request-state management |
| `build_index.py` | Build/rebuild the FAISS index |

---

## 3. Folder structure

```
mgc-ai-task/
├── app.py                       # Streamlit UI
├── build_index.py               # Build/rebuild the vector index
├── requirements.txt
├── README.md
├── .env.example                 # Config template (copy to .env)
├── .gitignore                   # .env and generated artifacts ignored
├── BRIEF.md                     # The task brief (reference copy)
├── data/
│   └── leads.csv                # For Parts 2 & 3 (unused by Part 1)
├── docs/
│   ├── 01_mgc_aurora_heights_brochure.md
│   ├── 02_price_list_payment_plan.md
│   └── 03_booking_policy_faq.md
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── guardrails.py
│   ├── calculator.py
│   └── assistant.py
├── ml/                          # Reserved for Part 3
│   └── __init__.py
├── database/                    # Reserved for Part 2 (.gitkeep)
├── vectorstore/                 # Generated FAISS index (git-ignored)
├── tests/
│   ├── check_cases.py            # LIVE 5-case verifier (needs API key)
│   ├── verify_pipeline.py        # Deterministic logic verifier (no key)
│   ├── smoke_app.py              # Streamlit boot / session-state smoke test
│   ├── smoke_app_live.py         # LIVE Streamlit chat + source isolation test
│   └── debug_retrieval.py        # Dev tool: inspect raw vs selected retrieval
```

---

## 4. Technology choices

### Why Gemini embeddings (`gemini-embedding-001`)
The task is pure Markdown text. A text embedding model is the correct and
sufficient choice; a multimodal embedding model would add cost and complexity
with zero benefit here. `gemini-embedding-001` is Google's text embedding model,
accessed through LangChain's `GoogleGenerativeAIEmbeddings`, using
`task_type="RETRIEVAL_DOCUMENT"` at index time and `RETRIEVAL_QUERY` at query
time so the model optimises its representation for each stage.

### Why Gemini Flash (`gemini-2.5-flash`)
Fast, cost-effective, strong instruction-following — ideal for a grounded
answer step where the heavy lifting is done by retrieval and the deterministic
calculator, not by the model's world knowledge. Temperature is set to 0 and the
prompt hard-restricts the model to the retrieved evidence to minimise
hallucination.

### Why LangChain
We only adopt the pieces with real value: the standard `Document` type with
metadata, the canonical embedding/LLM wrappers for the Google Gemini integration,
the FAISS vector-store wrapper, and similarity search with scores. Everything
that is MGC-specific (chunking, extraction, guardrails, calculator, conflict
detection) is plain Python we own and can explain line by line. LangChain is a
wrapper library here, not a framework we bend the app around.

### Why FAISS
An embeddable, local similarity index that needs no server. Given three small
Markdown documents, a full vector database (Pinecone/Qdrant/Weaviate) would be
infrastructure for its own sake. FAISS is fast, trivially persisted locally, and
easily swapped for a production vector DB later. We index over L2-normalised
embeddings with the inner-product index so the retrieved scores are cosine
similarities in [-1, 1] (higher = more similar) — a natural, interpretable
signal for the relevance gate.

### Why the Python calculator
Banks of figures are where LLMs go wrong. All price arithmetic is done in
`rag/calculator.py`: retrieve the verified base price and premium percentages
**from the documents**, pass them into the calculator, and return a transparent
breakdown (base + each premium + total). The LLM is never asked to multiply
numbers; it is only asked to narrate the verified breakdown.

### Why Streamlit
The task asks for a minimal web interface. Streamlit gives a clean chat UI with
session state for free, no frontend code. The UI is a thin shell — all logic
sits in the `rag` package so it can be reused by any interface (CLI, FastAPI, …).

### Why session state (and only session state)
Chat history is stored in Streamlit's `st.session_state`, which persists for the
life of a browser session. This matches the task scope exactly and keeps the
assistant service contract clean (stateless `Assistant.answer(question)`), so a
persistent store can be added later without reworking anything.

**Current implementation stores chat history in Streamlit session state.
Persistent conversation storage can be added later using a database such as
PostgreSQL/SQLite with conversation and message tables.**

### Why two-step RAG and NOT LangGraph / agents
The problem is deterministic retrieval-first RAG, not an agentic multi-step
workflow. There is no decision tree, tool-using loop or multi-agent handoff to
orchestrate; a linear pipeline (retrieve → validate → calculate → generate)
fully covers the five required behaviours. LangGraph/agentic RAG would add
runtime complexity, latency and a larger hallucination surface without changing
any required outcome. The relevant "intelligence" — relevance gating, conflict
detection, deterministic arithmetic — is deliberate plain-Python guardrail code,
which is exactly what a reviewer can read and reason about.

---

## 5. Document ingestion & chunking strategy

- Documents are read from `docs/` as Markdown. **Markdown is the source of
  truth**; it is never converted to PDF and back.
- `ingestion.py` splits each file on `#`/`##` headings, so every chunk is a
  coherent section (e.g. "Base Prices (Block B)", "Location Premiums",
  "Transfers").
- Sections that fit the chunk-size cap (1800 chars) are kept whole, so related
  facts — a price table and its header, the four premium bullets, a single FAQ
  answer — stay together. Oversized sections are split on sentence boundaries.
- Every chunk carries metadata: `source_file`, `document_name`, `section`,
  `chunk_id`, `document_type` (price_list / booking_policy / brochure), plus a
  `header` field so a chunk is self-describing when retrieved for citation.

22 chunks are produced across the three documents.

---

## 6. Retrieval strategy

Two-stage, intent-aware evidence selection (`rag/retriever.py`):

1. **Vector candidate pool** — the question is embedded and the top
   `pool_k` (default 10) chunks are returned with true cosine similarities
   (`MAX_INNER_PRODUCT` over L2-normalised vectors; higher = more similar).
2. **Intent-aware evidence selection** — the question's topic is detected with
   generic lexical rules (pricing, transfer fee, rental yield, anchor tenant,
   amenities, payment plan, possession, NOC/approval). For a detected intent,
   only candidates whose `document_type` / section / content match that intent
   are kept (metadata filtering + keyword affinity). The survivors are
   re-ordered by a *grounded score* — cosine + small per-section affinity boost
   + a block-entity boost ("Block B" in the question boosts chunks whose
   section is "Base Prices (Block B)") — and selection cuts off once a chunk
   falls more than `margin` (0.05) below the best.

The result: a "2-bed Block B price" question returns **only** the
`Base Prices (Block B)` chunk, a total-price question returns `Base Prices
(Block B)` + `Location Premiums`, a transfer-fee question returns the two
conflicting sources, and so on. No unrelated Block A / Amenities / Notes /
Commercial Podium chunks leak into the sources shown for an answer.

**No answers are hard-coded.** Prices, percentages and totals are always read
from the retrieved document text. The raw cosine score stays unchanged in each
chunk's `score` metadata, which is what the relevance gate uses.

Retrieval is a pure service, reusable and testable independently of the UI and
the assistant. Use `python tests/debug_retrieval.py` to inspect exactly what
the retriever returns for a question (raw candidates vs. final selection).

---

## 7. Grounding & abstention strategy

The system **never** blindly hands retrieved text to the model:

1. **Relevance gate** (`guardrails.validate_evidence`): if the best retrieved
   cosine similarity between the query and the closest chunk is below a
   threshold (default `MIN_RELEVANCE_SCORE=0.60`), the assistant abstains with:
   > "I don't have enough information in the provided MGC documents to answer
   > this reliably."
2. **Conflict detection** (`guardrails.detect_conflicts`): when retrieved
   documents state different values for the same fact (the transfer fee),
   the system surfaces both values with their sources instead of choosing one.
3. **Constrained generation**: the system prompt hard-restricts Gemini to the
   retrieved evidence, forbids world knowledge, and requires every claim to be
   tied to a cited source and section.

No web search, no external knowledge, no model-world-knowledge scaffolding.

---

## 8. Conflict handling (transfer fee)

The price list states **"Transfer fee (before possession): 2% of the current
list price"**, while the booking policy states **"Transfer fee is 2.5% of the
current list price."** `detect_conflicts` scans retrieved lines that mention
"transfer" + a percentage, groups them by source file, and where two sources
differ, produces a conflict object:

> The supplied MGC documents contain conflicting information about the transfer
> fee and do not establish which figure is currently authoritative. Please
> confirm with MGC before quoting the fee to a customer.

Both values and both source filenames are shown; the LLM is instructed never to
silently select one.

---

## 9. Deterministic pricing calculation

Question → retrieve the relevant price-list chunk(s) →
`_maybe_calculate` extracts:
- the **base price** for the requested unit type/block from the price table,
- the applicable **premium percentages** from the "Location Premiums" bullets
  (floor tier, corner, Margalla-facing),

and feeds them into `calculate_with_premiums`, which produces a transparent
breakdown:

```
Base price: PKR 22,425,000.00
- Floor floors 13-19: +4% = +PKR 897,000.00
- Corner: +3% = +PKR 672,750.00
- Margalla-facing: +6% = +PKR 1,345,500.00
Subtotal of premiums: PKR 2,915,250.00
Final total: PKR 25,340,250.00
```

The expected figure (25,340,250) is used **as a validation assertion in tests,
never as a hard-coded answer**. The calculator only contains generic summing
logic; every number it uses comes from the retrieved documents. Premium
additivity is verified against the price list's own statement:
"Premiums are cumulative. A Margalla-facing corner unit on floor 15 carries
+4% +3% +6% = +13% over base."

---

## 10. Session chat history & request state

### Per-message isolation

- Stored in Streamlit `st.session_state["messages"]`. **Each message is fully
  self-contained**; sources/breakdown/conflicts never leak across turns:

  ```
  {"role": "user",      "content": "..."}
  {"role": "assistant", "content": "...",
   "sources":   [{"source_file": "...", "section": "..."}],
   "breakdown": {...} | None,
   "conflicts": [{"values": {...}, "explanation": "..."}],
   "evidence":  [{score, grounded_score, section, preview, ...}]}
  ```

- `sources`/`breakdown`/`conflicts`/`evidence` are built fresh inside
  `_assistant_message()` for exactly one response and rendered only under that
  response. There is no global sources list, no `sources.extend(...)`, and no
  session-level source accumulation.

- The assistant service itself is stateless (`Assistant.answer(question)`).

### Request-state management (no duplicate submissions)

- `is_generating` and `pending_prompt` are the only request-state flags, both
  in `st.session_state` (never module globals).
- On submit: the user message is appended, `pending_prompt` is set,
  `is_generating = True`, then `st.rerun()`.
- In the generating rerun the chat input is **disabled** (its Send control is
  disabled too), a spinner ("Searching MGC documents and generating answer...")
  runs while `Assistant.answer()` executes synchronously, and the pending
  prompt is consumed exactly once.
- On success or error `is_generating` is cleared and `st.rerun()` re-enables
  the input/Send. Errors are shown via `st.error()` (API failures, etc.) and
  the UI recovers. A stale `is_generating` flag (e.g. an interrupted run)
  self-heals on the next rerun.

Double-clicks, repeated Enter presses, and re-renders during generation cannot
trigger a second request because the input is disabled while generating and the
pending prompt is consumed exactly once.

> **Current implementation stores chat history in Streamlit session state.
> Persistent conversation storage can be added later using a database such as
> PostgreSQL/SQLite with conversation and message tables.**

---

## 11. Environment setup & installation

Prerequisites: Python 3.11+ (tested on 3.13).

```bash
cd mgc-ai-task

# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key
copy .env.example .env        # Windows
cp .env.example .env          # Linux/macOS
# Then edit .env and add your real key:
#   GEMINI_API_KEY=AIza...
```

Get a key from [Google AI Studio](https://aistudio.google.com/app/apikey).

## 12. Build the index

```bash
python build_index.py
```

This loads `docs/*.md`, chunks them, embeds with Gemini and persists the FAISS
index to `vectorstore/index.faiss`. The app will auto-build the index on first
run if it's missing, but pre-building is recommended so the UI starts instantly.

## 13. Run Streamlit

```bash
streamlit run app.py
```

Then open the printed local URL (usually http://localhost:8501).

---

## 14. Required test questions & expected behaviour

Verified by `tests/check_cases.py` (live, reports selected evidence + pass/fail),
`tests/smoke_app_live.py` (live Streamlit session + per-message source
isolation) and `tests/verify_pipeline.py` (deterministic, no API key).

| # | Question | Expected |
|---|---|---|
| 1 | What is the base price of a 2-bed in Block B? | **PKR 22,425,000** — cited to `02_price_list_payment_plan.md` / Base Prices (Block B) |
| 2 | What is the total for a Margalla-facing corner unit, floor 15, 2-bed Block B? | **PKR 25,340,250** via deterministic breakdown (base +4%+3%+6%), cited |
| 3 | What is the transfer fee? | **Conflict surfaced**: Price List 2% vs Booking Policy 2.5%; no side chosen |
| 4 | What is the rental yield on a 1-bed? | **Abstain** — MGC does not publish rental yield projections |
| 5 | Who is the anchor tenant? | **State it is not confirmed** — "no anchor tenant has been confirmed" |

These values are **expected outcomes**, not hard-coded answers. They are the
validation target; the system derives them from the documents at runtime.

The five live questions additionally verify that the retrieved evidence is
**precisely** the relevant section(s) and contains no unrelated chunks, and
that the persisted chat history keeps each assistant message's sources isolated
from every other turn.

---

## 15. Error handling

- Missing `GEMINI_API_KEY` → clear startup message; `.env` guidance.
- Missing documents / no Markdown found → explicit `FileNotFoundError`.
- FAISS index missing → `build_index.py` guidance; app auto-builds on first run.
- Empty/too-weak retrieval → relevance gate abstains with an explicit message.
- Gemini API/network/model errors → surfaced with a descriptive message, not
  silently swallowed.
- Malformed model responses → handled by wrapping generation in a try/except
  that raises a clear `RuntimeError`.

---

## 16. Security

- The API key lives only in `.env`, which `.gitignore` explicitly excludes and
  which is never imported into version control.
- The key is never printed or logged; it is read once via `python-dotenv`.
- Generated artifacts (`vectorstore/*.faiss`, `index.pkl`) are git-ignored.
- Source documents are treated as read-only ground truth and are never modified
  by the application.

---

## 17. Known limitations

- Retrieval combines a vector candidate pool with metadata affinity and keyword
  intent filtering, but still has no BM25/hybrid score or model-based
  reranker. Near-synonym phrasing can occasionally surface a less relevant
  chunk; the relevance gate, margin cutoff and grounding layers are the safety
  net.
- Conflict detection is written for the specific but generalisable pattern of
  "two documents state different percentages for the same fact". It handles the
  required transfer-fee case and similar numeric disagreements, not every
  conceivable contradiction.
- The pricing extractor understands the structure of the supplied price list
  (pipe-table rows, "Location Premiums" bullets). A differently structured
  follow-up document would need its extraction wildcards revisited.
- Live Gemini behaviour (Tests 1–5 end-to-end, real similarity scores) is run
  via `tests/check_cases.py` and requires the API key; the no-key tests cover
  everything downstream of retrieval. The free Gemini tier is rate-limited
  (~20 generation requests/day per key) — if you see `ResourceExhausted: 429`,
  the app retries automatically; exceeding the daily quota requires waiting or
  using a higher-tier key.
- Chat history is session-scoped and is lost when the browser session ends
  (by design for this task).
- A stray dev index (`vectorstore/dummy_test_*.faiss`) may occasionally linger
  on OneDrive-synced folders until sync settles; it is git-ignored and has no
  effect on the application.

---

## 18. Future improvements

- **Persistent conversation storage** — PostgreSQL/SQLite with conversation and
  message tables (the service layer is already stateless and ready for it).
- **Hybrid retrieval** — combine FAISS with keyword (BM25) retrieval and merge
  results for robustness on exact-fact lookups.
- **Reranking** — a cross-encoder reranker on the top-k candidates.
- **Evaluation dataset** — a labelled set of sales questions with expected
  grounded answers plus retrieval metrics (recall@k, MRR) to track quality.
- **Production vector database** — replace the local FAISS file with
  PostgreSQL + pgvector or Qdrant when the corpus grows.
- **Observability** — log retrieval scores, gate decisions and latency per query.
- **Authentication & audit** — restrict access and log who asked what.
- **Parts 2–4** — the schema/queries (`database/`), ML baseline (`ml/`), and the
  combined web UI are intentionally left for the following parts.

---

## 19. Tests

```bash
python tests/verify_pipeline.py   # deterministic logic, no API key
python tests/smoke_app.py          # Streamlit boot + session state, no API key
python tests/check_cases.py        # live 5-case run (needs GEMINI_API_KEY in .env)
python tests/debug_retrieval.py    # inspect retrieval for any question (dev)
```

---

## 20. License / note

Built for the MGC Developments technical take-home task. All MGC document
content remains property of MGC Developments and is used here solely for the
purpose of the exercise.

## Part 2: Database Implementation (SQL)

### Database Schema
The schema (located in `database/schema.sql`) was designed to cleanly represent the `leads.csv` dataset in a relational format. It consists of a single `leads` table that tracks lead profiles, interaction metrics, and final conversion outcomes. 
- **Design Decisions:** Used appropriate data types (`DECIMAL` for budget/seconds, `INT` for counts, `BOOLEAN/INT` for flags).
- **Indexing:** Created indexes on `source` (for fast grouping) and `crm_record_hash` (for fast duplicate lookups).

### SQL Queries
Implemented in `database/queries.sql`:
1. **Query 1 (Conversion Rate):** Calculates the conversion rate grouped by lead source. It safely calculates the percentage and uses a `HAVING` clause to filter out sources with fewer than 200 leads.
2. **Query 2 (Duplicate Detection):** Finds duplicate leads by grouping on `crm_record_hash`. As noted in the SQL comments, these duplicates would ideally be prevented at the database level by enforcing a `UNIQUE` constraint on `crm_record_hash`.

---

## Part 3: ML Lead Scoring

### ML Pipeline Overview
We built a robust, scikit-learn based machine learning pipeline to predict whether a lead will convert, given their profile at creation time. The pipeline includes:
1. **Data Ingestion & Deduplication:** Loads `leads.csv` and removes duplicates based on `crm_record_hash` to strictly prevent train/test data contamination.
2. **Feature Engineering & Preprocessing:** Uses a `ColumnTransformer` to handle missing values (median imputation for numbers, constant imputation for categories) and applies One-Hot Encoding for categorical features (`source`, `city`, `area`, `property_type`).
3. **Model:** A `GradientBoostingClassifier` is used because it naturally handles non-linear relationships and mixed feature types.

### Addressing Data Leakage
A critical part of the pipeline was preventing **target leakage**. 
- **Features Used (Safe):** `source`, `city`, `area`, `property_type`, `budget_pkr_lac`, `bedrooms`, `is_overseas`, `referred_by_existing_client`, `has_financing_approved`.
- **Features Excluded (Leakage):** `token_amount_received_pkr`, `calls_made`, `total_call_seconds`, `whatsapp_replies`, `site_visits`, `first_response_minutes`, `agent_experience_years`. 
If we included post-contact features, the model would achieve an artificially high accuracy (>95%), but it would be entirely useless in production for predicting the outcome of *new* leads.

### Evaluation & Results
Because the dataset is heavily imbalanced (~6.9% converted), accuracy is a misleading metric (predicting '0' every time yields ~93% accuracy). We focused on **PR-AUC (Average Precision)**.
- **Accuracy:** 92.78%
- **ROC-AUC:** 68.04%
- **PR-AUC (Average Precision):** 14.93%

**Did we achieve 95% accuracy?** No. We intentionally prioritized ML correctness over an arbitrary accuracy target. The reported performance is an honest, production-ready reflection of the leakage-safe evaluation.

---

## Installation & Running Guide

### 1. Requirements & Setup
First, ensure you have Python 3.9+ installed. Then install the dependencies:
```bash
pip install -r requirements.txt
```

You must also set up your Gemini API key for the RAG Assistant to work:
1. Copy `.env.example` to `.env`
2. Add your Google AI Studio key: `GEMINI_API_KEY=your_key_here`

### 2. Running the SQL/Database Tests
The database schema and queries are located in the `database/` folder. You can test them against the CSV using an in-memory SQLite database (or any SQL runner):
```bash
# Example SQLite run
sqlite3 < database/schema.sql
```

### 3. Training the ML Model
To reproduce the ML pipeline, train the model, and generate the `model.joblib` file:
```bash
python -m ml.train
```
This will output the evaluation metrics and save the trained model to `ml/model.joblib`.

### 4. Running the Complete Application (Streamlit)
The application combines the Grounded Document AI (RAG) and the ML Lead Scoring interface (in the sidebar). Start it with:
```bash
streamlit run app.py
```
- **RAG Assistant:** Use the main chat interface to ask questions about MGC Aurora Heights.
- **Lead Scoring:** Open the sidebar, enter lead details, and click 'Predict Conversion' to test the ML model interactively.
