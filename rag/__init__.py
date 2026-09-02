"""MGC Sales Assistant - RAG core package.

Modules:
- config:       environment configuration (.env)
- ingestion:    Markdown loading and structure-aware chunking
- embeddings:   Gemini text embeddings (document / query)
- vectorstore:  FAISS index build / persist / load
- retriever:    semantic retrieval over the index
- guardrails:   evidence validation and conflict detection
- calculator:   deterministic price breakdowns
- assistant:    end-to-end grounded Q&A service
"""