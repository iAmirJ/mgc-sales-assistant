"""Vector store: build, persist and load a FAISS index over the chunks.

FAISS is a lightweight, embeddable similarity-search library. We store the
index and its chunk metadata on disk under ``vectorstore/`` so the application
can load an existing index instead of re-embedding the documents on every run.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from typing import Iterator, Optional

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.config import config

logger = logging.getLogger(__name__)

# IndexFlatIP over L2-normalised vectors. Scores returned by
# similarity_search_with_score are then inner products of unit vectors, i.e.
# cosine similarities in [-1, 1] where HIGHER means MORE similar - a natural,
# interpretable relevance signal for the evidence gate.
DISTANCE_STRATEGY = DistanceStrategy.MAX_INNER_PRODUCT


def build_index(
    documents: list[Document],
    embeddings: Embeddings,
    index_path: Optional[str] = None,
) -> FAISS:
    """Embed the documents and persist a FAISS index to ``index_path``."""
    index_path = index_path or str(config.faiss_index_path)
    with _suppress_cosine_warning():
        vectorstore = FAISS.from_documents(
            documents,
            embeddings,
            distance_strategy=DISTANCE_STRATEGY,
            normalize_L2=True,
        )
    vectorstore.save_local(index_path)
    logger.info("Built and saved FAISS index with %d chunks to %s", len(documents), index_path)
    return vectorstore


def load_index(
    embeddings: Embeddings,
    index_path: Optional[str] = None,
    allow_dangerous_deserialization: bool = True,
) -> Optional[FAISS]:
    """Load a previously built index, or return None if it does not exist."""
    index_path = index_path or str(config.faiss_index_path)
    if not config.faiss_index_path.exists():
        logger.warning("No FAISS index found at %s", index_path)
        return None
    with _suppress_cosine_warning():
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization,
            distance_strategy=DISTANCE_STRATEGY,
            normalize_L2=True,
        )
    return vectorstore


@contextlib.contextmanager
def _suppress_cosine_warning() -> Iterator[None]:
    """Suppress langchain's cosmetic warning for MAX_INNER_PRODUCT + L2 norm.

    LangChain warns that L2 normalisation is "not applicable" for any non-
    Euclidean strategy, but for MAX_INNER_PRODUCT it is exactly what turns the
    inner product into cosine similarity - which is what we want. The warning
    is noise; the behaviour is correct.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def index_exists(index_path: Optional[str] = None) -> bool:
    index_path = index_path or str(config.faiss_index_path)
    return config.faiss_index_path.exists()
