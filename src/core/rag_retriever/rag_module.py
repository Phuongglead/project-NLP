from __future__ import annotations
import json
import os
import uuid
from typing import List, Optional, Tuple

import numpy as np

from src.core.rag_retriever.retrieval import hybrid_score, mmr_select
from src.core.rag_retriever.skill_expansion import expand_cv_keywords
from src.shared.contracts.schemas import RetrievalHit, SkillEntity
from src.shared.corpus.schema import build_index_text, normalize_knowledge_record
from src.shared.utils.io_utils import get_logger, load_config
import faiss

logger = get_logger(__name__)

_embedder = None
_index = None
_reference_texts: List[str] = []
_reference_metadata: List[dict] = []
_loaded_index_path: Optional[str] = None


def _load_embedder():
    global _embedder
    from sentence_transformers import SentenceTransformer
    cfg = load_config()
    model_name = cfg["rag"]["embedding_model"]
    logger.info(f"Loading embedding model: {model_name}")
    _embedder = SentenceTransformer(model_name)


def _embed(texts: List[str]) -> np.ndarray:
    if _embedder is None:
        _load_embedder()
    return _embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _embedding_text(rec: dict) -> str:
    return rec.get("index_text") or build_index_text(rec)


def build_faiss_index(corpus_path: str = None, index_path: str = None) -> None:
    global _index, _reference_texts, _reference_metadata, _loaded_index_path
    cfg = load_config()["rag"]
    corpus_path = corpus_path or cfg["reference_corpus_path"]
    index_path = index_path or cfg["index_path"]

    logger.info(f"Building FAISS index from: {corpus_path}")

    _reference_texts = []
    _reference_metadata = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            rec = normalize_knowledge_record(rec)
            _reference_texts.append(_embedding_text(rec))
            _reference_metadata.append(rec)

    logger.info(f"Embedding {len(_reference_texts)} knowledge records...")
    embeddings = _embed(_reference_texts)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings.astype(np.float32))

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(_index, index_path + ".faiss")

    with open(index_path + "_texts.json", "w", encoding="utf-8") as f:
        json.dump({"texts": _reference_texts, "metadata": _reference_metadata}, f)

    _loaded_index_path = index_path
    logger.info(f"FAISS index saved to: {index_path}.faiss")


def _load_faiss_index(index_path: str = None):
    """Load FAISS index from disk into memory."""
    global _index, _reference_texts, _reference_metadata, _loaded_index_path
    cfg = load_config()["rag"]
    index_path = index_path or cfg["index_path"]

    if _loaded_index_path == index_path and _index is not None:
        return

    faiss_file = index_path + ".faiss"
    if not os.path.exists(faiss_file):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_file}'. "
            "Run build_faiss_index() first."
        )

    _index = faiss.read_index(faiss_file)
    with open(index_path + "_texts.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    _reference_texts = data["texts"]
    _reference_metadata = data["metadata"]
    _loaded_index_path = index_path
    logger.info(f"FAISS index loaded: {_index.ntotal} vectors.")


def _build_query_text(cv_text: str, job_description: str, expanded_keywords: List[str]) -> str:
    kw_part = ", ".join(expanded_keywords) if expanded_keywords else cv_text[:300]
    return f"Skills: {kw_part}\nJob: {job_description[:300]}"


def retrieve_candidates(
    cv_text: str,
    job_description: str,
    cv_skills: List[SkillEntity],
    top_k: int = None,
    exclude_ids: List[str] = None,
) -> List[RetrievalHit]:
    """
    Hybrid FAISS + keyword retrieval with MMR diversity.
    Returns top_k RetrievalHit objects.
    """
    global _index

    if _index is None:
        _load_faiss_index()

    cfg = load_config()["rag"]
    k = top_k or cfg.get("default_top_k", 5)
    pool_mult = cfg.get("candidate_pool_multiplier", 3)
    pool_size = min(k * pool_mult, len(_reference_metadata))

    skill_names = [s.entity for s in cv_skills]
    expanded = expand_cv_keywords(skill_names)
    query = _build_query_text(cv_text, job_description, expanded)
    query_embedding = _embed([query]).astype(np.float32)

    distances, indices = _index.search(query_embedding, pool_size)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        idx = int(idx)
        if idx < 0 or idx >= len(_reference_metadata):
            continue
        record = _reference_metadata[idx]
        candidates.append({
            "record": record,
            "distance": float(dist),
            "hybrid_score": hybrid_score(float(dist), expanded, record),
            "idx": idx,
        })

    candidates.sort(key=lambda c: c["hybrid_score"], reverse=True)
    return mmr_select(candidates, top_k=k, exclude_ids=exclude_ids)


def retrieve_reference(cv_text: str, job_description: str, top_k: int = None) -> str:
    """Legacy single-best retrieval (backward compatible)."""
    hits = retrieve_candidates(cv_text, job_description, cv_skills=[], top_k=1)
    if not hits:
        logger.warning("No retrieval hits; returning empty reference.")
        return ""
    reference = hits[0].record.get("reference_answer", "")
    logger.debug(f"Retrieved reference (score={hits[0].match_score}): {reference[:80]}...")
    return reference


def retrieve_with_metadata(
    cv_text: str,
    job_description: str,
    top_k: int = 3,
    cv_skills: List[SkillEntity] = None,
) -> List[Tuple[str, dict, float]]:
    """Retrieve top-k with metadata and distances (backward compatible)."""
    hits = retrieve_candidates(
        cv_text, job_description, cv_skills=cv_skills or [], top_k=top_k
    )
    return [
        (h.record.get("reference_answer", ""), h.record, h.semantic_distance)
        for h in hits
    ]


def create_sample_corpus(output_path: str = "data/knowledge_corpus.active.jsonl") -> None:
    """Create a small sample knowledge corpus for development/testing."""
    from src.shared.corpus.schema import legacy_to_knowledge_record, write_jsonl

    sample_answers = [
        {"topic": "kubernetes", "reference_answer": "Kubernetes uses a control plane consisting of the API server, etcd, scheduler, and controller manager to orchestrate containerized workloads across a cluster of nodes, enabling automated scaling, self-healing, and rolling updates."},
        {"topic": "docker", "reference_answer": "Docker containers encapsulate an application and its dependencies into a portable, lightweight unit that runs consistently across different environments by sharing the host OS kernel while maintaining process isolation."},
        {"topic": "cicd", "reference_answer": "CI/CD pipelines automate the build, test, and deployment process, reducing manual errors and enabling teams to deliver software changes rapidly and reliably through stages such as source control, automated testing, and deployment gates."},
        {"topic": "microservices", "reference_answer": "Microservices architecture decomposes an application into small, independently deployable services that communicate via APIs, enabling teams to develop, scale, and maintain each service independently while improving fault isolation."},
        {"topic": "python", "reference_answer": "Python's GIL (Global Interpreter Lock) prevents multiple threads from executing Python bytecode simultaneously, which limits CPU-bound parallelism in CPython but does not affect I/O-bound concurrency. Use multiprocessing or async I/O for parallelism."},
        {"topic": "machine_learning", "reference_answer": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters by computing the gradient of the loss function with respect to each parameter and taking a step proportional to the negative gradient, minimizing prediction error."},
        {"topic": "sql", "reference_answer": "Database indexing creates auxiliary data structures (B-trees or hash maps) that allow the query engine to locate rows without scanning the entire table, dramatically reducing query time at the cost of additional storage and slower writes."},
        {"topic": "api_design", "reference_answer": "RESTful API design follows stateless client-server communication using standard HTTP methods (GET, POST, PUT, DELETE), resource-based URLs, and HTTP status codes to represent the outcome of operations, enabling uniform interface and scalability."},
        {"topic": "distributed_systems", "reference_answer": "The CAP theorem states that a distributed system can guarantee at most two of three properties: Consistency, Availability, and Partition tolerance. In practice, network partitions are inevitable, so systems must choose between consistency and availability."},
        {"topic": "testing", "reference_answer": "Test-driven development (TDD) requires writing a failing test before writing production code, then writing the minimum code to pass the test, and refactoring. This ensures code is testable by design and reduces regression bugs."},
    ]
    records = []
    for ans in sample_answers:
        ans["id"] = str(uuid.uuid4())
        records.append(legacy_to_knowledge_record(ans))
    write_jsonl(records, output_path)
    logger.info(f"Sample corpus written to: {output_path} ({len(records)} entries)")
