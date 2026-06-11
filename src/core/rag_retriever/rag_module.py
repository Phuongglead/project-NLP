from __future__ import annotations
import json
import os
from typing import List, Optional, Tuple
import numpy as np
from src.shared.utils.io_utils import get_logger, load_config
import faiss

logger = get_logger(__name__)

_embedder = None
_index = None         
_chroma_collection = None
_reference_texts: List[str] = []
_reference_metadata: List[dict] = []


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

def build_faiss_index(corpus_path: str = None) -> None:
    global _index, _reference_texts, _reference_metadata
    cfg = load_config()["rag"]
    corpus_path = corpus_path or cfg["reference_corpus_path"]
    index_path = cfg["index_path"]

    logger.info(f"Building FAISS index from: {corpus_path}")

    _reference_texts = []
    _reference_metadata = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            _reference_texts.append(rec["reference_answer"])
            _reference_metadata.append(rec)

    logger.info(f"Embedding {len(_reference_texts)} reference answers...")
    embeddings = _embed(_reference_texts)

    dim = embeddings.shape[1]
    _index = faiss.IndexFlatL2(dim)
    _index.add(embeddings.astype(np.float32))

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(_index, index_path + ".faiss")

    with open(index_path + "_texts.json", "w", encoding="utf-8") as f:
        json.dump({"texts": _reference_texts, "metadata": _reference_metadata}, f)

    logger.info(f"FAISS index saved to: {index_path}.faiss")


def _load_faiss_index():
    """Load FAISS index from disk into memory."""
    global _index, _reference_texts, _reference_metadata
    cfg = load_config()["rag"]
    index_path = cfg["index_path"]

    if not os.path.exists(index_path + ".faiss"):
        raise FileNotFoundError(
            f"FAISS index not found at '{index_path}.faiss'. "
            "Run build_faiss_index() first."
        )

    _index = faiss.read_index(index_path + ".faiss")
    with open(index_path + "_texts.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    _reference_texts = data["texts"]
    _reference_metadata = data["metadata"]
    logger.info(f"FAISS index loaded: {_index.ntotal} vectors.")


# ── Public API: retrieve_reference() ─────────────────────────────────────────

def retrieve_reference(cv_text: str, job_description: str, top_k: int = None) -> str:
    """
    Retrieve the most relevant Reference Answer for a given CV + Job Description.

    Args:
        cv_text: Candidate's raw CV text.
        job_description: Job description or role context.
        top_k: Number of candidates to retrieve (default from config).

    Returns:
        The best-matching reference answer string.

    Example:
        >>> answer = retrieve_reference("5 years Kubernetes experience...", "Senior DevOps Engineer")
        >>> # "Kubernetes uses a control plane to orchestrate..."
    """
    global _index

    if _index is None:
        _load_faiss_index()

    cfg = load_config()["rag"]
    k = top_k or cfg["top_k"]

    # Combine CV text + job description for richer query
    query = f"Skills: {cv_text[:500]}\nJob: {job_description[:300]}"
    query_embedding = _embed([query]).astype(np.float32)

    distances, indices = _index.search(query_embedding, k)
    best_idx = int(indices[0][0])

    if best_idx < 0 or best_idx >= len(_reference_texts):
        logger.warning("FAISS returned invalid index; returning empty reference.")
        return ""

    reference = _reference_texts[best_idx]
    logger.debug(f"Retrieved reference (dist={distances[0][0]:.4f}): {reference[:80]}...")
    return reference


def retrieve_with_metadata(cv_text: str, job_description: str, top_k: int = 3) -> List[Tuple[str, dict, float]]:
    """
    Retrieve top-k reference answers with their metadata and distances.
    Returns list of (answer_text, metadata_dict, distance) tuples.
    """
    global _index
    if _index is None:
        _load_faiss_index()

    query = f"Skills: {cv_text[:500]}\nJob: {job_description[:300]}"
    query_embedding = _embed([query]).astype(np.float32)
    distances, indices = _index.search(query_embedding, top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if 0 <= idx < len(_reference_texts):
            results.append((_reference_texts[idx], _reference_metadata[idx], float(dist)))
    return results


# ── Corpus creation helper ────────────────────────────────────────────────────

def create_sample_corpus(output_path: str = "data/reference_answers.jsonl") -> None:
    """
    Create a small sample reference answer corpus for development/testing.
    In production, replace with a real technical interview answer corpus.
    """
    import uuid
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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for ans in sample_answers:
            ans["id"] = str(uuid.uuid4())
            f.write(json.dumps(ans) + "\n")
    logger.info(f"Sample corpus written to: {output_path} ({len(sample_answers)} entries)")
