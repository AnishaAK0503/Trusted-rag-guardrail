"""Phase 9 - retrieval evaluation.

Metrics: Precision@K, Recall@K, MRR, Hit Rate@K, plus trusted-source rate and
provenance completeness. A result is "relevant" if its URL/title contains one of
the query's relevant_keywords (config/eval_queries.json).

Run:  python -m app.evaluation
"""
import json

from app import rag
from app.config import settings
from app.ingestion import is_relevant
from app.source_manager import load_policy


def precision_at_k(flags: list[bool], k: int) -> float:
    return sum(flags[:k]) / k


def hit_rate_at_k(flags: list[bool], k: int) -> float:
    return 1.0 if any(flags[:k]) else 0.0


def reciprocal_rank(flags: list[bool]) -> float:
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1 / position
    return 0.0


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate(k: int = 5) -> dict:
    policy = load_policy(settings.sources_file)
    if not rag.load_index():
        raise rag.IndexNotReadyError("No index found. Run POST /api/v1/ingest first.")
    chunks = rag.get_chunks()
    rows = []
    for case in json.loads(settings.eval_file.read_text(encoding="utf-8")):
        keywords = case["relevant_keywords"]
        relevant_docs = {c["document_id"] for c in chunks if is_relevant(f"{c['url']} {c['title']}", keywords)}
        if not relevant_docs:
            rows.append({"query": case["query"], "skipped": "no relevant document in the index"})
            continue
        results = [r.model_dump() for r in rag.retrieve(case["query"], k, None, policy)]
        flags = [is_relevant(f"{r['url']} {r['title']}", keywords) for r in results]
        found_docs = {r["document_id"] for r, ok in zip(results, flags) if ok}
        rows.append({
            "query": case["query"],
            "precision@k": round(precision_at_k(flags, k), 4),
            "recall@k": round(len(found_docs) / len(relevant_docs), 4),  # distinct relevant documents found
            "reciprocal_rank": round(reciprocal_rank(flags), 4),
            "hit@k": hit_rate_at_k(flags, k),
            "trusted_rate": _mean([1.0 if policy.is_trusted(r["url"]) else 0.0 for r in results]),
            "provenance_complete": _mean([1.0 if all(r[f] for f in ("source", "url", "chunk_id", "document_id")) else 0.0
                                          for r in results]),
        })
    scored = [r for r in rows if "skipped" not in r]
    summary = {
        "k": k, "queries_scored": len(scored), "queries_skipped": len(rows) - len(scored),
        "Precision@K": _mean([r["precision@k"] for r in scored]),
        "Recall@K": _mean([r["recall@k"] for r in scored]),
        "MRR": _mean([r["reciprocal_rank"] for r in scored]),
        "HitRate@K": _mean([r["hit@k"] for r in scored]),
        "TrustedRate": _mean([r["trusted_rate"] for r in scored]),
        "ProvenanceComplete": _mean([r["provenance_complete"] for r in scored]),
    }
    return {"summary": summary, "per_query": rows}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
