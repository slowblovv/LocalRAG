from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from localrag.config import settings
from localrag.engine import RagEngine
from localrag.index import LocalIndex
from localrag.retrieval.rerank import create_reranker


def sentences_with_citations(answer: str):
    out = []
    for raw in re.split(r"(?<=[.!?])\s+", answer.strip()):
        ids = [int(x) for x in re.findall(r"\[(\d+)\]", raw)]
        text = re.sub(r"\s*\[(\d+)\]", "", raw).strip()
        if not text and ids and out:
            previous, previous_ids = out[-1]
            out[-1] = (previous, previous_ids + ids)
        else:
            out.append((text.casefold(), ids))
    return out


def claim_metrics(answer, citations, claims, relevant_ids, context_by_id):
    sentences = sentences_with_citations(answer)
    citation_map = {c["id"]: c for c in citations}
    cited_chunk_ids = {c["chunk_id"] for c in citations}
    valid_citations = len(cited_chunk_ids) / max(len(citations), 1) if citations else 0.0
    claim_correctness = []
    citation_coverage = []
    evidence_support = []
    for claim in claims:
        required_terms = [x.casefold() for x in claim["required_terms"]]
        wanted_evidence = [x.casefold() for x in claim.get("evidence_terms", [])]
        claim_hit = False
        claim_has_citation = False
        supported = False
        for sentence, ids in sentences:
            if not ids:
                continue
            sentence_hit = all(term in sentence for term in required_terms)
            cited_chunks = [citation_map.get(str(i)) for i in ids]
            cited_chunks = [c for c in cited_chunks if c is not None]
            if sentence_hit:
                claim_hit = True
                if cited_chunks:
                    claim_has_citation = True
            for citation in cited_chunks:
                chunk_id = citation["chunk_id"]
                if sentence_hit and chunk_id in relevant_ids:
                    evidence_text = context_by_id.get(chunk_id, "").casefold()
                    if all(term in evidence_text for term in wanted_evidence):
                        supported = True
                        break
            if supported:
                break
        claim_correctness.append(1.0 if claim_hit else 0.0)
        citation_coverage.append(1.0 if claim_has_citation else 0.0)
        evidence_support.append(1.0 if supported else 0.0)
    return {
        "claim_correctness": sum(claim_correctness) / max(len(claim_correctness), 1),
        "citation_coverage": sum(citation_coverage) / max(len(citation_coverage), 1),
        "citation_validity": valid_citations,
        "evidence_support": sum(evidence_support) / max(len(evidence_support), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deterministic", action="store_true", help="allow CI/test doubles")
    args = parser.parse_args()
    idx = LocalIndex(); idx.build("demo_corpus"); idx.load_current(); engine = RagEngine(idx, reranker=create_reranker(settings.reranker_model))
    if (idx.embedder.name.startswith("hash-") or engine.reranker.name == "overlap") and not args.deterministic:
        raise RuntimeError("answer evaluation must run with the configured real dense model and Cross-Encoder reranker")
    answer_items = json.loads(Path("evaluation/answers.json").read_text())
    rel = json.loads(Path("evaluation/relevance.json").read_text())
    rows = []
    for item, relevance in zip(answer_items, rel[::4]):
        result = engine.query(item["query"], "hybrid", 8, True)
        ctx = {cid: idx.chunks[cid].text for cid in result["trace"]["context_chunk_ids"]}
        metrics = claim_metrics(result["answer"], result["citations"], item["claims"], set(relevance["relevant_chunk_ids"]), ctx)
        rows.append({**metrics, "groundedness": metrics["evidence_support"], "refusal_correct": 0.0 if result["trace"]["citation_validation_result"] == "refused" else 1.0})
    no_items = json.loads(Path("evaluation/no_answer.json").read_text())
    no_refusals = sum(engine.query(x["query"], "hybrid", 8, True)["trace"]["citation_validation_result"] == "refused" for x in no_items)
    answerable_refusals = sum(engine.query(x["query"], "hybrid", 8, True)["trace"]["citation_validation_result"] == "refused" for x in answer_items)
    total_refusals = no_refusals + answerable_refusals
    summary = {
        "answerable_questions": len(rows),
        "claim_correctness": sum(x["claim_correctness"] for x in rows) / len(rows),
        "citation_coverage": sum(x["citation_coverage"] for x in rows) / len(rows),
        "citation_validity": sum(x["citation_validity"] for x in rows) / len(rows),
        "groundedness": sum(x["groundedness"] for x in rows) / len(rows),
        "no_answer_questions": len(no_items),
        "refusal_precision": no_refusals / max(total_refusals, 1),
        "refusal_recall": no_refusals / max(len(no_items), 1),
        "models": {"embedding": idx.embedder.name, "reranker": engine.reranker.name, "llm": settings.llm_model, "prompt": settings.prompt_version},
    }
    print(json.dumps(summary, indent=2))
    print("Evaluation is claim/evidence based. Automated fixture metrics are proxies, not an objective human or LLM-judge truth score.")


if __name__ == "__main__": main()
