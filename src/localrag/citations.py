import re


def extract_citation_ids(answer: str) -> list[int]:
    return [int(x) for x in re.findall(r"\[(\d+)\]", answer)]


def validate_citations(answer: str, context_count: int) -> tuple[list[int], bool]:
    ids = extract_citation_ids(answer)
    valid = [i for i in ids if 1 <= i <= context_count]
    return valid, bool(ids) and all(1 <= i <= context_count for i in ids)


def citation_source_coverage(citation_ids: list[int], context_count: int) -> float:
    return len(set(citation_ids)) / max(context_count, 1)
