"""Tools the agents use: reading PDFs and searching for research papers."""

import functools
import os
import time

import pymupdf
import requests

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_URL = "https://api.openalex.org/works"
TIMEOUT = 20


def read_pdf(file_bytes: bytes) -> str:
    """Extract plain text from an uploaded PDF."""
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def _search_semantic_scholar(query: str, limit: int) -> list:
    headers = {}
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        headers["x-api-key"] = key
    params = {
        "query": query,
        "limit": min(limit * 2, 40),  # ask for extra, some have no abstract
        "fields": "title,abstract,year,citationCount,url",
    }
    # The shared (no-key) pool often answers 429, so retry briefly with backoff.
    for wait in (0, 2, 4):
        time.sleep(wait)
        resp = requests.get(S2_URL, params=params, headers=headers, timeout=TIMEOUT)
        if resp.status_code == 429:
            continue
        resp.raise_for_status()
        papers = []
        for item in resp.json().get("data") or []:
            if not item.get("abstract"):
                continue
            papers.append({
                "title": item.get("title") or "Untitled",
                "abstract": item["abstract"],
                "year": item.get("year"),
                "citations": item.get("citationCount") or 0,
                "url": item.get("url") or "",
            })
        return papers[:limit]
    raise RuntimeError("Semantic Scholar rate limit")


def _rebuild_abstract(inverted_index: dict) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; put the words back in order."""
    if not inverted_index:
        return ""
    positions = [(i, word) for word, idxs in inverted_index.items() for i in idxs]
    return " ".join(word for _, word in sorted(positions))


def _search_openalex(query: str, limit: int) -> list:
    params = {
        "search": query,
        "per-page": min(limit * 2, 50),
        "filter": "has_abstract:true",
        "select": "id,doi,display_name,publication_year,cited_by_count,abstract_inverted_index",
    }
    email = os.getenv("OPENALEX_EMAIL", "").strip()
    if email:
        params["mailto"] = email
    resp = requests.get(OPENALEX_URL, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("results") or []:
        abstract = _rebuild_abstract(item.get("abstract_inverted_index"))
        if not abstract:
            continue
        papers.append({
            "title": item.get("display_name") or "Untitled",
            "abstract": abstract,
            "year": item.get("publication_year"),
            "citations": item.get("cited_by_count") or 0,
            "url": item.get("doi") or item.get("id") or "",
        })
    return papers[:limit]


@functools.lru_cache(maxsize=64)
def _cached_search(query: str, limit: int) -> tuple:
    """Try Semantic Scholar first, fall back to OpenAlex. Cached so re-runs don't repeat calls."""
    try:
        papers = _search_semantic_scholar(query, limit)
        if papers:
            return tuple(papers), "Semantic Scholar"
    except Exception:
        pass  # rate limited or down: fall back
    try:
        return tuple(_search_openalex(query, limit)), "OpenAlex"
    except Exception as exc:
        raise RuntimeError(f"Both paper sources failed for '{query}': {exc}") from exc


def search_papers(query: str, limit: int = 8):
    """Return (list of paper dicts, source name)."""
    papers, source = _cached_search(query.strip(), limit)
    return [dict(p) for p in papers], source
