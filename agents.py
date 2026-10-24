"""Reviewer 2: the agents and the LangGraph workflow that connects them.

Flow:
    Claim Extractor -> Evidence Agent -> (weak evidence? -> Query Refiner -> Evidence Agent) -> Meta-Reviewer
"""

import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from tools import search_papers

MAX_PAPER_CHARS = 12000   # keeps each request inside the free-tier token limits
ABSTRACT_CHARS = 700
MIN_RELEVANT = 3          # fewer relevant papers than this = weak evidence
DECISIONS = ["Accept", "Major Revision", "Reject"]


class ReviewState(TypedDict, total=False):
    paper_text: str
    title: str
    field: str
    claims: list
    refine_rounds: int
    review: dict
    log: Annotated[list, operator.add]   # every agent appends what it did


def log(agent: str, message: str) -> dict:
    return {"agent": agent, "message": message}


# ---------- Consensus scoring (plain Python, so it is transparent) ----------

def score_claim(papers: list) -> dict:
    relevant = [p for p in papers if p.get("relevant")]
    s = sum(p["stance"] == "supports" for p in relevant)
    o = sum(p["stance"] == "opposes" for p in relevant)
    m = sum(p["stance"] == "mixed" for p in relevant)
    n = len(relevant)
    if n < MIN_RELEVANT:
        label = "Insufficient evidence"
    elif s / n >= 0.6 and o / n < 0.2:
        label = "Well supported"
    elif o > s:
        label = "Goes against the literature"
    else:
        label = "Contested"
    agreement = round(100 * (s + 0.5 * m) / n) if n >= MIN_RELEVANT else None
    return {"supports": s, "opposes": o, "mixed": m, "relevant": n,
            "label": label, "agreement": agreement}


# ---------- Agent prompts ----------

EXTRACTOR_SYSTEM = """You are the Claim Extractor on an academic peer-review panel.
Read the draft paper and pick its most important claims that can be checked against published research.
Prefer claims about findings, effects, comparisons, and the paper's conclusions or recommendations. Skip definitions, background facts everyone agrees on, and future work.
Each claim must be about a DIFFERENT topic. Never pick two claims that are variations of the same finding; merge them into one.
Bold or surprising claims (e.g. "X has no effect on Y") are the most important to check, so include them.
For each claim give a short search query (3 to 8 keywords) that would find related papers.
Reply in JSON only, in this exact shape:
{"title": "paper title", "field": "research field",
 "claims": [{"claim": "the claim in one sentence", "quote": "short exact quote from the paper", "search_query": "keywords"}]}"""

EVIDENCE_SYSTEM = """You are the Evidence Agent on an academic peer-review panel.
You get one claim and a numbered list of published papers (title, year, abstract).
For every paper decide:
- relevant: true if the abstract studies the same relationship or effect as the claim, even in a somewhat different
  population or setting (e.g. "university students" is relevant to a claim about "engineering students").
  false if it is about a different relationship.
- stance (only matters if relevant): compare the paper's FINDINGS with the claim EXACTLY AS STATED, including its direction.
  "supports" = the findings agree with the claim. "opposes" = the findings contradict the claim.
  "mixed" = partly agrees, or results depend on conditions.
  Be careful with negative claims: if the claim says "X has no effect on Y" or "X is not related to Y" and the paper
  finds that X DOES affect Y, the stance is "opposes", not "supports".
- reason: one short sentence explaining the stance, based ONLY on the abstract text given. Do not use outside knowledge.
Reply in JSON only: {"papers": [{"index": 1, "relevant": true, "stance": "supports", "reason": "..."}]}"""

REFINER_SYSTEM = """You are the Query Refiner. A literature search for a claim found too few relevant papers.
Write ONE better search query: broader, using the standard academic terms for the topic, 3 to 7 keywords.
Reply in JSON only: {"search_query": "..."}"""

META_SYSTEM = """You are the Meta-Reviewer (the senior reviewer) on an academic peer-review panel.
You get the paper's opening text and the Evidence Agent's findings for each key claim.
Write the final review. Rules:
- decision must be exactly one of: "Accept", "Major Revision", "Reject".
  Accept: key claims are well supported and the work looks sound.
  Major Revision: some claims are contested, unsupported, or need more evidence.
  Reject: central claims go against the literature or the study has serious flaws.
- Only mention papers that appear in the findings, using their exact titles. Never invent papers or numbers.
- strengths and weaknesses: 2 to 4 items each, every item a separate short point (one idea per item, no semicolon lists).
- Fixes must be specific and ordered by importance (priority 1 = most important), at most 5.
Reply in JSON only:
{"decision": "...", "summary": "2-3 sentences",
 "strengths": ["..."], "weaknesses": ["..."],
 "fixes": [{"priority": 1, "fix": "what to change", "why": "reason, citing the evidence"}]}"""


# ---------- Build the graph ----------

def build_graph(llm, max_claims: int = 3, papers_per_claim: int = 8):

    def claim_extractor(state: ReviewState) -> dict:
        text = state["paper_text"][:MAX_PAPER_CHARS]
        out = llm.ask_json(EXTRACTOR_SYSTEM,
                           f"Extract at most {max_claims} key claims.\n\nDRAFT PAPER:\n{text}")
        claims = []
        for c in (out.get("claims") or [])[:max_claims]:
            if not c.get("claim"):
                continue
            claims.append({
                "claim": c["claim"].strip(),
                "quote": (c.get("quote") or "").strip(),
                "search_query": (c.get("search_query") or c["claim"]).strip(),
                "queries_tried": [],
                "attempts": 0,
                "status": "pending",
                "papers": [],
                "source": "",
                "score": {},
            })
        if not claims:
            raise RuntimeError("No checkable claims were found in this text.")
        return {
            "title": out.get("title") or "Untitled draft",
            "field": out.get("field") or "",
            "claims": claims,
            "refine_rounds": 0,
            "log": [log("Claim Extractor",
                        f"Found {len(claims)} key claim(s) to check: "
                        + "; ".join(f'"{c["claim"]}"' for c in claims))],
        }

    def evidence_agent(state: ReviewState) -> dict:
        claims = [dict(c) for c in state["claims"]]
        entries = []
        for i, c in enumerate(claims, 1):
            if c["status"] != "pending":
                continue
            c["queries_tried"] = c["queries_tried"] + [c["search_query"]]
            papers, source = search_papers(c["search_query"], papers_per_claim)
            c["source"] = source
            if papers:
                listing = "\n\n".join(
                    f"[{n}] {p['title']} ({p['year']})\n{p['abstract'][:ABSTRACT_CHARS]}"
                    for n, p in enumerate(papers, 1))
                out = llm.ask_json(EVIDENCE_SYSTEM, f"CLAIM: {c['claim']}\n\nPAPERS:\n{listing}")
                judged = {j.get("index"): j for j in out.get("papers") or [] if isinstance(j, dict)}
                for n, p in enumerate(papers, 1):
                    j = judged.get(n, {})
                    stance = j.get("stance") if j.get("stance") in ("supports", "opposes", "mixed") else "mixed"
                    p.update(relevant=bool(j.get("relevant")), stance=stance,
                             reason=(j.get("reason") or "").strip())
            c["papers"] = papers
            c["score"] = score_claim(papers)
            # Decision: is the evidence strong enough, or should we search again?
            weak = c["score"]["relevant"] < MIN_RELEVANT
            c["status"] = "weak" if weak and c["attempts"] == 0 else "done"
            msg = (f"Claim {i}: searched {source} for \"{c['search_query']}\", "
                   f"{len(papers)} paper(s) read, {c['score']['relevant']} relevant -> {c['score']['label']}")
            if c["status"] == "weak":
                msg += " (too little evidence, will search again)"
            entries.append(log("Evidence Agent", msg))
        return {"claims": claims, "log": entries}

    def needs_refining(state: ReviewState) -> str:
        weak = any(c["status"] == "weak" for c in state["claims"])
        return "refine" if weak and state.get("refine_rounds", 0) < 1 else "meta"

    def query_refiner(state: ReviewState) -> dict:
        claims = [dict(c) for c in state["claims"]]
        entries = []
        for i, c in enumerate(claims, 1):
            if c["status"] != "weak":
                continue
            out = llm.ask_json(REFINER_SYSTEM,
                               f"CLAIM: {c['claim']}\nQUERIES ALREADY TRIED: {c['queries_tried']}",
                               max_tokens=100)
            new_q = (out.get("search_query") or "").strip()
            if not new_q or new_q in c["queries_tried"]:
                c["status"] = "done"   # nothing new to try; keep what we have
                continue
            c.update(search_query=new_q, attempts=c["attempts"] + 1, status="pending")
            entries.append(log("Query Refiner", f'Claim {i}: retrying with "{new_q}"'))
        return {"claims": claims, "refine_rounds": state.get("refine_rounds", 0) + 1, "log": entries}

    def meta_reviewer(state: ReviewState) -> dict:
        findings = []
        for i, c in enumerate(state["claims"], 1):
            relevant = sorted((p for p in c["papers"] if p.get("relevant")),
                              key=lambda p: p["citations"], reverse=True)[:4]
            findings.append({
                "claim_number": i,
                "claim": c["claim"],
                "consensus": c["score"]["label"],
                "counts": {k: c["score"][k] for k in ("supports", "opposes", "mixed", "relevant")},
                "key_papers": [{"title": p["title"], "year": p["year"],
                                "stance": p["stance"], "reason": p["reason"]} for p in relevant],
            })
        user = (f"PAPER TITLE: {state.get('title')}\n\nOPENING TEXT:\n{state['paper_text'][:3000]}\n\n"
                f"EVIDENCE FINDINGS:\n{json.dumps(findings, indent=1)}")
        review = llm.ask_json(META_SYSTEM, user, max_tokens=1800)
        if review.get("decision") not in DECISIONS:
            review["decision"] = "Major Revision"
        review.setdefault("summary", "")
        for key in ("strengths", "weaknesses", "fixes"):
            if not isinstance(review.get(key), list):
                review[key] = []
        return {"review": review,
                "log": [log("Meta-Reviewer", f"Decision: {review['decision']}")]}

    g = StateGraph(ReviewState)
    g.add_node("claim_extractor", claim_extractor)
    g.add_node("evidence_agent", evidence_agent)
    g.add_node("query_refiner", query_refiner)
    g.add_node("meta_reviewer", meta_reviewer)
    g.add_edge(START, "claim_extractor")
    g.add_edge("claim_extractor", "evidence_agent")
    g.add_conditional_edges("evidence_agent", needs_refining,
                            {"refine": "query_refiner", "meta": "meta_reviewer"})
    g.add_edge("query_refiner", "evidence_agent")
    g.add_edge("meta_reviewer", END)
    return g.compile()
