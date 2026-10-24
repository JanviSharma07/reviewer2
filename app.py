"""Reviewer 2 - Streamlit interface. Run with:  streamlit run app.py"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents import build_graph
from llm import LLM, MODELS
from tools import read_pdf

load_dotenv()
st.set_page_config(page_title="Reviewer 2", layout="wide")

LABEL_COLOURS = {
    "Well supported": "green",
    "Contested": "orange",
    "Goes against the literature": "red",
    "Insufficient evidence": "gray",
}
DECISION_BOX = {"Accept": st.success, "Major Revision": st.warning, "Reject": st.error}
STANCE_MARK = {"supports": ":green[supports]", "opposes": ":red[opposes]", "mixed": ":orange[mixed]"}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key and not env_key.startswith("paste_"):
        # Key stays on the server: never sent to the browser or shown on screen.
        api_key = env_key
        st.caption("Groq API key loaded from .env")
    else:
        api_key = st.text_input("Groq API key", type="password",
                                help="Free key from console.groq.com/keys. Not saved anywhere.")
    default_model = os.getenv("GROQ_MODEL", MODELS[0])
    model = st.selectbox("Model", MODELS, index=MODELS.index(default_model) if default_model in MODELS else 0,
                         help="120B is smarter. 20B is faster and has higher free limits - use it while testing.")
    max_claims = st.slider("Claims to check", 1, 3, 3)
    papers_per_claim = st.slider("Papers per claim", 4, 10, 8)
    st.divider()
    st.caption("Agents: Claim Extractor -> Evidence Agent -> Query Refiner (only if evidence is weak) "
               "-> Meta-Reviewer. Papers come from Semantic Scholar, with OpenAlex as backup.")

# ---------------- Header and input ----------------
st.title("Reviewer 2")
st.write("A multi-agent peer reviewer. Upload a draft paper and a panel of AI agents checks its key claims "
         "against published research, then gives a review decision with a prioritised fix list.")

tab_pdf, tab_text, tab_sample = st.tabs(["Upload PDF", "Paste text", "Try the sample draft"])
paper_text = ""
with tab_pdf:
    pdf = st.file_uploader("Draft paper (PDF)", type="pdf")
    if pdf is not None:
        paper_text = read_pdf(pdf.getvalue())
        if not paper_text.strip():
            st.error("No text could be read from this PDF (it may be a scanned image). Paste the text instead.")
with tab_text:
    pasted = st.text_area("Paste the paper text", height=220)
    if pasted.strip() and not paper_text:
        paper_text = pasted
with tab_sample:
    sample_path = Path(__file__).parent / "sample_draft.txt"
    sample = sample_path.read_text(encoding="utf-8")
    st.caption("A short made-up student paper, written for demos. It mixes well-supported and doubtful claims.")
    st.text_area("Sample draft", sample, height=220, disabled=True)
    use_sample = st.checkbox("Review this sample")
    if use_sample and not paper_text:
        paper_text = sample

run = st.button("Run review", type="primary", disabled=not paper_text.strip())


# ---------------- Run the agents ----------------
def run_review(text: str) -> dict:
    graph = build_graph(LLM(api_key, model), max_claims, papers_per_claim)
    final = {"paper_text": text, "log": []}
    with st.status("Reviewer panel at work...", expanded=True) as status:
        for update in graph.stream({"paper_text": text, "log": []}, stream_mode="updates"):
            for node, changes in update.items():
                for entry in changes.get("log", []):
                    st.write(f"**{entry['agent']}:** {entry['message']}")
                    final["log"].append(entry)
                final.update({k: v for k, v in changes.items() if k != "log"})
        status.update(label="Review complete", state="complete", expanded=False)
    return final


if run:
    if not api_key:
        st.error("Add your Groq API key in the sidebar (or in the .env file) first.")
    else:
        try:
            st.session_state["result"] = run_review(paper_text)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")


# ---------------- Show results ----------------
def to_markdown(r: dict) -> str:
    rv = r["review"]
    lines = [f"# Review: {r.get('title', '')}", "", f"**Decision:** {rv['decision']}", "", rv["summary"], ""]
    lines += ["## Claims checked", ""]
    for i, c in enumerate(r["claims"], 1):
        s = c["score"]
        lines.append(f"### Claim {i}: {c['claim']}")
        lines.append(f"Consensus: **{s['label']}** ({s['supports']} support, {s['opposes']} oppose, "
                     f"{s['mixed']} mixed out of {s['relevant']} relevant papers)")
        for p in c["papers"]:
            if p.get("relevant"):
                lines.append(f"- {p['title']} ({p['year']}) - {p['stance']}: {p['reason']} {p['url']}")
        lines.append("")
    for heading, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses")):
        lines += [f"## {heading}", ""] + [f"- {x}" for x in rv[key]] + [""]
    lines += ["## Prioritised fixes", ""]
    for f in sorted(rv["fixes"], key=lambda f: f.get("priority", 99)):
        lines.append(f"{f.get('priority', '-')}. {f.get('fix', '')} - {f.get('why', '')}")
    return "\n".join(lines)


result = st.session_state.get("result")
if result and result.get("review"):
    rv = result["review"]
    st.divider()
    st.subheader(result.get("title", "Review"))
    DECISION_BOX[rv["decision"]](f"**Decision: {rv['decision']}**\n\n{rv['summary']}")

    st.markdown("### Claims checked against the literature")
    for i, c in enumerate(result["claims"], 1):
        s = c["score"]
        colour = LABEL_COLOURS[s["label"]]
        with st.container(border=True):
            st.markdown(f"**Claim {i}:** {c['claim']}")
            if c["quote"]:
                st.caption(f"From the paper: \"{c['quote']}\"")
            cols = st.columns(5)
            cols[0].markdown(f"Consensus\n\n:{colour}[**{s['label']}**]")
            cols[1].metric("Support", s["supports"])
            cols[2].metric("Oppose", s["opposes"])
            cols[3].metric("Mixed", s["mixed"])
            cols[4].metric("Agreement", f"{s['agreement']}%" if s["agreement"] is not None else "-")
            with st.expander(f"Evidence: {s['relevant']} relevant of {len(c['papers'])} papers read "
                             f"(source: {c['source']}, query: \"{c['search_query']}\")"):
                for p in sorted(c["papers"], key=lambda p: not p.get("relevant")):
                    if p.get("relevant"):
                        st.markdown(f"- {STANCE_MARK[p['stance']]} - [{p['title']}]({p['url']}) "
                                    f"({p['year']}, {p['citations']} citations)  \n  {p['reason']}")
                    else:
                        st.markdown(f"- :gray[not relevant - {p['title']} ({p['year']})]")

    left, right = st.columns(2)
    with left:
        st.markdown("### Strengths")
        for x in rv["strengths"]:
            st.markdown(f"- {x}")
    with right:
        st.markdown("### Weaknesses")
        for x in rv["weaknesses"]:
            st.markdown(f"- {x}")

    st.markdown("### Prioritised fixes")
    for f in sorted(rv["fixes"], key=lambda f: f.get("priority", 99)):
        st.markdown(f"**{f.get('priority', '-')}. {f.get('fix', '')}**  \n{f.get('why', '')}")

    with st.expander("Agent activity log"):
        for entry in result["log"]:
            st.markdown(f"**{entry['agent']}:** {entry['message']}")

    st.download_button("Download review (Markdown)", to_markdown(result),
                       file_name="reviewer2_review.md", mime="text/markdown")
