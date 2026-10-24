# Reviewer 2

A multi-agent AI peer reviewer. Upload a draft research paper and a panel of AI agents checks its key claims against published research, then gives a review decision (Accept / Major Revision / Reject) with a prioritised list of fixes.

## How it works

```
Claim Extractor -> Evidence Agent -> Meta-Reviewer
                        |    ^
            weak evidence|    |retry once
                        v    |
                    Query Refiner
```

| Agent | What it does |
|---|---|
| Claim Extractor | Reads the draft and picks up to 3 key, checkable claims, each with a search query |
| Evidence Agent | Searches Semantic Scholar (OpenAlex as backup) for related papers, reads their abstracts, and labels each paper as supports / opposes / mixed / not relevant |
| Query Refiner | Runs only if a claim has fewer than 3 relevant papers: writes a better search query, then the Evidence Agent tries again (once) |
| Meta-Reviewer | Combines all findings into a decision, strengths, weaknesses and prioritised fixes, citing only papers that were actually found |

The consensus label for each claim (Well supported / Contested / Goes against the literature / Insufficient evidence) is calculated in plain Python from the paper counts, not guessed by the model.

## Tech stack

- Python, LangGraph (agent workflow and the retry loop)
- GPT-OSS 120B / 20B (OpenAI open-weight models) via the Groq free API
- Semantic Scholar API and OpenAlex API (free paper search)
- PyMuPDF (reading PDFs), Streamlit (interface)

## Run it (Windows, Command Prompt)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
notepad .env
streamlit run app.py
```

Put your free Groq key (from https://console.groq.com/keys) in `.env` before running.

## Limitations

- Judges papers from their abstracts only, not full text, so subtle disagreements can be missed.
- Checks at most 3 claims and 10 papers per claim to stay within free API limits.
- The model can occasionally mislabel a paper's stance; every label shows its reason and a link so a person can check it.
- Free tiers are rate-limited; heavy use may need a short wait between runs.
- It assists a human reviewer and does not replace one.
