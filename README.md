# DSA Doubt Solver

A small RAG app that answers data-structures-and-algorithms questions **only** from a
fixed set of sources, cites them, and says "I don't know" when the sources don't cover
the question.

Live demo: <ADD LINK>

## Product note  (TODO: write this section in your own words before coding)
- **User:** who is this for, and what are they doing when they get stuck?
- **Problem:** why is a grounded answer better than asking a general chatbot?
- **Non-goals:** what this deliberately does not do (agents, login, vector DB, ...).
- **Success looks like:** how I judge it works (answer accuracy, citation rate, correct refusals).

## How it works
1. `ingest.py` downloads 12 Wikipedia articles into `data/`.
2. `rag.py` splits them into overlapping ~180-word chunks and indexes them with BM25.
3. For each question, the top 4 chunks go to the LLM with strict instructions to answer
   only from them, cite `[n]`, or reply with a fixed refusal.
4. If the best retrieval score is below a threshold, the app refuses without calling the model.

## Decisions log  (TODO: your reasoning)
- Why BM25 instead of embeddings:
- Why this chunk size / overlap:
- Why a retrieval-score threshold, and how I picked it:

## Evals
15 questions: 10 answerable, 3 out of scope, 2 ambiguous. Run `python eval.py`.

| Metric | Result |
|---|---|
| Answerable questions answered with citations | _/10 |
| Out-of-scope questions correctly refused | _/3 |

**Failures I found:** (TODO)
**What I changed because of them:** (TODO)

## Run locally
```
pip install -r requirements.txt
python ingest.py
export ANTHROPIC_API_KEY=your_key      # Windows: set ANTHROPIC_API_KEY=your_key
streamlit run app.py
```

## Deploy
Streamlit Community Cloud: connect the repo, add `ANTHROPIC_API_KEY` under app secrets.
Never commit keys. Set a monthly spend limit in the API console.

## What I'd do next
Embeddings + hybrid retrieval, a larger eval set, answer-quality scoring by a second model
check, and per-user rate limiting.

## Attribution
Source text from Wikipedia, licensed CC BY-SA 4.0.
