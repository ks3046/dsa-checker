# DSA Doubt Solver

A small RAG app that answers data-structures-and-algorithms questions **only** from a
fixed set of sources, cites them, and says "I don't know" when the sources don't cover
the question.


## Product note  
User: A student learning DSA, for example someone preparing for coding interviews, who gets stuck on a concept while studying alone and wants a quick explanation they can trust.
Problem: A general chatbot answers fluently even when it’s wrong, and the learner can’t check it. A grounded answer comes only from known sources, shows citations the learner can verify, and refuses when the sources don’t cover the question. It trades coverage for trust.
Non-goals: No agents, login, personalization, code execution or vector database. It isn’t a replacement for a course, and it only covers 12 articles.
Success looks like: For in-scope questions, correct answers that carry citations. For out-of-scope questions, a correct refusal.

## How it works
1. `ingest.py` downloads 12 Wikipedia articles into `data/`.
2. `rag.py` splits them into overlapping ~180-word chunks and indexes them with BM25.
3. For each question, the top 4 chunks go to the LLM with strict instructions to answer
   only from them, cite `[n]`, or reply with a fixed refusal.
4. If the best retrieval score is below a threshold, the app refuses without calling the model.

## Decisions log  
BM25 instead of embeddings: It needs no embedding API or extra cost, it was buildable in a 3-hour budget, and it’s transparent (I can see which words made a chunk match). It suits keyword-heavy technical queries like “quicksort”. The trade-off is that it misses paraphrases and synonyms, so hybrid or embedding retrieval is the next step.

Chunk size and overlap: I started with 180 words and 40 overlap. That’s big enough for a self-contained explanation and small enough to fit 4 chunks cheaply in the prompt, and the overlap avoids cutting an explanation in half. This was a starting guess, not a tuned value. Don’t write “chosen after testing” unless you test it: try 120, 180 and 300, rerun eval.py, and record the results here.

Retrieval-score threshold: 1.0 is also a starting guess. To pick it from data, run this snippet on your eval questions and choose a value between the scores of answerable and out-of-scope questions:


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
