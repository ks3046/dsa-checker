"""Run the eval set and write results.csv.  Usage: python eval.py

Auto-scoring is deliberately simple. READ the answers yourself and record real
failures in the README: that analysis is the product work.
  answerable   -> pass if not refused AND has a [n] citation AND hits an expected keyword
  out_of_scope -> pass if the system refused
  ambiguous    -> left blank for manual review
"""
import csv
import json
import re

from rag import Retriever, answer

retriever = Retriever()
rows = []
with open("evals.json", encoding="utf-8") as f:
    cases = json.load(f)

for case in cases:
    r = answer(case["question"], retriever)
    text = r["answer"]
    cited = bool(re.search(r"\[\d+\]", text))
    keyword_hit = any(k.lower() in text.lower() for k in case.get("expect", []))
    if case["type"] == "answerable":
        passed = (not r["refused"]) and cited and keyword_hit
    elif case["type"] == "out_of_scope":
        passed = r["refused"]
    else:
        passed = ""
    rows.append(
        {
            "type": case["type"],
            "question": case["question"],
            "refused": r["refused"],
            "cited": cited,
            "keyword_hit": keyword_hit,
            "auto_pass": passed,
            "answer": text.replace("\n", " "),
        }
    )
    print(f"[{case['type']}] {case['question']} -> pass={passed}")

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

scored = [r for r in rows if r["auto_pass"] != ""]
print(f"\nAuto-pass: {sum(bool(r['auto_pass']) for r in scored)}/{len(scored)}. Details in results.csv")
