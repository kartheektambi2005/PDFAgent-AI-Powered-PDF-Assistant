"""Live check against the real Gemini API.

  python tests/live_scenarios.py --models                      # list embedding/generation models for your key
  python tests/live_scenarios.py doc.pdf tests/scenarios.json  # run scenarios

scenarios.json: [{"name": "...", "question": "...", "expect": "answer"|"refuse", "pages": [3]}]
`pages` (optional) = pages a correct answer must cite. The last step re-asks the first question and
checks that it costs zero API calls.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import config
from rag.answer import answer_question
from rag.gemini import Gemini
from rag.pipeline import index_pdf
from rag.retrieve import retrieve
from rag.store import Store


def main():
    client = Gemini()
    if sys.argv[1] == "--models":
        for m in client.list_models():
            methods = m.get("supportedGenerationMethods", [])
            if "embedContent" in methods or "batchEmbedContents" in methods or "generateContent" in methods:
                print(m["name"], methods)
        return

    pdf_path, scen_path = sys.argv[1], sys.argv[2]
    scenarios = json.load(open(scen_path, encoding="utf-8"))
    store = Store(os.path.join(tempfile.mkdtemp(), "live.db"))  # fresh DB: nothing cached from earlier runs
    data = open(pdf_path, "rb").read()

    doc = index_pdf(store, client, os.path.basename(pdf_path), data)
    print(f"Indexed {doc['name']}: {doc['pages']} pages with text, {doc['chunks']} chunks, embed calls={client.calls['embed']}")
    again = index_pdf(store, client, doc["name"], data)
    print(f"Re-upload reused={again['reused']}, embed calls still {client.calls['embed']}\n")

    failures = 0
    for s in scenarios + [{"name": "repeat first question", "question": scenarios[0]["question"], "expect": "cache"}]:
        before = dict(client.calls)
        hits, best = retrieve(store, client, doc["id"], s["question"]) if s["expect"] != "cache" else ([], 0)
        before = dict(client.calls) if s["expect"] != "cache" else before  # retrieval embed is cached for the real call
        r = answer_question(store, client, doc["id"], s["question"])
        used = {k: client.calls[k] - before[k] for k in client.calls}
        cited = [c["page"] for c in r["citations"]]

        ok = True
        if s["expect"] == "refuse":
            ok = r["answer"] == config.REFUSAL
        elif s["expect"] == "answer":
            ok = r["answerable"] and (not s.get("pages") or set(s["pages"]) <= set(cited))
        elif s["expect"] == "cache":
            ok = r["cached"] and used == {"embed": 0, "generate": 0}
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {s['name']}\n  Q: {s['question']}\n  A: {r['answer']}\n"
              f"  cited pages={cited} best_score={r['best_score']} retrieved_pages={sorted({h.page for h in hits})}\n"
              f"  api calls this question={used} tokens in/out={r['usage']['input']}/{r['usage']['output']} cached={r['cached']}\n")

    print("Totals:", json.dumps(store.usage_totals()))
    print(f"{failures} scenario(s) failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
