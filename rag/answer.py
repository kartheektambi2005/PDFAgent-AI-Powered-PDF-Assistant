"""Generation: one short, grounded Gemini call over retrieved chunks, with citation validation."""
from . import config
from .retrieve import retrieve
from .store import Store, question_key

SYSTEM = (
    "Answer the question using ONLY the numbered PDF excerpts provided. Use no outside knowledge. "
    "If the excerpts do not explicitly contain the answer, set answerable=false. "
    "If the question assumes something the excerpts do not support, set answerable=false, "
    "unless an excerpt directly contradicts the assumption, in which case answer with that correction. "
    "Be concise; never invent facts, figures or names. "
    "In pages, list the page numbers of the excerpts you relied on."
)

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "answerable": {"type": "BOOLEAN"},
        "answer": {"type": "STRING"},
        "pages": {"type": "ARRAY", "items": {"type": "INTEGER"}},
    },
    "required": ["answerable", "answer", "pages"],
}


def _refusal(best: float, **extra) -> dict:
    return {"answerable": False, "answer": config.REFUSAL, "citations": [], "best_score": round(best, 3),
            "usage": {"input": 0, "output": 0}, **extra}


def answer_question(store: Store, client, doc_id: str, question: str) -> dict:
    # Retrieval settings are part of the key so tuning MIN_SCORE/TOP_K never serves stale answers.
    key = f"{question_key(question)}|{config.TOP_K}|{config.MIN_SCORE}|{config.GEN_MODEL}"
    cached = store.get_answer(doc_id, key)
    if cached:
        return {**cached, "cached": True, "usage": {"input": 0, "output": 0}}

    hits, best = retrieve(store, client, doc_id, question)
    if not hits:  # retrieval alone is enough to refuse: no LLM call
        result = _refusal(best, llm_called=False)
        store.put_answer(doc_id, key, result)
        return {**result, "cached": False}

    context = "\n\n".join(f"[p.{h.page}] {h.text}" for h in hits)
    prompt = f"Excerpts:\n{context}\n\nQuestion: {question}"
    parsed, usage = client.generate(SYSTEM, prompt, SCHEMA, config.MAX_OUTPUT_TOKENS)
    store.log_usage("generate", 1, usage["input"], usage["output"])

    # Only pages we actually sent can be cited; a hallucinated page is dropped.
    by_page = {}
    for h in hits:
        if h.page not in by_page or h.score > by_page[h.page].score:
            by_page[h.page] = h
    pages = sorted({p for p in parsed.get("pages", []) if p in by_page})

    if not parsed.get("answerable") or not pages or not parsed.get("answer", "").strip():
        result = _refusal(best, llm_called=True, usage=usage)
    else:
        citations = [{"page": p, "snippet": by_page[p].text[:240], "score": round(by_page[p].score, 3)} for p in pages]
        result = {"answerable": True, "answer": parsed["answer"].strip(), "citations": citations,
                  "best_score": round(best, 3), "usage": usage, "llm_called": True}
    store.put_answer(doc_id, key, result)
    return {**result, "cached": False}
