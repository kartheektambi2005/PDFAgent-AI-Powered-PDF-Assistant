"""Offline tests: fake embedder + scripted LLM, so no network or API key is needed."""
import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rag import config
from rag.answer import answer_question
from rag.ingest import IngestError, chunk_pages, extract_pages
from rag.pipeline import index_pdf
from rag.store import Store


def make_pdf(pages: list[str]) -> bytes:
    """Minimal valid PDF, one text line per page (enough for pypdf extraction)."""
    objs = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        n = len(objs) + 1
        kids.append(f"{n} 0 R")
        esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 50 700 Td ({esc}) Tj ET" if text else ""
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {n + 1} 0 R >>")
        objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


class FakeClient:
    """Hashed bag-of-words embeddings (related text => high cosine) and a scripted generate()."""
    DIM = 256

    def __init__(self, reply=None):
        self.calls = {"embed": 0, "generate": 0}
        self.reply = reply or {"answerable": True, "answer": "ok", "pages": [1]}

    def embed(self, texts, task):
        self.calls["embed"] += 1
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in "".join(c if c.isalnum() else " " for c in t.lower()).split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1
        return out

    def generate(self, system, prompt, schema, max_tokens):
        self.calls["generate"] += 1
        self.last_prompt = prompt
        return dict(self.reply), {"input": 100, "output": 10}


@pytest.fixture
def store(tmp_path):
    return Store(str(tmp_path / "t.db"))


@pytest.fixture(autouse=True)
def lenient_threshold(monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.2)  # bag-of-words cosines are lower than real embeddings


PAGES = ["The reactor core operates at 320 degrees Celsius during normal operation.",
         "Maintenance crews inspect the cooling pumps every 90 days.",
         "The annual budget for the safety program is 4 million dollars."]


# --- ingestion -------------------------------------------------------------
def test_extract_pages_keeps_page_numbers():
    pages = extract_pages(make_pdf(PAGES))
    assert [p for p, _ in pages] == [1, 2, 3]
    assert "320 degrees" in pages[0][1]


def test_blank_page_skipped_but_numbering_preserved():
    pages = extract_pages(make_pdf(["First page text.", "", "Third page text."]))
    assert [p for p, _ in pages] == [1, 3]


@pytest.mark.parametrize("data", [b"hello", b"%PDF-1.4 garbage"])
def test_bad_files_raise_ingest_error(data):
    with pytest.raises(IngestError):
        extract_pages(data)


def test_image_only_pdf_raises():
    with pytest.raises(IngestError, match="No extractable text"):
        extract_pages(make_pdf(["", ""]))


def test_chunks_never_cross_pages_and_respect_size():
    long = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(80))
    chunks = chunk_pages([(1, long), (2, "Short second page.")], size=300, overlap=60)
    assert {c.page for c in chunks} == {1, 2}
    assert all(len(c.text) <= 300 for c in chunks)
    assert sum(c.page == 1 for c in chunks) > 1
    assert [c.idx for c in chunks] == list(range(len(chunks)))
    assert "Short second page." not in " ".join(c.text for c in chunks if c.page == 1)


def test_chunk_overlap_repeats_tail_of_previous_chunk():
    text = " ".join(f"Sentence {i} is here." for i in range(40))
    a, b = chunk_pages([(1, text)], size=200, overlap=60)[:2]
    assert a.text.split(". ")[-1] in b.text or a.text.split(".")[-2].strip() in b.text


# --- storage / reuse -------------------------------------------------------
def test_same_pdf_indexed_once(store):
    c, pdf = FakeClient(), make_pdf(PAGES)
    first = index_pdf(store, c, "a.pdf", pdf)
    second = index_pdf(store, c, "renamed.pdf", pdf)
    assert not first["reused"] and second["reused"]
    assert c.calls["embed"] == 1


def test_embeddings_persist_across_store_instances(store, tmp_path):
    c = FakeClient()
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    again = Store(str(tmp_path / "t.db"))
    rows, mat = again.load_chunks(doc["id"])
    assert len(rows) == doc["chunks"] and mat.shape[0] == len(rows)
    assert index_pdf(again, c, "a.pdf", make_pdf(PAGES))["reused"]
    assert c.calls["embed"] == 1


# --- answering -------------------------------------------------------------
def test_supported_answer_has_correct_page_citation(store, monkeypatch):
    monkeypatch.setattr(config, "MIN_SCORE", 0.4)  # stopword-only overlap (e.g. "the") must not count as relevant
    c = FakeClient({"answerable": True, "answer": "Every 90 days.", "pages": [2]})
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    r = answer_question(store, c, doc["id"], "How often do crews inspect the cooling pumps?")
    assert r["answerable"] and [x["page"] for x in r["citations"]] == [2]
    assert "cooling pumps" in r["citations"][0]["snippet"]
    assert "[p.2]" in c.last_prompt and "[p.3]" not in c.last_prompt  # only relevant chunks sent


def test_unrelated_question_refuses_without_llm_call(store):
    c = FakeClient()
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    r = answer_question(store, c, doc["id"], "Who won the football world cup in 1998?")
    assert r["answer"] == config.REFUSAL and not r["answerable"] and r["citations"] == []
    assert c.calls["generate"] == 0


def test_model_says_unanswerable_gives_exact_refusal(store):
    c = FakeClient({"answerable": False, "answer": "made-up text", "pages": [1]})
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    r = answer_question(store, c, doc["id"], "What temperature does the reactor core run at?")
    assert r["answer"] == config.REFUSAL and r["citations"] == []


def test_fabricated_page_is_stripped_and_unsupported_answer_refused(store):
    doc_pdf = make_pdf(PAGES)
    c = FakeClient({"answerable": True, "answer": "320 C", "pages": [1, 99]})
    doc = index_pdf(store, c, "a.pdf", doc_pdf)
    r = answer_question(store, c, doc["id"], "What temperature does the reactor core run at?")
    assert [x["page"] for x in r["citations"]] == [1]

    c2 = FakeClient({"answerable": True, "answer": "something", "pages": [99]})
    r2 = answer_question(store, c2, doc["id"], "How often do crews inspect the cooling pumps?")
    assert r2["answer"] == config.REFUSAL


def test_repeated_question_costs_zero_calls(store):
    c = FakeClient({"answerable": True, "answer": "320", "pages": [1]})
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    q = "What temperature does the reactor core run at?"
    r1 = answer_question(store, c, doc["id"], q)
    before = dict(c.calls)
    r2 = answer_question(store, c, doc["id"], "  what temperature does the REACTOR core run at? ")
    assert c.calls == before and r2["cached"] and not r1["cached"]
    assert r2["answer"] == r1["answer"]


def test_new_phrasing_reuses_document_embeddings(store):
    c = FakeClient()
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    answer_question(store, c, doc["id"], "reactor core temperature")
    assert c.calls["embed"] == 2  # 1 for document, 1 for the question; document never re-embedded


def test_duplicate_chunks_are_not_sent_twice(store):
    dup = ["The warranty period is twelve months from delivery."] * 2 + ["Unrelated shipping notes."]
    c = FakeClient({"answerable": True, "answer": "12 months", "pages": [1, 2]})
    doc = index_pdf(store, c, "a.pdf", make_pdf(dup))
    answer_question(store, c, doc["id"], "How long is the warranty period?")
    assert c.last_prompt.count("The warranty period is twelve months") == 1


def test_usage_is_tracked(store):
    c = FakeClient({"answerable": True, "answer": "320", "pages": [1]})
    doc = index_pdf(store, c, "a.pdf", make_pdf(PAGES))
    answer_question(store, c, doc["id"], "What temperature does the reactor core run at?")
    u = store.usage_totals()
    assert u["generate"]["input_tokens"] == 100 and u["generate"]["output_tokens"] == 10
    assert u["embed_doc"]["estimated"]
