"""Wires document processing to the store: PDF bytes -> chunks -> embeddings -> SQLite (once per unique file)."""
import hashlib

from .ingest import chunk_pages, extract_pages
from .store import Store


def index_pdf(store: Store, client, name: str, data: bytes) -> dict:
    doc_id = hashlib.sha256(data).hexdigest()[:16]
    existing = store.get_doc(doc_id)
    if existing:  # same bytes already indexed: reuse stored embeddings, zero API calls
        return {**existing, "name": name, "reused": True}

    pages = extract_pages(data)
    chunks = chunk_pages(pages)
    embeddings = client.embed([c.text for c in chunks], "RETRIEVAL_DOCUMENT")
    store.add_doc(doc_id, name, len(pages), chunks, embeddings)
    store.log_usage("embed_doc", 1, sum(len(c.text) for c in chunks) // 4, 0, estimated=True)
    return {"id": doc_id, "name": name, "pages": len(pages), "chunks": len(chunks), "reused": False}
