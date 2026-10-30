"""Flask entry point: routes only. Logic lives in rag/."""
from flask import Flask, jsonify, render_template, request

from rag import config
from rag.answer import answer_question
from rag.gemini import Gemini, GeminiError
from rag.ingest import IngestError
from rag.pipeline import index_pdf
from rag.store import Store

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_PDF_BYTES

store = Store(config.DB_PATH)
_client = None


class _LazyGemini:
    """Builds the real client on first API use, so file validation errors surface before key errors."""

    def __getattr__(self, name):
        global _client
        if _client is None:
            _client = Gemini()
        return getattr(_client, name)


client = _LazyGemini()


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"File too large (max {config.MAX_PDF_BYTES // (1024 * 1024)} MB)."), 413


@app.errorhandler(GeminiError)
def gemini_error(e):
    return jsonify(error=str(e)), 502


@app.errorhandler(IngestError)
def ingest_error(e):
    return jsonify(error=str(e)), 400


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify(error="Choose a PDF file first."), 400
    return jsonify(index_pdf(store, client, f.filename, f.read()))


@app.post("/ask")
def ask():
    body = request.get_json(silent=True) or {}
    question = (body.get("question") or "").strip()
    doc_id = body.get("doc_id") or ""
    if not question:
        return jsonify(error="Type a question."), 400
    if len(question) > config.MAX_QUESTION_CHARS:
        return jsonify(error=f"Question too long (max {config.MAX_QUESTION_CHARS} characters)."), 400
    if not store.has_doc(doc_id):
        return jsonify(error="Upload a PDF first."), 400
    return jsonify(answer_question(store, client, doc_id, question))


@app.get("/usage")
def usage():
    return jsonify(store.usage_totals())


if __name__ == "__main__":
    app.run(debug=False, port=5000)
