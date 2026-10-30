"""Settings read from environment (.env is loaded for local use). No secrets live in source."""
import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-3.5-flash-lite")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

TOP_K = int(os.getenv("TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))  # best-chunk cosine below this => refuse without an LLM call
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "400"))

CHUNK_CHARS = 900
CHUNK_OVERLAP = 120
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_QUESTION_CHARS = 500

# Vercel's project directory is read-only; only the temp dir (/tmp) is writable there.
_DB_DIR = tempfile.gettempdir() if os.getenv("VERCEL") else os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.getenv("DB_PATH", os.path.join(_DB_DIR, "rag.db"))

REFUSAL = "I couldn't find sufficient information in the uploaded document to answer this question."
