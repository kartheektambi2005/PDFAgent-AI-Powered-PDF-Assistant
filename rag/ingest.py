"""Document processing: PDF bytes -> per-page text -> page-aware chunks."""
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from . import config


class IngestError(Exception):
    """User-facing problem with the uploaded file."""


@dataclass
class Chunk:
    page: int  # 1-based PDF page
    idx: int  # order within the document
    text: str


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that contain text."""
    if not data.startswith(b"%PDF"):
        raise IngestError("That file is not a PDF.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("This PDF is password-protected.")
        pages = []
        for n, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # one bad page shouldn't sink the whole document
                text = ""
            text = _clean(text)
            if text:
                pages.append((n, text))
    except IngestError:
        raise
    except (PyPdfError, ValueError, OSError, KeyError) as e:
        raise IngestError(f"Could not read this PDF ({type(e).__name__}).") from e
    if not pages:
        raise IngestError("No extractable text found (scanned/image-only PDFs are not supported).")
    return pages


def _clean(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join hard-wrapped lines, keep paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def _units(text: str, limit: int) -> list[str]:
    """Split into sentence-ish units, hard-splitting anything longer than `limit`."""
    out = []
    for piece in re.split(r"(?<=[.!?])\s+|\n\n", text):
        piece = piece.strip()
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            out.append(piece)
    return out


def chunk_pages(pages: list[tuple[int, str]], size: int = config.CHUNK_CHARS,
                overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Chunk each page separately so every chunk maps to exactly one page."""
    chunks: list[Chunk] = []
    for page, text in pages:
        cur: list[str] = []
        cur_len = 0
        fresh = 0  # units in `cur` that are new (not carried-over overlap)

        def flush():
            chunks.append(Chunk(page, len(chunks), " ".join(cur)))

        for unit in _units(text, size):
            if cur and cur_len + len(unit) + 1 > size:
                flush()
                keep, kept_len = [], 0  # carry the tail as overlap
                for u in reversed(cur):
                    if kept_len + len(u) > overlap:
                        break
                    keep.insert(0, u)
                    kept_len += len(u) + 1
                cur, cur_len, fresh = keep, kept_len, 0
            cur.append(unit)
            cur_len += len(unit) + 1
            fresh += 1
        if fresh:  # skip a trailing chunk that is pure overlap
            flush()
    return chunks
