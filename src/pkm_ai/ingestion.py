"""Utility functions for document ingestion and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:  # pragma: no cover - import used only for typing
    from pypdf import PdfReader as _PdfReaderType
else:  # pragma: no cover - fallback for runtime without typing support
    _PdfReaderType = Any

try:  # pragma: no cover - dependency availability is environment-specific
    from pypdf import PdfReader as _PdfReader  # type: ignore[assignment]
except ImportError:  # pragma: no cover - handled at runtime
    _PdfReader = None

PdfReader: _PdfReaderType | None = _PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


@dataclass
class Document:
    """Simple container for loaded document content."""

    content: str
    metadata: dict


class DocumentLoaderError(Exception):
    """Raised when a document cannot be loaded."""


def _load_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - unlikely in tests
        raise DocumentLoaderError(f"Impossibile decodificare il file di testo: {path}") from exc


def _load_pdf_file(path: Path) -> str:
    if PdfReader is None:
        raise DocumentLoaderError("Supporto PDF non disponibile: installa 'pypdf'.")

    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - pypdf already validates input
        raise DocumentLoaderError(f"Impossibile aprire il PDF: {path}") from exc

    contents: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        contents.append(text.strip())
    return "\n".join(chunk for chunk in contents if chunk)


def load_document(path: Path | str) -> Document:
    """Load a document (PDF, TXT, MD) and return its content with metadata."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise DocumentLoaderError(f"File non trovato: {file_path}")

    extension = file_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise DocumentLoaderError(
            f"Formato non supportato: {extension}. Formati supportati: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if extension == ".pdf":
        content = _load_pdf_file(file_path)
    else:
        content = _load_text_file(file_path)

    return Document(
        content=content,
        metadata={
            "path": str(file_path),
            "extension": extension,
            "num_chars": len(content),
        },
    )


def split_text(
    text: str,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Split raw text into overlapping chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size deve essere positivo")
    if overlap < 0:
        raise ValueError("overlap non può essere negativo")
    if overlap >= chunk_size:
        raise ValueError("overlap deve essere inferiore a chunk_size")

    normalized = text.strip()
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(normalized)
    while start < text_length:
        end = start + chunk_size
        chunk = normalized[start:end]
        chunks.append(chunk)
        if end >= text_length:
            break
        start = end - overlap  # reintroduce parte del testo per il contesto
    return chunks


__all__ = ["Document", "DocumentLoaderError", "load_document", "split_text"]
