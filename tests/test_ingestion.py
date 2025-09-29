from pathlib import Path

import pytest

from pkm_ai.ingestion import DocumentLoaderError, PdfReader, load_document, split_text


def test_load_document_txt(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello world!", encoding="utf-8")

    doc = load_document(file_path)

    assert doc.content == "Hello world!"
    assert doc.metadata["extension"] == ".txt"
    assert doc.metadata["num_chars"] == len("Hello world!")


def test_load_document_md(tmp_path):
    file_path = tmp_path / "notes.md"
    file_path.write_text("# Title\nSome content.", encoding="utf-8")

    doc = load_document(file_path)

    assert doc.content.startswith("# Title")
    assert doc.metadata["extension"] == ".md"


def test_load_document_pdf(monkeypatch, tmp_path):
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4 test content")

    if PdfReader is None:
        pytest.skip("pypdf non installato")

    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage("Hello"), FakePage(None), FakePage("PDF")]  # type: ignore[arg-type]

    monkeypatch.setattr("pkm_ai.ingestion.PdfReader", lambda _: FakeReader(None))

    doc = load_document(file_path)

    assert doc.content == "Hello\nPDF"
    assert doc.metadata["extension"] == ".pdf"


def test_load_document_unsupported(tmp_path):
    file_path = tmp_path / "data.csv"
    file_path.write_text("id,value\n1,42", encoding="utf-8")

    with pytest.raises(DocumentLoaderError):
        load_document(file_path)


def test_split_text_basic():
    text = "abcdefghij" * 10
    chunks = split_text(text, chunk_size=10, overlap=2)

    assert chunks
    assert all(len(chunk) <= 10 for chunk in chunks[:-1])
    assert chunks[0][-2:] == chunks[1][:2]


@pytest.mark.parametrize(
    "chunk_size,overlap,expected_exception",
    [
        (0, 0, ValueError),
        (-1, 0, ValueError),
        (10, -1, ValueError),
        (10, 10, ValueError),
    ],
)
def test_split_text_invalid_params(chunk_size, overlap, expected_exception):
    with pytest.raises(expected_exception):
        split_text("sample", chunk_size=chunk_size, overlap=overlap)


def test_split_text_empty():
    assert split_text("   ") == []
