from pathlib import Path

import pytest

from pkm_ai.storage import (
    ChunkInput,
    JSONMetadataStore,
    SQLiteMetadataStore,
)


def test_sqlite_metadata_store_upsert_and_replace(tmp_path):
    db_path = tmp_path / "metadata.db"
    store = SQLiteMetadataStore(db_path)

    doc = store.upsert_document(path="/tmp/doc.txt", extension=".txt", num_chars=42)
    assert Path(doc.path).as_posix() == "/tmp/doc.txt"

    chunks = [ChunkInput(text="chunk-1", position=0), ChunkInput(text="chunk-2", position=1)]
    chunk_records = store.replace_document_chunks(doc.id, chunks)

    assert len(chunk_records) == 2
    assert [c.position for c in chunk_records] == [0, 1]

    stored_chunks = store.list_document_chunks(doc.id)
    assert [c.text for c in stored_chunks] == ["chunk-1", "chunk-2"]

    # Replace with a single chunk and ensure the old ones are gone
    new_records = store.replace_document_chunks(doc.id, [ChunkInput(text="replaced", position=0)])
    assert len(new_records) == 1

    stored_chunks = store.list_document_chunks(doc.id)
    assert [c.text for c in stored_chunks] == ["replaced"]

    # Re-upsert same path should reuse document id
    updated_doc = store.upsert_document(path="/tmp/doc.txt", extension=".txt", num_chars=100)
    assert updated_doc.id == doc.id
    assert updated_doc.num_chars == 100

    store.close()


def test_json_metadata_store_replace(tmp_path):
    json_path = tmp_path / "metadata.json"
    store = JSONMetadataStore(json_path)

    doc = store.upsert_document(path="/tmp/doc.md", extension=".md", num_chars=10)

    records = store.replace_document_chunks(
        doc.id,
        [ChunkInput(text="chunk-a", position=1), ChunkInput(text="chunk-b", position=0)],
    )
    assert len(records) == 2

    listed = store.list_document_chunks(doc.id)
    assert [chunk.text for chunk in listed] == ["chunk-b", "chunk-a"]

    store.replace_document_chunks(doc.id, [])
    assert store.list_document_chunks(doc.id) == []

    # Re-upsert updates metadata but keeps same document id
    updated_doc = store.upsert_document(path="/tmp/doc.md", extension=".md", num_chars=25)
    assert updated_doc.id == doc.id
    assert updated_doc.num_chars == 25


@pytest.mark.parametrize("StoreCls", [SQLiteMetadataStore, JSONMetadataStore])
def test_store_handles_multiple_documents(tmp_path, StoreCls):
    store_path = tmp_path / ("data.db" if StoreCls is SQLiteMetadataStore else "data.json")
    store = StoreCls(store_path)

    doc1 = store.upsert_document(path="/tmp/doc1.txt", extension=".txt", num_chars=5)
    doc2 = store.upsert_document(path="/tmp/doc2.txt", extension=".txt", num_chars=7)

    store.replace_document_chunks(doc1.id, [ChunkInput(text="a", position=0)])
    store.replace_document_chunks(doc2.id, [ChunkInput(text="b", position=0)])

    chunks1 = store.list_document_chunks(doc1.id)
    chunks2 = store.list_document_chunks(doc2.id)

    assert chunks1[0].text == "a"
    assert chunks2[0].text == "b"

    if isinstance(store, SQLiteMetadataStore):
        store.close()
