"""Metadata persistence layer for PKM AI."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence
from uuid import uuid4


@dataclass
class DocumentRecord:
    id: str
    path: str
    extension: str
    num_chars: int


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    position: int
    text: str


@dataclass
class ChunkInput:
    text: str
    position: int


class MetadataStoreError(Exception):
    """Raised when metadata operations fail."""


class BaseMetadataStore:
    """Abstract metadata store interface."""

    def upsert_document(self, *, path: str, extension: str, num_chars: int) -> DocumentRecord:
        raise NotImplementedError

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[ChunkInput],
    ) -> List[ChunkRecord]:
        raise NotImplementedError

    def list_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        raise NotImplementedError


class SQLiteMetadataStore(BaseMetadataStore):
    """SQLite-backed metadata store."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                extension TEXT NOT NULL,
                num_chars INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                text TEXT NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_position
                ON chunks(document_id, position);
            """
        )
        self._conn.commit()

    def upsert_document(self, *, path: str, extension: str, num_chars: int) -> DocumentRecord:
        cur = self._conn.cursor()
        cur.execute("SELECT id FROM documents WHERE path = ?", (path,))
        row = cur.fetchone()
        if row:
            document_id = row["id"]
            cur.execute(
                """
                UPDATE documents
                   SET extension = ?, num_chars = ?, updated_at = CURRENT_TIMESTAMP
                 WHERE id = ?
                """,
                (extension, num_chars, document_id),
            )
        else:
            document_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO documents (id, path, extension, num_chars)
                VALUES (?, ?, ?, ?)
                """,
                (document_id, path, extension, num_chars),
            )
        self._conn.commit()
        return DocumentRecord(id=document_id, path=path, extension=extension, num_chars=num_chars)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[ChunkInput],
    ) -> List[ChunkRecord]:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))

        chunk_records: List[ChunkRecord] = []
        for chunk in chunks:
            chunk_id = str(uuid4())
            cur.execute(
                """
                INSERT INTO chunks (id, document_id, position, text)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, document_id, chunk.position, chunk.text),
            )
            chunk_records.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    position=chunk.position,
                    text=chunk.text,
                )
            )
        self._conn.commit()
        return chunk_records

    def list_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, document_id, position, text FROM chunks WHERE document_id = ? ORDER BY position",
            (document_id,),
        )
        rows = cur.fetchall()
        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                position=row["position"],
                text=row["text"],
            )
            for row in rows
        ]

    def close(self) -> None:
        self._conn.close()


class JSONMetadataStore(BaseMetadataStore):
    """JSON-backed metadata store (simple, file-based)."""

    def __init__(self, json_path: Path | str) -> None:
        self._path = Path(json_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(json.dumps({"documents": {}, "chunks": {}}, indent=2), encoding="utf-8")

    def _load(self) -> dict:
        with self._path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write(self, data: dict) -> None:
        with self._path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, ensure_ascii=False)

    def upsert_document(self, *, path: str, extension: str, num_chars: int) -> DocumentRecord:
        data = self._load()
        documents: dict = data.setdefault("documents", {})

        document_id = None
        for doc_id, doc in documents.items():
            if doc["path"] == path:
                document_id = doc_id
                break

        if document_id is None:
            document_id = str(uuid4())

        documents[document_id] = {
            "path": path,
            "extension": extension,
            "num_chars": num_chars,
        }
        self._write(data)
        return DocumentRecord(id=document_id, path=path, extension=extension, num_chars=num_chars)

    def replace_document_chunks(
        self,
        document_id: str,
        chunks: Sequence[ChunkInput],
    ) -> List[ChunkRecord]:
        data = self._load()
        chunks_data: dict = data.setdefault("chunks", {})

        # Remove previous chunks for the document
        to_delete = [chunk_id for chunk_id, chunk in chunks_data.items() if chunk["document_id"] == document_id]
        for chunk_id in to_delete:
            del chunks_data[chunk_id]

        chunk_records: List[ChunkRecord] = []
        for chunk in chunks:
            chunk_id = str(uuid4())
            chunks_data[chunk_id] = {
                "document_id": document_id,
                "position": chunk.position,
                "text": chunk.text,
            }
            chunk_records.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    position=chunk.position,
                    text=chunk.text,
                )
            )

        self._write(data)
        return chunk_records

    def list_document_chunks(self, document_id: str) -> List[ChunkRecord]:
        data = self._load()
        chunks_data: dict = data.get("chunks", {})
        relevant = [
            (chunk_id, chunk)
            for chunk_id, chunk in chunks_data.items()
            if chunk["document_id"] == document_id
        ]
        sorted_chunks = sorted(relevant, key=lambda item: item[1]["position"])
        return [
            ChunkRecord(
                id=chunk_id,
                document_id=document_id,
                position=chunk["position"],
                text=chunk["text"],
            )
            for chunk_id, chunk in sorted_chunks
        ]


__all__ = [
    "DocumentRecord",
    "ChunkRecord",
    "ChunkInput",
    "MetadataStoreError",
    "BaseMetadataStore",
    "SQLiteMetadataStore",
    "JSONMetadataStore",
]
