from pathlib import Path

from pkm_ai.chat import ChatResponse, RetrievedChunk
from pkm_ai.export import (
    SummarySection,
    export_context_to_json,
    export_summary_to_markdown,
)


def build_response() -> ChatResponse:
    return ChatResponse(
        answer="Summary answer",
        prompt="prompt",
        chunks=[
            RetrievedChunk(
                text="Important chunk text",
                metadata={"path": "doc.txt", "position": 0},
                score=0.9,
            )
        ],
    )


def test_export_markdown(tmp_path):
    response = build_response()
    md = export_summary_to_markdown(
        response,
        sections=[SummarySection(title="Notes", content="Extra detail")],
    )

    assert "Summary answer" in md
    assert "Important chunk text" in md
    assert "Notes" in md


def test_export_json(tmp_path):
    response = build_response()
    output = export_context_to_json(response, tmp_path / "context.json")
    data = output.read_text(encoding="utf-8")

    assert "Summary answer" in data
    assert "doc.txt" in data
    assert Path(output).exists()
