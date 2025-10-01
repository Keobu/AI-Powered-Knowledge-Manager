"""Utilities for exporting summaries and highlighting source snippets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:  # pragma: no cover - optional dependency
    from reportlab.lib.pagesizes import A4  # type: ignore[attr-defined]
    from reportlab.pdfgen import canvas  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - handled when PDF export is called
    A4 = None
    canvas = None

from .chat import ChatResponse


@dataclass
class SummarySection:
    title: str
    content: str


class ExportError(Exception):
    """Raised when an export cannot be completed."""


def export_summary_to_markdown(
    response: ChatResponse,
    *,
    sections: Sequence[SummarySection] | None = None,
    highlight_format: str = "**%s**",
) -> str:
    """Return a Markdown string with the answer and highlighted supporting chunks."""

    lines = ["# Summary", "", response.answer, "", "## Supporting Context"]
    for idx, chunk in enumerate(response.chunks, start=1):
        highlighted_text = highlight_format % chunk.text if "%s" in highlight_format else chunk.text
        lines.append(f"### Chunk {idx}")
        lines.append(f"- Path: {chunk.metadata.get('path', 'unknown')}")
        lines.append(f"- Position: {chunk.metadata.get('position')}")
        lines.append(f"- Score: {chunk.score:.3f}")
        lines.append("")
        lines.append(highlighted_text)
        lines.append("")

    if sections:
        lines.append("## Extra Notes")
        for section in sections:
            lines.append(f"### {section.title}")
            lines.append(section.content)
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def export_summary_to_pdf(
    response: ChatResponse,
    output_path: Path | str,
    *,
    sections: Sequence[SummarySection] | None = None,
) -> Path:
    """Create a simple PDF summarizing the answer and context."""

    if canvas is None or A4 is None:
        raise ExportError("reportlab is required for PDF export. Install 'reportlab'.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pdf = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    y = height - 72

    def draw_line(text: str) -> None:
        nonlocal y
        pdf.drawString(72, y, text)
        y -= 18
        if y < 72:
            pdf.showPage()
            y = height - 72

    draw_line("PKM AI Summary")
    draw_line("")
    draw_line("Answer:")
    for line in response.answer.splitlines():
        draw_line(line)
    draw_line("")
    draw_line("Supporting Context:")
    for idx, chunk in enumerate(response.chunks, start=1):
        draw_line(f"Chunk {idx} | Score: {chunk.score:.3f}")
        draw_line(f"Path: {chunk.metadata.get('path', 'unknown')} | Position: {chunk.metadata.get('position')}")
        for line in chunk.text.splitlines():
            draw_line(line)
        draw_line("")

    if sections:
        draw_line("Extra Notes:")
        for section in sections:
            draw_line(section.title)
            for line in section.content.splitlines():
                draw_line(line)
            draw_line("")

    pdf.save()
    return path


def export_context_to_json(response: ChatResponse, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "answer": response.answer,
        "prompt": response.prompt,
        "chunks": [
            {
                "text": chunk.text,
                "metadata": chunk.metadata,
                "score": chunk.score,
            }
            for chunk in response.chunks
        ],
    }

    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


__all__ = [
    "SummarySection",
    "ExportError",
    "export_summary_to_markdown",
    "export_summary_to_pdf",
    "export_context_to_json",
]
