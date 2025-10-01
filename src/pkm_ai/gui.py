"""CustomTkinter desktop GUI for the PKM AI project."""

from __future__ import annotations

import datetime
import os
import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from dotenv import load_dotenv, set_key
from PIL import Image
from tkinter import filedialog, messagebox, ttk

from .app_state import AppState
from .chat import ChatEngine, ChatResponse
from .embeddings import build_vector_store
from .export import (
    ExportError,
    SummarySection,
    export_context_to_json,
    export_summary_to_markdown,
    export_summary_to_pdf,
)
from .pipeline import DocumentIngestionPipeline
from .storage import SQLiteMetadataStore


DATA_DIR = Path(os.environ.get("PKM_AI_DATA_DIR", "./data"))
EMBEDDING_DIM = 384
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
ACCENT_COLOR = "#38bdf8"


class AIKnowledgeManagerApp(ctk.CTk):
    """Main window for the desktop knowledge manager."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("AI Knowledge Manager")
        self.geometry("1180x760")
        self.minsize(980, 640)
        self.configure(fg_color="#0f172a")

        self._last_response: Optional[ChatResponse] = None
        self._current_embedding_backend = "SentenceTransformers"

        self.status_var = ctk.StringVar(value="Ready")
        self.progress_var = ctk.DoubleVar(value=0.0)

        load_dotenv()
        self._initialize_backend()
        self._build_layout()
        self._populate_documents()

    # ------------------------------------------------------------------
    # Backend wiring
    # ------------------------------------------------------------------
    def _initialize_backend(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        metadata_store = SQLiteMetadataStore(DATA_DIR / "metadata.db")
        vector_store = build_vector_store("faiss", dim=EMBEDDING_DIM)

        pipeline = DocumentIngestionPipeline(metadata_store, vector_store)
        chat_engine = ChatEngine(
            vector_store,
            metadata_store=metadata_store,
            llm=lambda prompt: "(Demo response) Connect a real LLM in settings.",
        )

        self.app_state = AppState(
            metadata_store=metadata_store,
            ingestion_pipeline=pipeline,
            chat_engine=chat_engine,
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(self, fg_color="#1e293b")
        hero.grid(row=0, column=0, padx=24, pady=(24, 12), sticky="ew")
        hero.grid_columnconfigure(1, weight=1)

        logo_label = ctk.CTkLabel(hero, text="🧠", font=ctk.CTkFont(size=36))
        logo_label.grid(row=0, column=0, padx=(16, 8), pady=16)

        title_font = ctk.CTkFont(size=26, weight="bold")
        subtitle_font = ctk.CTkFont(size=14)
        ctk.CTkLabel(hero, text="AI Knowledge Manager", font=title_font).grid(
            row=0, column=1, sticky="w", padx=(0, 8), pady=(18, 0)
        )
        ctk.CTkLabel(
            hero,
            text="Ingest. Search. Ask. Export. All your personal knowledge in one assistant.",
            font=subtitle_font,
            text_color="#94a3b8",
        ).grid(row=1, column=1, sticky="w", padx=(0, 8), pady=(0, 16))

        self.tab_view = ctk.CTkTabview(self, segmented_button_fg_color="#1e293b")
        self.tab_view.grid(row=1, column=0, padx=24, pady=(0, 12), sticky="nsew")

        self.dashboard_tab = self.tab_view.add("Dashboard")
        self.search_tab = self.tab_view.add("Search & Chat")
        self.settings_tab = self.tab_view.add("Settings")

        for tab in (self.dashboard_tab, self.search_tab, self.settings_tab):
            tab.configure(fg_color="#0f172a")

        self._build_dashboard_tab()
        self._build_search_tab()
        self._build_settings_tab()

        status_frame = ctk.CTkFrame(self, fg_color="#1e293b")
        status_frame.grid(row=2, column=0, padx=24, pady=(0, 16), sticky="ew")
        status_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=0, column=0, padx=16, pady=(10, 4), sticky="ew")

        self.status_progress = ctk.CTkProgressBar(
            status_frame, variable=self.progress_var, height=6, fg_color="#1e293b", progress_color=ACCENT_COLOR
        )
        self.status_progress.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")
        self.status_progress.set(0.0)

    def _build_dashboard_tab(self) -> None:
        self.dashboard_tab.grid_rowconfigure(1, weight=1)
        self.dashboard_tab.grid_columnconfigure((0, 1), weight=1)

        panel = ctk.CTkFrame(self.dashboard_tab, fg_color="#172554")
        panel.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 12), sticky="ew")
        panel.grid_columnconfigure(1, weight=1)

        upload_button = ctk.CTkButton(
            panel,
            text="📁 Upload Documents",
            command=self._on_upload_click,
            corner_radius=14,
            fg_color=ACCENT_COLOR,
            hover_color="#0ea5e9",
        )
        upload_button.grid(row=0, column=0, padx=(16, 8), pady=16)

        remove_button = ctk.CTkButton(
            panel,
            text="🗑️ Remove Selected",
            command=self._on_remove_click,
            fg_color="#ef4444",
            hover_color="#b91c1c",
            corner_radius=14,
        )
        remove_button.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Custom.Treeview",
            background="#0f172a",
            fieldbackground="#0f172a",
            foreground="#e2e8f0",
            rowheight=28,
            borderwidth=0,
        )
        style.configure("Custom.Treeview.Heading", background="#1e293b", foreground="#38bdf8", font=("Helvetica", 12, "bold"))
        style.map("Custom.Treeview", background=[("selected", "#1d4ed8")])

        tree_frame = ctk.CTkFrame(self.dashboard_tab, fg_color="#0f172a")
        tree_frame.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="nsew")
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)

        columns = ("name", "size", "uploaded")
        self.tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="browse",
            height=12,
            style="Custom.Treeview",
        )
        self.tree.heading("name", text="Document")
        self.tree.heading("size", text="Size (KB)")
        self.tree.heading("uploaded", text="Uploaded")
        self.tree.column("name", minwidth=200, width=360, anchor="w")
        self.tree.column("size", width=100, anchor="center")
        self.tree.column("uploaded", width=170, anchor="center")

        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")

    def _build_search_tab(self) -> None:
        self.search_tab.grid_rowconfigure(2, weight=1)
        self.search_tab.grid_columnconfigure((0, 1), weight=1)

        query_frame = ctk.CTkFrame(self.search_tab, fg_color="#172554")
        query_frame.grid(row=0, column=0, columnspan=2, padx=18, pady=(18, 12), sticky="ew")
        query_frame.grid_columnconfigure(0, weight=1)

        self.query_entry = ctk.CTkEntry(
            query_frame,
            placeholder_text="Ask a question or search your knowledge base…",
            height=40,
        )
        self.query_entry.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="ew")

        search_button = ctk.CTkButton(
            query_frame,
            text="🔍 Search",
            command=self._on_search,
            corner_radius=14,
            fg_color="#22c55e",
            hover_color="#16a34a",
        )
        search_button.grid(row=0, column=1, padx=4, pady=16)

        ask_button = ctk.CTkButton(
            query_frame,
            text="🤖 Ask AI",
            command=self._on_ask_ai,
            corner_radius=14,
            fg_color=ACCENT_COLOR,
            hover_color="#0ea5e9",
        )
        ask_button.grid(row=0, column=2, padx=(4, 16), pady=16)

        result_frame = ctk.CTkFrame(self.search_tab, fg_color="#0f172a")
        result_frame.grid(row=1, column=0, columnspan=2, padx=18, pady=(0, 18), sticky="nsew")
        result_frame.grid_rowconfigure(0, weight=1)
        result_frame.grid_columnconfigure(0, weight=1)

        self.results_text = ctk.CTkTextbox(result_frame, wrap="word", font=ctk.CTkFont(size=13), fg_color="#0f172a")
        self.results_text.grid(row=0, column=0, padx=16, pady=16, sticky="nsew")
        self.results_text.configure(state="disabled")

    def _build_settings_tab(self) -> None:
        self.settings_tab.grid_columnconfigure(0, weight=1)

        model_frame = ctk.CTkFrame(self.settings_tab, fg_color="#172554")
        model_frame.grid(row=0, column=0, padx=18, pady=(18, 12), sticky="ew")
        model_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(model_frame, text="Embeddings Provider", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=(16, 8), pady=16, sticky="w"
        )
        self.model_option = ctk.CTkSegmentedButton(
            model_frame,
            values=["SentenceTransformers", "OpenAI"],
            command=self._on_model_change,
        )
        self.model_option.set(self._current_embedding_backend)
        self.model_option.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="ew")

        api_frame = ctk.CTkFrame(self.settings_tab, fg_color="#172554")
        api_frame.grid(row=1, column=0, padx=18, pady=12, sticky="ew")
        api_frame.grid_columnconfigure(1, weight=1)

        self.openai_entry = ctk.CTkEntry(api_frame, placeholder_text="OpenAI API Key", show="*")
        self.hf_entry = ctk.CTkEntry(api_frame, placeholder_text="HuggingFace Token", show="*")
        openai_value = os.environ.get("OPENAI_API_KEY", "")
        hf_value = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")
        if openai_value:
            self.openai_entry.insert(0, openai_value)
        if hf_value:
            self.hf_entry.insert(0, hf_value)

        ctk.CTkLabel(api_frame, text="🔑 OpenAI API Key", text_color="#e2e8f0").grid(
            row=0, column=0, padx=(16, 8), pady=(18, 6), sticky="w"
        )
        self.openai_entry.grid(row=0, column=1, padx=(8, 16), pady=(18, 6), sticky="ew")
        ctk.CTkLabel(api_frame, text="🤝 HuggingFace Token", text_color="#e2e8f0").grid(
            row=1, column=0, padx=(16, 8), pady=6, sticky="w"
        )
        self.hf_entry.grid(row=1, column=1, padx=(8, 16), pady=6, sticky="ew")

        save_button = ctk.CTkButton(
            api_frame,
            text="💾 Save Credentials",
            command=self._on_save_credentials,
            fg_color=ACCENT_COLOR,
            hover_color="#0ea5e9",
            corner_radius=16,
        )
        save_button.grid(row=2, column=0, columnspan=2, padx=16, pady=(14, 18), sticky="ew")

        export_frame = ctk.CTkFrame(self.settings_tab, fg_color="#172554")
        export_frame.grid(row=2, column=0, padx=18, pady=(0, 18), sticky="ew")
        export_frame.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            export_frame,
            text="Export last AI summary",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, columnspan=3, padx=16, pady=(18, 4))

        ctk.CTkButton(export_frame, text="📝 Markdown", command=self._export_markdown).grid(
            row=1, column=0, padx=12, pady=12, sticky="ew"
        )
        ctk.CTkButton(export_frame, text="📄 PDF", command=self._export_pdf).grid(
            row=1, column=1, padx=12, pady=12, sticky="ew"
        )
        ctk.CTkButton(export_frame, text="🗄️ JSON", command=self._export_json).grid(
            row=1, column=2, padx=12, pady=12, sticky="ew"
        )

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------
    def _populate_documents(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        documents = self.app_state.refresh_documents()
        for doc in documents:
            path = Path(doc.record.path)
            size_kb = path.stat().st_size / 1024 if path.exists() else 0
            uploaded = datetime.datetime.fromtimestamp(path.stat().st_mtime) if path.exists() else datetime.datetime.now()
            self.tree.insert(
                "",
                "end",
                iid=doc.record.id,
                values=(path.name, f"{size_kb:.1f}", uploaded.strftime("%Y-%m-%d %H:%M")),
            )

    def _on_upload_click(self) -> None:
        file_paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[("Documents", "*.pdf *.txt *.md")],
        )
        if not file_paths:
            return

        def task() -> None:
            try:
                self._set_progress(0.15)
                for idx, file_path in enumerate(file_paths, start=1):
                    if Path(file_path).suffix.lower() not in SUPPORTED_EXTENSIONS:
                        self._show_error(f"Unsupported file type: {file_path}")
                        continue
                    self.app_state.ingest_file(file_path)
                    self._set_progress(0.15 + (idx / max(len(file_paths), 1)) * 0.75)
                self._populate_documents()
                self._set_status("Upload completed")
            except Exception as exc:  # pragma: no cover - GUI safeguard
                self._show_error(str(exc))
            finally:
                self._set_status("Ready")
                self._set_progress(0.0)

        self._set_status("Uploading…")
        self._set_progress(0.05)
        threading.Thread(target=task, daemon=True).start()

    def _on_remove_click(self) -> None:
        selection = self.tree.selection()
        if not selection:
            self._show_error("Select a document to remove.")
            return
        doc_id = selection[0]
        confirm = messagebox.askyesno("Confirm", "Remove the selected document? This cannot be undone.")
        if not confirm:
            return

        def task() -> None:
            try:
                if isinstance(self.app_state.metadata_store, SQLiteMetadataStore):
                    conn = self.app_state.metadata_store._conn  # type: ignore[attr-defined]
                    cur = conn.cursor()
                    cur.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                    cur.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    conn.commit()
                if doc_id in self.app_state.documents:
                    del self.app_state.documents[doc_id]
                self._populate_documents()
                self._set_status("Document removed")
            except Exception as exc:
                self._show_error(f"Unable to remove document: {exc}")
            finally:
                self._set_status("Ready")
                self._set_progress(0.0)

        self._set_status("Removing…")
        self._set_progress(0.2)
        threading.Thread(target=task, daemon=True).start()

    # ------------------------------------------------------------------
    # Search & Chat
    # ------------------------------------------------------------------
    def _on_search(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            self._show_error("Enter a query to search.")
            return

        def task() -> None:
            try:
                response = self.app_state.chat_engine.ask(query, top_k=5)
                self._display_response(response, mode="Semantic Search")
                self._set_status("Search complete")
            except Exception as exc:  # pragma: no cover
                self._show_error(str(exc))
            finally:
                self._set_status("Ready")
                self._set_progress(0.0)

        self._set_status("Searching…")
        self._set_progress(0.25)
        threading.Thread(target=task, daemon=True).start()

    def _on_ask_ai(self) -> None:
        query = self.query_entry.get().strip()
        if not query:
            self._show_error("Enter a question before asking the AI.")
            return

        def task() -> None:
            try:
                response = self.app_state.chat(query)
                self._last_response = response
                self._display_response(response, mode="AI Answer")
                self._set_status("AI responded")
            except Exception as exc:  # pragma: no cover
                self._show_error(str(exc))
            finally:
                self._set_status("Ready")
                self._set_progress(0.0)

        self._set_status("Contacting AI…")
        self._set_progress(0.35)
        threading.Thread(target=task, daemon=True).start()

    def _display_response(self, response: ChatResponse, *, mode: str) -> None:
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", "end")
        header = f"=== {mode} ===\n\n"
        self.results_text.insert("end", header)
        self.results_text.insert("end", response.answer + "\n\n", "answer")
        self.results_text.insert("end", "Context\n", "subtitle")
        for idx, chunk in enumerate(response.chunks, start=1):
            meta = chunk.metadata
            block = (
                f"[{idx}] Score: {chunk.score:.3f} | Path: {meta.get('path', 'unknown')} | Position: {meta.get('position')}\n"
                f"{chunk.text}\n\n"
            )
            self.results_text.insert("end", block, "context")
        self.results_text.tag_config("answer", font=ctk.CTkFont(size=14, weight="bold"))
        self.results_text.tag_config("subtitle", foreground=ACCENT_COLOR)
        self.results_text.tag_config("context", foreground="#e2e8f0")
        self.results_text.configure(state="disabled")
        self.results_text.see("end")

    # ------------------------------------------------------------------
    # Settings & exports
    # ------------------------------------------------------------------
    def _on_model_change(self, choice: str) -> None:
        self._current_embedding_backend = choice
        self._set_status(f"Embedding backend set to {choice}.")

    def _on_save_credentials(self) -> None:
        openai_key = self.openai_entry.get().strip()
        hf_key = self.hf_entry.get().strip()
        env_path = Path(".env")

        if openai_key:
            set_key(env_path, "OPENAI_API_KEY", openai_key)
        if hf_key:
            set_key(env_path, "HUGGINGFACEHUB_API_TOKEN", hf_key)

        self._set_status("Credentials saved locally.")
        messagebox.showinfo("Saved", "API credentials stored in .env file.")

    def _export_markdown(self) -> None:
        if not self._require_response():
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".md", filetypes=[("Markdown", "*.md")])
        if not file_path:
            return
        try:
            markdown = export_summary_to_markdown(self._last_response)
            Path(file_path).write_text(markdown, encoding="utf-8")
            self._set_status("Markdown export complete")
        except Exception as exc:  # pragma: no cover
            self._show_error(str(exc))

    def _export_pdf(self) -> None:
        if not self._require_response():
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF", "*.pdf")])
        if not file_path:
            return
        try:
            export_summary_to_pdf(self._last_response, file_path)
            self._set_status("PDF export complete")
        except ExportError as exc:
            self._show_error(str(exc))
        except Exception as exc:  # pragma: no cover
            self._show_error(str(exc))

    def _export_json(self) -> None:
        if not self._require_response():
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not file_path:
            return
        try:
            export_context_to_json(self._last_response, file_path)
            self._set_status("JSON export complete")
        except Exception as exc:  # pragma: no cover
            self._show_error(str(exc))

    def _require_response(self) -> bool:
        if self._last_response is None:
            self._show_error("Ask the AI first to generate a summary.")
            return False
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _set_status(self, message: str) -> None:
        self.status_var.set(message)
        self.update_idletasks()

    def _set_progress(self, value: float) -> None:
        self.progress_var.set(max(0.0, min(1.0, value)))
        self.update_idletasks()

    @staticmethod
    def _show_error(message: str) -> None:
        messagebox.showerror("Error", message)


if __name__ == "__main__":
    app = AIKnowledgeManagerApp()
    app.mainloop()
