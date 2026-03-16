"""
rag_engine.py — RAG pipeline

Models used (all free, local):
  • Embeddings : sentence-transformers/all-MiniLM-L6-v2
  • Generation : google/flan-t5-large via AutoModelForSeq2SeqLM
                 (transformers 5.x removed text2text-generation pipeline task,
                  so we load the model directly instead)

Extra feature: keyword/topic extraction from the uploaded document.
"""

from __future__ import annotations

import re
import tempfile
from collections import Counter
from typing import Any

import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ─── In-memory store: one FAISS index per chat_id ─────────────────────────────
_vector_dbs: dict[str, Any] = {}
_doc_page_counts: dict[str, int] = {}

# ─── Load models once ─────────────────────────────────────────────────────────
_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

_MODEL_NAME = "google/flan-t5-large"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL_NAME)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = _model.to(_device)


def _generate(prompt: str, max_new_tokens: int = 300) -> str:
    """Run flan-t5-large inference and return the decoded answer string."""
    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(_device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True,
        )
    return _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ─── Document processing ──────────────────────────────────────────────────────

def process_document(file_bytes: bytes, chat_id: str) -> int:
    """
    Load a PDF, chunk it, embed it, and store the FAISS index for chat_id.
    Returns the total number of pages.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)

    _vector_dbs[chat_id] = FAISS.from_documents(chunks, _embeddings)
    page_count = len(docs)
    _doc_page_counts[chat_id] = page_count
    return page_count


def has_document(chat_id: str) -> bool:
    return chat_id in _vector_dbs


# ─── Q&A ──────────────────────────────────────────────────────────────────────

def ask_question(query: str, chat_id: str) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate an answer.

    Returns:
        formatted_answer : Markdown string with bullet points + sources
        sources          : list of {page, snippet} dicts
    """
    vdb = _vector_dbs.get(chat_id)
    if vdb is None:
        return (
            "⚠️ No document loaded for this chat. Please upload a PDF first.",
            [],
        )

    # Retrieve top-4 chunks
    result_docs = vdb.similarity_search(query, k=4)

    context = "\n\n".join([d.page_content for d in result_docs])

    prompt = (
        f"You are a helpful document assistant. "
        f"Answer the following question using ONLY the context provided. "
        f"Give a clear, detailed answer in bullet points.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer (in bullet points):"
    )

    raw = _generate(prompt)

    # ── Format bullet points ──────────────────────────────────────────────────
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    bullets: list[str] = []
    for line in lines:
        clean = re.sub(r"^[-•*]\s*", "", line).strip()
        if clean:
            bullets.append(f"- {clean}")

    # ── Deduplicate sources by page ───────────────────────────────────────────
    seen_pages: set = set()
    sources: list[dict] = []
    for doc in result_docs:
        page = doc.metadata.get("page", None)
        page_display = (page + 1) if isinstance(page, int) else "?"
        if page_display not in seen_pages:
            seen_pages.add(page_display)
            snippet = doc.page_content[:120].replace("\n", " ").strip()
            sources.append({"page": page_display, "snippet": snippet})

    # ── Build final Markdown ──────────────────────────────────────────────────
    answer_block = "\n".join(bullets) if bullets else f"- {raw}"

    source_lines = "\n".join(
        [f"- 📄 **Page {s['page']}** — {s['snippet']}..." for s in sources]
    )

    formatted = (
        f"## 🤖 Answer\n\n"
        f"{answer_block}\n\n"
        f"---\n\n"
        f"## 📌 Sources\n\n"
        f"{source_lines}"
    )

    return formatted, sources


# ─── EXTRA FEATURE: Document keyword / topic extraction ───────────────────────

def extract_keywords(chat_id: str, top_n: int = 10) -> str:
    """
    Extract the top N keywords/topics from the entire document.
    Returns a formatted Markdown string.
    """
    vdb = _vector_dbs.get(chat_id)
    if vdb is None:
        return "⚠️ No document loaded."

    # Pull all stored texts from the FAISS docstore
    all_text = " ".join(
        doc.page_content
        for doc in vdb.docstore._dict.values()  # type: ignore[attr-defined]
    )

    # Simple frequency-based keyword extraction (no extra deps)
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "is", "are", "was", "were", "be", "been",
        "by", "that", "this", "it", "as", "from", "its", "their", "they",
        "we", "our", "has", "have", "had", "not", "also", "which", "more",
        "can", "will", "may", "such", "these", "those", "into", "than",
        "i", "you", "he", "she", "us", "do", "does", "did", "so", "if",
        "about", "all", "each", "other", "over", "after", "figure", "table",
    }
    words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())
    filtered = [w for w in words if w not in stop_words]
    counts = Counter(filtered)
    top_keywords = counts.most_common(top_n)

    lines = [f"• **{kw}** ({freq} occurrences)" for kw, freq in top_keywords]

    return (
        f"### 🏷️ Top Keywords / Topics in Document\n\n"
        + "\n".join(lines)
    )


# ─── Document summary ─────────────────────────────────────────────────────────

def summarise_document(chat_id: str) -> str:
    """Generate a concise summary of the document."""
    return ask_question(
        "Give a comprehensive summary of this document. What is it about? "
        "What are the main topics, findings, or conclusions?",
        chat_id,
    )[0]