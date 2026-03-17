"""
rag_engine.py - RAG pipeline

Models used (all free, local):
  - Embeddings : sentence-transformers/all-MiniLM-L6-v2
  - Generation : google/flan-t5-large via AutoModelForSeq2SeqLM

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

# --- In-memory store: one FAISS index per chat_id ---
_vector_dbs: dict[str, Any] = {}
_doc_page_counts: dict[str, int] = {}

# --- Load models once ---
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


# --- Document processing ---

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


# --- Q&A ---

def ask_question(query: str, chat_id: str) -> tuple[str, list[dict]]:
    """
    Retrieve relevant chunks and generate an answer.
    Returns formatted_answer and sources list.
    """
    vdb = _vector_dbs.get(chat_id)
    if vdb is None:
        return (
            "No document loaded for this chat. Please upload a PDF first.",
            [],
        )

    result_docs = vdb.similarity_search(query, k=4)
    context = "\n\n".join([d.page_content for d in result_docs])

    # Build a more specific prompt depending on the question type
    q_lower = query.lower()
    if any(w in q_lower for w in ["about", "overview", "describe", "what is this", "summarize", "summary"]):
        instruction = (
            "Read the context carefully and describe what this document is about. "
            "Mention the person's name, their qualification, skills, and experience. "
            "Give the answer in clear bullet points."
        )
    elif any(w in q_lower for w in ["skill", "skills", "technologies", "tools", "tech stack"]):
        instruction = (
            "List all the technical skills, programming languages, tools, and technologies "
            "mentioned in the context. Give the answer in bullet points."
        )
    elif any(w in q_lower for w in ["experience", "work", "job", "company", "project"]):
        instruction = (
            "List all work experience, projects, and companies mentioned in the context. "
            "Give the answer in bullet points."
        )
    elif any(w in q_lower for w in ["education", "degree", "college", "university", "qualification"]):
        instruction = (
            "List all educational qualifications, degrees, colleges and universities "
            "mentioned in the context. Give the answer in bullet points."
        )
    else:
        instruction = (
            "Answer the following question using ONLY the context provided. "
            "Give a clear, detailed answer in bullet points."
        )

    prompt = (
        "You are a helpful document assistant. "
        + instruction + "\n\n"
        "Context:\n" + context + "\n\n"
        "Question: " + query + "\n\n"
        "Answer (in bullet points):"
    )

    raw = _generate(prompt)

    # Format bullet points
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    bullets = []
    for line in lines:
        clean = re.sub(r"^[-*]\s*", "", line).strip()
        if clean:
            bullets.append("- " + clean)

    # Deduplicate sources by page
    seen_pages: set = set()
    sources: list[dict] = []
    for doc in result_docs:
        page = doc.metadata.get("page", None)
        page_display = (page + 1) if isinstance(page, int) else "?"
        if page_display not in seen_pages:
            seen_pages.add(page_display)
            snippet = doc.page_content[:120].replace("\n", " ").strip()
            sources.append({"page": page_display, "snippet": snippet})

    answer_block = "\n".join(bullets) if bullets else "- " + raw

    source_lines = "\n".join(
        ["- Page **" + str(s["page"]) + "** — " + s["snippet"] + "..." for s in sources]
    )

    formatted = (
        "## Answer\n\n"
        + answer_block
        + "\n\n---\n\n"
        "## Sources\n\n"
        + source_lines
    )

    return formatted, sources


# --- EXTRA FEATURE: Document keyword / topic extraction ---

def extract_keywords(chat_id: str, top_n: int = 10) -> str:
    """
    Extract the top N keywords/topics from the entire document.
    Returns a formatted Markdown string.
    """
    vdb = _vector_dbs.get(chat_id)
    if vdb is None:
        return "No document loaded."

    all_text = " ".join(
        doc.page_content
        for doc in vdb.docstore._dict.values()
    )

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

    lines = ["- **" + kw + "** (" + str(freq) + " occurrences)" for kw, freq in top_keywords]

    return "## Top Keywords / Topics in Document\n\n" + "\n".join(lines)


# --- Document summary (extractive - no LLM needed) ---

def summarise_document(chat_id: str) -> str:
    """
    Extractive summarization - scores every sentence by keyword frequency
    and picks the top sentences. Fast and accurate without needing the LLM.
    """
    vdb = _vector_dbs.get(chat_id)
    if vdb is None:
        return "No document loaded."

    all_docs = sorted(
        vdb.docstore._dict.values(),
        key=lambda d: d.metadata.get("page", 0)
    )
    all_text = " ".join(doc.page_content for doc in all_docs)
    page_count = _doc_page_counts.get(chat_id, len(all_docs))

    stop_words = {
        "the","a","an","and","or","but","in","on","at","to","for","of","with",
        "is","are","was","were","be","been","by","that","this","it","as","from",
        "its","their","they","we","our","has","have","had","not","also","which",
        "more","can","will","may","such","these","those","into","than","i","you",
        "he","she","us","do","does","did","so","if","about","all","each","other",
        "over","after","figure","table","page","section","chapter","using","used",
        "based","result","results","show","shows","however","therefore","thus",
        "would","could","should","been","have","very","just","like","well",
    }

    # Word frequency map
    words = re.findall(r"[a-zA-Z]{4,}", all_text.lower())
    filtered = [w for w in words if w not in stop_words]
    freq = Counter(filtered)
    max_freq = max(freq.values()) if freq else 1
    freq_norm = {w: v / max_freq for w, v in freq.items()}

    # Score every sentence
    sentences = re.split(r"(?<=[.!?])\s+", all_text)
    sentences = [s.strip() for s in sentences if 20 < len(s.strip()) < 300]

    scored = []
    for sent in sentences:
        words_in_sent = re.findall(r"[a-zA-Z]{4,}", sent.lower())
        score = sum(freq_norm.get(w, 0) for w in words_in_sent if w not in stop_words)
        score = score / (len(words_in_sent) + 1)
        scored.append((score, sent))

    scored.sort(reverse=True)

    # Pick top 8 unique sentences
    top_sents = []
    seen_lower = set()
    for score, sent in scored[:40]:
        key = sent.lower()[:60]
        if key not in seen_lower:
            seen_lower.add(key)
            top_sents.append(sent)
        if len(top_sents) == 8:
            break

    # Restore original document order
    sent_order = {s: i for i, s in enumerate(sentences)}
    top_sents.sort(key=lambda s: sent_order.get(s, 9999))

    # Top keywords for header
    top_topics = [w for w, _ in Counter(filtered).most_common(6)]
    topics_str = "  |  ".join(top_topics)

    bullet_points = "\n".join("- " + s for s in top_sents)

    return (
        "## Document Summary\n\n"
        "Pages: **" + str(page_count) + "**\n\n"
        "Key topics: " + topics_str + "\n\n"
        "---\n\n"
        "### Key Points\n\n"
        + bullet_points
    )