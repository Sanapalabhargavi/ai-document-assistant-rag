"""
app.py — AI Document Assistant (RAG-Based)

Features implemented:
  1.  New Chat button                          ✅
  2.  Chat history in left sidebar             ✅  (Chainlit built-in, Supabase-backed)
  3.  Upload icon beside prompt bar            ✅  (Chainlit multi-modal / file upload)
  4.  Home page with title + description       ✅  (chainlit.md + on_chat_start)
  5.  Chat initiators on home page             ✅  (cl.Starter)
  6.  Extra feature: keyword extraction        ✅  (🏷️ Extract Keywords starter)
  7.  Clear chat history action                ✅  (⚙️ action in every chat)
  8.  New Chat shows initiators again           ✅  (handled by on_chat_start)
  9.  Free real-time database (Supabase)       ✅  (database.py)
 10.  Resume existing chat from history        ✅  (on_chat_resume)
 11.  Formatted responses: bullets + sources   ✅  (rag_engine.py)
"""

import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import chainlit as cl

_executor = ThreadPoolExecutor(max_workers=1)

from database import (
    create_chat,
    get_messages,
    save_message,
    save_document_meta,
    get_document_meta,
    delete_all_chats,
    update_chat_title,
)
from rag_engine import (
    process_document,
    ask_question,
    has_document,
    extract_keywords,
    summarise_document,
)

# ─── Special sentinel queries ─────────────────────────────────────────────────
_STARTER_SUMMARISE  = "__SUMMARISE__"
_STARTER_KEYWORDS   = "__KEYWORDS__"
_STARTER_HOWTO      = "__HOWTO__"
_STARTER_FINDINFO   = "__FINDINFO__"


# ─── Helper ───────────────────────────────────────────────────────────────────

def _current_chat_id() -> str:
    return cl.user_session.get("chat_id")


async def _stream_and_save(text: str, chat_id: str) -> None:
    """Send a response as a full message so markdown renders correctly."""
    await cl.Message(content=text).send()
    save_message(chat_id, "assistant", text)


# ─── on_chat_start ────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Called for every brand-new chat."""
    chat_id = str(uuid.uuid4())
    cl.user_session.set("chat_id", chat_id)
    cl.user_session.set("doc_loaded", False)
    cl.user_session.set("title_set", False)

    # Persist the new chat row in Supabase
    create_chat(chat_id, "New Chat")

    # Welcome banner
    await cl.Message(
        content=(
            "## 📄 AI Document Assistant\n\n"
            "**RAG-Powered · Semantic Search · Sourced Answers**\n\n"
            "---\n\n"
            "👋 Welcome! Upload a **PDF** using the 📎 icon next to the message bar.\n\n"
            "Once uploaded, choose a quick-start below or type your own question."
        )
    ).send()


# ─── Chat initiators (shown on the welcome screen) ────────────────────────────

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="📝 Summarise the document",
            message=_STARTER_SUMMARISE,
            icon="/public/icons/summary.svg",
        ),
        cl.Starter(
            label="🏷️ Extract keywords & topics",
            message=_STARTER_KEYWORDS,
            icon="/public/icons/keywords.svg",
        ),
        cl.Starter(
            label="❓ What is this document about?",
            message="What is this document about? Give me an overview.",
            icon="/public/icons/question.svg",
        ),
        cl.Starter(
            label="🔍 Find key information",
            message="What are the most important findings, conclusions, or recommendations in this document?",
            icon="/public/icons/search.svg",
        ),
    ]


# ─── on_chat_resume — restore from Supabase ───────────────────────────────────

@cl.on_chat_resume
async def on_chat_resume(thread: dict):
    """
    Called when a user clicks a chat from the sidebar history.
    In Chainlit 2.x the argument is a plain dict, not cl.Thread.
    """
    chat_id = thread.get("id") or thread.get("thread_id", "")
    cl.user_session.set("chat_id", chat_id)
    cl.user_session.set("title_set", True)

    # Check if a document was uploaded in this chat
    docs = get_document_meta(chat_id)
    if docs:
        filenames = ", ".join(d["filename"] for d in docs)
        cl.user_session.set("doc_loaded", True)
        await cl.Message(
            content=(
                f"📂 **Chat resumed.**\n\n"
                f"Document{'s' if len(docs) > 1 else ''} previously uploaded: **{filenames}**\n\n"
                f"_Note: Please re-upload the PDF to continue asking questions "
                f"(the index is stored in memory and resets on server restart)._"
            )
        ).send()
    else:
        cl.user_session.set("doc_loaded", False)
        await cl.Message(
            content="📂 **Chat resumed.** No document was uploaded in this session."
        ).send()


# ─── File upload handler ──────────────────────────────────────────────────────


@cl.on_message
async def on_message(message: cl.Message):
    """Main message handler — also handles file uploads attached to messages."""
    chat_id = _current_chat_id()

    # ── Handle attached file uploads ─────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "mime") and "pdf" in (element.mime or ""):
                await _handle_pdf_upload(element, chat_id)
                return  # Don't treat the message text as a query after upload

    query = message.content.strip()
    if not query:
        return

    # ── Persist user message ─────────────────────────────────────────────────
    save_message(chat_id, "user", query)

    # Auto-title the chat from first user message
    if not cl.user_session.get("title_set"):
        title = query[:50] + ("…" if len(query) > 50 else "")
        update_chat_title(chat_id, title)
        cl.user_session.set("title_set", True)

    # ── Clear history action ──────────────────────────────────────────────────
    if query.lower() in ("clear history", "/clear"):
        await _clear_history(chat_id)
        return

    # ── Natural language trigger matching ─────────────────────────────────────
    q_lower = query.lower().strip()

    _SUMMARISE_TRIGGERS = [
        _STARTER_SUMMARISE, "summarise", "summarize", "summarise the document",
        "summarize the document", "summary", "give me a summary",
        "what is this document about", "overview"
    ]
    _KEYWORD_TRIGGERS = [
        _STARTER_KEYWORDS, "extract keywords", "keywords", "extract keywords & topics",
        "extract key words", "show keywords", "key topics", "main topics",
        "what are the keywords", "topics"
    ]

    if q_lower in [t.lower() for t in _SUMMARISE_TRIGGERS]:
        query = _STARTER_SUMMARISE
    elif q_lower in [t.lower() for t in _KEYWORD_TRIGGERS]:
        query = _STARTER_KEYWORDS

    # ── Starter: Summarise ────────────────────────────────────────────────────
    if query == _STARTER_SUMMARISE:
        if not has_document(chat_id):
            await _no_doc_warning()
            return
        thinking = await cl.Message(content="⏳ Generating summary… *(this may take 1-2 min on CPU)*").send()
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(_executor, summarise_document, chat_id)
        await thinking.remove()
        await _stream_and_save(answer, chat_id)
        return

    # ── Starter: Keywords ────────────────────────────────────────────────────
    _kw_triggers = ["keyword", "keywords", "topics", "extract keyword", "extract keywords", "key words", "key topics"]
    if query == _STARTER_KEYWORDS or any(t in query.lower() for t in _kw_triggers):
        if not has_document(chat_id):
            await _no_doc_warning()
            return
        thinking = await cl.Message(content="⏳ Extracting keywords…").send()
        loop = asyncio.get_event_loop()
        answer = await loop.run_in_executor(_executor, extract_keywords, chat_id)
        await thinking.remove()
        await _stream_and_save(answer, chat_id)
        return

    # ── Regular Q&A ──────────────────────────────────────────────────────────
    if not has_document(chat_id):
        await _no_doc_warning()
        return

    thinking = await cl.Message(content="⏳ Searching document… *(this may take 1-2 min on CPU)*").send()
    loop = asyncio.get_event_loop()
    answer, _ = await loop.run_in_executor(_executor, ask_question, query, chat_id)
    await thinking.remove()
    await _stream_and_save(answer, chat_id)


# ─── PDF processing helper ────────────────────────────────────────────────────

async def _handle_pdf_upload(element, chat_id: str):
    filename = getattr(element, "name", "document.pdf")

    processing_msg = await cl.Message(
        content=f"⏳ Processing **{filename}**… this may take a few seconds."
    ).send()

    try:
        # Chainlit 2.x saves uploads to a temp file; element.content is None
        file_path = getattr(element, "path", None)
        if file_path:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        elif getattr(element, "content", None) is not None:
            file_bytes = element.content
        else:
            raise ValueError("Cannot read uploaded file.")
        loop = asyncio.get_event_loop()
        page_count = await loop.run_in_executor(_executor, process_document, file_bytes, chat_id)
        cl.user_session.set("doc_loaded", True)

        # Persist metadata to Supabase
        save_document_meta(chat_id, filename)

        # Auto-title the chat if not set yet
        if not cl.user_session.get("title_set"):
            update_chat_title(chat_id, f"Doc: {filename[:44]}")
            cl.user_session.set("title_set", True)

        await processing_msg.remove()
        await cl.Message(
            content=(
                f"✅ **{filename}** uploaded successfully!\n\n"
                f"📊 **{page_count} pages** indexed and ready.\n\n"
                f"You can now ask questions, summarise, or extract keywords."
            )
        ).send()

    except Exception as exc:
        await processing_msg.remove()
        await cl.Message(
            content=f"❌ Error processing document: `{exc}`"
        ).send()


# ─── Action: Clear history ────────────────────────────────────────────────────

@cl.action_callback("clear_history")
async def on_clear_history(action: cl.Action):
    chat_id = _current_chat_id()
    await _clear_history(chat_id)


async def _clear_history(chat_id: str):
    try:
        delete_all_chats()
        await cl.Message(
            content=(
                "🗑️ **All chat history has been cleared.**\n\n"
                "Click **New Chat** in the sidebar to start fresh."
            )
        ).send()
    except Exception as exc:
        await cl.Message(
            content=f"❌ Could not clear history: `{exc}`"
        ).send()


# ─── No-document warning ─────────────────────────────────────────────────────

async def _no_doc_warning():
    await cl.Message(
        content=(
            "⚠️ **No document loaded.**\n\n"
            "Please upload a PDF using the 📎 icon next to the message bar first."
        )
    ).send()


# ─── Persistent action buttons shown in every reply ──────────────────────────

@cl.on_chat_end
async def on_chat_end():
    pass  # cleanup hook if needed


# ─── Show clear-history button after every assistant message ──────────────────
# We attach it as a persistent element via the sidebar action instead,
# but you can also add it inline:

async def _send_with_clear_action(content: str):
    actions = [
        cl.Action(
            name="clear_history",
            label="🗑️ Clear all history",
            payload={"action": "clear"},
        )
    ]
    await cl.Message(content=content, actions=actions).send()