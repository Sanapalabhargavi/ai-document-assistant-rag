"""
database.py — Supabase (free tier) persistence layer.

Tables required (run the SQL in supabase_setup.sql to create them):
  chats       — one row per chat session
  messages    — one row per message, linked to a chat
  documents   — one row per uploaded document, linked to a chat
"""

import os
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _client


# ─── Chats ────────────────────────────────────────────────────────────────────

def create_chat(chat_id: str, title: str = "New Chat") -> dict:
    """Insert a new chat session row."""
    row = {
        "id": chat_id,
        "title": title,
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
    }
    get_client().table("chats").insert(row).execute()
    return row


def update_chat_title(chat_id: str, title: str) -> None:
    """Update the chat title (first user message truncated to 50 chars)."""
    get_client().table("chats").update({
        "title": title,
        "updated_at": datetime.utcnow().isoformat(),
    }).eq("id", chat_id).execute()


def get_all_chats() -> list[dict]:
    """Return all chats ordered by most-recently updated, max 50."""
    res = (
        get_client()
        .table("chats")
        .select("id, title, created_at, updated_at")
        .order("updated_at", desc=True)
        .limit(50)
        .execute()
    )
    return res.data or []


def delete_all_chats() -> None:
    """Delete every chat (cascades to messages + documents)."""
    # Delete messages first (FK constraint), then chats
    get_client().table("messages").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    get_client().table("documents").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
    get_client().table("chats").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()


# ─── Messages ─────────────────────────────────────────────────────────────────

def save_message(chat_id: str, role: str, content: str) -> None:
    """Append a message to a chat."""
    get_client().table("messages").insert({
        "chat_id": chat_id,
        "role": role,
        "content": content,
        "created_at": datetime.utcnow().isoformat(),
    }).execute()
    # bump chat updated_at
    get_client().table("chats").update({
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", chat_id).execute()


def get_messages(chat_id: str) -> list[dict]:
    """Return all messages for a chat ordered oldest-first."""
    res = (
        get_client()
        .table("messages")
        .select("role, content, created_at")
        .eq("chat_id", chat_id)
        .order("created_at")
        .execute()
    )
    return res.data or []


# ─── Documents ────────────────────────────────────────────────────────────────

def save_document_meta(chat_id: str, filename: str) -> None:
    """Record that a PDF was uploaded for this chat."""
    get_client().table("documents").insert({
        "chat_id": chat_id,
        "filename": filename,
        "uploaded_at": datetime.utcnow().isoformat(),
    }).execute()


def get_document_meta(chat_id: str) -> list[dict]:
    """Return uploaded documents for a chat."""
    res = (
        get_client()
        .table("documents")
        .select("filename, uploaded_at")
        .eq("chat_id", chat_id)
        .order("uploaded_at")
        .execute()
    )
    return res.data or []
