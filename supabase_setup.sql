-- ============================================================
-- Supabase Setup SQL
-- Run this in your Supabase project → SQL Editor
-- ============================================================

-- 1. chats table
CREATE TABLE IF NOT EXISTS chats (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. messages table
CREATE TABLE IF NOT EXISTS messages (
    id          BIGSERIAL PRIMARY KEY,
    chat_id     TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. documents table
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    chat_id      TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    filename     TEXT NOT NULL,
    uploaded_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_messages_chat_id  ON messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_documents_chat_id ON documents(chat_id);
CREATE INDEX IF NOT EXISTS idx_chats_updated_at  ON chats(updated_at DESC);

-- Row Level Security (optional but recommended for free tier)
ALTER TABLE chats     ENABLE ROW LEVEL SECURITY;
ALTER TABLE messages  ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;

-- Allow anon key full access (adjust if you add auth)
CREATE POLICY "allow_all_chats"     ON chats     FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_messages"  ON messages  FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_documents" ON documents FOR ALL USING (true) WITH CHECK (true);
