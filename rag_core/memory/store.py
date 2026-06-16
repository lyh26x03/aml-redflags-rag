"""Thread-safe, in-process registry of per-session conversation memory.

Local and in-process by design (constraints: no Redis/SQL/vector DB/external
memory service). FastAPI runs sync endpoints in a worker threadpool, so the
store is guarded by a lock. The number of retained sessions is bounded; the
least-recently-updated session is evicted first so a long-running demo cannot
grow without limit.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional

from rag_core.memory.state import ConversationMemory

DEFAULT_MAX_SESSIONS = 256


class ConversationMemoryStore:
    def __init__(self, max_sessions: int = DEFAULT_MAX_SESSIONS):
        self._sessions: Dict[str, ConversationMemory] = {}
        self._lock = threading.Lock()
        self._max_sessions = max(1, max_sessions)

    def get(self, session_id: str) -> Optional[ConversationMemory]:
        with self._lock:
            return self._sessions.get(session_id)

    def exists(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._sessions

    def get_or_create(self, session_id: str) -> ConversationMemory:
        with self._lock:
            memory = self._sessions.get(session_id)
            if memory is None:
                memory = ConversationMemory(session_id=session_id)
                self._sessions[session_id] = memory
                self._evict_if_needed()
            return memory

    def reset(self, session_id: str) -> bool:
        """Delete a session's memory. Returns True if something was removed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def snapshot(self, session_id: str) -> Optional[dict]:
        with self._lock:
            memory = self._sessions.get(session_id)
            return memory.to_dict() if memory is not None else None

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    # --- internal (callers already hold the lock) ---

    def _evict_if_needed(self) -> None:
        while len(self._sessions) > self._max_sessions:
            oldest_id = min(
                self._sessions,
                key=lambda sid: self._sessions[sid].updated_at,
            )
            self._sessions.pop(oldest_id, None)
