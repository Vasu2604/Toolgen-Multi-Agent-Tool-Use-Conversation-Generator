"""
memory.py — MemoryStore
=======================

Spec requirement:
    from mem0 import Memory
    m = Memory()
    Wrap inside MemoryStore so the rest of the codebase is decoupled from mem0.

This file contains:
  1. SimpleMemoryFallback  — pure Python, used in tests + when mem0 fails
  2. MemoryStore           — the ONLY class the rest of the codebase uses

Scopes:
  "session" — within one conversation (tool output grounding)
  "corpus"  — across all conversations (diversity / planner diversity)

mem0 version compatibility:
  mem0ai ≥ 0.1.29 + qdrant-client == 1.7.3 works perfectly.
  If there is a version mismatch, we fall back to SimpleMemoryFallback
  so generation never crashes. The fallback fully satisfies scope isolation.

  To fix version issues: pip install "mem0ai==0.1.29" "qdrant-client==1.7.3"
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── SimpleMemoryFallback ───────────────────────────────────────

class SimpleMemoryFallback:
    """
    Pure-Python keyword-overlap memory store.
    No external dependencies. Always works.
    Scope isolation guaranteed by filtering on the scope field.
    Used in tests (use_fallback=True) and as safety net when mem0 fails.
    """

    def __init__(self):
        self._store: list[dict] = []

    def add(self, content: str, scope: str, metadata: dict) -> None:
        self._store.append({"content": content, "scope": scope, "metadata": metadata})

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        scoped  = [e for e in self._store if e["scope"] == scope]
        q_words = set(query.lower().split())
        scored  = [(len(q_words & set(e["content"].lower().split())), e) for e in scoped]
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k]]


# ── MemoryStore ────────────────────────────────────────────────

class MemoryStore:
    """
    The ONLY memory class the rest of the codebase imports.

    Exposes exactly the two methods the spec requires:
        add(content, scope, metadata) -> None
        search(query, scope, top_k)   -> list[dict]

    Internally backed by mem0 (Memory()) exactly as the spec instructs.
    Falls back to SimpleMemoryFallback if mem0 is unavailable or broken.
    """

    def __init__(self, use_fallback: bool = False):
        """
        Args:
            use_fallback: Force SimpleMemoryFallback (used in tests).
                          Default False = attempt mem0 first.
        """
        self._mem0         = None
        self._fallback     = SimpleMemoryFallback()
        self._use_fallback = use_fallback

        if not use_fallback:
            self._init_mem0()   # always attempt mem0 when use_fallback=False
        else:
            print("[Memory] Using SimpleMemoryFallback (test mode)")

    def _init_mem0(self) -> None:
        """
        Initialise mem0 as the spec instructs:
            from mem0 import Memory
            m = Memory()

        Tries plain Memory() first (spec-compliant, uses Qdrant embedded).
        Falls back to explicit Ollama config for newer mem0 versions.
        Falls back to SimpleMemoryFallback if both fail — never crashes.
        """
        # ── Attempt 1: Plain Memory() — exactly what the spec says ──
        try:
            from mem0 import Memory          # spec line 1
            candidate = Memory()             # spec line 2
            # Smoke-test: catches silent Qdrant attribute errors immediately
            candidate.add(
                messages=[{"role": "user", "content": "smoke_test"}],
                user_id="__smoke__",
            )
            candidate.search(query="smoke_test", user_id="__smoke__")
            self._mem0 = candidate
            print("[Memory] mem0 initialized with Memory() — Qdrant embedded, no external service")
            return
        except ImportError:
            logger.warning(
                "mem0 not installed. Run: pip install mem0ai\n"
                "Using SimpleMemoryFallback."
            )
            self._use_fallback = True
            return
        except Exception as e1:
            logger.debug(f"mem0 plain Memory() failed: {e1}")

        # ── Attempt 2: Explicit Ollama config (newer mem0 versions) ──
        try:
            from mem0 import Memory
            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "llama3.2",
                        "ollama_base_url": "http://localhost:11434",
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text",
                        "ollama_base_url": "http://localhost:11434",
                        "embedding_model_dims": 768,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "toolgen_memory",
                        "embedding_model_dims": 768,
                        "on_disk": False,
                    },
                },
            }
            candidate = Memory.from_config(config)
            candidate.add(
                messages=[{"role": "user", "content": "smoke_test"}],
                user_id="__smoke__",
            )
            candidate.search(query="smoke_test", user_id="__smoke__")
            self._mem0 = candidate
            print("[Memory] mem0 initialized with Ollama config")
            return
        except Exception as e2:
            logger.debug(f"mem0 Ollama config also failed: {e2}")

        # ── Attempt 3: graceful fallback ──────────────────────
        logger.warning(
            "mem0 could not be initialized (likely qdrant-client version conflict). "
            "Fix: pip install 'mem0ai==0.1.29' 'qdrant-client==1.7.3' --force-reinstall\n"
            "Falling back to SimpleMemoryFallback — pipeline continues normally."
        )
        self._use_fallback = True

    # ── Public interface — the ONLY two methods other code uses ──

    def add(self, content: str, scope: str, metadata: dict) -> None:
        """
        Store a memory entry.
        scope = "session" (within one conversation) or
                "corpus"  (across all conversations).
        """
        self._fallback.add(content, scope, metadata)   # always write to fallback

        if self._mem0 is not None and not self._use_fallback:
            try:
                self._mem0.add(
                    messages=[{"role": "user", "content": content}],
                    user_id=scope,
                    metadata={**metadata, "scope": scope},
                )
            except Exception as e:
                logger.debug(f"mem0.add() failed: {e}")

    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve relevant entries for a query within a scope.
        Returns list of {"content", "scope", "metadata"} dicts.
        Spec: count a retrieval as present whenever search() returns results,
              regardless of score threshold.
        """
        if self._mem0 is not None and not self._use_fallback:
            try:
                results    = self._mem0.search(query=query, user_id=scope, limit=top_k)
                normalized = []
                for r in results:
                    content      = r.get("memory", r.get("content", str(r)))
                    meta         = r.get("metadata", {})
                    result_scope = meta.get("scope", scope)
                    if result_scope == scope:
                        normalized.append({"content": content,
                                           "scope": result_scope,
                                           "metadata": meta})
                if normalized:
                    return normalized[:top_k]
            except Exception as e:
                logger.debug(f"mem0.search() failed: {e}")

        return self._fallback.search(query, scope, top_k)
