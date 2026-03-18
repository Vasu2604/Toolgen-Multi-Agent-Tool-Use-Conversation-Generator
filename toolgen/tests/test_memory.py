"""
test_memory.py — Unit tests for MemoryStore

REQUIRED spec tests:
  1. add() followed by search() returns the stored entry
  2. Entries in one scope are NOT returned when querying another scope

Works with: python -m unittest tests/test_memory.py
        and: pytest tests/test_memory.py
"""

import json, os, sys, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from toolgen.memory import MemoryStore, SimpleMemoryFallback


class TestSimpleMemoryFallback(unittest.TestCase):
    """Tests for the pure-Python fallback store (no Ollama needed)."""

    def setUp(self):
        self.store = SimpleMemoryFallback()

    # ── Required spec test 1 ──────────────────────────────────
    def test_add_then_search_returns_entry(self):
        """add() followed by search() must return the stored entry."""
        self.store.add(
            content="Weather in Paris is sunny, 22 degrees",
            scope="session",
            metadata={"step": 1},
        )
        results = self.store.search(query="Paris weather", scope="session")
        self.assertGreater(len(results), 0)
        self.assertTrue(any("Paris" in r["content"] for r in results))

    # ── Required spec test 2 ──────────────────────────────────
    def test_scope_isolation_session_vs_corpus(self):
        """Entries added to 'session' must NOT appear in 'corpus' results, and vice versa."""
        self.store.add(
            content="session data: hotel_id=abc123",
            scope="session",
            metadata={"conversation_id": "c1"},
        )
        self.store.add(
            content="corpus data: travel planning conversation",
            scope="corpus",
            metadata={"conversation_id": "c1"},
        )
        session_results = self.store.search(query="data", scope="session")
        corpus_results  = self.store.search(query="data", scope="corpus")

        # Every session result must have scope == "session"
        for r in session_results:
            self.assertEqual(r["scope"], "session",
                             f"corpus entry leaked into session results: {r}")
        # Every corpus result must have scope == "corpus"
        for r in corpus_results:
            self.assertEqual(r["scope"], "corpus",
                             f"session entry leaked into corpus results: {r}")

        # Confirm corpus content not in session results
        self.assertFalse(any("corpus" in r["content"] for r in session_results))
        # Confirm session content not in corpus results
        self.assertFalse(any("session" in r["content"] for r in corpus_results))

    def test_top_k_respected(self):
        for i in range(10):
            self.store.add(f"entry {i} about search topic", "session", {})
        results = self.store.search("search topic", "session", top_k=3)
        self.assertLessEqual(len(results), 3)

    def test_empty_store_returns_empty_list(self):
        results = self.store.search("anything", "session")
        self.assertEqual(results, [])

    def test_multiple_entries_coexist(self):
        for i in range(5):
            self.store.add(f"Flight result {i}: airline=AA price={100+i*10}",
                           "session", {"step": i})
        results = self.store.search("flight airline", "session", top_k=5)
        self.assertGreater(len(results), 0)

    def test_different_scopes_independent(self):
        """Adding to one scope does not affect the other scope's result count."""
        self.store.add("only in session", "session", {})
        corpus_results = self.store.search("only in session", "corpus")
        self.assertEqual(len(corpus_results), 0)


class TestMemoryStore(unittest.TestCase):
    """Tests for the main MemoryStore interface (use_fallback=True)."""

    def setUp(self):
        self.store = MemoryStore(use_fallback=True)

    # ── Required spec test 1 ──────────────────────────────────
    def test_add_then_search_returns_stored_entry(self):
        """The primary contract: add() then search() returns the entry."""
        self.store.add(
            content="Tool output: 3 hotels in Paris, cheapest Hotel Lumiere $89/night",
            scope="session",
            metadata={"conversation_id": "conv_t1", "step": 1,
                      "endpoint": "hotel_booking::search_hotels"},
        )
        results = self.store.search(query="hotels Paris", scope="session")
        self.assertGreater(len(results), 0)
        contents = [r["content"] for r in results]
        self.assertTrue(any("Paris" in c or "hotel" in c.lower() for c in contents))

    # ── Required spec test 2 ──────────────────────────────────
    def test_scope_isolation(self):
        """Entries in scope A must not appear when querying scope B."""
        self.store.add(
            content="session entry: flight_id=FL123 departure=JFK arrival=CDG",
            scope="session",
            metadata={"step": 1},
        )
        self.store.add(
            content="corpus entry: travel planning with flight and hotel tools",
            scope="corpus",
            metadata={"conversation_id": "c1"},
        )
        session_results = self.store.search("flight hotel travel", "session")
        corpus_results  = self.store.search("flight hotel travel", "corpus")

        for r in session_results:
            self.assertEqual(r["scope"], "session",
                             f"Wrong scope in session results: {r}")
        for r in corpus_results:
            self.assertEqual(r["scope"], "corpus",
                             f"Wrong scope in corpus results: {r}")

    def test_session_memory_write_format(self):
        """Verify the exact write format used in the generator works correctly."""
        tool_output = {"hotel_id": "H456", "name": "Grand Hotel", "price_per_night": 150}
        self.store.add(
            content=json.dumps({"endpoint": "hotel_booking::search_hotels",
                                "output": tool_output}),
            scope="session",
            metadata={"conversation_id": "conv_42", "step": 0,
                      "endpoint": "hotel_booking::search_hotels"},
        )
        results = self.store.search("hotel search results", "session")
        self.assertGreater(len(results), 0)

    def test_corpus_memory_write_format(self):
        """Verify the exact corpus write format used in the generator works."""
        summary = ("Tools: weather_api, flight_search. "
                   "Domain: travel. Pattern: sequential.")
        self.store.add(
            content=summary,
            scope="corpus",
            metadata={"conversation_id": "conv_1",
                      "tools": ["weather_api", "flight_search"],
                      "pattern_type": "sequential"},
        )
        results = self.store.search("travel weather flight", "corpus")
        self.assertGreater(len(results), 0)

    def test_metadata_preserved_in_results(self):
        """Metadata passed to add() must be present in search results."""
        self.store.add("some content", "session",
                       {"conversation_id": "conv99", "step": 5})
        results = self.store.search("some content", "session")
        self.assertGreater(len(results), 0)
        self.assertIn("metadata", results[0])

    def test_multiple_scopes_independent(self):
        """Operations on one scope do not affect result counts from another scope."""
        for i in range(5):
            self.store.add(f"session item {i}", "session", {})
        corpus_results = self.store.search("session item", "corpus")
        self.assertEqual(len(corpus_results), 0)


if __name__ == "__main__":
    unittest.main()
