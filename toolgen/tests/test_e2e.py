"""
test_e2e.py — End-to-end test

Builds registry + graph, generates ≥50 conversations using a MockLLM
(no Ollama required), and verifies the full output format.
"""

import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from toolgen.registry import ToolRegistry
from toolgen.graph import ToolGraph, ToolChainSampler
from toolgen.memory import MemoryStore
from toolgen.generator import ConversationGenerator, serialize_conversation
from toolgen.agents import ValidatorAgent

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sample_tools")


# ── Mock LLM — no Ollama needed ───────────────────────────────

class MockLLM:
    """Deterministic fake LLM for testing."""

    def generate(self, prompt, system=""):
        p = prompt.lower()
        if "next tool to call" in p:          return "NONE"
        if "first message" in p:              return "Hi, I need help planning a trip to Paris."
        if "respond" in p or "follow" in p:   return "That sounds great, thank you!"
        return "I found great options for your Paris trip!"

    def generate_json(self, prompt, system=""):
        p = prompt.lower()
        if "user_goal" in p and "domain" in p:
            return {"user_goal": "Plan a Paris trip", "user_persona": "A traveler",
                    "context": "Needs flights and hotels.", "ambiguities": ["dates"],
                    "domain": "travel"}
        # Covers ALL required params across all 8 sample tools
        if "fill in the arguments" in p or "schema" in p:
            return {
                "city": "Paris", "country_code": "FR",
                "origin": "JFK", "destination": "CDG",
                "departure_date": "2025-06-01", "return_date": "2025-06-08",
                "check_in": "2025-06-01", "check_out": "2025-06-07",
                "query": "restaurants", "location": "Paris",
                "from_currency": "USD", "to_currency": "EUR", "amount": 100,
                "date": "2025-06-01", "start_time": "10:00", "end_time": "11:00",
                "title": "Paris Trip", "party_size": 2, "time": "19:00",
                "restaurant_id": "R001", "hotel_id": "H001", "flight_id": "F001",
                "result_id": "R001", "limit": 10, "passengers": 1,
            }
        return {
            "status": "success", "hotel_id": "H_456", "flight_id": "F_789",
            "restaurant_id": "R_321", "name": "Mock Result", "price": 150.0,
            "price_per_night": 120.0, "rating": 4.5, "stars": 4, "available": True,
            "temperature": 22, "condition": "sunny", "humidity": 60, "wind_speed": 10,
            "feels_like": 20, "airline": "Air France", "duration": "7h30m", "stops": 0,
            "rate": 0.92, "converted_amount": 92.0, "address": "1 Rue de Paris",
            "distance_km": 1.5, "from_currency": "USD", "to_currency": "EUR",
        }


# ── Shared setup (runs once) ──────────────────────────────────

_registry = None
_graph = None
_sampler = None

def get_components():
    global _registry, _graph, _sampler
    if _registry is None:
        _registry = ToolRegistry()
        _registry.load_from_directory(TOOLS_DIR)
        _graph   = ToolGraph(_registry)
        _sampler = ToolChainSampler(graph=_graph, registry=_registry, seed=42)
    return _registry, _graph, _sampler


def make_generator(corpus_memory=True, seed=42):
    reg, graph, sampler = get_components()
    llm    = MockLLM()
    memory = MemoryStore(use_fallback=True)
    gen    = ConversationGenerator(
        registry=reg, sampler=sampler, memory=memory, llm=llm,
        corpus_memory_enabled=corpus_memory, seed=seed,
    )
    gen.planner_agent.llm = llm
    gen.user_proxy.llm    = llm
    gen.assistant.llm     = llm
    gen.executor.llm      = llm
    return gen


# ── Build tests ───────────────────────────────────────────────

class TestBuildArtifacts(unittest.TestCase):

    def test_registry_loads_at_least_3_tools(self):
        reg, _, _ = get_components()
        self.assertGreaterEqual(len(reg.get_all_tools()), 3)

    def test_registry_has_endpoints(self):
        reg, _, _ = get_components()
        self.assertGreaterEqual(len(reg.get_all_endpoints()), 5)

    def test_graph_has_nodes(self):
        _, graph, _ = get_components()
        self.assertGreater(graph.graph.number_of_nodes(), 0)

    def test_graph_has_edges(self):
        _, graph, _ = get_components()
        self.assertGreater(graph.graph.number_of_edges(), 0)

    def test_sampler_sequential_chain(self):
        _, _, sampler = get_components()
        chain = sampler.sample_chain("sequential")
        self.assertGreaterEqual(len(chain), 2)

    def test_sampler_parallel_chain(self):
        _, _, sampler = get_components()
        chain = sampler.sample_chain("parallel")
        self.assertGreaterEqual(len(chain), 2)

    def test_sampler_mixed_chain(self):
        _, _, sampler = get_components()
        chain = sampler.sample_chain("mixed")
        self.assertGreaterEqual(len(chain), 2)

    def test_sampler_chain_endpoints_valid(self):
        reg, _, sampler = get_components()
        chain = sampler.sample_chain()
        for ep_id in chain:
            self.assertIsNotNone(reg.get_endpoint(ep_id),
                                 f"Invalid endpoint_id in chain: {ep_id}")

    def test_registry_save(self):
        reg, _, _ = get_components()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            reg.save(path)
            data = json.load(open(path))
            self.assertGreater(len(data), 0)
        finally:
            os.unlink(path)


# ── Single conversation tests ─────────────────────────────────

class TestSingleConversation(unittest.TestCase):

    def setUp(self):
        self.gen = make_generator()

    def test_generate_one_returns_conversation(self):
        conv = self.gen.generate_one(conv_id="tc_001")
        self.assertIsNotNone(conv)
        self.assertEqual(conv.id, "tc_001")

    def test_conversation_has_messages(self):
        conv = self.gen.generate_one()
        self.assertGreater(len(conv.messages), 0)

    def test_conversation_has_tool_calls(self):
        conv = self.gen.generate_one()
        self.assertGreater(len(conv.tool_calls), 0)

    def test_tool_calls_have_required_fields(self):
        conv = self.gen.generate_one()
        for tc in conv.tool_calls:
            self.assertIn("endpoint",  tc, "tool_call missing 'endpoint'")
            self.assertIn("arguments", tc, "tool_call missing 'arguments'")
            self.assertIn("output",    tc, "tool_call missing 'output'")
            self.assertIn("tool_name", tc, "tool_call missing 'tool_name'")

    def test_messages_have_valid_roles(self):
        conv = self.gen.generate_one()
        valid_roles = {"user", "assistant", "tool"}
        for m in conv.messages:
            self.assertIn(m.role, valid_roles, f"Invalid role: {m.role}")

    def test_serialize_has_all_metadata_fields(self):
        conv   = self.gen.generate_one()
        record = serialize_conversation(conv)
        self.assertIn("id",         record)
        self.assertIn("messages",   record)
        self.assertIn("tool_calls", record)
        self.assertIn("metadata",   record)
        required_meta = ["seed", "tool_ids_used", "num_turns",
                         "num_clarification_questions",
                         "memory_grounding_rate", "corpus_memory_enabled"]
        for field in required_meta:
            self.assertIn(field, record["metadata"],
                          f"Missing metadata field: {field}")

    def test_memory_grounding_rate_type(self):
        conv = self.gen.generate_one()
        mgr = conv.memory_grounding_rate
        self.assertTrue(mgr is None or isinstance(mgr, float),
                        f"Invalid memory_grounding_rate type: {type(mgr)}")


# ── End-to-end 50-sample test ─────────────────────────────────

class TestE2E50Samples(unittest.TestCase):
    """
    The primary E2E test: generate ≥50 valid conversations and
    verify the JSONL dataset meets all output requirements.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()
        cls.output_path = os.path.join(cls.tmpdir, "dataset.jsonl")

        gen = make_generator(corpus_memory=True, seed=42)
        cls.conversations = gen.generate_batch(50)

        with open(cls.output_path, "w") as f:
            for conv in cls.conversations:
                f.write(json.dumps(serialize_conversation(conv)) + "\n")

        with open(cls.output_path) as f:
            cls.records = [json.loads(line) for line in f if line.strip()]

    def test_generates_at_least_50_conversations(self):
        self.assertGreaterEqual(len(self.conversations), 50,
                                f"Expected ≥50 conversations, got {len(self.conversations)}")

    def test_jsonl_file_created(self):
        self.assertTrue(os.path.exists(self.output_path))

    def test_jsonl_has_at_least_50_records(self):
        self.assertGreaterEqual(len(self.records), 50)

    def test_all_records_valid_json(self):
        # Already parsed in setUpClass — if we got here, they're all valid JSON
        self.assertEqual(len(self.records), len(self.conversations))

    def test_all_records_have_required_metadata(self):
        required = ["seed", "tool_ids_used", "num_turns",
                    "num_clarification_questions", "memory_grounding_rate",
                    "corpus_memory_enabled"]
        for i, r in enumerate(self.records):
            meta = r.get("metadata", {})
            for field in required:
                self.assertIn(field, meta,
                              f"Record {i} missing metadata field: '{field}'")

    def test_majority_use_multiple_tools(self):
        multi = sum(
            1 for c in self.conversations
            if len({tc["tool_name"] for tc in c.tool_calls}) >= 2
        )
        pct = multi / len(self.conversations)
        self.assertGreaterEqual(pct, 0.5,
                                f"Only {multi}/{len(self.conversations)} use ≥2 distinct tools")

    def test_majority_have_3plus_tool_calls(self):
        three_plus = sum(1 for c in self.conversations if len(c.tool_calls) >= 3)
        pct = three_plus / len(self.conversations)
        self.assertGreaterEqual(pct, 0.5,
                                f"Only {three_plus}/{len(self.conversations)} have ≥3 tool calls")

    def test_memory_grounding_rate_valid_range(self):
        for conv in self.conversations:
            mgr = conv.memory_grounding_rate
            self.assertTrue(
                mgr is None or (isinstance(mgr, float) and 0.0 <= mgr <= 1.0),
                f"memory_grounding_rate out of range: {mgr}"
            )

    def test_pattern_types_diverse(self):
        """At least two distinct pattern types should appear across 50 conversations."""
        patterns = {c.pattern_type for c in self.conversations}
        self.assertGreaterEqual(len(patterns), 2,
                                f"Only one pattern type seen: {patterns}")

    def test_messages_not_empty(self):
        for i, r in enumerate(self.records):
            self.assertGreater(len(r.get("messages", [])), 0,
                               f"Record {i} has no messages")

    def test_tool_calls_not_empty(self):
        for i, r in enumerate(self.records):
            self.assertGreater(len(r.get("tool_calls", [])), 0,
                               f"Record {i} has no tool_calls")


if __name__ == "__main__":
    unittest.main()
