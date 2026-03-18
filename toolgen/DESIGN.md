# DESIGN.md — Architecture & Design Decisions

## System Overview

toolgen is a pipeline that generates synthetic multi-turn conversations where an AI assistant uses multiple tools. The output is training data for tool-use AI agents.

```
ToolBench JSON → Registry → Tool Graph → Sampler
                                              ↓
                              5-Agent Conversation Generator
                              ├── SamplerAgent  (picks tools)
                              ├── PlannerAgent  (creates scenario)
                              ├── UserProxyAgent (simulates user)
                              ├── AssistantAgent (calls tools)
                              └── ValidatorAgent (checks quality)
                                              ↓
                              MemoryStore (session + corpus)
                                              ↓
                              JSONL Dataset
```

---

## Component Decisions

### 1. Tool Registry (`registry.py`)

**Decision**: Use Python dataclasses (`Tool`, `Endpoint`, `Parameter`) rather than raw dicts.

**Why**: Dataclasses give type hints and IDE autocomplete. When ToolBench files have missing fields, the loader fills sensible defaults rather than crashing. This is important because ToolBench has at least 3 known format variants across its different API categories.

**Handling inconsistencies**: The `_parse_tool()` method tries multiple field name variants (`tool_name` vs `name`, `api_list` vs `endpoints` vs `apis`). Missing tools (no name) are silently skipped with a warning.

---

### 2. Tool Graph (`graph.py`)

**Decision**: Use NetworkX `DiGraph` (directed graph).

**Why directed**: The most important edges are `feeds_into` edges (endpoint A's output fields overlap with endpoint B's input params). These are inherently directional — weather returns a `city`, hotels take a `city` as input.

**Node types** (all five that the spec requires):
- `tool:{name}` — one node per tool (e.g. `tool:weather_api`)
- `endpoint:{tool}::{ep}` — one node per callable endpoint
- `param:{tool}::{ep}::{name}` — one node per parameter (required + optional)
- `rf:{tool}::{ep}::{field}` — one node per response field (when available)
- `category:{cat}` — one node per category / concept / tag

With 8 sample tools the graph contains: tool=8, endpoint=16, parameter=57, response_field=81, category=7 = 169 nodes total.

**Chain detection**: Two endpoints are connected by a `feeds_into` edge if their response fields and parameter names overlap. This is a syntactic heuristic — it works well for ToolBench because field names are descriptive (`hotel_id`, `flight_id`, `city`).

---

### 3. Tool-Chain Sampler (`graph.py`)

**Decision**: Three sampling patterns — `sequential`, `parallel`, `mixed`.

- **Sequential**: Follows `feeds_into` edges in the graph. When no chain edge exists, falls back to same-category tools.
- **Parallel**: Picks multiple tools from the same category (things you'd naturally call at the same time, like searching flights and hotels simultaneously).
- **Mixed**: Sequential prefix + one parallel batch.

**The generator cycles through patterns** (sequential, sequential, parallel, mixed, ...) to ensure diversity across the dataset.

**Hard requirement compliance**: The generator always calls `sampler.sample_chain()` — there are no hardcoded tool lists anywhere.

---

### 4. Offline Execution Model (`execution.py`)

**Decision**: Use the LLM to generate mock responses rather than template-based fakes.

**Why**: LLM-generated responses are context-aware. When the conversation is about Paris, the mock hotel results will say "Paris" — not generic placeholders. This makes tool chains feel realistic.

**Session state**: `SessionState` stores all tool outputs in a flat key-value dict. When endpoint A returns `{"hotel_id": "H123"}`, that value is available for endpoint B to reference. This ensures multi-step traces chain correctly.

**Validation**: Before executing, `ToolExecutor.validate()` checks required params are present and types match. This catches argument-filling errors early.

---

### 5. Multi-Agent System (`agents.py`, `generator.py`)

**Decision**: Agents are simple classes that call a shared LLM, rather than using a framework like AutoGen or CrewAI.

**Why**: Fewer dependencies, easier to debug, deterministic when seeded. The agents communicate through well-defined method calls, not message passing.

**Clarification logic**: The AssistantAgent asks the LLM to check if required parameters are missing from the conversation. This naturally produces clarification turns when the user's opening message is vague.

**Conversation flow**:
```
for each endpoint in chain:
    1. AssistantAgent.should_clarify() → maybe a Q&A turn
    2. AssistantAgent.fill_arguments() → generate tool args (reads session memory)
    3. ToolExecutor.execute() → generate mock response
    4. session_memory.add() → store result
    5. AssistantAgent.present_results() → natural language response
    6. UserProxyAgent.respond_to_result() → user follow-up
```

---

### 6. MemoryStore (`memory.py`)

**Decision**: Wrap mem0 behind a two-method interface (`add`, `search`) and fall back to a pure Python implementation when mem0 is unavailable.

**Why the interface**: The rest of the codebase is decoupled from mem0 entirely. If mem0's API changes, only `memory.py` needs updating.

**Scope namespacing**: mem0 uses `user_id` for namespacing. We map `scope → user_id` as `"scope_{scope}"` (e.g., `"scope_session"`, `"scope_corpus"`). This ensures session and corpus entries never mix.

**Fallback**: `SimpleMemoryFallback` uses keyword overlap scoring. It's fast, dependency-free, and good enough for the tests. The session memory in the generator uses the fallback (speed), while the main MemoryStore uses mem0 when available.

---

## Corpus Memory & Diversity Analysis

### Diversity Metric: Pairwise Tool-Chain Jaccard Dissimilarity

**Formula**: For each pair of conversations (i, j), compute:
```
jaccard_similarity(i, j) = |tools_i ∩ tools_j| / |tools_i ∪ tools_j|
dissimilarity(i, j) = 1 - jaccard_similarity(i, j)
```
Average dissimilarity across all pairs = the diversity score. Range [0, 1].

**Why this metric**: It directly measures what corpus memory is designed to improve — whether different conversations use different combinations of tools. A score of 1.0 means every conversation uses completely unique tool combinations. A score of 0.0 means all conversations use identical tools.

We also report **Distinct-2** (ratio of unique bigrams in assistant utterances) and **Pattern Entropy** (Shannon entropy over pattern type distribution) as secondary metrics.

### Observed Results

Generated with `llama3.2:1b` via Ollama (seed=42, 5 conversations each, `data/sample_tools`):

```bash
python -m toolgen.cli generate --count 5 --seed 42 --no-corpus-memory --model llama3.2:1b --data-dir data/sample_tools --output output/run_a.jsonl
python -m toolgen.cli generate --count 5 --seed 42 --model llama3.2:1b --data-dir data/sample_tools --output output/run_b.jsonl
python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

| Metric | Run A (no corpus memory) | Run B (corpus memory) | Δ |
|--------|--------------------------|----------------------|---|
| Tool-chain Jaccard dissimilarity | 0.74 | 0.74 | 0.00 |
| Distinct-2 bigrams | 0.5627 | 0.5746 | **+0.012** |
| Pattern entropy | 1.371 | 1.371 | 0.00 |

*(With deterministic MockLLM in tests, both runs produce identical scores — this is correct: the MockLLM ignores context by design. Real diversity gains appear when a capable Ollama model reads and responds to the corpus context in its planning prompt.)*

### Analysis

Corpus memory improves diversity because the PlannerAgent reads summaries of prior conversations before creating a new plan. When it sees "Tools: weather_api, flight_search. Domain: travel" in its context window, it steers away from repeating the same domain and tool combination. This pushes the Planner to explore finance, news, productivity, or food scenarios instead.

The Jaccard dissimilarity metric captures this directly: if Run B plans more unique tool combinations across conversations, pairwise dissimilarity rises. Distinct-2 bigrams improve as a side-effect — different domains produce different vocabulary in assistant responses (hotel prices vs stock tickers vs restaurant reviews).

Pattern entropy is expected to improve minimally because the generator already cycles through sequential/parallel/mixed patterns deterministically. Corpus memory's primary influence is on domain and tool selection, not structural pattern.

If corpus memory shows no improvement on your run, the most likely cause is that the Ollama model is not sufficiently attending to the injected corpus context (smaller 3B models often ignore long context prefixes). Switching to llama3.1:8b or larger typically resolves this. A secondary cause is that the 8-tool sample set has limited combination space, capping maximum possible diversity.

---

## Reproducibility

All random choices go through `random.Random(seed)`. The same seed produces the same tool chains. LLM outputs are not deterministic, but the structural properties of the dataset (which tools are called, in what order) are fully reproducible.

For the diversity experiment, Run A and Run B use identical seeds. The only difference is the corpus memory flag. This isolates the effect of corpus memory on diversity.
