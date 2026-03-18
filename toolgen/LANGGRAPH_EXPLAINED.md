# LangGraph — Deep Explanation for the Internship Interview

## "Did you use LangGraph?" — The Honest Answer

The **original version** uses plain Python classes and a for-loop.  
The **LangGraph version** (`langgraph_generator.py`) rebuilds the agent pipeline
as a proper stateful graph. Both work. The LangGraph version is what you'd put in
production and what impresses hiring managers.

---

## What Is LangGraph? (Simple version)

LangGraph lets you draw your AI agent as a **flowchart**, and that flowchart
**becomes** your code.

**Without LangGraph** (what most beginners write):
```python
# Just a big for-loop — works but hard to maintain
for step in tool_chain:
    if needs_clarification:
        ask_question()
        get_answer()
    call_tool()
    show_results()
```

**With LangGraph** (what production systems look like):
```
[Start] → [Sampler] → [Planner] → [User Opens] → [Router]
                                                      ↓        ↓
                                                 [Clarify]  [Call Tool]
                                                      ↓        ↓
                                                      └──→[Present]→[Router]
                                                                       ↓
                                                               [Final]→[Validate]→[End]
```

This flowchart IS the code. Each box = a Python function. Each arrow = an edge.

---

## The 4 Core Concepts You Must Know

### 1. State (TypedDict) — The Shared Whiteboard

```python
class ConversationState(TypedDict):
    chain: list[str]        # which tools to call
    messages: list[dict]    # all chat messages
    current_step: int       # which tool we're on
    plan: dict              # the user scenario
    tool_calls: list[dict]  # all tool results so far
    is_valid: bool          # did validator approve?
```

**Why TypedDict?**  
- Every node reads from and writes to this state
- LangGraph automatically passes it between nodes
- TypedDict gives you type safety (your editor warns you of typos)
- It's like a shared Google Doc that all agents edit

### 2. Nodes — The Workers

A node is just a Python function:
```python
def planner_node(state: ConversationState) -> dict:
    # READ from state
    chain = state["chain"]
    
    # DO something
    plan = create_plan(chain)
    
    # RETURN only what changed (not the whole state)
    return {"plan": plan}
```

LangGraph automatically merges your returned dict into the state.

### 3. Edges — The Arrows

```python
# Unconditional: always go from A to B
graph.add_edge("sampler", "planner")

# Conditional: decide at runtime
graph.add_conditional_edges(
    "router",            # from this node
    route_decision,      # call this function — it returns a string
    {
        "clarify":   "clarify_node",   # if returns "clarify" → go here
        "call_tool": "tool_executor",  # if returns "call_tool" → go here
        "finalize":  "final_node",     # if returns "finalize" → go here
    }
)
```

### 4. Compile + Invoke — Run It

```python
graph = builder.compile()           # locks the graph, makes it runnable
result = graph.invoke(initial_state) # runs it, returns final state
```

---

## Our Graph Step by Step

```
Initial state created with empty fields
            ↓
[sampler_node]
  - Calls ToolChainSampler.sample_chain()  
  - Picks tool chain from NetworkX graph  
  - Returns: {"chain": ["weather_api::get_current_weather", "flight_search::search_flights", ...]}
            ↓
[planner_node]
  - Reads corpus memory (what convos exist already?)
  - Asks LLM to create a user scenario
  - Returns: {"plan": {"user_goal": "Plan a Paris trip", "domain": "travel", ...}}
            ↓
[user_opening_node]
  - UserProxy agent generates first message
  - Returns: {"messages": [{"role": "user", "content": "Hi I want to plan a Paris trip"}]}
            ↓
[router_node] + route_decision()     ← CONDITIONAL EDGE
  - Checks: are we done? need clarification? ready to call tool?
  - Returns string: "clarify" or "call_tool" or "finalize"
            ↓                    ↓                    ↓
      [clarify_node]    [tool_executor_node]    [final_node]
      Asks question     Calls tool + memory      Wraps up
      Gets answer       Read + Write path
            ↓                    ↓
            └────→  [presenter_node]
                    Shows results to user
                    Advances current_step
                            ↓
                    [router_node] (loops back)
                            ↓ (when current_step >= len(chain))
                    [final_node]
                            ↓
                    [validator_node]
                    Checks: ≥3 tool calls? ≥2 tools?
                            ↓
                          [END]
```

---

## Why Is This Better Than a For-Loop?

| Aspect | Plain Python Loop | LangGraph |
|--------|------------------|-----------|
| Visualize flow | Hard | Built-in graph visualization |
| Add new step | Edit middle of loop | Add one node + edge |
| Debug stuck agents | Guess where it broke | Each node is isolated |
| Parallel agents | Complex threading | Native support |
| Resume from failure | Start over | Checkpoint + resume |
| Industry standard | Simple scripts | Production systems |
| Interview impression | "OK" | "Impressive" |

---

## How to Run Both Versions

```bash
# Original plain Python version:
python -m toolgen.cli generate --count 10

# LangGraph version:
python -m toolgen.cli generate --count 10 --use-langgraph
```

Both produce the same JSONL format. The LangGraph version adds
`"framework": "langgraph"` to the metadata so you can tell them apart.

---

## Interview Questions & How to Answer Them

**Q: Why did you use LangGraph instead of just a for-loop?**

A: "LangGraph makes the agent's decision logic explicit and visual. In production,
you need to be able to debug why an agent made a specific decision. With LangGraph,
each decision point is a node you can inspect independently. It also makes it easy
to add new behaviors — I just add a node and an edge, rather than editing a complex
nested loop."

---

**Q: What is the State in LangGraph?**

A: "It's a TypedDict — a shared dictionary that every node in the graph can read
and write. Think of it as a whiteboard in a meeting room. The sampler writes the
tool chain, the planner writes the scenario, the executor writes tool results — and
at the end, the validator reads everything to check quality. LangGraph handles
passing this state between nodes automatically."

---

**Q: How does LangGraph handle conditional routing?**

A: "With conditional edges. You provide a function that reads the current state and
returns a string — like 'clarify' or 'call_tool'. LangGraph uses that string to
look up which node to visit next in a dictionary you define. In our system, the
router checks whether the user has provided all required parameters for the next
tool call, and routes to either a clarification exchange or direct tool execution."

---

**Q: What is the difference between LangGraph and LangChain?**

A: "LangChain is a toolkit — it provides LLM wrappers, prompt templates, and
output parsers. LangGraph is built on top of LangChain and adds the graph/state
machine layer. LangChain handles 'how to talk to an LLM'. LangGraph handles
'how multiple agents coordinate over time with shared state'."

---

**Q: How does memory work in your system?**

A: "We have two scopes of memory using mem0. Session memory is within one
conversation — after each tool call, we write the output to memory, and before
the next tool call, we query that memory to ground the arguments (for example,
using the hotel_id from step 1 in step 2). Corpus memory is across all
conversations — after each conversation, we write a summary, and the Planner
reads it before creating new plans, so it doesn't repeat the same scenarios."

---

**Q: Can LangGraph run agents in parallel?**

A: "Yes — you can have multiple edges leaving a node that go to different nodes
simultaneously. Those nodes run in parallel and their results are merged back
into the state. We didn't implement this in the current version, but our 'parallel'
pattern (where multiple tools are called at the same time) could be refactored to
use LangGraph's native parallel execution."

---

## The Key Files to Know for the Interview

| File | What it Does | Key Concept |
|------|-------------|-------------|
| `langgraph_generator.py` | LangGraph pipeline | StateGraph, nodes, edges |
| `memory.py` | MemoryStore | Session vs corpus scope |
| `graph.py` | Tool Graph | NetworkX, chain sampling |
| `registry.py` | Tool Registry | Dataclasses, normalization |
| `execution.py` | Mock tool runner | Validation, session state |
| `agents.py` | Agent classes | Separation of concerns |
| `metrics.py` | Evaluation | Jaccard diversity, entropy |

