"""
langgraph_generator.py — LangGraph Multi-Agent Conversation Generator
======================================================================

WHY LANGGRAPH?
--------------
Our original generator.py uses a plain Python for-loop to run agents.
This works, but it's not how production AI systems are built.

LangGraph lets you define agents as a GRAPH (flowchart):
  - Each agent = a NODE (a Python function)
  - Each arrow = an EDGE (what happens next)
  - Branching logic = CONDITIONAL EDGES (if/else routing)

This means:
  1. You can VISUALIZE the entire agent pipeline
  2. Each node is isolated, easy to test and swap
  3. State flows automatically between nodes (no manual passing)
  4. Industry standard — companies like LinkedIn, Replit use this

HOW THE GRAPH LOOKS:
--------------------

    [START]
       ↓
  [sampler_node]        ← Agent 1: picks tool chain from the graph
       ↓
  [planner_node]        ← Agent 2: creates user scenario (reads corpus memory)
       ↓
  [user_opening_node]   ← Agent 3 (UserProxy): generates first message
       ↓
  [router_node]         ← decides: need clarification? or call tool?
       ↓              ↓
  [clarify_node]   [tool_executor_node]   ← Agent 4 (Assistant): calls tool
       ↓                   ↓
  [user_reply_node]  [presenter_node]     ← presents results
       ↓                   ↓
       └──────────────→ [step_advance]    ← move to next tool
                              ↓
                    (loop back to router_node
                     or go to final_node)
                              ↓
                      [final_node]        ← generate summary
                              ↓
                      [validator_node]    ← Agent 5: quality check
                              ↓
                          [END]

STATE:
------
All nodes share a single ConversationState dictionary.
Think of it like a whiteboard in a meeting room —
every agent reads from it and writes to it.
"""

import json
import uuid
from typing import Annotated, Any, Optional

# ── LangGraph imports ──────────────────────────────────────────
# StateGraph: the main graph builder
# END: special node meaning "we're done"
from langgraph.graph import END, StateGraph

# TypedDict: Python's way of defining a dict with known keys + types
# Annotated: lets us attach extra info to a type (used for the messages list)
from typing import TypedDict

# LangChain message types — the standard way to represent chat messages
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# LangChain Ollama integration — wraps Ollama in LangChain's interface
# This gives us: structured output, tool-calling support, streaming, etc.
from langchain_ollama import ChatOllama

# Our existing modules — LangGraph replaces the ORCHESTRATION, not the tools
from toolgen.execution import SessionState, ToolExecutor
from toolgen.graph import ToolChainSampler
from toolgen.memory import MemoryStore
from toolgen.registry import ToolRegistry
from toolgen.agents import ValidatorAgent, Conversation, Message


# ════════════════════════════════════════════════════════════════
# STEP 1: Define the STATE
# ════════════════════════════════════════════════════════════════
#
# The State is a TypedDict — a Python dict where every key has a type.
# LangGraph automatically passes this between nodes.
#
# Think of it as the "whiteboard" all agents share.
# Every node READS from state and RETURNS updates to state.

class ConversationState(TypedDict):
    """
    The shared state that flows through every node in our graph.

    Every field here is like a row on the whiteboard:
    - Some fields are set early (like 'chain', 'plan') and never change
    - Some fields grow over time (like 'messages', 'tool_calls')
    - Some fields are counters (like 'current_step', 'clarification_count')
    """

    # ── Set by sampler_node ────────────────────────────────────
    chain: list[str]              # Ordered list of endpoint_ids to call
                                  # e.g. ["weather_api::get_current_weather",
                                  #        "flight_search::search_flights"]

    conversation_id: str          # Unique ID for this conversation
    pattern_type: str             # "sequential", "parallel", or "mixed"

    # ── Set by planner_node ────────────────────────────────────
    plan: dict                    # {"user_goal": "...", "domain": "...", ...}

    # ── Grows throughout the conversation ─────────────────────
    messages: list[dict]          # All chat messages (role + content)
                                  # We store as plain dicts for simplicity
    tool_calls: list[dict]        # All tool calls made (endpoint, args, output)

    # ── Counters that nodes update ─────────────────────────────
    current_step: int             # Which tool in the chain we're on (0, 1, 2...)
    clarification_count: int      # How many clarifications we've asked
    memory_grounded_steps: int    # How many non-first tool calls used memory
    non_first_tool_calls: int     # Total non-first tool calls (for rate calculation)

    # ── Config passed in at start ──────────────────────────────
    corpus_memory_enabled: bool
    seed: int

    # ── Set by router when clarification is needed ─────────────
    pending_clarification: str   # question to ask user before calling tool

    # ── Set at the end ─────────────────────────────────────────
    is_valid: bool                # Did the validator approve this conversation?
    validation_issues: list[str]  # List of problems found (empty = valid)


# ════════════════════════════════════════════════════════════════
# STEP 2: Define helper to build the LLM
# ════════════════════════════════════════════════════════════════

def make_llm(model: str = "llama3.2", temperature: float = 0.7) -> ChatOllama:
    """
    Create a LangChain ChatOllama instance.

    ChatOllama wraps Ollama in LangChain's interface.
    This gives us:
    - .invoke(messages) → AIMessage
    - .with_structured_output(schema) → always returns JSON
    - Compatible with ALL LangChain tools and chains

    temperature=0.7 means: some creativity but not too random.
    temperature=0.0 means: always the same output (deterministic).
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url="http://localhost:11434",
    )


def llm_json(llm: ChatOllama, prompt: str, system: str = "") -> dict:
    """
    Ask the LLM and parse the response as JSON.
    Uses LangChain message format: SystemMessage + HumanMessage.
    """
    import re
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))

    response = llm.invoke(msgs)
    text = response.content.strip()

    # Try to parse JSON, stripping markdown fences if needed
    for attempt in [text,
                    re.sub(r"```(?:json)?", "", text).strip().rstrip("`"),
                    ]:
        try:
            return json.loads(attempt)
        except json.JSONDecodeError:
            pass

    # Last resort: find the first {...} block
    import re
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    return {"error": "could_not_parse", "raw": text[:200]}


def llm_text(llm: ChatOllama, prompt: str, system: str = "") -> str:
    """Ask the LLM and get a plain text response."""
    msgs = []
    if system:
        msgs.append(SystemMessage(content=system))
    msgs.append(HumanMessage(content=prompt))
    return llm.invoke(msgs).content.strip()


# ════════════════════════════════════════════════════════════════
# STEP 3: Define the NODES
# ════════════════════════════════════════════════════════════════
#
# A node is just a Python function that:
#   - Takes the current state as input
#   - Returns a dict of UPDATES to the state
#
# LangGraph merges your returned dict into the state automatically.
# You only return the fields that CHANGED — not the whole state.

class ConversationGraphBuilder:
    """
    Builds the LangGraph StateGraph for conversation generation.

    We put everything inside a class so nodes can share:
    - self.llm        (the language model)
    - self.registry   (tool definitions)
    - self.memory     (session + corpus memory)
    - self.executor   (tool execution)
    - self.sampler    (tool chain sampler)
    - self.validator  (quality checker)

    Each method (sampler_node, planner_node, etc.) becomes a NODE.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        sampler: ToolChainSampler,
        memory: MemoryStore,
        model: str = "llama3.2",
    ):
        self.registry  = registry
        self.sampler   = sampler
        self.memory    = memory
        self.llm       = make_llm(model)
        self.executor  = ToolExecutor(registry=registry, llm=None)
        # Give executor a simple wrapper so it works without our custom LLM class
        self.executor.llm = _LLMAdapter(self.llm)
        self.validator = ValidatorAgent()

    # ── NODE 1: Sampler ───────────────────────────────────────
    def sampler_node(self, state: ConversationState) -> dict:
        """
        NODE 1 — Sampler Agent

        Uses the Tool Graph to propose a tool chain.
        This is where the GRAPH SAMPLER requirement is satisfied.

        Input:  state["pattern_type"]
        Output: state["chain"]  ← list of endpoint_ids
        """
        pattern = state.get("pattern_type", "sequential")

        # Call our ToolChainSampler (which uses NetworkX graph internally)
        raw_chain = self.sampler.sample_chain(pattern=pattern)

        # Flatten — parallel patterns return lists of lists
        chain = []
        for item in raw_chain:
            if isinstance(item, list):
                chain.extend(item)
            else:
                chain.append(item)

        print(f"[LangGraph:Sampler] Chain: {chain}")

        # Return only the fields we're updating
        return {"chain": chain}

    # ── NODE 2: Planner ───────────────────────────────────────
    def planner_node(self, state: ConversationState) -> dict:
        """
        NODE 2 — Planner Agent

        Given the tool chain, creates a realistic user scenario.
        Also reads CORPUS MEMORY to avoid repeating past conversations.

        Input:  state["chain"], state["corpus_memory_enabled"]
        Output: state["plan"]
        """
        chain = state["chain"]

        # Describe the tools for the LLM
        tool_descriptions = []
        for ep_id in chain:
            ep = self.registry.get_endpoint(ep_id)
            if ep:
                tool_descriptions.append(f"- {ep.endpoint_id}: {ep.description}")
        tools_text = "\n".join(tool_descriptions)

        # ── Read corpus memory ─────────────────────────────────
        # This is the "corpus memory read path" from the spec
        corpus_prefix = ""
        if state.get("corpus_memory_enabled", True):
            prior = self.memory.search(
                query=tools_text[:200], scope="corpus", top_k=3
            )
            if prior:
                summaries = "\n".join(f"- {e['content']}" for e in prior)
                corpus_prefix = (
                    f"[Prior conversations in corpus — avoid duplicating]\n"
                    f"{summaries}\n\n"
                )

        plan = llm_json(
            self.llm,
            prompt=(
                f"{corpus_prefix}"
                f"Given these tools, create a specific realistic user scenario.\n\n"
                f"Tools:\n{tools_text}\n\n"
                f"Return JSON with:\n"
                f'{{"user_goal":"...","user_persona":"...","context":"...",'
                f'"ambiguities":["..."],"domain":"travel/food/finance/etc"}}'
            ),
            system=(
                "You create realistic scenarios for AI assistant training data. "
                "Be specific and varied. Return ONLY valid JSON."
            ),
        )

        # Fallback if LLM fails
        if "error" in plan:
            plan = {
                "user_goal": f"Use tools: {', '.join(chain[:2])}",
                "user_persona": "A regular user",
                "context": "User needs help with a task.",
                "ambiguities": ["location", "date"],
                "domain": "general",
            }

        print(f"[LangGraph:Planner] Goal: {plan.get('user_goal','?')[:60]}")
        return {"plan": plan}

    # ── NODE 3: User Opening ───────────────────────────────────
    def user_opening_node(self, state: ConversationState) -> dict:
        """
        NODE 3 — UserProxy: first message

        Generates the user's opening message based on the plan.
        Deliberately vague to create room for clarification.

        Input:  state["plan"]
        Output: state["messages"]  ← adds one user message
        """
        plan = state["plan"]

        opening = llm_text(
            self.llm,
            prompt=(
                f"You are a user with this goal: {plan['user_goal']}\n"
                f"Persona: {plan['user_persona']}\n"
                f"Write a natural, slightly vague first message to an AI assistant.\n"
                f"1-2 sentences only. Write ONLY the message."
            ),
        ).strip('"')

        new_msg = {"role": "user", "content": opening}
        return {
            "messages": state.get("messages", []) + [new_msg],
        }

    # ── NODE 4: Router ─────────────────────────────────────────
    def router_node(self, state: ConversationState) -> dict:
        """
        NODE 4 — Router (decision maker)

        This node doesn't produce output — it just exists so the
        CONDITIONAL EDGE can read the state and decide where to go.

        The actual routing logic is in `route_decision()` below.
        LangGraph calls router_node first, then calls route_decision
        to determine which node to go to next.

        Think of it as a traffic cop standing at an intersection.
        """
        # Nothing to compute here — the routing happens in the edge function
        return {}

    def route_decision(self, state: ConversationState) -> str:
        """
        CONDITIONAL EDGE FUNCTION

        This is called AFTER router_node to decide which node to visit next.
        Returns a STRING that must match one of the edges defined in the graph.

        Decision tree:
          1. Are we past the last tool? → "finalize"
          2. Have we clarified too many times? → "call_tool"
          3. Is clarification needed? → "clarify" or "call_tool"
        """
        current_step = state.get("current_step", 0)
        chain = state.get("chain", [])
        clarification_count = state.get("clarification_count", 0)

        # Check if we've processed all tools
        if current_step >= len(chain):
            return "finalize"

        # Don't ask too many clarifications
        if clarification_count >= 2:
            return "call_tool"

        # Ask LLM if clarification is needed for next tool
        endpoint_id = chain[current_step]
        ep = self.registry.get_endpoint(endpoint_id)
        if not ep:
            return "call_tool"

        recent_msgs = state.get("messages", [])[-4:]
        conversation_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent_msgs)
        required = [p.name for p in ep.required_params]

        if not required:
            return "call_tool"

        response = llm_text(
            self.llm,
            prompt=(
                f"Conversation so far:\n{conversation_text}\n\n"
                f"Next tool: {ep.name} — requires: {required}\n\n"
                f"Is any required info MISSING from the conversation?\n"
                f"If yes, write ONE short question. If no, write exactly: NONE"
            ),
        )

        if "NONE" in response.upper() or len(response) > 200:
            return "call_tool"
        else:
            # Store in proper state field via router_node return value
            # (We set it here directly since route_decision has access to state)
            state["pending_clarification"] = response
            return "clarify"

    # ── NODE 5: Clarify ────────────────────────────────────────
    def clarify_node(self, state: ConversationState) -> dict:
        """
        NODE 5 — Clarification exchange

        Adds TWO messages: the assistant's question + user's answer.
        This creates the multi-turn disambiguation the spec requires.

        Input:  state["plan"]["_pending_clarification"]
        Output: state["messages"]  ← adds Q&A pair
                state["clarification_count"]  ← incremented
        """
        plan = state["plan"]
        question = state.get("pending_clarification", "Could you provide more details?")

        # User responds to the clarification question
        user_answer = llm_text(
            self.llm,
            prompt=(
                f"You are a user with goal: {plan['user_goal']}\n"
                f"The assistant asked: '{question}'\n"
                f"Write a brief, helpful answer. 1-2 sentences only."
            ),
        ).strip('"')

        new_messages = [
            {"role": "assistant", "content": question},
            {"role": "user",      "content": user_answer},
        ]

        print(f"[LangGraph:Clarify] Q: {question[:50]}...")

        return {
            "messages":            state.get("messages", []) + new_messages,
            "clarification_count": state.get("clarification_count", 0) + 1,
        }

    # ── NODE 6: Tool Executor ──────────────────────────────────
    def tool_executor_node(self, state: ConversationState) -> dict:
        """
        NODE 6 — Tool Executor (the core of the pipeline)

        This node:
          1. Reads session memory (grounding)
          2. Fills in the tool arguments using LLM
          3. Executes the tool (generates mock response)
          4. Writes output to session memory
          5. Updates state with tool call record

        This is where the SESSION MEMORY read + write paths happen.
        """
        chain        = state["chain"]
        current_step = state.get("current_step", 0)

        if current_step >= len(chain):
            return {}

        endpoint_id = chain[current_step]
        ep = self.registry.get_endpoint(endpoint_id)
        if not ep:
            return {"current_step": current_step + 1}

        plan = state["plan"]
        messages = state.get("messages", [])
        session = SessionState()  # fresh session — we'll replay history

        # Rebuild session from prior tool calls
        for i, tc in enumerate(state.get("tool_calls", [])):
            session.store_output(tc["endpoint"], i, tc["output"])

        # ── Memory READ path ───────────────────────────────────
        # Before filling args (non-first step), query session memory
        memory_context = ""
        memory_used = False
        non_first = state.get("non_first_tool_calls", 0)
        grounded  = state.get("memory_grounded_steps", 0)

        if current_step > 0:
            non_first += 1
            prior = self.memory.search(
                query=f"{ep.name} {ep.description}",
                scope="session",
                top_k=3,
            )
            if prior:
                memory_used = True
                grounded += 1
                memory_context = "[Memory context]\n"
                memory_context += "\n".join(f"- {e['content'][:200]}" for e in prior)
                memory_context += "\n\n"

        # Fill arguments
        recent_text = "\n".join(f"{m['role']}: {m['content']}"
                                for m in messages[-6:])
        param_lines = (
            [f"  REQUIRED: {p.name} ({p.type}) — {p.description}"
             for p in ep.required_params] +
            [f"  optional: {p.name} ({p.type}) — {p.description}"
             for p in ep.optional_params[:3]]
        )

        args = llm_json(
            self.llm,
            prompt=(
                f"{memory_context}"
                f"Given the above context and the current tool schema, "
                f"fill in the arguments for {ep.name}.\n\n"
                f"User goal: {plan.get('user_goal','')}\n\n"
                f"Conversation:\n{recent_text}\n\n"
                f"Tool schema:\n" + "\n".join(param_lines) +
                f"\n\nReturn ONLY a JSON object of argument values."
            ),
        )

        if "error" in args:
            args = {p.name: f"example_{p.name}" for p in ep.required_params}

        # Execute the tool (generate mock response)
        try:
            output = self.executor._generate_mock_response(ep, args, session)
        except Exception as e:
            output = {"error": str(e), "status": "failed"}

        # ── Memory WRITE path ──────────────────────────────────
        # After every tool call completes, write its output to session memory
        self.memory.add(
            content=json.dumps({"endpoint": endpoint_id, "output": output}),
            scope="session",
            metadata={
                "conversation_id": state["conversation_id"],
                "step":            current_step,
                "endpoint":        endpoint_id,
            },
        )

        # Build the tool call record
        tool_call_record = {
            "endpoint":  endpoint_id,
            "tool_name": ep.tool_name,
            "arguments": args,
            "output":    output,
            "step":      current_step,
        }

        print(f"[LangGraph:Executor] Step {current_step}: {endpoint_id}")

        return {
            "tool_calls":           state.get("tool_calls", []) + [tool_call_record],
            "messages":             state.get("messages", []) + [
                {"role": "assistant",
                 "content": f"[Calling {endpoint_id}]",
                 "tool_call": {"endpoint": endpoint_id, "arguments": args}},
                {"role": "tool",
                 "content": json.dumps(output),
                 "tool_output": output},
            ],
            "non_first_tool_calls": non_first,
            "memory_grounded_steps": grounded,
        }

    # ── NODE 7: Result Presenter ───────────────────────────────
    def presenter_node(self, state: ConversationState) -> dict:
        """
        NODE 7 — Assistant presents tool results to user

        After calling a tool, the assistant explains what it found
        in natural language (not raw JSON).

        Then the USER sends a follow-up message.
        Both messages are added to the conversation.
        """
        tool_calls   = state.get("tool_calls", [])
        current_step = state.get("current_step", 0)
        plan         = state["plan"]
        chain        = state["chain"]

        if not tool_calls:
            return {}

        last_tc = tool_calls[-1]
        ep = self.registry.get_endpoint(last_tc["endpoint"])

        # Assistant presents results
        result_msg = llm_text(
            self.llm,
            prompt=(
                f"Present these tool results naturally to the user.\n"
                f"Tool: {last_tc['endpoint']}\n"
                f"User goal: {plan.get('user_goal','')}\n"
                f"Result: {json.dumps(last_tc['output'])[:400]}\n\n"
                f"Write a helpful 2-3 sentence response. Be conversational."
            ),
        )

        new_messages = [{"role": "assistant", "content": result_msg}]

        # User follow-up (only if there are more tools to call)
        is_last_step = (current_step + 1 >= len(chain))
        if not is_last_step:
            user_followup = llm_text(
                self.llm,
                prompt=(
                    f"You are a user. The assistant said:\n'{result_msg[:200]}'\n"
                    f"Write a brief natural follow-up. 1 sentence."
                ),
            ).strip('"')
            new_messages.append({"role": "user", "content": user_followup})

        return {
            "messages":     state.get("messages", []) + new_messages,
            "current_step": current_step + 1,  # ← advance to next tool
        }

    # ── NODE 8: Final Response ────────────────────────────────
    def final_node(self, state: ConversationState) -> dict:
        """
        NODE 8 — Generate the final summary

        After all tools are called, the assistant wraps up with a
        comprehensive response answering the user's original goal.

        Also writes the conversation summary to CORPUS MEMORY.
        """
        plan       = state["plan"]
        tool_calls = state.get("tool_calls", [])

        # Build summary of what was accomplished
        tc_summary = "\n".join(
            f"- {tc['endpoint']}: {str(tc['output'])[:100]}"
            for tc in tool_calls
        )

        final_msg = llm_text(
            self.llm,
            prompt=(
                f"User wanted: {plan.get('user_goal','')}\n\n"
                f"You made these tool calls:\n{tc_summary}\n\n"
                f"Write a helpful final summary (3-4 sentences). Be conversational."
            ),
        )

        # ── Corpus memory WRITE path ───────────────────────────
        # After each conversation, write a summary to corpus memory
        # so future conversations can diversify from this one
        if state.get("corpus_memory_enabled", True):
            tools_used = list({tc["tool_name"] for tc in tool_calls})
            summary = (
                f"Tools: {', '.join(tools_used)}. "
                f"Domain: {plan.get('domain','general')}. "
                f"Pattern: {state.get('pattern_type','sequential')}. "
                f"Goal: {plan.get('user_goal','')[:80]}."
            )
            self.memory.add(
                content=summary,
                scope="corpus",
                metadata={
                    "conversation_id": state["conversation_id"],
                    "tools":           tools_used,
                    "pattern_type":    state.get("pattern_type", "sequential"),
                },
            )

        return {
            "messages": state.get("messages", []) + [
                {"role": "assistant", "content": final_msg}
            ],
        }

    # ── NODE 9: Validator ─────────────────────────────────────
    def validator_node(self, state: ConversationState) -> dict:
        """
        NODE 9 — Validator Agent (quality gate)

        Checks if the conversation meets the minimum requirements:
        - At least 3 tool calls
        - At least 2 distinct tools

        This is the LAST node before END.
        """
        tool_calls     = state.get("tool_calls", [])
        num_tools      = len(tool_calls)
        distinct_tools = len({tc.get("tool_name","") for tc in tool_calls})
        issues         = []

        if num_tools < 3:
            issues.append(f"Only {num_tools} tool calls (need ≥ 3)")
        if distinct_tools < 2:
            issues.append(f"Only {distinct_tools} distinct tools (need ≥ 2)")
        if not state.get("messages"):
            issues.append("No messages")

        is_valid = len(issues) == 0
        print(f"[LangGraph:Validator] {'✓ VALID' if is_valid else '✗ INVALID'} "
              f"— {num_tools} tool calls, {distinct_tools} tools")

        return {
            "is_valid":         is_valid,
            "validation_issues": issues,
        }

    # ════════════════════════════════════════════════════════════
    # STEP 4: BUILD THE GRAPH
    # ════════════════════════════════════════════════════════════
    #
    # This is where we connect all the nodes with edges.
    # After build(), call compile() to get a runnable graph.

    def build(self) -> "CompiledGraph":
        """
        Assemble all nodes and edges into a compiled LangGraph.

        After this, you can call:
            graph.invoke(initial_state)

        Or stream step by step:
            for step in graph.stream(initial_state):
                print(step)
        """

        # Create the graph builder, telling it what the State looks like
        g = StateGraph(ConversationState)

        # ── Add all nodes ─────────────────────────────────────
        # .add_node(name, function)
        # name = string identifier used in edges
        # function = the node function (takes state, returns dict)
        g.add_node("sampler",    self.sampler_node)
        g.add_node("planner",    self.planner_node)
        g.add_node("user_open",  self.user_opening_node)
        g.add_node("router",     self.router_node)
        g.add_node("clarify",    self.clarify_node)
        g.add_node("call_tool",  self.tool_executor_node)
        g.add_node("present",    self.presenter_node)
        g.add_node("finalize",   self.final_node)
        g.add_node("validate",   self.validator_node)

        # ── Set the entry point ───────────────────────────────
        # This is the first node the graph runs
        g.set_entry_point("sampler")

        # ── Add simple (unconditional) edges ──────────────────
        # A → B means: after A finishes, always go to B
        g.add_edge("sampler",   "planner")    # after sampling, plan
        g.add_edge("planner",   "user_open")  # after planning, user speaks
        g.add_edge("user_open", "router")     # after user speaks, decide next
        g.add_edge("clarify",   "call_tool")  # after clarifying, call the tool
        g.add_edge("call_tool", "present")    # after calling tool, present results
        g.add_edge("finalize",  "validate")   # after final summary, validate
        g.add_edge("validate",  END)          # after validation, done!

        # ── Add conditional edges ─────────────────────────────
        # After "router" node runs, call route_decision(state) to decide next node
        # The return value of route_decision must be a KEY in the dict below
        g.add_conditional_edges(
            "router",             # ← from this node
            self.route_decision,  # ← call this function to decide
            {
                # route_decision return value → next node name
                "clarify":   "clarify",    # needs clarification
                "call_tool": "call_tool",  # ready to call tool
                "finalize":  "finalize",   # all tools done
            },
        )

        # After presenting results, go back to router (for next tool)
        # or finalize if we've done all tools
        # We use presenter_node to advance current_step, then router decides
        g.add_edge("present", "router")

        # Compile and return — this "locks" the graph and makes it runnable
        return g.compile()


# ════════════════════════════════════════════════════════════════
# STEP 5: High-level runner
# ════════════════════════════════════════════════════════════════

class LangGraphConversationGenerator:
    """
    High-level runner that uses the compiled LangGraph to generate
    conversations. This replaces generator.py's ConversationGenerator.

    Usage:
        gen = LangGraphConversationGenerator(registry, sampler, memory)
        conv = gen.generate_one()
        batch = gen.generate_batch(50)
    """

    # Cycle through patterns for variety
    _PATTERNS = ["sequential", "sequential", "parallel", "mixed"]

    def __init__(
        self,
        registry: ToolRegistry,
        sampler: ToolChainSampler,
        memory: MemoryStore,
        model: str = "llama3.2",
        corpus_memory_enabled: bool = True,
        seed: int = 42,
    ):
        self.registry = registry
        self.memory   = memory
        self.corpus_memory_enabled = corpus_memory_enabled
        self.seed     = seed
        self._pattern_idx = 0

        # Build and compile the graph ONCE (reused for all conversations)
        builder    = ConversationGraphBuilder(registry, sampler, memory, model)
        self.graph = builder.build()

        print("[LangGraph] Graph compiled successfully!")
        print("[LangGraph] Nodes:", ["sampler","planner","user_open","router",
                                      "clarify","call_tool","present","finalize","validate"])

    def _next_pattern(self) -> str:
        p = self._PATTERNS[self._pattern_idx % len(self._PATTERNS)]
        self._pattern_idx += 1
        return p

    def generate_one(self, conv_id: Optional[str] = None) -> dict:
        """
        Run the compiled graph for one conversation.

        .invoke(state) runs the full graph from start to END
        and returns the final state.

        Returns a dict ready for serialize_conversation().
        """
        conv_id = conv_id or str(uuid.uuid4())[:8]

        # Initial state — every field must be initialized
        # (LangGraph will raise errors for missing required fields)
        initial_state: ConversationState = {
            "chain":                   [],
            "conversation_id":         conv_id,
            "pattern_type":            self._next_pattern(),
            "plan":                    {},
            "messages":                [],
            "tool_calls":              [],
            "current_step":            0,
            "clarification_count":     0,
            "memory_grounded_steps":   0,
            "non_first_tool_calls":    0,
            "corpus_memory_enabled":   self.corpus_memory_enabled,
            "seed":                    self.seed,
            "is_valid":                False,
            "validation_issues":       [],
            "pending_clarification":   "",   # set by router, read by clarify_node
        }

        # Run the entire graph — this calls every node in order
        final_state = self.graph.invoke(initial_state)

        # Compute memory grounding rate
        nf  = final_state.get("non_first_tool_calls", 0)
        mgr = (final_state.get("memory_grounded_steps", 0) / nf
               if nf > 0 else None)

        # Convert to the same output format as the original generator
        return {
            "id":           conv_id,
            "messages":     final_state.get("messages", []),
            "tool_calls":   final_state.get("tool_calls", []),
            "is_valid":     final_state.get("is_valid", False),
            "metadata": {
                "seed":                      self.seed,
                "tool_ids_used":             list({
                    tc["tool_name"]
                    for tc in final_state.get("tool_calls", [])
                }),
                "num_turns":                 sum(
                    1 for m in final_state.get("messages", [])
                    if m["role"] in ("user", "assistant")
                ),
                "num_clarification_questions": final_state.get("clarification_count", 0),
                "memory_grounding_rate":       mgr,
                "corpus_memory_enabled":       self.corpus_memory_enabled,
                "pattern_type":                final_state.get("pattern_type", "sequential"),
                "framework":                   "langgraph",  # ← shows we used LangGraph
            },
        }

    def generate_batch(self, count: int) -> list[dict]:
        """Generate `count` valid conversations."""
        results  = []
        attempts = 0
        max_att  = count * 3

        while len(results) < count and attempts < max_att:
            attempts += 1
            try:
                conv = self.generate_one()
                if conv["is_valid"]:
                    results.append(conv)
                    print(f"[LangGraph] ✓ {len(results)}/{count} "
                          f"({len(conv['tool_calls'])} tool calls)")
                else:
                    print(f"[LangGraph] ✗ Invalid: {conv.get('validation_issues','?')}")
            except Exception as e:
                print(f"[LangGraph] Error: {e}")

        return results


# ════════════════════════════════════════════════════════════════
# Internal adapter — bridges LangChain LLM to our executor
# ════════════════════════════════════════════════════════════════

class _LLMAdapter:
    """
    The ToolExecutor expects our custom LocalLLM class.
    This adapter wraps a ChatOllama so it works with ToolExecutor
    without changing executor.py at all.

    This is the "adapter pattern" — a common design pattern.
    """
    def __init__(self, chat_llm: ChatOllama):
        self._llm = chat_llm

    def generate_json(self, prompt: str, system: str = "") -> dict:
        return llm_json(self._llm, prompt, system)

    def generate(self, prompt: str, system: str = "") -> str:
        return llm_text(self._llm, prompt, system)
