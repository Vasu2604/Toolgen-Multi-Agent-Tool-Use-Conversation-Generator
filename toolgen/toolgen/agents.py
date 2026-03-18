"""
agents.py — The 5 Multi-Agent System

These 5 agents work together like a team to generate one conversation:

  1. SamplerAgent    → "What tools should we use?" (uses Tool Graph)
  2. PlannerAgent    → "What's the user's goal?" (creates the scenario)
  3. UserProxyAgent  → "Plays the user" (generates user messages)
  4. AssistantAgent  → "Plays the AI assistant" (makes tool calls, responds)
  5. ValidatorAgent  → "Quality check" (is this conversation good enough?)

The flow:
  Sampler picks tools → Planner creates scenario → UserProxy + Assistant
  have a back-and-forth → Validator checks it → Done!
"""

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

from toolgen.execution import SessionState, ToolExecutor, ValidationError
from toolgen.graph import ToolChainSampler
from toolgen.llm import LocalLLM
from toolgen.memory import MemoryStore
from toolgen.registry import Endpoint, ToolRegistry


# ──────────────────────────────────────────────────
# Data classes for a conversation
# ──────────────────────────────────────────────────

@dataclass
class Message:
    role: str   # "user" | "assistant" | "tool"
    content: str
    tool_call: Optional[dict] = None   # if assistant, the tool call made
    tool_output: Optional[dict] = None  # if tool, the result


@dataclass
class Conversation:
    id: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)   # all tool calls made
    tool_ids_used: list[str] = field(default_factory=list)
    num_turns: int = 0
    num_clarification_questions: int = 0
    memory_grounding_rate: Optional[float] = None
    corpus_memory_enabled: bool = False
    seed: int = 42
    pattern_type: str = "sequential"


# ──────────────────────────────────────────────────
# Agent 1: Sampler Agent
# ──────────────────────────────────────────────────

class SamplerAgent:
    """
    Uses the Tool Graph to propose a realistic tool chain.
    This is the agent that decides WHICH tools will be used in the conversation.
    """

    def __init__(self, sampler: ToolChainSampler):
        self.sampler = sampler

    def propose_chain(self, pattern: str = "sequential") -> list[str]:
        """
        Returns an ordered list of endpoint_ids.
        e.g. ["weather_api::get_current_weather", "flight_search::search_flights"]
        """
        chain = self.sampler.sample_chain(pattern=pattern)
        print(f"[Sampler] Proposed chain ({pattern}): {chain}")
        return chain


# ──────────────────────────────────────────────────
# Agent 2: Planner Agent
# ──────────────────────────────────────────────────

class PlannerAgent:
    """
    Given a tool chain, creates a realistic user scenario.

    Example: given [weather_api, flight_search, hotel_booking],
    it might plan: "User wants to plan a vacation to Tokyo — check weather
    there, find flights from NYC, and search for hotels."

    It also reads Corpus Memory to avoid duplicating past conversations.
    """

    def __init__(self, llm: LocalLLM, registry: ToolRegistry, memory: MemoryStore):
        self.llm = llm
        self.registry = registry
        self.memory = memory

    def plan(self, chain: list[str], corpus_memory_enabled: bool = True) -> dict:
        """
        Create a conversation plan.

        Returns dict with:
          user_goal: str         — What the user wants
          user_persona: str      — Brief description of the user
          context: str           — Any relevant background
          ambiguities: list[str] — Things that are unclear and need clarification
        """
        # Describe the tools in the chain (flatten in case of parallel batches)
        tool_descriptions = []
        for ep_id in chain:
            if isinstance(ep_id, list):
                ep_id = ep_id[0]  # just use first of parallel batch for description
            ep = self.registry.get_endpoint(ep_id)
            if ep:
                tool_descriptions.append(f"- {ep.endpoint_id}: {ep.description}")

        tools_text = "\n".join(tool_descriptions)

        # Read corpus memory to diversify
        corpus_context = ""
        if corpus_memory_enabled:
            prior_convos = self.memory.search(
                query=tools_text[:200],
                scope="corpus",
                top_k=3,
            )
            if prior_convos:
                summaries = "\n".join(f"- {e['content']}" for e in prior_convos)
                corpus_context = f"""
[Prior conversations in corpus — AVOID duplicating these scenarios]
{summaries}

"""

        prompt = f"""{corpus_context}Given the following tool chain, create a realistic and specific user scenario.

Available tools in this conversation:
{tools_text}

Create a scenario where a user would naturally need to use ALL of these tools in sequence.
Make the scenario specific and interesting (not generic). The user should have a concrete goal.

Return JSON with exactly these fields:
{{
  "user_goal": "one sentence describing what the user wants to achieve",
  "user_persona": "brief description: age, profession, situation",
  "context": "2-3 sentences of background context",
  "ambiguities": ["list", "of", "things", "that", "are", "unclear"],
  "domain": "travel/food/finance/news/productivity/etc"
}}"""

        try:
            plan = self.llm.generate_json(prompt)
        except Exception:
            # Fallback plan
            plan = {
                "user_goal": f"Use {', '.join(chain[:2])} to accomplish a task",
                "user_persona": "A regular user looking for help",
                "context": "The user needs assistance with a multi-step task.",
                "ambiguities": ["location", "date"],
                "domain": "general",
            }

        print(f"[Planner] Plan created: {plan.get('user_goal', 'N/A')[:80]}")
        return plan


# ──────────────────────────────────────────────────
# Agent 3: User Proxy Agent
# ──────────────────────────────────────────────────

class UserProxyAgent:
    """
    Simulates user messages throughout the conversation.

    - Generates the opening message
    - Responds to clarification questions from the assistant
    - Responds to tool results when the assistant presents them
    """

    def __init__(self, llm: LocalLLM):
        self.llm = llm

    def generate_opening(self, plan: dict) -> str:
        """Generate the user's first message based on the plan."""
        prompt = f"""You are a user with this goal: {plan['user_goal']}
Your persona: {plan['user_persona']}
Context: {plan['context']}

Write a natural, conversational first message to an AI assistant.
- Be realistic — don't be overly formal
- Don't list everything explicitly; leave some things vague (this creates room for clarification)
- 1-3 sentences max
- Write ONLY the message, nothing else"""

        try:
            return self.llm.generate(prompt).strip().strip('"')
        except Exception:
            return f"Hi, I need help with: {plan['user_goal']}"

    def respond_to_clarification(self, question: str, plan: dict) -> str:
        """Generate a user response to an assistant's clarification question."""
        prompt = f"""You are a user with this goal: {plan['user_goal']}
Your persona: {plan['user_persona']}
Context: {plan['context']}

The assistant asked you: "{question}"

Write a natural, helpful response that provides the requested information.
Keep it brief (1-2 sentences). Write ONLY the response, nothing else."""

        try:
            return self.llm.generate(prompt).strip().strip('"')
        except Exception:
            return "Let me clarify — " + plan.get("context", "")[:100]

    def respond_to_result(self, assistant_message: str, plan: dict) -> str:
        """Generate a follow-up user message after the assistant presents results."""
        prompt = f"""You are a user with this goal: {plan['user_goal']}

The assistant said: "{assistant_message[:300]}"

Write a brief natural follow-up message. You might:
- Confirm you're happy with the results
- Ask a follow-up question
- Request more details about one item
Keep it to 1-2 sentences. Write ONLY the message."""

        try:
            return self.llm.generate(prompt).strip().strip('"')
        except Exception:
            return "That looks great, thank you!"


# ──────────────────────────────────────────────────
# Agent 4: Assistant Agent
# ──────────────────────────────────────────────────

class AssistantAgent:
    """
    The main AI assistant that:
    - Decides whether to ask for clarification or call a tool
    - Generates tool call arguments
    - Uses session memory to ground arguments in prior results
    - Presents results to the user
    """

    def __init__(self, llm: LocalLLM, executor: ToolExecutor, memory: MemoryStore, registry: ToolRegistry):
        self.llm = llm
        self.executor = executor
        self.memory = memory
        self.registry = registry

    def should_clarify(self, messages: list[Message], plan: dict, endpoint: Endpoint) -> Optional[str]:
        """
        Decide if we need to ask the user for clarification before calling the tool.
        Returns a clarification question string, or None if no clarification needed.

        Clarification is needed when:
        - The user hasn't provided a required parameter
        - The intent is ambiguous
        """
        # Check what required params we have from the conversation
        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in messages[-4:]  # last 4 messages
        )
        required_params = [p.name for p in endpoint.required_params]

        if not required_params:
            return None  # no required params = no need to clarify

        prompt = f"""Look at this conversation and decide if there's a clarification question to ask.

Conversation so far:
{conversation_text}

Next tool to call: {endpoint.name} ({endpoint.description})
Required parameters: {required_params}

If any required parameter is missing or ambiguous, write ONE specific clarification question.
If all required info is already in the conversation, write "NONE".

Write ONLY the question or "NONE"."""

        try:
            response = self.llm.generate(prompt).strip()
            if "NONE" in response.upper() or len(response) > 200:
                return None
            return response
        except Exception:
            return None

    def fill_arguments(self, endpoint: Endpoint, messages: list[Message],
                       session: SessionState, plan: dict, step: int) -> dict:
        """
        Generate the argument values for a tool call.
        Uses session memory to ground arguments in prior results.
        """
        # --- Memory: read path ---
        # Before filling args (if not the first step), query session memory
        memory_context = ""
        memory_used = False
        if step > 0:
            prior_results = self.memory.search(
                query=f"{endpoint.name} {endpoint.description}",
                scope="session",
                top_k=3,
            )
            if prior_results:
                memory_used = True
                memory_context = "\n[Memory context]\n"
                for entry in prior_results:
                    memory_context += f"- {entry['content'][:300]}\n"
                memory_context += "\n"

        # Build the schema description for the LLM
        conversation_text = "\n".join(
            f"{m.role}: {m.content}" for m in messages[-6:]
        )
        param_descriptions = []
        for p in endpoint.required_params:
            param_descriptions.append(f"  - {p.name} (REQUIRED, {p.type}): {p.description}")
        for p in endpoint.optional_params[:3]:  # limit optional params shown
            param_descriptions.append(f"  - {p.name} (optional, {p.type}): {p.description}")

        params_text = "\n".join(param_descriptions)

        prompt = f"""{memory_context}Given the above context and the current tool schema, fill in the arguments for {endpoint.name}.

User goal: {plan.get('user_goal', '')}

Conversation:
{conversation_text}

Tool schema for {endpoint.name}:
{params_text}

Return a JSON object with the argument values.
Use information from the conversation and memory context.
Make up reasonable values for anything not explicitly mentioned.
Return ONLY the JSON arguments object."""

        try:
            args = self.llm.generate_json(prompt)
        except Exception:
            # Fallback: fill required params with dummy values
            args = {}
            for p in endpoint.required_params:
                if p.type == "integer":
                    args[p.name] = 1
                elif p.type == "number":
                    args[p.name] = 1.0
                else:
                    args[p.name] = f"example_{p.name}"

        return args, memory_used

    def present_results(self, endpoint: Endpoint, tool_output: dict, plan: dict) -> str:
        """Generate a human-readable response presenting the tool results."""
        prompt = f"""You are a helpful assistant. Present these tool results naturally to the user.

Tool called: {endpoint.name}
User goal: {plan.get('user_goal', '')}
Tool result: {json.dumps(tool_output, indent=2)[:600]}

Write a natural, helpful response that:
- Summarizes the key findings
- Highlights the most relevant information for the user's goal
- Is conversational (not a bullet-point list)
- Is 2-4 sentences

Write ONLY the response."""

        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return f"I found some results from {endpoint.name}. Here's what I got: {str(tool_output)[:200]}"

    def generate_final_response(self, plan: dict, tool_calls: list[dict], messages: list[Message]) -> str:
        """Generate the assistant's final summary response."""
        tool_summary = "\n".join(
            f"- Called {tc['endpoint']} and got: {str(tc['output'])[:150]}"
            for tc in tool_calls
        )

        prompt = f"""You are a helpful assistant. The user wanted: {plan.get('user_goal', '')}

You made these tool calls and got results:
{tool_summary}

Write a concise, helpful final summary for the user.
- Bring together all the information
- Answer their original goal
- Be conversational
- 3-5 sentences max

Write ONLY the final response."""

        try:
            return self.llm.generate(prompt).strip()
        except Exception:
            return "I've gathered all the information you needed. Let me know if you have any more questions!"


# ──────────────────────────────────────────────────
# Agent 5: Validator Agent
# ──────────────────────────────────────────────────

class ValidatorAgent:
    """
    Checks whether a generated conversation meets quality requirements.

    Requirements (from the task spec):
    - At least 3 tool calls total
    - At least 2 distinct tools used
    - At least one clarification question (for ambiguous scenarios)
    """

    def validate(self, conv: Conversation) -> tuple[bool, list[str]]:
        """
        Returns (is_valid, list_of_issues).
        is_valid = True means the conversation is good enough to keep.
        """
        issues = []

        num_tool_calls = len(conv.tool_calls)
        num_distinct_tools = len(set(tc.get("tool_name", "") for tc in conv.tool_calls))

        if num_tool_calls < 3:
            issues.append(f"Only {num_tool_calls} tool calls (need ≥ 3)")

        if num_distinct_tools < 2:
            issues.append(f"Only {num_distinct_tools} distinct tools (need ≥ 2)")

        if not conv.messages:
            issues.append("No messages in conversation")

        # Check that tool calls have required fields
        for i, tc in enumerate(conv.tool_calls):
            if "endpoint" not in tc:
                issues.append(f"Tool call {i} missing 'endpoint' field")
            if "arguments" not in tc:
                issues.append(f"Tool call {i} missing 'arguments' field")
            if "output" not in tc:
                issues.append(f"Tool call {i} missing 'output' field")

        is_valid = len(issues) == 0
        return is_valid, issues
