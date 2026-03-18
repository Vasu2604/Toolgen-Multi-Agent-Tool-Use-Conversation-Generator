"""
generator.py — Conversation Generator (with verbose progress + LLM fallback protection)
"""

import json
import random
import sys
import uuid
from typing import Optional

from toolgen.agents import (
    AssistantAgent, Conversation, Message,
    PlannerAgent, SamplerAgent, UserProxyAgent, ValidatorAgent,
)
from toolgen.execution import SessionState, ToolExecutor
from toolgen.graph import ToolChainSampler
from toolgen.llm import LocalLLM
from toolgen.memory import MemoryStore
from toolgen.registry import ToolRegistry


def _log(msg: str):
    """Print and immediately flush so Cursor terminal shows progress."""
    print(msg, flush=True)
    sys.stdout.flush()


# ── Fallback plan when LLM can't generate one ─────────────────

_FALLBACK_PLANS = [
    {"user_goal": "Plan a trip to Paris",
     "user_persona": "A traveler planning a vacation",
     "context": "Needs flights, hotels, and weather info.",
     "ambiguities": ["dates", "budget"], "domain": "travel"},
    {"user_goal": "Find the best restaurants in Tokyo",
     "user_persona": "A food enthusiast visiting Tokyo",
     "context": "Looking for authentic local dining experiences.",
     "ambiguities": ["cuisine type", "budget"], "domain": "food"},
    {"user_goal": "Check stock prices and convert currency",
     "user_persona": "An investor monitoring markets",
     "context": "Needs financial data and currency conversion.",
     "ambiguities": ["which stocks", "which currencies"], "domain": "finance"},
    {"user_goal": "Get latest tech news and schedule a meeting",
     "user_persona": "A tech professional staying updated",
     "context": "Wants news summaries and calendar management.",
     "ambiguities": ["specific topics", "meeting time"], "domain": "news"},
    {"user_goal": "Find hotels and check the weather at destination",
     "user_persona": "A business traveler",
     "context": "Needs accommodation and weather for a conference.",
     "ambiguities": ["check-in date", "city"], "domain": "travel"},
]

_FALLBACK_OPENINGS = [
    "Hi, I need some help planning my next trip.",
    "Can you help me find some information about my destination?",
    "I'm looking for recommendations and need some data.",
    "Could you help me research a few things?",
    "I need assistance with planning and information gathering.",
]

_plan_idx = 0
_opening_idx = 0

def _get_fallback_plan() -> dict:
    global _plan_idx
    p = _FALLBACK_PLANS[_plan_idx % len(_FALLBACK_PLANS)]
    _plan_idx += 1
    return dict(p)

def _get_fallback_opening() -> str:
    global _opening_idx
    o = _FALLBACK_OPENINGS[_opening_idx % len(_FALLBACK_OPENINGS)]
    _opening_idx += 1
    return o


class ConversationGenerator:

    def __init__(
        self,
        registry: ToolRegistry,
        sampler: ToolChainSampler,
        memory: MemoryStore,
        llm: LocalLLM,
        corpus_memory_enabled: bool = True,
        seed: int = 42,
    ):
        self.registry = registry
        self.memory   = memory
        self.llm      = llm
        self.corpus_memory_enabled = corpus_memory_enabled
        self.seed     = seed

        executor           = ToolExecutor(registry=registry, llm=llm)
        self.sampler_agent = SamplerAgent(sampler=sampler)
        self.planner_agent = PlannerAgent(llm=llm, registry=registry, memory=memory)
        self.user_proxy    = UserProxyAgent(llm=llm)
        self.assistant     = AssistantAgent(llm=llm, executor=executor,
                                            memory=memory, registry=registry)
        self.validator     = ValidatorAgent()
        self.executor      = executor

        self._pattern_cycle = ["sequential", "sequential", "parallel", "mixed"]
        self._pattern_idx   = 0

    def _next_pattern(self) -> str:
        p = self._pattern_cycle[self._pattern_idx % len(self._pattern_cycle)]
        self._pattern_idx += 1
        return p

    def generate_one(self, conv_id: Optional[str] = None) -> Conversation:
        conv_id = conv_id or str(uuid.uuid4())[:8]
        pattern = self._next_pattern()

        _log(f"  [conv:{conv_id}] pattern={pattern}")

        session_state = SessionState()
        conv = Conversation(
            id=conv_id,
            corpus_memory_enabled=self.corpus_memory_enabled,
            seed=self.seed,
            pattern_type=pattern,
        )

        # ── Step 1: Sample tool chain ──────────────────────────
        _log(f"  [conv:{conv_id}] sampling chain...")
        raw_chain = self.sampler_agent.propose_chain(pattern=pattern)
        chain = []
        for item in raw_chain:
            if isinstance(item, list):
                chain.extend(item)
            else:
                chain.append(item)
        if not chain:
            _log(f"  [conv:{conv_id}] empty chain — skipping")
            return conv
        _log(f"  [conv:{conv_id}] chain={chain}")

        # ── Step 2: Plan the scenario ──────────────────────────
        _log(f"  [conv:{conv_id}] planning scenario (LLM call 1)...")
        try:
            plan = self.planner_agent.plan(
                chain=chain,
                corpus_memory_enabled=self.corpus_memory_enabled,
            )
            # If LLM returned empty/error plan, use fallback
            if not plan or not plan.get("user_goal"):
                _log(f"  [conv:{conv_id}] plan empty — using fallback plan")
                plan = _get_fallback_plan()
        except Exception as e:
            _log(f"  [conv:{conv_id}] planner failed ({e}) — using fallback plan")
            plan = _get_fallback_plan()

        _log(f"  [conv:{conv_id}] goal: {plan.get('user_goal','?')[:50]}")

        # ── Step 3: Opening message ────────────────────────────
        _log(f"  [conv:{conv_id}] generating opening (LLM call 2)...")
        try:
            opening = self.user_proxy.generate_opening(plan)
            if not opening or len(opening) < 5:
                opening = _get_fallback_opening()
        except Exception as e:
            _log(f"  [conv:{conv_id}] opening failed ({e}) — using fallback")
            opening = _get_fallback_opening()

        conv.messages.append(Message(role="user", content=opening))
        conv.num_turns += 1

        non_first_tool_calls = 0
        grounded_tool_calls  = 0

        # ── Step 4: Tool call loop ─────────────────────────────
        for step, endpoint_id in enumerate(chain):
            endpoint = self.registry.get_endpoint(endpoint_id)
            if not endpoint:
                _log(f"  [conv:{conv_id}] endpoint not found: {endpoint_id}")
                continue

            _log(f"  [conv:{conv_id}] step {step+1}/{len(chain)}: {endpoint_id}")

            # 4a: Clarification?
            try:
                clarification = self.assistant.should_clarify(conv.messages, plan, endpoint)
            except Exception:
                clarification = None

            if clarification and conv.num_clarification_questions < 2:
                _log(f"  [conv:{conv_id}]   → asking clarification")
                conv.messages.append(Message(role="assistant", content=clarification))
                conv.num_clarification_questions += 1
                conv.num_turns += 1
                try:
                    user_resp = self.user_proxy.respond_to_clarification(clarification, plan)
                    if not user_resp:
                        user_resp = "Yes, please proceed."
                except Exception:
                    user_resp = "Yes, please proceed."
                conv.messages.append(Message(role="user", content=user_resp))
                conv.num_turns += 1

            # 4b: Fill arguments
            _log(f"  [conv:{conv_id}]   → filling args (LLM call)...")
            try:
                args, memory_used = self.assistant.fill_arguments(
                    endpoint=endpoint,
                    messages=conv.messages,
                    session=session_state,
                    plan=plan,
                    step=step,
                )
                # Ensure required params are present even if LLM failed
                for param in endpoint.required_params:
                    if param.name not in args:
                        args[param.name] = f"example_{param.name}"
            except Exception as e:
                _log(f"  [conv:{conv_id}]   → fill_arguments failed ({e}) — using defaults")
                args = {p.name: f"example_{p.name}" for p in endpoint.required_params}
                memory_used = False

            if step > 0:
                non_first_tool_calls += 1
                if memory_used:
                    grounded_tool_calls += 1

            # 4c: Execute tool
            _log(f"  [conv:{conv_id}]   → executing tool (LLM call)...")
            try:
                output = self.executor.execute(
                    endpoint=endpoint,
                    arguments=args,
                    session=session_state,
                    step=step,
                )
            except Exception as e:
                _log(f"  [conv:{conv_id}]   → execution error ({e}) — using fallback output")
                output = self._fallback_output(endpoint)

            # Record
            conv.tool_calls.append({
                "endpoint":  endpoint_id,
                "tool_name": endpoint.tool_name,
                "arguments": args,
                "output":    output,
                "step":      step,
            })
            conv.messages.append(Message(
                role="assistant",
                content=f"[Calling {endpoint_id}]",
                tool_call={"endpoint": endpoint_id, "arguments": args},
            ))
            conv.messages.append(Message(
                role="tool",
                content=json.dumps(output),
                tool_output=output,
            ))

            # Memory write
            try:
                self.memory.add(
                    content=json.dumps({"endpoint": endpoint_id, "output": output}),
                    scope="session",
                    metadata={"conversation_id": conv_id, "step": step, "endpoint": endpoint_id},
                )
            except Exception:
                pass

            # 4d: Present results — SKIP for the last tool to avoid two
            # consecutive [ASSISTANT] messages. generate_final_response (Step 5)
            # closes the conversation instead.
            is_last_step = (step == len(chain) - 1)

            if not is_last_step:
                _log(f"  [conv:{conv_id}]   → presenting results (LLM call)...")
                try:
                    result_msg = self.assistant.present_results(endpoint, output, plan)
                    if not result_msg:
                        result_msg = f"I found results from {endpoint.name}. Here's what I got."
                except Exception:
                    result_msg = f"I found results from {endpoint.name}. Here's what I got."

                conv.messages.append(Message(role="assistant", content=result_msg))
                conv.num_turns += 1

                try:
                    user_fu = self.user_proxy.respond_to_result(result_msg, plan)
                    if not user_fu:
                        user_fu = "Thanks! Can you continue?"
                except Exception:
                    user_fu = "Thanks! Can you continue?"
                conv.messages.append(Message(role="user", content=user_fu))
                conv.num_turns += 1

        # ── Step 5: Final response (single closing assistant message) ───
        if conv.tool_calls:
            _log(f"  [conv:{conv_id}] generating final response (LLM call)...")
            try:
                final = self.assistant.generate_final_response(
                    plan=plan, tool_calls=conv.tool_calls, messages=conv.messages,
                )
                if not final:
                    final = "I've gathered all the information you needed. Let me know if you need anything else!"
            except Exception:
                final = "I've gathered all the information you needed. Let me know if you need anything else!"
            conv.messages.append(Message(role="assistant", content=final))

        # ── Step 6: Memory grounding rate ─────────────────────
        conv.memory_grounding_rate = (
            grounded_tool_calls / non_first_tool_calls
            if non_first_tool_calls > 0 else None
        )

        # ── Step 7: Corpus memory ──────────────────────────────
        if self.corpus_memory_enabled and conv.tool_calls:
            tools_used   = list({tc["tool_name"] for tc in conv.tool_calls})
            summary_text = (
                f"Tools: {', '.join(tools_used)}. "
                f"Domain: {plan.get('domain', 'general')}. "
                f"Pattern: {pattern}. "
                f"Goal: {plan.get('user_goal', '')[:80]}."
            )
            try:
                self.memory.add(
                    content=summary_text,
                    scope="corpus",
                    metadata={"conversation_id": conv_id,
                              "tools": tools_used, "pattern_type": pattern},
                )
            except Exception:
                pass

        conv.tool_ids_used = list({tc["tool_name"] for tc in conv.tool_calls})
        _log(f"  [conv:{conv_id}] DONE — {len(conv.tool_calls)} tool calls, {conv.num_turns} turns")
        return conv

    def _fallback_output(self, endpoint) -> dict:
        """Always-valid fallback tool output when LLM mock generation fails."""
        output = {"status": "success"}
        for field in (endpoint.response_fields or []):
            if "id" in field.lower():
                output[field] = f"mock_{field}_001"
            elif "price" in field.lower() or "rate" in field.lower():
                output[field] = 99.99
            elif "name" in field.lower():
                output[field] = "Mock Result"
            elif "available" in field.lower():
                output[field] = True
            elif "temp" in field.lower():
                output[field] = 22
            else:
                output[field] = f"mock_{field}"
        return output

    def generate_batch(self, count: int) -> list[Conversation]:
        """Generate `count` valid conversations with detailed progress."""
        conversations = []
        attempts      = 0
        max_attempts  = count * 3

        _log(f"\n[Generator] Starting batch: need {count} valid conversations")
        _log(f"[Generator] Max attempts: {max_attempts}")
        _log(f"[Generator] Validator requires: ≥3 tool calls, ≥2 distinct tools\n")

        while len(conversations) < count and attempts < max_attempts:
            attempts += 1
            _log(f"\n[Generator] ── Attempt {attempts}/{max_attempts} "
                 f"(have {len(conversations)}/{count}) ──")
            try:
                conv     = self.generate_one()
                is_valid, issues = self.validator.validate(conv)
                if is_valid:
                    conversations.append(conv)
                    _log(f"[Generator] ✓ VALID #{len(conversations)}: "
                         f"{len(conv.tool_calls)} tool calls, "
                         f"{len(set(tc['tool_name'] for tc in conv.tool_calls))} tools, "
                         f"{conv.num_turns} turns, "
                         f"mgr={conv.memory_grounding_rate}")
                else:
                    _log(f"[Generator] ✗ INVALID: {issues}")
            except Exception as e:
                _log(f"[Generator] ERROR in attempt {attempts}: {e}")
                import traceback
                traceback.print_exc()

        _log(f"\n[Generator] Batch complete: "
             f"{len(conversations)} valid / {attempts} attempts")
        return conversations


def serialize_conversation(conv: Conversation) -> dict:
    return {
        "id": conv.id,
        "messages": [
            {
                "role":    m.role,
                "content": m.content,
                **({"tool_call":   m.tool_call}   if m.tool_call   else {}),
                **({"tool_output": m.tool_output} if m.tool_output else {}),
            }
            for m in conv.messages
        ],
        "tool_calls": conv.tool_calls,
        "metadata": {
            "seed":                      conv.seed,
            "tool_ids_used":             conv.tool_ids_used,
            "num_turns":                 conv.num_turns,
            "num_clarification_questions": conv.num_clarification_questions,
            "memory_grounding_rate":     conv.memory_grounding_rate,
            "corpus_memory_enabled":     conv.corpus_memory_enabled,
            "pattern_type":              conv.pattern_type,
        },
    }