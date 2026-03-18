"""
execution.py — Offline Tool Execution Model

When the assistant "calls" a tool, we can't hit a real API.
Instead this module:
  1. Validates that the arguments match the endpoint schema
  2. Generates a realistic-looking fake response using the LLM
  3. Stores results in a session state so later tool calls can reference
     things like hotel_id, flight_id, etc. from earlier steps

Think of it as a "simulator" — like flight simulators that feel real
but don't actually fly planes.
"""

import json
import random
import re
import uuid
from typing import Optional

from toolgen.llm import LocalLLM
from toolgen.registry import Endpoint, ToolRegistry


class ValidationError(Exception):
    """Raised when tool arguments don't match the schema."""
    pass


class SessionState:
    """
    Lightweight in-memory store that remembers results from this conversation.

    When tool A returns a hotel_id, and tool B needs that hotel_id,
    the session state bridges the gap.
    """

    def __init__(self):
        self._data: dict[str, any] = {}  # flat key-value store
        self._history: list[dict] = []   # ordered list of all tool outputs

    def store_output(self, endpoint_id: str, step: int, output: dict):
        """Save a tool output to session."""
        # Flatten all output values into the state for easy lookup
        for key, value in output.items():
            self._data[key] = value
            # Also store with endpoint prefix for disambiguation
            self._data[f"{endpoint_id}__{key}"] = value

        self._history.append({
            "step": step,
            "endpoint_id": endpoint_id,
            "output": output,
        })

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def get_all(self) -> dict:
        return dict(self._data)

    def get_history(self) -> list[dict]:
        return list(self._history)

    def find_ids(self) -> dict:
        """Return all values that look like IDs (contain 'id' in their key)."""
        return {k: v for k, v in self._data.items() if "id" in k.lower()}


class ToolExecutor:
    """
    Validates arguments against the schema and generates mock responses.
    """

    def __init__(self, registry: ToolRegistry, llm: LocalLLM):
        self.registry = registry
        self.llm = llm

    def validate(self, endpoint: Endpoint, arguments: dict) -> list[str]:
        """
        Check that all required params are present and types match.
        Returns a list of error strings (empty = valid).
        """
        errors = []
        for param in endpoint.required_params:
            if param.name not in arguments:
                errors.append(f"Missing required parameter: '{param.name}'")
            else:
                val = arguments[param.name]
                # Basic type checking
                if param.type == "integer" and not isinstance(val, (int, float)):
                    try:
                        int(val)  # can it be converted?
                    except (ValueError, TypeError):
                        errors.append(f"Parameter '{param.name}' should be integer, got {type(val).__name__}")
                if param.enum and str(val) not in [str(e) for e in param.enum]:
                    errors.append(f"Parameter '{param.name}' must be one of {param.enum}")
        return errors

    def execute(self, endpoint: Endpoint, arguments: dict, session: SessionState, step: int) -> dict:
        """
        Validate args, generate a mock response, store it in session.
        Returns the mock response dict.
        """
        # Validate first
        errors = self.validate(endpoint, arguments)
        if errors:
            raise ValidationError(f"Invalid arguments for {endpoint.endpoint_id}: {errors}")

        # Generate a realistic mock response
        response = self._generate_mock_response(endpoint, arguments, session)

        # Store in session for later use
        session.store_output(endpoint.endpoint_id, step, response)

        return response

    def _generate_mock_response(self, endpoint: Endpoint, arguments: dict, session: SessionState) -> dict:
        """
        Ask the LLM to generate a realistic response for this tool call.
        We include the argument values so the response feels grounded.
        """
        # Include session context so responses chain correctly
        session_context = ""
        history = session.get_history()
        if history:
            recent = history[-2:]  # last 2 results
            session_context = f"\n\nPrior tool results in this session: {json.dumps(recent, indent=2)}"

        prompt = f"""Generate a realistic mock API response for this tool call.

Tool: {endpoint.tool_name}
Endpoint: {endpoint.name}
Description: {endpoint.description}
Arguments provided: {json.dumps(arguments, indent=2)}
Expected response fields: {endpoint.response_fields or 'any relevant fields'}
{session_context}

Rules:
- Return a valid JSON object with realistic values
- If response_fields are specified, include ALL of them
- Generate plausible values based on the arguments (e.g., if city is Paris, use Paris-appropriate data)
- Include an ID field (like hotel_id, flight_id, etc.) when appropriate
- If prior tool results exist, make your response reference them naturally (e.g., same city)
- Keep values realistic (prices, times, names should make sense)"""

        try:
            response = self.llm.generate_json(prompt)
            # Ensure IDs are included where expected
            if endpoint.response_fields and any("id" in f.lower() for f in endpoint.response_fields):
                for f in endpoint.response_fields:
                    if "id" in f.lower() and f not in response:
                        response[f] = str(uuid.uuid4())[:8]
            return response
        except Exception as e:
            # Fallback: generate a simple response without LLM
            return self._fallback_response(endpoint, arguments)

    def _fallback_response(self, endpoint: Endpoint, arguments: dict) -> dict:
        """Simple deterministic fallback if LLM fails."""
        response = {}
        for field_name in (endpoint.response_fields or ["result", "status"]):
            if "id" in field_name.lower():
                response[field_name] = f"mock_{field_name}_{random.randint(1000, 9999)}"
            elif "price" in field_name.lower() or "rate" in field_name.lower():
                response[field_name] = round(random.uniform(10, 500), 2)
            elif "name" in field_name.lower():
                response[field_name] = "Mock Result"
            elif "available" in field_name.lower():
                response[field_name] = True
            else:
                response[field_name] = f"mock_{field_name}"

        response["status"] = "success"
        return response
