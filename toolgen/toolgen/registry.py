"""
registry.py — Tool Registry

This is the "filing cabinet" of the system.
It loads tool definitions from ToolBench JSON files and stores them
in a clean, normalized format that the rest of the system can use.

ToolBench has slightly inconsistent formatting across different tool files,
so this loader handles missing/extra fields gracefully.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# Data classes — these are the shapes we store
# ──────────────────────────────────────────────

@dataclass
class Parameter:
    """Represents one input parameter of an API endpoint."""
    name: str
    type: str                   # string, integer, number, boolean
    description: str
    required: bool
    default: Optional[str] = None
    enum: list[str] = field(default_factory=list)   # allowed values if restricted


@dataclass
class Endpoint:
    """One callable API endpoint (like a specific function within a tool)."""
    name: str                           # e.g. "get_current_weather"
    tool_name: str                      # parent tool name
    description: str
    method: str                         # GET, POST, etc.
    required_params: list[Parameter] = field(default_factory=list)
    optional_params: list[Parameter] = field(default_factory=list)
    response_fields: list[str] = field(default_factory=list)  # what the output contains

    @property
    def endpoint_id(self) -> str:
        """Unique identifier: tool_name::endpoint_name"""
        return f"{self.tool_name}::{self.name}"

    @property
    def all_params(self) -> list[Parameter]:
        return self.required_params + self.optional_params


@dataclass
class Tool:
    """
    A tool = a collection of related endpoints.
    Think of it as an API provider (e.g. "weather_api" has
    get_current_weather, get_forecast, etc.)
    """
    name: str
    description: str
    category: str
    endpoints: list[Endpoint] = field(default_factory=list)


# ──────────────────────────────────────────────
# Registry class
# ──────────────────────────────────────────────

class ToolRegistry:
    """
    Loads, stores, and provides access to all tools.

    Usage:
        registry = ToolRegistry()
        registry.load_from_directory("data/sample_tools")
        tools = registry.get_all_tools()
        ep = registry.get_endpoint("weather_api::get_current_weather")
    """

    def __init__(self):
        # tool_name -> Tool
        self._tools: dict[str, Tool] = {}
        # endpoint_id -> Endpoint  (for fast lookup)
        self._endpoints: dict[str, Endpoint] = {}

    # ── Loading ──────────────────────────────

    def load_from_directory(self, directory: str) -> int:
        """
        Scan a directory for .json files and load all tools found.
        Returns the number of tools loaded.
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Tool data directory not found: {directory}")

        count = 0
        for json_file in path.rglob("*.json"):
            count += self._load_file(json_file)

        print(f"[Registry] Loaded {count} tools with {len(self._endpoints)} endpoints "
              f"from {directory}")
        return count

    def _load_file(self, filepath: Path) -> int:
        """Load one JSON file. Returns number of tools added."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ToolBench files can be a single tool dict OR a list of tools
        if isinstance(data, dict):
            data = [data]

        count = 0
        for raw_tool in data:
            tool = self._parse_tool(raw_tool)
            if tool:
                self._tools[tool.name] = tool
                for ep in tool.endpoints:
                    self._endpoints[ep.endpoint_id] = ep
                count += 1
        return count

    def _parse_tool(self, raw: dict) -> Optional[Tool]:
        """
        Convert a raw ToolBench dict into our clean Tool dataclass.
        Handles missing fields gracefully.
        """
        # ToolBench uses various field names — try them all
        name = (raw.get("tool_name") or raw.get("name") or raw.get("api_name", "")).strip()
        if not name:
            return None  # skip malformed entries

        description = (raw.get("tool_description") or raw.get("description", "")).strip()
        category = (raw.get("category") or raw.get("tool_category", "general")).strip()

        # endpoints are in "api_list" in standard ToolBench
        raw_endpoints = raw.get("api_list") or raw.get("endpoints") or raw.get("apis", [])

        endpoints = []
        for raw_ep in raw_endpoints:
            ep = self._parse_endpoint(raw_ep, tool_name=name)
            if ep:
                endpoints.append(ep)

        return Tool(name=name, description=description, category=category, endpoints=endpoints)

    def _parse_endpoint(self, raw: dict, tool_name: str) -> Optional[Endpoint]:
        """Parse one endpoint from its raw ToolBench dict."""
        name = (raw.get("name") or raw.get("api_name", "")).strip()
        if not name:
            return None

        description = raw.get("description", "").strip()
        method = raw.get("method", "GET").upper()
        response_fields = raw.get("response_fields", [])

        # Build parameter lists
        required_params = [
            self._parse_param(p, required=True)
            for p in raw.get("required_parameters", [])
        ]
        optional_params = [
            self._parse_param(p, required=False)
            for p in raw.get("optional_parameters", [])
        ]

        return Endpoint(
            name=name,
            tool_name=tool_name,
            description=description,
            method=method,
            required_params=[p for p in required_params if p],
            optional_params=[p for p in optional_params if p],
            response_fields=response_fields,
        )

    def _parse_param(self, raw: dict, required: bool) -> Optional[Parameter]:
        """Parse one parameter."""
        name = raw.get("name", "").strip()
        if not name:
            return None
        return Parameter(
            name=name,
            type=raw.get("type", "string"),
            description=raw.get("description", ""),
            required=required,
            default=raw.get("default"),
            enum=raw.get("enum", []),
        )

    # ── Query methods ──────────────────────────

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def get_endpoint(self, endpoint_id: str) -> Optional[Endpoint]:
        return self._endpoints.get(endpoint_id)

    def get_all_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_all_endpoints(self) -> list[Endpoint]:
        return list(self._endpoints.values())

    def get_endpoints_for_tool(self, tool_name: str) -> list[Endpoint]:
        tool = self._tools.get(tool_name)
        return tool.endpoints if tool else []

    def get_tools_by_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category.lower() == category.lower()]

    def summary(self) -> dict:
        """Return a summary dict for debugging."""
        return {
            "total_tools": len(self._tools),
            "total_endpoints": len(self._endpoints),
            "categories": list({t.category for t in self._tools.values()}),
            "tools": [t.name for t in self._tools.values()],
        }

    def save(self, path: str) -> None:
        """Save registry index as JSON (for the `build` CLI command)."""
        data = []
        for tool in self._tools.values():
            data.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "endpoints": [
                    {
                        "endpoint_id": ep.endpoint_id,
                        "name": ep.name,
                        "description": ep.description,
                        "method": ep.method,
                        "required_params": [p.name for p in ep.required_params],
                        "optional_params": [p.name for p in ep.optional_params],
                        "response_fields": ep.response_fields,
                    }
                    for ep in tool.endpoints
                ],
            })
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Registry] Saved to {path}")
