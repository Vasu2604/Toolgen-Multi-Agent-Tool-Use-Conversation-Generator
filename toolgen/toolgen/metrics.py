"""
metrics.py — Evaluation Metrics

Computes quality and diversity metrics on generated datasets.

Key metrics:
  - coverage: what % of conversations meet the min tool-call requirements
  - diversity: how different the conversations are from each other
    (used for the Run A vs Run B corpus memory comparison)
  - memory_grounding_rate: average across all conversations
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Optional


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Structural metrics ─────────────────────────────────────────

def compute_coverage(records: list[dict]) -> dict:
    """
    What % of conversations meet the minimum requirements:
    - ≥ 3 tool calls
    - ≥ 2 distinct tools
    - Has clarification questions
    """
    total = len(records)
    if total == 0:
        return {}

    has_3_tools = sum(1 for r in records if len(r.get("tool_calls", [])) >= 3)
    has_2_distinct = sum(
        1 for r in records
        if len({tc.get("tool_name", "") for tc in r.get("tool_calls", [])}) >= 2
    )
    has_clarification = sum(
        1 for r in records
        if r.get("metadata", {}).get("num_clarification_questions", 0) > 0
    )

    return {
        "total_conversations": total,
        "pct_with_3plus_tool_calls": round(has_3_tools / total * 100, 1),
        "pct_with_2plus_distinct_tools": round(has_2_distinct / total * 100, 1),
        "pct_with_clarification": round(has_clarification / total * 100, 1),
        "avg_tool_calls": round(
            sum(len(r.get("tool_calls", [])) for r in records) / total, 2
        ),
        "avg_turns": round(
            sum(r.get("metadata", {}).get("num_turns", 0) for r in records) / total, 2
        ),
    }


def compute_memory_grounding(records: list[dict]) -> dict:
    """Average memory grounding rate across conversations that have it."""
    rates = [
        r["metadata"]["memory_grounding_rate"]
        for r in records
        if r.get("metadata", {}).get("memory_grounding_rate") is not None
    ]
    if not rates:
        return {"avg_memory_grounding_rate": None, "conversations_with_memory": 0}
    return {
        "avg_memory_grounding_rate": round(sum(rates) / len(rates), 3),
        "conversations_with_memory": len(rates),
    }


# ── Diversity metrics ─────────────────────────────────────────

def tool_chain_jaccard_dissimilarity(records: list[dict]) -> float:
    """
    Measures how DIFFERENT the tool chains are across conversations.

    For each pair of conversations, compute Jaccard similarity of their tool sets.
    Then dissimilarity = 1 - similarity.
    Average across all pairs.

    A value close to 1.0 = very diverse (all conversations use different tools).
    A value close to 0.0 = very repetitive (all conversations use the same tools).

    This is our primary diversity metric for the Run A vs Run B comparison.
    """
    # Get tool sets for each conversation
    tool_sets = []
    for r in records:
        tools = frozenset(r.get("metadata", {}).get("tool_ids_used", []))
        if tools:
            tool_sets.append(tools)

    if len(tool_sets) < 2:
        return 0.0

    total_dissimilarity = 0.0
    pairs = 0
    for i in range(len(tool_sets)):
        for j in range(i + 1, len(tool_sets)):
            a, b = tool_sets[i], tool_sets[j]
            intersection = len(a & b)
            union = len(a | b)
            if union > 0:
                jaccard_similarity = intersection / union
                total_dissimilarity += (1.0 - jaccard_similarity)
                pairs += 1

    return round(total_dissimilarity / pairs, 4) if pairs > 0 else 0.0


def distinct_ngrams(records: list[dict], n: int = 2) -> float:
    """
    Distinct-N: ratio of unique n-grams to total n-grams in assistant utterances.
    Higher = more diverse language used.

    This captures lexical diversity — are the assistant responses
    saying different things or repeating the same phrases?
    """
    all_ngrams = []
    unique_ngrams = set()

    for r in records:
        for msg in r.get("messages", []):
            if msg.get("role") == "assistant":
                words = msg.get("content", "").lower().split()
                ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
                all_ngrams.extend(ngrams)
                unique_ngrams.update(ngrams)

    if not all_ngrams:
        return 0.0
    return round(len(unique_ngrams) / len(all_ngrams), 4)


def domain_entropy(records: list[dict]) -> float:
    """
    Shannon entropy over the pattern_type distribution.
    Higher entropy = more evenly distributed patterns = more diverse.

    Entropy formula: H = -sum(p * log2(p)) for each pattern type.
    Max entropy = log2(num_patterns).
    """
    patterns = [
        r.get("metadata", {}).get("pattern_type", "sequential")
        for r in records
    ]
    if not patterns:
        return 0.0

    counts = Counter(patterns)
    total = len(patterns)
    entropy = -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
        if count > 0
    )
    return round(entropy, 4)


# ── Main metrics function ──────────────────────────────────────

def compute_all_metrics(path: str) -> dict:
    """Load a JSONL file and compute all metrics."""
    records = load_jsonl(path)
    if not records:
        return {"error": "No records found"}

    return {
        "file": path,
        "coverage": compute_coverage(records),
        "memory": compute_memory_grounding(records),
        "diversity": {
            "tool_chain_jaccard_dissimilarity": tool_chain_jaccard_dissimilarity(records),
            "distinct_bigrams": distinct_ngrams(records, n=2),
            "pattern_entropy": domain_entropy(records),
        },
    }


def compare_runs(path_a: str, path_b: str) -> dict:
    """
    Compare Run A (no corpus memory) vs Run B (corpus memory enabled).
    Reports diversity metrics for both.
    """
    records_a = load_jsonl(path_a)
    records_b = load_jsonl(path_b)

    div_a = {
        "tool_chain_jaccard_dissimilarity": tool_chain_jaccard_dissimilarity(records_a),
        "distinct_bigrams": distinct_ngrams(records_a, n=2),
        "pattern_entropy": domain_entropy(records_a),
    }
    div_b = {
        "tool_chain_jaccard_dissimilarity": tool_chain_jaccard_dissimilarity(records_b),
        "distinct_bigrams": distinct_ngrams(records_b, n=2),
        "pattern_entropy": domain_entropy(records_b),
    }

    return {
        "run_a_no_corpus_memory": div_a,
        "run_b_with_corpus_memory": div_b,
        "improvement": {
            k: round(div_b[k] - div_a[k], 4)
            for k in div_a
        },
    }
