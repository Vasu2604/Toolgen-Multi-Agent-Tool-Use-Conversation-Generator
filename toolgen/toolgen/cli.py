"""
cli.py — Command Line Interface

Commands:
  build     — build tool registry and graph from JSON files
  generate  — generate conversations (plain Python or LangGraph)
  validate  — validate a generated dataset
  metrics   — compute evaluation metrics

LangGraph mode:
  python -m toolgen.cli generate --count 10 --use-langgraph

Run A vs Run B diversity experiment:
  python -m toolgen.cli generate --count 20 --no-corpus-memory --output output/run_a.jsonl
  python -m toolgen.cli generate --count 20 --output output/run_b.jsonl
  python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
"""

import json
import os
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))


@click.group()
def cli():
    """Multi-Agent Tool-Use Conversation Generator"""
    pass


# ── build ──────────────────────────────────────────────────────

@cli.command()
@click.option("--data-dir",   default="data/sample_tools", show_default=True)
@click.option("--output-dir", default="artifacts",         show_default=True)
@click.option("--model",      default="llama3.2",          show_default=True)
def build(data_dir, output_dir, model):
    """Build the tool registry and graph index."""
    from toolgen.registry import ToolRegistry
    from toolgen.graph import ToolGraph

    click.echo(f"[build] Loading tools from: {data_dir}")
    os.makedirs(output_dir, exist_ok=True)

    registry = ToolRegistry()
    count = registry.load_from_directory(data_dir)
    if count == 0:
        click.echo(f"[build] ERROR: No tools found in {data_dir}")
        sys.exit(1)

    registry_path = os.path.join(output_dir, "registry.json")
    registry.save(registry_path)
    click.echo(f"[build] Registry saved: {registry_path}")
    click.echo(f"[build] Summary: {json.dumps(registry.summary(), indent=2)}")

    click.echo("[build] Building Tool Graph...")
    graph = ToolGraph(registry)
    graph.save(os.path.join(output_dir, "graph.json"))

    click.echo(f"\n✅ Build complete! Artifacts in: {output_dir}")


# ── generate ───────────────────────────────────────────────────

@cli.command()
@click.option("--count",           default=10,                    show_default=True)
@click.option("--seed",            default=42,                    show_default=True)
@click.option("--output",          default="output/dataset.jsonl",show_default=True)
@click.option("--data-dir",        default="data/sample_tools",   show_default=True)
@click.option("--model",           default="llama3.2",            show_default=True)
@click.option("--no-corpus-memory",is_flag=True, default=False,
              help="Disable corpus memory (Run A)")
@click.option("--use-langgraph",   is_flag=True, default=False,
              help="Use LangGraph multi-agent pipeline instead of plain Python")
def generate(count, seed, output, data_dir, model, no_corpus_memory, use_langgraph):
    """Generate multi-turn tool-use conversations."""
    from toolgen.registry import ToolRegistry
    from toolgen.graph import ToolGraph, ToolChainSampler
    from toolgen.memory import MemoryStore

    corpus_memory_enabled = not no_corpus_memory
    framework = "LangGraph" if use_langgraph else "Plain Python"

    click.echo(f"\n{'='*55}")
    click.echo(f"Generating {count} conversations")
    click.echo(f"Framework:     {framework}")
    click.echo(f"Seed:          {seed}")
    click.echo(f"Model:         {model}")
    click.echo(f"Corpus memory: {'enabled' if corpus_memory_enabled else 'DISABLED'}")
    click.echo(f"Output:        {output}")
    click.echo(f"{'='*55}\n")

    click.echo("[1/4] Loading registry...")
    registry = ToolRegistry()
    registry.load_from_directory(data_dir)

    click.echo("[2/4] Building graph...")
    graph   = ToolGraph(registry)
    sampler = ToolChainSampler(graph=graph, registry=registry, seed=seed)

    click.echo("[3/4] Setting up memory...")
    memory = MemoryStore()

    click.echo(f"[4/4] Starting generation with {framework}...\n")

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)

    if use_langgraph:
        # ── LangGraph pipeline ─────────────────────────────────
        from toolgen.langgraph_generator import LangGraphConversationGenerator

        generator = LangGraphConversationGenerator(
            registry=registry,
            sampler=sampler,
            memory=memory,
            model=model,
            corpus_memory_enabled=corpus_memory_enabled,
            seed=seed,
        )
        conversations = generator.generate_batch(count)

        with open(output, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        click.echo(f"\n✅ Generated {len(conversations)} conversations → {output}")
        click.echo(f"   Framework: LangGraph ⚡")

    else:
        # ── Plain Python pipeline ──────────────────────────────
        from toolgen.llm import LocalLLM
        from toolgen.generator import ConversationGenerator, serialize_conversation

        llm = LocalLLM(model=model)
        generator = ConversationGenerator(
            registry=registry, sampler=sampler, memory=memory, llm=llm,
            corpus_memory_enabled=corpus_memory_enabled, seed=seed,
        )
        conversations = generator.generate_batch(count)

        with open(output, "w") as f:
            for conv in conversations:
                f.write(json.dumps(serialize_conversation(conv)) + "\n")

        click.echo(f"\n✅ Generated {len(conversations)} conversations → {output}")

    if conversations:
        def _get_tool_calls(c):
            if isinstance(c, dict):
                return c.get("tool_calls", [])
            return getattr(c, "tool_calls", [])

        all_tcs = [_get_tool_calls(c) for c in conversations]
        avg_tc  = sum(len(t) for t in all_tcs) / len(conversations)
        click.echo(f"   Average tool calls: {avg_tc:.1f}")


# ── validate ───────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_path")
def validate(dataset_path):
    """Validate a generated JSONL dataset."""
    click.echo(f"\nValidating: {dataset_path}")

    with open(dataset_path) as f:
        records = [json.loads(l) for l in f if l.strip()]

    click.echo(f"Total records: {len(records)}")
    passed, failed, issues_found = 0, 0, []

    for i, r in enumerate(records):
        record_issues = []
        tcs  = r.get("tool_calls", [])
        meta = r.get("metadata", {})

        if len(tcs) < 3:
            record_issues.append(f"Only {len(tcs)} tool calls")
        if len({tc.get("tool_name","") for tc in tcs}) < 2:
            record_issues.append("Fewer than 2 distinct tools")
        if not r.get("messages"):
            record_issues.append("No messages")
        for field in ["seed","num_turns","memory_grounding_rate"]:
            if field not in meta:
                record_issues.append(f"Missing: {field}")

        if record_issues:
            failed += 1
            issues_found += [f"Record {i}: {x}" for x in record_issues]
        else:
            passed += 1

    click.echo(f"\n✅ Passed: {passed}/{len(records)}")
    if failed:
        click.echo(f"❌ Failed: {failed}")
        for issue in issues_found[:20]:
            click.echo(f"   {issue}")


# ── metrics ────────────────────────────────────────────────────

@cli.command()
@click.argument("dataset_path")
@click.option("--compare", default=None,
              help="Second dataset for Run A vs Run B comparison")
def metrics(dataset_path, compare):
    """Compute evaluation metrics."""
    from toolgen.metrics import compute_all_metrics, compare_runs

    click.echo(f"\nMetrics for: {dataset_path}")
    result = compute_all_metrics(dataset_path)
    click.echo(json.dumps(result, indent=2))

    if compare:
        click.echo(f"\n\n── Run A vs Run B Diversity Comparison ──")
        click.echo(f"  Run A (no corpus memory): {dataset_path}")
        click.echo(f"  Run B (corpus memory):    {compare}")
        click.echo(json.dumps(compare_runs(dataset_path, compare), indent=2))


if __name__ == "__main__":
    cli()
