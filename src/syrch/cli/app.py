from __future__ import annotations

import logging
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from syrch.core.config import ExecutionConfig, LLMConfig, merge_config
from syrch.core.logging import LogConfig, setup_logging
from syrch.core.models import FinalSolution, ProblemSpec
from syrch.eval.report import export_markdown, print_eval_result, print_benchmark_report, export_report
from syrch.eval.runner import (
    BenchmarkProblem,
    _create_executor,
    load_benchmark,
    run_benchmark,
    run_single,
)
from syrch.executors.cached_executor import CachedExecutor
from syrch.executors.sqlite_executor import SQLiteExecutor
from syrch.llm.base import BaseLLM
from syrch.llm.cache import CachedLLM, CentralCache
from syrch.llm.openai_llm import OpenAILLM
from syrch.llm.anthropic_llm import AnthropicLLM
from syrch.search.pipeline import run_pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="syrch",
    help="NL \u2192 ProblemSpec \u2192 Search(D&C+RLM) \u2192 SQL Evaluation \u2192 Optimal Solution",
)
console = Console()


def _create_llm(config: LLMConfig) -> BaseLLM:
    if config.provider == "openai":
        return OpenAILLM(model=config.model, api_key=config.api_key, base_url=config.base_url)
    if config.provider == "anthropic":
        return AnthropicLLM(model=config.model, api_key=config.api_key)
    raise ValueError(f"Unknown LLM provider: {config.provider}")


def _create_cache(config: ExecutionConfig) -> CentralCache | None:
    if not config.cache_enabled:
        return None
    return CentralCache(ttl=config.cache_ttl)


def _build_config(
    question: str,
    db: str,
    executor: str,
    max_depth: int,
    max_attempts: int,
    high_confidence: float,
    budget: int,
    llm_provider: str,
    llm_model: str,
    api_key: str | None,
    base_url: str | None,
    verbose: bool,
    cache_enabled: bool = True,
    cache_ttl: int = 86400,
    interactive: bool = False,
    config_file: str | None = None,
) -> ExecutionConfig:
    return merge_config(
        cli_overrides=dict(
            question=question,
            db_path=db,
            executor_type=executor,
            max_depth=max_depth,
            max_attempts_per_node=max_attempts,
            high_confidence=high_confidence,
            token_budget=budget,
            verbose=verbose,
            cache_enabled=cache_enabled,
            cache_ttl=cache_ttl,
            interactive=interactive,
            llm=dict(provider=llm_provider, model=llm_model, api_key=api_key, base_url=base_url),
        ),
        config_file_path=config_file,
    )


@app.command()
def search(
    question: str = typer.Option(..., "-q", "--question", help="Natural language problem"),
    db: str = typer.Option("orders_10dim.sqlite", "--db", help="Database path or connection string"),
    max_depth: int = typer.Option(3, "--max-depth", help="Max D&C recursion depth"),
    executor_type: str = typer.Option("sqlite", "--executor", help="sqlite | databricks | jdbc"),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Max RLM attempts per node"),
    high_confidence: float = typer.Option(0.85, "--high-conf", help="Confidence threshold for greedy stop"),
    budget: int = typer.Option(100_000, "--budget", help="Token budget"),
    llm_provider: str = typer.Option("openai", "--llm", help="openai | anthropic"),
    llm_model: str = typer.Option("qwen3.5-4b-4bit", "--model", help="LLM model name"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY",
                                          help="API key (or OPENAI_API_KEY env var)"),
    base_url: Optional[str] = typer.Option("http://localhost:8000/v1", "--base-url",
                                           help="OpenAI-compatible API base URL"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable LLM/SQL cache"),
    cache_ttl: int = typer.Option(86400, "--cache-ttl", help="Cache TTL in seconds"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Show reasoning traces"),
    interactive: bool = typer.Option(False, "--interactive/--no-interactive",
                                     help="Ask clarification questions when ambiguous"),
    grid: bool = typer.Option(False, "--grid", help="Run grid search over hyperparameters"),
    grid_parallel: bool = typer.Option(True, "--grid-parallel/--grid-sequential", help="Run grid cells in parallel"),
    grid_max_workers: int = typer.Option(3, "--grid-max-workers", help="Max concurrent API calls during grid search"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Solve a problem using D&C + RLM search over SQL execution."""
    if grid:
        from syrch.search.grid import GridSearchConfig, run_grid_search

        problem = BenchmarkProblem(id="grid", question=question, db=db)
        llm_config = LLMConfig(provider=llm_provider, model=llm_model, api_key=api_key, base_url=base_url)
        grid_config = GridSearchConfig(parallel=grid_parallel, max_workers=grid_max_workers)
        report = run_grid_search(problem, llm_config, grid_config)

        console.print(f"[bold]Grid Search Complete[/bold] — {len(report.cells)} cells")
        if report.best:
            console.print(f"[green]Best:[/green] {report.best.params}")
        console.print(f"[dim]Report:[/dim] {report.output_dir}")
        console.print("  [dim]config.json[/dim]  — grid configuration")
        console.print("  [dim]results.json[/dim] — all cell results")
        console.print("  [dim]best.json[/dim]    — best config")
        console.print("  [dim]summary.md[/dim]   — human-readable report")
        return

    log_level = "INFO" if verbose else "WARNING"
    setup_logging(LogConfig(level=log_level))

    config = _build_config(question, db, executor_type, max_depth, max_attempts,
                           high_confidence, budget, llm_provider, llm_model,
                           api_key, base_url, verbose,
                           cache_enabled=cache, cache_ttl=cache_ttl,
                           interactive=interactive,
                           config_file=config_file)

    console.print(f"[bold]syrch[/bold] \u2014 searching: [cyan]{question}[/cyan]")
    console.print(f"  db={db}  executor={executor_type}  max_depth={max_depth}")

    cache_obj = _create_cache(config)
    llm = _create_llm(config.llm)
    if cache_obj:
        llm = CachedLLM(llm, cache_obj, model=config.llm.model, temperature=config.llm.temperature)
    db_executor = _create_executor(executor_type, db)
    if cache_obj:
        db_executor = CachedExecutor(db_executor, cache_obj)

    try:
        schema = db_executor.get_schema()
    except Exception as e:
        console.print(f"[red]Error connecting to database: {e}[/red]")
        raise typer.Exit(1)

    from syrch.search.clarify import find_worst_ambiguity, generate_question

    problem = ProblemSpec(question=question, schema=schema)
    clarification_qa: list[tuple[str, str]] = []

    for _ in range(2):
        with console.status("[bold green]Planning decomposition...[/bold green]"):
            solution, dag, _results = run_pipeline(llm, db_executor, config, problem)

        worst = find_worst_ambiguity(_results)
        if not (interactive and worst and worst[1] > config.ambiguity_threshold):
            break

        worst_id, worst_score = worst
        q = generate_question(llm, problem, dag, worst_id, _results)
        answer = console.input(f"\n[bold yellow]?[/bold yellow] {q}\n> ")
        clarification_qa.append((q, answer))

        problem = ProblemSpec(
            question=f"{problem.question}\n[User clarification: {answer}]",
            schema=schema,
            all_schemas=problem.all_schemas,
        )

    solution.clarification_qa = clarification_qa
    solution.clarified = bool(clarification_qa)

    if verbose:
        _show_plan(dag)

    _show_solution(solution)
    if solution.clarified:
        console.print(f"[dim]Clarified: {len(solution.clarification_qa)} round(s) of user interaction[/dim]")

    if verbose:
        _show_tree(solution)


@app.command()
def eval(
    question: str = typer.Option(..., "-q", "--question", help="Natural language problem"),
    db: str = typer.Option("orders_10dim.sqlite", "--db", help="Database path"),
    expected: Optional[str] = typer.Option(None, "--expected", help="Expected result CSV path"),
    max_depth: int = typer.Option(3, "--max-depth", help="Max D&C recursion depth"),
    executor_type: str = typer.Option("sqlite", "--executor", help="sqlite | databricks | jdbc"),
    max_attempts: int = typer.Option(3, "--max-attempts", help="Max RLM attempts per node"),
    high_confidence: float = typer.Option(0.85, "--high-conf", help="Confidence threshold"),
    budget: int = typer.Option(100_000, "--budget", help="Token budget"),
    llm_provider: str = typer.Option("openai", "--llm", help="openai | anthropic"),
    llm_model: str = typer.Option("qwen3.5-4b-4bit", "--model", help="LLM model name"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY",
                                          help="API key (or OPENAI_API_KEY env var)"),
    base_url: Optional[str] = typer.Option("http://localhost:8000/v1", "--base-url",
                                           help="OpenAI-compatible API base URL"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable LLM/SQL cache"),
    cache_ttl: int = typer.Option(86400, "--cache-ttl", help="Cache TTL in seconds"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Solve and evaluate a problem against expected results."""
    setup_logging(LogConfig(level="WARNING"))

    config = _build_config(question, db, executor_type, max_depth, max_attempts,
                           high_confidence, budget, llm_provider, llm_model,
                           api_key, base_url, verbose=False,
                           cache_enabled=cache, cache_ttl=cache_ttl,
                           config_file=config_file)

    problem = BenchmarkProblem(id="eval", question=question, db=db, expected_data=expected)
    result = run_single(problem, llm_config=config.llm, config_overrides=dict(
        db_path=db, executor_type=executor_type, max_depth=max_depth,
        max_attempts_per_node=max_attempts,
        high_confidence=high_confidence, token_budget=budget,
        cache_enabled=cache, cache_ttl=cache_ttl,
    ))

    print_eval_result(console, result)
    if expected is not None and result.metrics:
        raise typer.Exit(0 if result.metrics.exact_match else 1)


@app.command()
def benchmark(
    file: str = typer.Option(..., "--file", help="Benchmark JSONL path"),
    report: Optional[str] = typer.Option(None, "--report", help="Output report path"),
    report_format: str = typer.Option("md", "--report-format", help="md | json"),
    max_depth: int = typer.Option(3, "--max-depth"),
    max_attempts: int = typer.Option(3, "--max-attempts"),
    high_confidence: float = typer.Option(0.85, "--high-conf"),
    executor_type: str = typer.Option("sqlite", "--executor", help="sqlite | databricks | jdbc"),
    llm_provider: str = typer.Option("openai", "--llm"),
    llm_model: str = typer.Option("qwen3.5-4b-4bit", "--model"),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    base_url: Optional[str] = typer.Option("http://localhost:8000/v1", "--base-url"),
    cache: bool = typer.Option(True, "--cache/--no-cache", help="Enable LLM/SQL cache"),
    cache_ttl: int = typer.Option(86400, "--cache-ttl", help="Cache TTL in seconds"),
    config_file: Optional[str] = typer.Option(None, "--config", help="Path to config file"),
) -> None:
    """Run a benchmark suite from a JSONL file."""
    setup_logging(LogConfig(level="WARNING"))

    console.print(f"[bold]Benchmark:[/bold] {file}")

    problems = load_benchmark(file)
    console.print(f"Loaded [cyan]{len(problems)}[/cyan] problems")

    llm_config = LLMConfig(provider=llm_provider, model=llm_model, api_key=api_key, base_url=base_url)
    overrides = dict(max_depth=max_depth, max_attempts_per_node=max_attempts, high_confidence=high_confidence, executor_type=executor_type, cache_enabled=cache, cache_ttl=cache_ttl)

    results = run_benchmark(problems, llm_config=llm_config, config_overrides=overrides)
    print_benchmark_report(console, results)

    if report:
        if report_format == "md":
            with open(report, "w") as f:
                f.write(export_markdown(results))
        else:
            export_report(results, report)
        console.print(f"Report saved to [cyan]{report}[/cyan]")


@app.command()
def schema(
    db: str = typer.Argument(..., help="Database path"),
    table: Optional[str] = typer.Option(None, "-t", "--table", help="Specific table"),
) -> None:
    """Inspect database schema."""
    executor = SQLiteExecutor(db)
    tables = [table] if table else executor.list_tables()

    for t in tables:
        schema = executor.get_schema(t)
        console.print(f"\n[bold]{schema.name}[/bold]")
        col_table = Table("Column", "Type", "Nullable")
        for col in schema.columns:
            col_table.add_row(col.name, col.type, "Yes" if col.nullable else "No")
        console.print(col_table)

    executor.close()


@app.command()
def config(db: str = typer.Option("orders_10dim.sqlite", "--db", help="Database path")) -> None:
    """Show current configuration defaults."""
    cfg = ExecutionConfig(question="", db_path=db)
    t = Table("Option", "Default")
    t.add_row("max_depth", str(cfg.max_depth))
    t.add_row("max_attempts_per_node", str(cfg.max_attempts_per_node))
    t.add_row("high_confidence", str(cfg.high_confidence))
    t.add_row("token_budget", str(cfg.token_budget))
    t.add_row("executor_type", cfg.executor_type)
    t.add_row("calibration_enabled", str(cfg.calibration_enabled))
    console.print(t)


def _show_plan(dag) -> None:
    tree = Tree("[bold]Decomposition Plan[/bold]")
    for layer_idx, layer in enumerate(dag.topo_layers):
        branch = tree.add(f"[yellow]Layer {layer_idx}[/yellow]")
        for nid in layer:
            node = dag.nodes[nid]
            label = f"[cyan]{nid}[/cyan]: {node.description}"
            if node.depends_on:
                label += f" (depends: {', '.join(node.depends_on)})"
            if node.is_atomic:
                label += " [dim]\u26a1 atomic[/dim]"
            branch.add(label)
    console.print(tree)


def _show_solution(solution: FinalSolution) -> None:
    console.print()
    console.print("[bold green]\u2550\u2550\u2550 Solution \u2550\u2550\u2550[/bold green]")
    console.print(solution.answer)
    console.print()
    console.print(f"[dim]Confidence: {solution.confidence:.2f} | "
                  f"Token cost: {solution.token_cost}[/dim]")
    if solution.sql:
        console.print("[bold]SQL:[/bold]")
        console.print(f"[dim]{solution.sql}[/dim]")


def _show_tree(solution: FinalSolution) -> None:
    if not solution.tree:
        return
    tree = Tree("[bold]Execution Tree[/bold]")
    for res in solution.tree:
        status = "[green]OK[/green]" if res.error is None else f"[red]FAIL: {res.error}[/red]"
        data_info = ""
        if res.data is not None and not res.data.empty:
            data_info = f" ({len(res.data)} rows \u00d7 {len(res.data.columns)} cols)"
        branch = tree.add(f"[cyan]{res.node_id}[/cyan] {status} "
                          f"conf={res.confidence:.2f} cost={res.cost_tokens}{data_info}")
        for path in res.reasoning_paths:
            branch.add(f"path {path.path_id}: conf={path.confidence:.2f} "
                       f"sql={path.sql[:80]}...")
    console.print(tree)


if __name__ == "__main__":
    app()
