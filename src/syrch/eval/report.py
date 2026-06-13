from __future__ import annotations

import json
from datetime import datetime

from rich.console import Console
from rich.table import Table

from syrch.eval.runner import BenchmarkResult


def print_eval_result(console: Console, result: BenchmarkResult) -> None:
    console.print()
    if result.error:
        console.print(f"[red]ERROR: {result.error}[/red]")
        return

    console.print("[bold green]═══ Evaluation ═══[/bold green]")
    console.print(f"Question: [cyan]{result.problem.question}[/cyan]")
    console.print(f"Duration: {result.duration:.2f}s")

    m = result.metrics
    if m:
        table = Table("Metric", "Value")
        table.add_row("Confidence", f"{m.solution.confidence:.2f}")
        table.add_row("Token cost", str(m.token_cost))
        if m.expected_provided:
            table.add_row("Exact match", "[green]PASS[/green]" if m.exact_match else "[red]FAIL[/red]")
            table.add_row("Row count match", "[green]PASS[/green]" if m.row_count_match else "[red]FAIL[/red]")
            table.add_row("Column match", "[green]PASS[/green]" if m.column_match else "[red]FAIL[/red]")
            table.add_row("Expected rows", str(m.expected_rows))
        else:
            table.add_row("Exact match", "\u2014")
            table.add_row("Row count match", "\u2014")
            table.add_row("Column match", "\u2014")
            table.add_row("Expected rows", "\u2014")
        table.add_row("Actual rows", str(m.actual_rows))
        table.add_row("Reasoning paths", str(m.num_reasoning_paths))
        table.add_row("Sub-tasks", str(m.num_subtasks))
        console.print(table)

    if result.solution and result.solution.answer:
        console.print()
        console.print("[bold]Answer:[/bold]")
        console.print(result.solution.answer)


def print_benchmark_report(console: Console, results: list[BenchmarkResult]) -> None:
    table = Table(title="Benchmark Report")
    table.add_column("ID", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Exact", justify="center")
    table.add_column("Rows", justify="center")
    table.add_column("Cols", justify="center")
    table.add_column("Conf", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Time", justify="right")

    for r in results:
        if r.error:
            table.add_row(
                r.problem.id, "[red]ERROR[/red]", "\u2014", "\u2014",
                "\u2014", "\u2014", "\u2014", f"{r.duration:.1f}s",
            )
        elif r.metrics:
            m = r.metrics
            if m.expected_provided:
                exact = "[green]\u2713[/green]" if m.exact_match else "[red]\u2717[/red]"
                rows = "[green]\u2713[/green]" if m.row_count_match else "[red]\u2717[/red]"
                cols = "[green]\u2713[/green]" if m.column_match else "[red]\u2717[/red]"
            else:
                exact = rows = cols = "\u2014"
            table.add_row(
                r.problem.id, "[green]OK[/green]", exact, rows, cols,
                f"{m.solution.confidence:.2f}", str(m.token_cost), f"{r.duration:.1f}s",
            )
        else:
            table.add_row(
                r.problem.id, "[yellow]SKIP[/yellow]", "\u2014", "\u2014",
                "\u2014", "\u2014", "\u2014", f"{r.duration:.1f}s",
            )

    console.print(table)

    total = sum(1 for r in results if r.metrics is not None or r.error is not None)
    passed = sum(1 for r in results if r.metrics and r.metrics.expected_provided and r.metrics.exact_match)
    console.print(f"\n[bold]Passed: {passed}/{total}[/bold]")


def export_markdown(results: list[BenchmarkResult]) -> str:
    lines = ["# Benchmark Report", f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    lines.append("| ID | Status | Exact | Rows | Cols | Conf | Tokens | Time |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        if r.error:
            status = "ERROR"
            exact = rows = cols = conf = tokens = "\u2014"
        elif r.metrics:
            m = r.metrics
            status = "OK"
            if m.expected_provided:
                exact = "\u2713" if m.exact_match else "\u2717"
                rows = "\u2713" if m.row_count_match else "\u2717"
                cols = "\u2713" if m.column_match else "\u2717"
            else:
                exact = rows = cols = "\u2014"
            conf = f"{m.solution.confidence:.2f}"
            tokens = str(m.token_cost)
        else:
            status = "SKIP"
            exact = rows = cols = conf = tokens = "\u2014"
        lines.append(f"| {r.problem.id} | {status} | {exact} | {rows} | {cols} | {conf} | {tokens} | {r.duration:.1f}s |")
    total = sum(1 for r in results if r.metrics is not None or r.error is not None)
    passed = sum(1 for r in results if r.metrics and r.metrics.expected_provided and r.metrics.exact_match)
    lines.append("")
    lines.append(f"**Passed:** {passed}/{total}")
    return "\n".join(lines)


def export_report(results: list[BenchmarkResult], path: str) -> None:
    data = {
        "timestamp": datetime.now().isoformat(),
        "results": [
            {
                "id": r.problem.id,
                "question": r.problem.question,
                "duration": r.duration,
                "error": r.error,
                "metrics": r.metrics.to_dict() if r.metrics else None,
            }
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
