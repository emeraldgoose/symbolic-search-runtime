from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

from syrch.core.config import LLMConfig
from syrch.eval.runner import BenchmarkProblem, BenchmarkResult, run_single


_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORTS_DIR = os.path.join(_PROJECT_ROOT, "autoresearch", "reports")


@dataclass
class GridSearchConfig:
    max_depth_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    high_conf_values: list[float] = field(default_factory=lambda: [0.7, 0.85, 0.95])
    max_attempts_values: list[int] = field(default_factory=lambda: [1, 3, 5])
    calibration_values: list[bool] = field(default_factory=lambda: [True, False])
    parallel: bool = True
    max_workers: int = 3

    @property
    def param_grid(self) -> list[dict[str, Any]]:
        keys = ["max_depth", "high_confidence", "max_attempts_per_node", "calibration_enabled"]
        result: list[dict[str, Any]] = []
        for md in self.max_depth_values:
            for hc in self.high_conf_values:
                for ma in self.max_attempts_values:
                    for cal in self.calibration_values:
                        result.append(dict(zip(keys, (md, hc, ma, cal))))
        return result

    @property
    def total_cells(self) -> int:
        return len(self.param_grid)


@dataclass
class GridCellResult:
    params: dict[str, Any]
    result: BenchmarkResult | None = None
    error: str | None = None


@dataclass
class GridSearchReport:
    problem: BenchmarkProblem
    config: GridSearchConfig
    cells: list[GridCellResult]
    best: GridCellResult | None = None
    timestamp: str = ""
    output_dir: str = ""

    def to_dict(self) -> dict:
        return {
            "problem": asdict(self.problem),
            "config": {
                "max_depth_values": self.config.max_depth_values,
                "high_conf_values": self.config.high_conf_values,
                "max_attempts_values": self.config.max_attempts_values,
                "calibration_values": self.config.calibration_values,
                "parallel": self.config.parallel,
                "max_workers": self.config.max_workers,
                "total_cells": self.config.total_cells,
            },
            "cells": [
                {
                    "params": c.params,
                    "metrics": c.result.metrics.to_dict() if c.result and c.result.metrics else None,
                    "error": c.error or (c.result.error if c.result and c.result.error else None),
                    "duration": c.result.duration if c.result else None,
                }
                for c in self.cells
            ],
            "best": {
                "params": self.best.params,
                "metrics": self.best.result.metrics.to_dict() if self.best and self.best.result and self.best.result.metrics else None,
            } if self.best else None,
            "timestamp": self.timestamp,
        }


def _build_overrides(params: dict[str, Any]) -> dict[str, Any]:
    overrides = {
        "max_depth": params["max_depth"],
        "high_confidence": params["high_confidence"],
        "max_attempts_per_node": params["max_attempts_per_node"],
    }
    if "calibration_enabled" in params:
        overrides["calibration_enabled"] = params["calibration_enabled"]
    return overrides


def _run_cell(args: tuple) -> GridCellResult:
    params, problem, llm_config = args
    overrides = _build_overrides(params)
    try:
        result = run_single(problem, llm_config, overrides)
        if result.error:
            return GridCellResult(params=params, result=result, error=result.error)
        return GridCellResult(params=params, result=result)
    except Exception as e:
        return GridCellResult(params=params, error=str(e))


def _pick_best(cells: list[GridCellResult]) -> GridCellResult | None:
    best: GridCellResult | None = None
    for cell in cells:
        if cell.error or not (cell.result and cell.result.metrics):
            continue
        if best is None:
            best = cell
            continue
        assert best.result is not None and best.result.metrics is not None
        assert cell.result is not None and cell.result.metrics is not None
        bm = best.result.metrics
        cm = cell.result.metrics
        if cm.exact_match and not bm.exact_match:
            best = cell
        elif cm.exact_match == bm.exact_match:
            if cm.solution.confidence > bm.solution.confidence:
                best = cell
    return best


def _write_reports(report: GridSearchReport) -> None:
    out = report.output_dir
    data = report.to_dict()

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(data["config"], f, indent=2, default=str)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(data["cells"], f, indent=2, default=str)
    if data.get("best"):
        with open(os.path.join(out, "best.json"), "w") as f:
            json.dump(data["best"], f, indent=2, default=str)

    lines = [
        f"# Grid Search Report — {report.timestamp}",
        "",
        f"**Problem:** {report.problem.question}",
        f"**Database:** {report.problem.db}",
        "",
        "## Config",
        "| Parameter | Values |",
        "|-----------|--------|",
        f"| max_depth | {report.config.max_depth_values} |",
        f"| high_confidence | {report.config.high_conf_values} |",
        f"| max_attempts_per_node | {report.config.max_attempts_values} |",
        f"| calibration_enabled | {report.config.calibration_values} |",
        f"| max_workers | {report.config.max_workers} |",
        f"| total cells | {report.config.total_cells} |",
        "",
        "## Best Configuration",
    ]
    if report.best:
        lines.append("| Param | Value |")
        lines.append("|-------|-------|")
        for k, v in report.best.params.items():
            lines.append(f"| {k} | {v} |")
        if report.best.result and report.best.result.metrics:
            m = report.best.result.metrics
            lines.append("")
            lines.append(f"**Metrics:** exact_match={m.exact_match}, "
                         f"confidence={m.solution.confidence:.2f}, "
                         f"token_cost={m.token_cost}, "
                         f"duration={report.best.result.duration:.1f}s")
    else:
        lines.append("No successful results found.")

    lines.append("")
    lines.append("## All Results")
    lines.append("| Params | exact_match | confidence | token_cost | duration | error |")
    lines.append("|--------|-------------|------------|------------|----------|-------|")
    for cell in report.cells:
        params_str = " ".join(f"{k}={v}" for k, v in cell.params.items())
        err = cell.error or (cell.result.error if cell.result and cell.result.error else None)
        if cell.result and cell.result.metrics:
            m = cell.result.metrics
            lines.append(
                f"| {params_str} | {m.exact_match} | "
                f"{m.solution.confidence:.2f} | {m.token_cost} | "
                f"{cell.result.duration:.1f}s | {err or ''} |"
            )
        else:
            lines.append(f"| {params_str} | | | | | {err or ''} |")

    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def run_grid_search(
    problem: BenchmarkProblem,
    llm_config: LLMConfig,
    grid: GridSearchConfig | None = None,
) -> GridSearchReport:
    if grid is None:
        grid = GridSearchConfig()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(REPORTS_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    param_grid = grid.param_grid
    cells: list[GridCellResult] = []

    if grid.parallel and len(param_grid) > 1:
        cell_args = [(p, problem, llm_config) for p in param_grid]
        with ProcessPoolExecutor(max_workers=grid.max_workers) as executor:
            cells = list(executor.map(_run_cell, cell_args))
    else:
        for params in param_grid:
            overrides = _build_overrides(params)
            try:
                result = run_single(problem, llm_config, overrides)
                cell = GridCellResult(params=params, result=result)
                if result.error:
                    cell.error = result.error
                cells.append(cell)
            except Exception as e:
                cells.append(GridCellResult(params=params, error=str(e)))

    best = _pick_best(cells)
    report = GridSearchReport(
        problem=problem,
        config=grid,
        cells=cells,
        best=best,
        timestamp=timestamp,
        output_dir=output_dir,
    )
    _write_reports(report)
    return report
