from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from syrch.core.config import ExecutionConfig
from syrch.core.models import NodeResult, TaskDAG, TaskNode
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.planner import compute_layers
from syrch.search.rlm_engine import RLMAgent

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(
        self,
        llm: BaseLLM,
        executor: BaseExecutor,
        config: ExecutionConfig,
        agent: RLMAgent | None = None,
        compressed_schemas: list | None = None,
        replan_callback: Callable | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.config = config
        self.agent = agent or RLMAgent(llm, executor, config)
        if compressed_schemas is not None:
            self.agent.set_compressed_schemas(compressed_schemas)
        self.replan_callback = replan_callback
        max_expected = config.llm.timeout_seconds * config.max_attempts_per_node
        self.node_timeout = max(max_expected * 2, 300)

    def run(self, dag: TaskDAG) -> dict[str, NodeResult]:
        results: dict[str, NodeResult] = {}
        total_tokens = 0

        consumed = set()

        for layer_idx, layer in enumerate(dag.topo_layers):
            ready: list[TaskNode] = []
            for nid in layer:
                node = dag.nodes[nid]
                if not node.is_atomic:
                    warnings.warn(f"Skipping non-atomic node {nid} in scheduler")
                    continue
                deps_met = all(dep in results for dep in node.depends_on)
                if deps_met and nid not in consumed:
                    ready.append(node)
                    consumed.add(nid)

            if not ready:
                continue

            if self.config.verbose:
                logger.info("Layer %d: dispatching %s", layer_idx, [n.id for n in ready])

            with ThreadPoolExecutor(max_workers=min(len(ready), self.config.max_concurrency)) as pool:
                future_map: dict = {}
                for node in ready:
                    ctx = self._build_context(node, results)
                    future = pool.submit(self.agent.solve, node, ctx)
                    future_map[future] = node

                for future in as_completed(future_map, timeout=self.node_timeout):
                    node = future_map[future]
                    try:
                        result = future.result(timeout=self.node_timeout)
                    except TimeoutError:
                        result = NodeResult(
                            node_id=node.id,
                            data=None,
                            sql="",
                            confidence=0.0,
                            error=f"Node timed out after {self.node_timeout}s",
                        )
                    except Exception as e:
                        result = NodeResult(
                            node_id=node.id,
                            data=None,
                            sql="",
                            confidence=0.0,
                            error=str(e),
                        )
                    results[node.id] = result
                    total_tokens += result.cost_tokens

                    if self.config.verbose:
                        status = "OK" if result.error is None else "FAIL"
                        logger.info("  [%s] %s confidence=%.2f tokens=%d",
                                    node.id, status, result.confidence, result.cost_tokens)

                    # Handle replan request
                    if self.replan_callback is not None and result.replan_request is not None:
                        if self.config.verbose:
                            logger.info("  [%s] replan requested: %s", node.id, result.replan_request[:100])
                        dag = self.replan_callback(dag, node.id, result)
                        dag.topo_layers = compute_layers(dag.nodes)

                    self._handle_alternative_joins(dag, node, results, consumed)

            if total_tokens > self.config.token_budget:
                logger.warning("Token budget exceeded (%d > %d), stopping",
                               total_tokens, self.config.token_budget)
                break

        return results

    def _build_context(
        self,
        node: TaskNode,
        results: dict[str, NodeResult],
    ) -> dict[str, NodeResult]:
        ctx: dict[str, NodeResult] = {}
        for dep_id in node.depends_on:
            if dep_id in results:
                ctx[dep_id] = results[dep_id]
        return ctx

    def _handle_alternative_joins(
        self,
        dag: TaskDAG,
        completed_node: TaskNode,
        results: dict[str, NodeResult],
        consumed: set[str],
    ) -> None:
        for nid, node in dag.nodes.items():
            if node.join_type == "any_of" and nid not in results:
                deps_met = all(dep in results for dep in node.depends_on)
                if deps_met:
                    any_success = any(
                        results[dep].error is None and results[dep].confidence > 0
                        for dep in node.depends_on
                        if dep in results
                    )
                    if any_success:
                        consumed.add(nid)
                        results[nid] = NodeResult(
                            node_id=nid,
                            data=None,
                            sql="",
                            confidence=1.0,
                            error=None,
                        )
