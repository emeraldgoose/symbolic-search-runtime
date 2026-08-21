from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import pandas as pd

from syrch.core.config import ExecutionConfig
from syrch.core.models import (
    ColumnSchema,
    NodeResult,
    NodeStatus,
    ParentContext,
    ReplanType,
    ScoredTable,
    TaskDAG,
    TaskNode,
)
from syrch.executors.base import BaseExecutor
from syrch.llm.base import BaseLLM
from syrch.search.planner import compute_layers
from syrch.search.data_probe import ProbeRegistry
from syrch.search.retriever import Retriever
from syrch.search.rlm_engine import RLMAgent

logger = logging.getLogger(__name__)


def _dtype_to_sql_type(dtype: str) -> str:
    """Best-effort mapping of a pandas dtype string to a SQL type label."""
    d = dtype.lower()
    if d.startswith("int"):
        return "INTEGER"
    if d.startswith("float"):
        return "FLOAT"
    if d.startswith("bool"):
        return "BOOLEAN"
    if "datetime" in d or d.startswith("timestamp"):
        return "TIMESTAMP"
    if d.startswith("date"):
        return "DATE"
    return "TEXT"


class Scheduler:
    def __init__(
        self,
        llm: BaseLLM,
        executor: BaseExecutor,
        config: ExecutionConfig,
        agent: RLMAgent | None = None,
        compressed_schemas: list | None = None,
        replan_callback: Callable | None = None,
        retriever: Retriever | None = None,
        all_schemas: list | None = None,
        alias_map: dict[str, list[tuple[str, str, str | None]]] | None = None,
        candidate_pool: list[ScoredTable] | None = None,
        probe_registry: ProbeRegistry | None = None,
    ):
        self.llm = llm
        self.executor = executor
        self.config = config
        self._base_agent = agent or RLMAgent(
            llm, executor, config,
            retriever=retriever, all_schemas=all_schemas,
            alias_map=alias_map, candidate_pool=candidate_pool,
            probe_registry=probe_registry,
        )
        self.retriever = retriever
        self.all_schemas = all_schemas
        self.alias_map = alias_map
        self.candidate_pool = candidate_pool
        self.probe_registry = probe_registry or ProbeRegistry()
        if compressed_schemas is not None:
            self._base_agent.set_compressed_schemas(compressed_schemas)
        self.replan_callback = replan_callback
        max_expected = config.llm.timeout_seconds * config.max_attempts_per_node
        self.node_timeout = max(max_expected * 2, 300)

    def _node_agent(self) -> RLMAgent:
        agent_cls = type(self._base_agent)
        return agent_cls(
            self.llm, self.executor, self.config,
            retriever=self.retriever, all_schemas=self.all_schemas,
            alias_map=self.alias_map, candidate_pool=self.candidate_pool,
            probe_registry=self.probe_registry,
        )

    def run(self, dag: TaskDAG) -> dict[str, NodeResult]:
        results: dict[str, NodeResult] = {}
        total_tokens = 0
        materialized: set[str] = set()

        consumed: set[str] = set()
        replanned_nodes: set[str] = set()

        while True:
            layers = compute_layers(dag.nodes)
            if not layers:
                break
            progressed = False

            for layer_idx, layer in enumerate(layers):
                ready: list[TaskNode] = []
                for nid in layer:
                    node = dag.nodes[nid]
                    if not node.is_atomic:
                        warnings.warn(f"Skipping non-atomic node {nid} in scheduler")
                        continue
                    deps_met = all(dep in results for dep in node.depends_on)
                    if deps_met and nid not in consumed:
                        consumed.add(nid)
                        if self._any_dep_blocked(node, results):
                            if self.config.verbose:
                                logger.info("  [%s] blocked by failed dependency", nid)
                            results[nid] = NodeResult(
                                node_id=nid,
                                data=pd.DataFrame(),
                                sql="",
                                confidence=0.0,
                                status=NodeStatus.BLOCKED,
                                error="Dependency failed or blocked",
                            )
                            progressed = True
                            continue
                        ready.append(node)

                if not ready:
                    continue

                if self.config.verbose:
                    logger.info("Layer %d: dispatching %s", layer_idx, [n.id for n in ready])

                with ThreadPoolExecutor(max_workers=min(len(ready), self.config.max_concurrency)) as pool:
                    future_map: dict = {}
                    for node in ready:
                        agent = self._node_agent()
                        if node._compressed_schemas is not None:
                            agent.set_compressed_schemas(node._compressed_schemas)
                        ctx = self._build_context(node, results)
                        future = pool.submit(agent.solve, node, ctx)
                        future_map[future] = (node, ctx)

                    for future in as_completed(future_map, timeout=self.node_timeout):
                        node, ctx = future_map[future]
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
                        result.had_context = bool(ctx)
                        result.context_used = (
                            bool(ctx) and RLMAgent._extract_context_refs(result.sql or "")
                        )
                        total_tokens += result.cost_tokens
                        progressed = True

                        if (self.config.materialize_context
                                and result.status == NodeStatus.SOLVED
                                and result.data is not None
                                and not result.data.empty):
                            parent_ctx = self._to_parent_context(node.id, result)
                            try:
                                self.executor.materialize_context(parent_ctx)
                                materialized.add(parent_ctx.table_name)
                                parent_ctx.materialized = True
                                if self.config.verbose:
                                    logger.info("  [%s] materialized %s", node.id, parent_ctx.table_name)
                            except Exception as e:
                                logger.warning("  [%s] context materialize failed: %s", node.id, e)

                        if self.config.verbose:
                            status = "OK" if result.error is None else "FAIL"
                            logger.info("  [%s] %s confidence=%.2f tokens=%d",
                                        node.id, status, result.confidence, result.cost_tokens)

                        # Handle STRUCTURAL replan request (DAG change)
                        if (self.replan_callback is not None
                                and result.replan_request is not None
                                and node.id not in replanned_nodes):
                            rtype, reason = result.replan_request
                            if rtype == ReplanType.STRUCTURAL:
                                if self.config.verbose:
                                    logger.info("  [%s] structural replan: %s", node.id, reason[:100])
                                dag = self.replan_callback(dag, node.id, result)
                                dag.topo_layers = compute_layers(dag.nodes)
                                replanned_nodes.add(node.id)
                                consumed.discard(node.id)

                if total_tokens > self.config.token_budget:
                    logger.warning("Token budget exceeded (%d > %d), stopping",
                                   total_tokens, self.config.token_budget)
                    self._drop_materialized(materialized)
                    return results

            if not progressed:
                break

        self._drop_materialized(materialized)
        return results

    def _drop_materialized(self, materialized: set[str]) -> None:
        for name in materialized:
            try:
                self.executor.drop_context(name)
            except Exception as e:
                logger.warning("  context cleanup failed for %s: %s", name, e)

    def _build_context(
        self,
        node: TaskNode,
        results: dict[str, NodeResult],
    ) -> dict[str, ParentContext]:
        """Wrap each dependency's result in a first-class ParentContext.

        The RLM receives explicit column/type/source metadata instead of a bare
        DataFrame preview, so it never mistakes a dependency output for a
        physical table (S3/S4). v0.3.5b: SOLVED dependencies are materialized
        (Executor.materialize_context) into real `_task_context_*` tables the
        dependent task can JOIN; only SOLVED parents are marked materialized.
        """
        ctx: dict[str, ParentContext] = {}
        for dep_id in node.depends_on:
            res = results.get(dep_id)
            if res is None or res.data is None or res.data.empty:
                continue
            parent = self._to_parent_context(dep_id, res)
            parent.materialized = (
                self.config.materialize_context
                and res.status == NodeStatus.SOLVED
            )
            ctx[dep_id] = parent
        return ctx

    @staticmethod
    def _to_parent_context(node_id: str, res: NodeResult) -> ParentContext:
        df = res.data
        columns = [
            ColumnSchema(name=str(c), type=_dtype_to_sql_type(str(dt)))
            for c, dt in df.dtypes.items()
        ]
        return ParentContext(
            node_id=node_id,
            table_name=f"_task_context_{node_id}",
            columns=columns,
            source_tables=sorted(RLMAgent._extract_table_names(res.sql or "")),
            row_count=len(df),
            preview=df.head(3).to_string() if not df.empty else "",
            sql=res.sql or "",
            data=df,
        )

    @staticmethod
    def _any_dep_blocked(
        node: TaskNode,
        results: dict[str, NodeResult],
    ) -> bool:
        """FAILED/BLOCKED dependencies propagate as BLOCKED (no SQL executed)."""
        for dep in node.depends_on:
            dep_res = results.get(dep)
            if dep_res is not None and dep_res.status in (
                NodeStatus.FAILED,
                NodeStatus.BLOCKED,
            ):
                return True
        return False
