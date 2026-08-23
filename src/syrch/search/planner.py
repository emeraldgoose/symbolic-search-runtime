from __future__ import annotations

import logging

from syrch.core.models import (
    FilterRequirement, JoinKey, MetricRequirement, NodeResult, ProblemSpec,
    RequirementSpec, SelectionReason, SupportingRelation, TableSchema, TaskDAG, TaskNode,
)

from syrch.core.config import ExecutionConfig
from syrch.llm.base import BaseLLM
from syrch.search.retriever import Retriever

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """You are a task decomposition planner. Given a problem and a ranked list of candidate tables, decompose it into a DAG of sub-tasks.

Rules:
- Each sub-task must be independently solvable
- Use depends_on to express data dependencies between sub-tasks
- Max recursion depth: {max_depth}
- A sub-task is atomic if a single SQL query can fully and cleanly answer it.
  Prefer decomposition when:
  * The question involves multiple analytical dimensions or drill-downs
  * Intermediate results carry independent insight
  * Multiple aggregation strategies, join combinations, or filtering
    approaches must be explored independently
  Keep it atomic when the question maps directly to a single query.
- Non-atomic sub-tasks will be further decomposed recursively (up to max_depth).
- Sub-tasks must be MECE (Mutually Exclusive, Collectively Exhaustive)
- Every sub-task must have: id, description, depends_on, is_atomic, expected_output.

TABLE SELECTION RULES (CRITICAL):
- Select hint_tables ONLY from the candidate list below.
- The candidates are ranked by relevance score. Higher score = more relevant.
- If you believe a table outside the candidate list is necessary, you MUST explain why.
- Prioritize tables with relevant metrics (revenue, count, etc.) and grain (daily, monthly, etc.).

SEMANTIC SLOTS (fill these for every sub-task):
- hint_tables (REQUIRED): the database table(s) most likely to contain the needed data. From candidates only.
- hint_columns (optional): columns you anticipate needing. The SQL generator will verify actual names.
- metric_columns (optional): columns containing numerical measures/metrics (e.g. "total_amount", "revenue", "count").
- grain (optional): row-level granularity — e.g. "per order", "daily", "per customer", "monthly", "per product".
- time_columns (optional): date/time columns relevant to the question's time filter or trend.

- If a sub-task depends on another and their results should be joined by column,
  specify join_keys. Each join_key has: left, left_col, right, right_col, how.
  Example: "join_keys": [{{"left": "B", "left_col": "customer_id",
                          "right": "A", "right_col": "id", "how": "inner"}}]

REQUIREMENTS (fill for every sub-task):
- requirements.metrics: list of business metric names the SQL must compute (e.g. ["revenue", "count"]).
- requirements.metric_details (optional): structured metric definitions for semantic metrics.
  Each has: name, aggregation ("sum"/"count"/"avg"/null), expression_semantics
  (natural-language meaning of the metric, e.g. "time between payment and shipment").
  IMPORTANT: do NOT put physical column names here — the SQL generator maps the
  semantics to columns. Use this for computed/derived metrics, not simple ones.
- requirements.dimensions: list of dimension columns for GROUP BY (e.g. ["category", "region"]).
- requirements.aggregation: "sum", "count", "avg", or null if no aggregation.
- requirements.grain: "total", "per_order", "daily", "monthly", "per_customer", or null.
- requirements.time_range: [start_date, end_date] as ISO strings if question has a time filter, else null.
- requirements.filters (optional): semantic filters the SQL must apply.
  Each has: semantic (what is excluded/filtered, e.g. "holidays", "refunded orders"),
  predicate (optional expression hint, e.g. "is_holiday = false"), relation
  (optional supporting table that carries the filter, e.g. "dim_date").
  Use only for filters that change the answer set, not for obvious time filters.
- requirements.supporting_relations (optional): helper tables the RLM should join to
  satisfy a filter/grain. Each has: table (from candidates), purpose (why needed),
  join_hint (optional, e.g. "date = sale_date"). NOT answer tables.

Output a JSON object with:
{{
  "subtasks": [
    {{
      "id": "A",
      "description": "clear description of this sub-problem",
      "depends_on": [],
      "is_atomic": true,
      "expected_output": "what data this sub-task produces",
      "hint_tables": ["table_name"],
      "hint_columns": ["column_name"],
      "metric_columns": ["amount", "revenue"],
      "grain": "per order",
      "time_columns": ["order_date"],
      "requirements": {{
        "metrics": ["revenue"],
        "metric_details": [],
        "dimensions": [],
        "aggregation": "sum",
        "grain": "total",
        "time_range": null,
        "filters": [],
        "supporting_relations": []
      }}
    }}
  ]
}}
"""


def compute_layers(nodes: dict[str, TaskNode]) -> list[list[str]]:
    indeg: dict[str, int] = {}
    for nid in nodes:
        indeg.setdefault(nid, 0)
        for dep in nodes[nid].depends_on:
            indeg[nid] = indeg.get(nid, 0) + 1
    layers: list[list[str]] = []
    remaining = set(nodes.keys())
    while remaining:
        layer = [nid for nid in remaining if indeg.get(nid, 0) == 0]
        if not layer:
            break
        layers.append(layer)
        for nid in layer:
            remaining.remove(nid)
            for other in remaining:
                if nid in nodes[other].depends_on:
                    indeg[other] -= 1
    return layers


class Planner:
    def __init__(self, llm: BaseLLM, config: ExecutionConfig, retriever: Retriever | None = None):
        self.llm = llm
        self.config = config
        self.retriever = retriever

    def decompose(self, problem: ProblemSpec) -> TaskDAG:
        dag = self._decompose_level(problem, depth=0)
        effective_max = self._resolve_max_depth(problem)
        self._recursive_expand(dag, depth=0, schemas=problem.all_schemas or [problem.schema],
                               max_depth_override=effective_max)
        pool_names = self._pool_names(problem)
        self._fold_supporting_relation_tasks(dag, pool_names)
        return dag

    @staticmethod
    def _pool_names(problem: ProblemSpec) -> set[str]:
        if problem.evidence and problem.evidence.candidates:
            return {c.schema.name for c in problem.evidence.candidates}
        return {s.name for s in (problem.all_schemas or [problem.schema])}

    @staticmethod
    def _is_simple_question(question: str) -> bool:
        q = question.lower()
        simple_keywords = {"show", "list", "what is", "what are", "how many", "how much",
                           "give me", "find", "tell me", "calculate", "compute", "get"}
        has_simple_prefix = any(q.startswith(k) for k in simple_keywords)
        word_count = len(q.split())
        return has_simple_prefix and word_count < 12

    def _resolve_max_depth(self, problem: ProblemSpec) -> int:
        base = self.config.max_depth
        if self._is_simple_question(problem.question):
            return min(base, 2)
        return base

    MAX_SUBTASKS = 6
    MAX_TOTAL_NODES = 10

    def _decompose_level(self, problem: ProblemSpec, depth: int = 0) -> TaskDAG:
        system = DECOMPOSE_SYSTEM.format(max_depth=self.config.max_depth)
        user = self._build_user_prompt(problem)
        result = self.llm.generate_json(system, user)
        known_columns = self._known_column_names(problem)
        nodes: dict[str, TaskNode] = {}
        for item in result.get("subtasks", [])[:self.MAX_SUBTASKS]:
            if not isinstance(item, dict):
                continue
            node_id = item.get("id")
            if not isinstance(node_id, str):
                continue
            depends_on_raw = item.get("depends_on", [])
            depends_on: list[str] = [d for d in depends_on_raw if isinstance(d, str)]

            join_keys = None
            if isinstance(item.get("join_keys"), list):
                join_keys = []
                for k in item["join_keys"]:
                    if isinstance(k, dict):
                        try:
                            join_keys.append(JoinKey(**k))
                        except Exception:
                            pass

            hint_tables_raw = item.get("hint_tables", [])
            hint_tables: list[str] | None = [t for t in hint_tables_raw if isinstance(t, str)] if isinstance(hint_tables_raw, list) else None
            hint_columns_raw = item.get("hint_columns", [])
            hint_columns: list[str] | None = self._filter_known_columns(
                [c for c in hint_columns_raw if isinstance(c, str)], known_columns
            )
            metric_columns_raw = item.get("metric_columns", [])
            metric_columns: list[str] | None = self._filter_known_columns(
                [c for c in metric_columns_raw if isinstance(c, str)], known_columns
            )
            grain: str | None = item.get("grain") if isinstance(item.get("grain"), str) else None
            time_columns_raw = item.get("time_columns", [])
            time_columns: list[str] | None = [c for c in time_columns_raw if isinstance(c, str)] if isinstance(time_columns_raw, list) else None

            requirements: RequirementSpec | None = None
            req_raw = item.get("requirements")
            if isinstance(req_raw, dict):
                try:
                    time_range = req_raw.get("time_range")
                    if isinstance(time_range, list) and len(time_range) >= 2:
                        tr = (str(time_range[0]), str(time_range[1]))
                    else:
                        tr = None
                    metric_details: list[MetricRequirement] = []
                    for md in req_raw.get("metric_details", []):
                        if isinstance(md, dict) and isinstance(md.get("name"), str):
                            metric_details.append(MetricRequirement(
                                name=md["name"],
                                aggregation=md.get("aggregation") if isinstance(md.get("aggregation"), str) else None,
                                expression_semantics=md.get("expression_semantics") if isinstance(md.get("expression_semantics"), str) else None,
                            ))
                    filters: list[FilterRequirement] = []
                    for fr in req_raw.get("filters", []):
                        if isinstance(fr, dict) and isinstance(fr.get("semantic"), str):
                            filters.append(FilterRequirement(
                                semantic=fr["semantic"],
                                relation=fr.get("relation") if isinstance(fr.get("relation"), str) else None,
                                predicate=fr.get("predicate") if isinstance(fr.get("predicate"), str) else None,
                            ))
                    supporting: list[SupportingRelation] = []
                    for sr in req_raw.get("supporting_relations", []):
                        if isinstance(sr, dict) and isinstance(sr.get("table"), str):
                            supporting.append(SupportingRelation(
                                table=sr["table"],
                                purpose=sr.get("purpose") if isinstance(sr.get("purpose"), str) else None,
                                join_hint=sr.get("join_hint") if isinstance(sr.get("join_hint"), str) else None,
                            ))
                    requirements = RequirementSpec(
                        metrics=req_raw.get("metrics", []),
                        metric_details=metric_details,
                        dimensions=req_raw.get("dimensions", []),
                        aggregation=req_raw.get("aggregation"),
                        grain=req_raw.get("grain"),
                        time_range=tr,
                        must_use_columns=req_raw.get("must_use_columns", []),
                        filters=filters,
                        supporting_relations=supporting,
                    )
                    self._apply_grain_fallback(requirements, item.get("description", ""))
                except Exception:
                    pass

            node = TaskNode(
                id=node_id,
                description=item.get("description", ""),
                depends_on=depends_on,
                depth=depth,
                is_atomic=item.get("is_atomic", True),
                expected_output_desc=item.get("expected_output", ""),
                join_keys=join_keys,
                hint_tables=hint_tables,
                hint_columns=hint_columns,
                metric_columns=metric_columns,
                grain=grain,
                time_columns=time_columns,
                requirements=requirements,
            )
            if hint_tables and problem.evidence and problem.evidence.candidates:
                for rank, ct in enumerate(problem.evidence.candidates, 1):
                    if ct.schema.name in hint_tables:
                        used = []
                        if ct.features.matched_metrics:
                            used.append("metric")
                        if ct.features.matched_keywords:
                            used.append("keyword")
                        if ct.features.matched_grains:
                            used.append("grain")
                        if ct.features.layer != "unknown":
                            used.append("layer")
                        node.selection_reason = SelectionReason(
                            selected_from_candidate=True,
                            top_k_rank=rank,
                            score=ct.score,
                            used_features=used[:5],
                        )
                        break
            nodes[node.id] = node
        if not nodes:
            root = TaskNode(id="A", description=str(problem.question), is_atomic=True, depth=depth)
            nodes["A"] = root
            return TaskDAG(nodes=nodes, root_id="A", topo_layers=[["A"]])
        root_id = self._resolve_root(nodes)
        layers = self._compute_layers(nodes)
        dag = TaskDAG(nodes=nodes, root_id=root_id, topo_layers=layers)
        self._validate(dag)
        return dag

    @staticmethod
    def _apply_grain_fallback(requirements: RequirementSpec, description: str) -> None:
        """S5 guard: deterministic grain extraction when the planner omits it.

        `monthly`/`daily`/`yearly` in the question text are *requirement*
        evidence (structure), not answer-table selection. If the planner did
        not already set a grain, derive it from explicit lexical markers so
        `grain_match` is not silently neutralized by a planner miss.
        """
        if requirements.grain:
            return
        text = description.lower()
        markers = [
            ("monthly", "month"),
            ("per month", "month"),
            ("by month", "month"),
            ("weekly", "week"),
            ("per week", "week"),
            ("by week", "week"),
            ("daily", "day"),
            ("per day", "day"),
            ("by day", "day"),
            ("yearly", "year"),
            ("annual", "year"),
            ("per year", "year"),
            ("by year", "year"),
            ("quarterly", "quarter"),
            ("per quarter", "quarter"),
        ]
        for token, canonical in markers:
            if token in text:
                requirements.grain = canonical
                return

    @staticmethod
    def _is_supporting_relation_task(node: TaskNode) -> bool:
        """True when a node's only role is to enumerate a dimension/lookup
        table (a supporting relation) rather than compute an answer (S15).

        Conservative: requires all hint tables to be lookup-layer (dim_/ref_/
        stg_/code_) AND no answer evidence (no metrics, no metric_details, no
        aggregation, no metric_columns). A task that computes anything, or
        whose hints include an answer table, is never folded.
        """
        hints = node.hint_tables or []
        if not hints:
            return False
        if not all(h.lower().startswith(("dim_", "ref_", "stg_", "code_")) for h in hints):
            return False
        req = node.requirements
        if req and (req.metrics or req.metric_details or req.aggregation):
            return False
        if node.metric_columns:
            return False
        return True

    @staticmethod
    def _add_supporting_relation(node: TaskNode, table: str, purpose: str) -> None:
        if node.requirements is None:
            node.requirements = RequirementSpec(metrics=[], aggregation=None)
        if not any(sr.table == table for sr in node.requirements.supporting_relations):
            node.requirements.supporting_relations.append(
                SupportingRelation(table=table, purpose=purpose)
            )

    @staticmethod
    def _add_exclusion_filter(node: TaskNode, filter_req: FilterRequirement) -> None:
        """Transfer the lookup task's filter *semantic* (e.g. 'holidays') to the
        consumer so the RLM knows what to exclude — without the selection
        predicate (a lookup 'is_holiday = true' must not become an exclusion
        predicate verbatim)."""
        if not filter_req.semantic:
            return
        if node.requirements is None:
            node.requirements = RequirementSpec(metrics=[], aggregation=None)
        if any(f.semantic == filter_req.semantic for f in node.requirements.filters):
            return
        node.requirements.filters.append(
            FilterRequirement(semantic=filter_req.semantic, relation=filter_req.relation, predicate=None)
        )

    def _fold_supporting_relation_tasks(
        self, dag: TaskDAG, pool_names: set[str]
    ) -> TaskDAG:
        """S15: fold lookup/filter tasks into their consumers as supporting
        relations instead of running them as independent DAG nodes.

        A supporting-relation task enumerates a dimension/lookup table (e.g.
        'holiday dates' from dim_date) purely to feed another node's filter.
        Executed independently it produces a fragile empty/lookup result that
        blocks the consumer. Folding gives the consumer the lookup table as a
        JOIN-AVAILABLE supporting relation so the filter becomes part of a
        single relational operation (the GT for S15 is one JOIN query).

        Only tables present in the candidate pool are folded in, so the fold
        can never introduce a supporting relation the RLM cannot reach (which
        would be rejected by the validator as an unmet requirement).
        """
        nodes = dag.nodes
        while True:
            changed = False
            for nid, node in list(nodes.items()):
                if not node.is_atomic:
                    continue
                consumers = [o for o, other in nodes.items() if nid in other.depends_on]
                if not consumers:
                    continue
                if not self._is_supporting_relation_task(node):
                    continue
                tables_in_pool = [t for t in (node.hint_tables or []) if t in pool_names]
                if not tables_in_pool:
                    # None of the lookup tables is reachable by the RLM — keep
                    # the node rather than demand an unreachable relation.
                    continue
                purpose = node.description or "supporting relation"
                filter_reqs = list((node.requirements.filters if node.requirements else []))
                for cid in consumers:
                    consumer = nodes[cid]
                    for table in tables_in_pool:
                        self._add_supporting_relation(consumer, table, purpose)
                    for fr in filter_reqs:
                        self._add_exclusion_filter(consumer, fr)
                    consumer.depends_on = [d for d in consumer.depends_on if d != nid]
                if self.config.verbose:
                    logger.info(
                        "  folding supporting-relation task %s (%s) into %s",
                        nid, ", ".join(node.hint_tables or []), consumers,
                    )
                del nodes[nid]
                changed = True
                break
            if not changed:
                break
        dag.topo_layers = self._compute_layers(nodes)
        dag.root_id = self._resolve_root(nodes)
        self._validate(dag)
        return dag

    def _recursive_expand(self, dag: TaskDAG, depth: int, schemas: list[TableSchema],
                          max_depth_override: int | None = None) -> None:
        if not schemas:
            return
        max_depth = max_depth_override if max_depth_override is not None else self.config.max_depth
        if depth >= max_depth - 1 or len(dag.nodes) >= self.MAX_TOTAL_NODES:
            for node in dag.nodes.values():
                node.is_atomic = True
            return
        for node in list(dag.nodes.values()):
            if node.is_atomic:
                continue
            sub_problem = ProblemSpec(
                question=node.description,
                schema=schemas[0],
                all_schemas=schemas,
            )
            if self.retriever and sub_problem.scored_schemas is None:
                sub_evidence = self.retriever.score(node.description)
                sub_problem.scored_schemas = sub_evidence.candidates
                sub_problem.evidence = sub_evidence
            sub_dag = self._decompose_level(sub_problem, depth=depth + 1)
            self._recursive_expand(sub_dag, depth + 1, schemas)
            self._merge_sub_dag(dag, node, sub_dag)
            if len(dag.nodes) >= self.MAX_TOTAL_NODES:
                break
        dag.topo_layers = self._compute_layers(dag.nodes)

    def _merge_sub_dag(self, dag: TaskDAG, parent_node: TaskNode, sub_dag: TaskDAG) -> None:
        prefix = parent_node.id + "."
        renamed: dict[str, TaskNode] = {}
        id_map: dict[str, str] = {}
        for nid, node in sub_dag.nodes.items():
            new_id = prefix + nid
            id_map[nid] = new_id
            node.id = new_id
            node.depth = parent_node.depth + 1
            node.depends_on = [id_map.get(d, d) for d in node.depends_on]
            if node.join_keys:
                for jk in node.join_keys:
                    jk.left = id_map.get(jk.left, jk.left)
                    jk.right = id_map.get(jk.right, jk.right)
            renamed[new_id] = node

        parent_node._children = list(id_map.values())

        del dag.nodes[parent_node.id]
        dag.nodes.update(renamed)

        sub_leaf_ids = self._find_sub_leaves(renamed)
        for node in list(dag.nodes.values()):
            if parent_node.id in node.depends_on:
                node.depends_on.remove(parent_node.id)
                for leaf in sub_leaf_ids:
                    if leaf not in node.depends_on:
                        node.depends_on.append(leaf)

    def _find_sub_leaves(self, nodes: dict[str, TaskNode]) -> list[str]:
        all_dependents: set[str] = set()
        for node in nodes.values():
            all_dependents.update(node.depends_on)
        return [nid for nid in nodes if nid not in all_dependents]

    @staticmethod
    def _known_column_names(problem: ProblemSpec) -> set[str]:
        """Lower-cased set of every physical column the planner can see.

        Hallucinated planner hints (`metric_columns`, `hint_columns`) must be
        grounded against the real schema before they reach the RLM. A column
        name that does not exist in any candidate (e.g. the model inventing
        `inventory` for a `total_items_sold` table) would otherwise be emitted
        as a hard hint the RLM trusts, then refuses when validation proves it
        absent (S10/S14 empty-SQL failures).
        """
        cols: set[str] = set()
        schemas: list = list(problem.all_schemas or [])
        if problem.schema is not None:
            schemas.append(problem.schema)
        if problem.evidence:
            schemas += [c.schema for c in problem.evidence.candidates]
        for s in schemas:
            for c in s.columns:
                cols.add(c.name.lower())
        return cols

    @classmethod
    def _filter_known_columns(
        cls, names: list[str], known_columns: set[str]
    ) -> list[str] | None:
        if not names:
            return None
        filtered = [n for n in names if n.lower() in known_columns]
        return filtered or None

    def _build_user_prompt(self, problem: ProblemSpec) -> str:
        evidence = problem.evidence

        hint_lines: list[str] = []
        if evidence:
            if evidence.grain_hints:
                hint_lines.append(f"Detected grain/aggregation hints: {', '.join(evidence.grain_hints)}")
            if evidence.metric_hints:
                hint_lines.append(f"Detected metric hints: {', '.join(evidence.metric_hints)}")
            if evidence.time_columns:
                hint_lines.append(f"Time/date columns available: {', '.join(evidence.time_columns[:6])}")

        sections = [f"Question: {problem.question}"]
        if hint_lines:
            sections.append("\n".join(hint_lines))

        if evidence and evidence.candidates:
            table_lines: list[str] = []
            for i, ct in enumerate(evidence.candidates, 1):
                feat = ct.features
                layer_info = f"Layer: {feat.layer.upper()}" if feat.layer and feat.layer != "unknown" else ""
                purpose = ct.schema.description or ""

                keyword_str = ""
                if feat.matched_keywords:
                    keyword_str = f"  Matched keywords: {', '.join(feat.matched_keywords[:5])}"
                col_str = ""
                if feat.representative_columns:
                    col_str = f"  Key columns: {', '.join(feat.representative_columns)}"
                metric_str = ""
                if feat.matched_metrics:
                    metric_str = f"  Metrics: {', '.join(feat.matched_metrics[:3])}"
                grain_str = ""
                if feat.matched_grains:
                    grain_str = f"  Grain: {', '.join(feat.matched_grains[:3])}"

                parts = [
                    f"Candidate {i}: {ct.schema.name} (score: {ct.score:.2f})",
                ]
                if layer_info:
                    parts.append(f"  {layer_info}")
                if purpose:
                    parts.append(f"  Purpose: {purpose}")
                if keyword_str:
                    parts.append(keyword_str)
                if col_str:
                    parts.append(col_str)
                if metric_str:
                    parts.append(metric_str)
                if grain_str:
                    parts.append(grain_str)

                table_lines.append("\n".join(parts))
            sections.append("Candidate tables (ranked by relevance):\n" + "\n\n".join(table_lines))
        else:
            all_schemas = problem.all_schemas or [problem.schema]
            table_lines = []
            for tbl in all_schemas:
                col_parts = []
                for c in tbl.columns:
                    desc = f" ({c.description})" if c.description else ""
                    col_parts.append(f"{c.name} ({c.type}){desc}")
                cols_str = ", ".join(col_parts)
                table_lines.append(f"Table: {tbl.name}\n  Columns: {cols_str}")
            sections.append("Available tables:\n" + "\n\n".join(table_lines))

        sections.append(
            "Decompose this into sub-tasks. Follow the JSON output format exactly. "
            "Each sub-task MUST include hint_tables from the candidate list above."
        )

        return "\n\n".join(sections)

    def _resolve_root(self, nodes: dict[str, TaskNode]) -> str:
        for nid, node in nodes.items():
            if not node.depends_on:
                return nid
        return list(nodes.keys())[0]

    def _compute_layers(self, nodes: dict[str, TaskNode]) -> list[list[str]]:
        return compute_layers(nodes)

    def _validate(self, dag: TaskDAG) -> None:
        for nid, node in dag.nodes.items():
            for dep in node.depends_on:
                if dep not in dag.nodes:
                    raise ValueError(f"Node {nid} depends on unknown node {dep}")
        VISITING, VISITED = 1, 2
        state: dict[str, int] = {}

        def dfs(nid: str) -> None:
            if nid in state:
                if state[nid] == VISITING:
                    raise ValueError(f"Cycle detected involving node {nid}")
                return
            state[nid] = VISITING
            for dep in dag.nodes[nid].depends_on:
                dfs(dep)
            state[nid] = VISITED

        for nid in dag.nodes:
            dfs(nid)

    def replan(
        self,
        dag: TaskDAG,
        failed_node_id: str,
        sql: str,
        error: str,
        node_result: NodeResult,
        scored_schemas: list,
        probe_registry=None,
    ) -> TaskDAG:
        if self.config.verbose:
            logger.info("Planner.replan (STRUCTURAL): node=%s error=%s", failed_node_id, error[:100])
        failed_node = dag.nodes.get(failed_node_id)
        if failed_node is None:
            return dag

        self._drop_infeasible_supporting_relations(failed_node, probe_registry)

        tried = self._collect_tried_tables(failed_node, node_result)
        alternatives = self._suggest_alternative_tables(failed_node, tried, scored_schemas)
        if not alternatives:
            if self.config.verbose:
                logger.info("  no alternative tables found for node %s", failed_node_id)
            return dag

        from syrch.core.models import NodeStatus
        is_ambiguous = node_result.status == NodeStatus.AMBIGUOUS

        if is_ambiguous:
            # AMBIGUOUS: viable candidates exist on current hints — merge to preserve them (S14)
            current = [t for t in (failed_node.hint_tables or []) if t not in alternatives]
            merged = current + alternatives
            if self.config.verbose:
                logger.info("  [AMBIGUOUS] merging hint_tables: %s -> %s", failed_node.hint_tables, merged)
            failed_node.hint_tables = merged[:8]
        else:
            # FAILED/BLOCKED: no viable results — REPLACE hints with fresh alternatives
            if self.config.verbose:
                logger.info("  [FAILED] replacing hint_tables: %s -> %s", failed_node.hint_tables, alternatives)
            failed_node.hint_tables = alternatives[:8]

        failed_node.selection_reason = None
        failed_node._compressed_schemas = None
        return dag

    def _drop_infeasible_supporting_relations(self, node: TaskNode, probe_registry) -> bool:
        """Remove supporting relations whose filter value the probe registry
        already verified does not exist in the table.

        The planner picks supporting relations from schema alone, so it can
        demand a decoy table (e.g. rpt_customer_ltv for 'filter for VIP
        customers') that the probe found to be empty of that value. Such a
        relation can never be satisfied — dropping it lets the node run against
        real candidate tables instead of retrying an impossible JOIN. Returns
        True when at least one relation was removed."""
        if probe_registry is None or node.requirements is None:
            return False
        facts = probe_registry.all()
        if not facts:
            return False
        kept = []
        removed: list[str] = []
        for sr in node.requirements.supporting_relations:
            base = sr.table.split(".")[-1].lower()
            purpose = (sr.purpose or "").lower()
            infeasible = False
            if purpose:
                for fact in facts:
                    if fact.exists:
                        continue
                    if fact.table.split(".")[-1].lower() != base:
                        continue
                    if fact.value.lower() in purpose:
                        infeasible = True
                        break
            if infeasible:
                removed.append(sr.table)
            else:
                kept.append(sr)
        if removed:
            if self.config.verbose:
                logger.info("  dropping infeasible supporting_relations: %s", removed)
            node.requirements.supporting_relations = kept
            return True
        return False

    def _collect_tried_tables(
        self,
        node: TaskNode,
        node_result: NodeResult,
    ) -> set[str]:
        tried: set[str] = set(node.hint_tables or [])
        for path in node_result.reasoning_paths:
            tried.update(self._extract_tables(path.sql))
        return tried

    @staticmethod
    def _extract_tables(sql: str) -> list[str]:
        if not sql:
            return []
        try:
            from sqlglot import parse_one
            from sqlglot.expressions import Table
            return [t.name for t in parse_one(sql).find_all(Table)]
        except Exception:
            return []

    def _suggest_alternative_tables(
        self,
        node: TaskNode,
        tried: set[str],
        scored_schemas: list,
    ) -> list[str] | None:
        if self.retriever is not None:
            evidence = self.retriever.score(node.description)
            scored = evidence.candidates
        else:
            scored = list(scored_schemas)

        fresh = [c for c in scored if c.schema.name not in tried]
        if not fresh:
            return None
        fresh.sort(key=lambda c: -c.score)
        return [c.schema.name for c in fresh[:3]]
