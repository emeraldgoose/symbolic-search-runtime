from __future__ import annotations

import logging

from syrch.core.models import JoinKey, NodeResult, ProblemSpec, TableSchema, TaskDAG, TaskNode
from syrch.core.config import ExecutionConfig
from syrch.llm.base import BaseLLM

logger = logging.getLogger(__name__)

DECOMPOSE_SYSTEM = """You are a task decomposition planner. Given a problem and a database schema, decompose it into a DAG of sub-tasks.

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
- VERY IMPORTANT: For each sub-task, specify hint_tables (the database table(s) most likely to contain the needed data). This is a list of table names. Be specific.
- If you know which columns are needed, also specify hint_columns (optional, as a hint only — the SQL generator will verify).
- If a sub-task depends on another and their results should be joined by column,
  specify join_keys. Each join_key has: left, left_col, right, right_col, how.
  Example: "join_keys": [{{"left": "B", "left_col": "customer_id",
                          "right": "A", "right_col": "id", "how": "inner"}}]

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
      "hint_columns": ["column_name"]
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
    def __init__(self, llm: BaseLLM, config: ExecutionConfig):
        self.llm = llm
        self.config = config

    def decompose(self, problem: ProblemSpec) -> TaskDAG:
        dag = self._decompose_level(problem, depth=0)
        effective_max = self._resolve_max_depth(problem)
        self._recursive_expand(dag, depth=0, schemas=problem.all_schemas or [problem.schema],
                               max_depth_override=effective_max)
        return dag

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
            hint_columns: list[str] | None = [c for c in hint_columns_raw if isinstance(c, str)] if isinstance(hint_columns_raw, list) else None

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
            )
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

    def _build_user_prompt(self, problem: ProblemSpec) -> str:
        scored = problem.scored_schemas
        if scored:
            table_lines: list[str] = []
            for s in scored:
                cols_str = ", ".join(f"{c.name}({c.type})" for c in s.schema.columns)
                reason_str = "; ".join(s.match_reasons[:2])
                table_lines.append(
                    f"Table: {s.schema.name}  score={s.score:.2f}  [{reason_str}]\n"
                    f"  Columns: {cols_str}"
                )
            tables_str = "\n".join(table_lines)
        else:
            all_schemas = problem.all_schemas or [problem.schema]
            tables_str = ""
            for s in all_schemas:
                cols_str = ", ".join(f"{c.name} ({c.type})" for c in s.columns)
                tables_str += f"Table: {s.name}\n  Columns: {cols_str}\n"
        return f"""Question: {problem.question}

{tables_str}

Decompose this into sub-tasks. Follow the JSON output format exactly. Each sub-task MUST include hint_tables."""

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
    ) -> TaskDAG:
        if self.config.verbose:
            logger.info("Planner.replan: node=%s error=%s", failed_node_id, error[:100])
        failed_node = dag.nodes.get(failed_node_id)
        if failed_node is None or failed_node.hint_tables:
            new_hints = self._suggest_alternative_tables(error, failed_node, scored_schemas) if failed_node else None
            if new_hints and failed_node:
                if self.config.verbose:
                    logger.info("  updating hint_tables: %s -> %s", failed_node.hint_tables, new_hints)
                failed_node.hint_tables = new_hints
        return dag

    @staticmethod
    def _suggest_alternative_tables(
        error: str,
        node: TaskNode,
        scored_schemas: list,
    ) -> list[str] | None:
        error_lower = error.lower()
        missing_table = None
        for phrase in ["no such table", "table not found", "doesn't exist"]:
            if phrase in error_lower:
                import re
                match = re.search(r"(?:table\s+)?['\"]?(\w+)['\"]?", error_lower)
                if match:
                    missing_table = match.group(1)
                break
        if missing_table and scored_schemas:
            alternatives = [
                s.schema.name for s in scored_schemas[:5]
                if s.schema.name.lower() != missing_table
            ]
            return alternatives[:3] if alternatives else None
        return None
