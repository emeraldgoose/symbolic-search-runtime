from __future__ import annotations

import json
from typing import Any

from syrch.core.models import JoinKey, ProblemSpec, TableSchema, TaskDAG, TaskNode
from syrch.core.config import ExecutionConfig
from syrch.llm.base import BaseLLM

DECOMPOSE_SYSTEM = """You are a task decomposition planner. Given a problem and a database schema, decompose it into a DAG of sub-tasks.

Rules:
- Each sub-task must be independently solvable
- Use depends_on to express data dependencies between sub-tasks
- Max recursion depth: {max_depth}
- A sub-task is atomic if a single SQL query can fully and cleanly answer it.
  Prefer decomposition when:
  * The question involves multiple analytical dimensions or drill-downs
    (e.g., aggregate at one granularity, then break down by another)
  * Intermediate results carry independent insight
    (e.g., yearly totals are meaningful on their own)
  * Multiple aggregation strategies, join combinations, or filtering
    approaches must be explored independently
  Keep it atomic when the question maps directly to a single query
  without losing clarity or completeness.
- Non-atomic sub-tasks will be further decomposed recursively (up to max_depth).
  Provide clear descriptions so recursive decomposition can proceed correctly.
- Sub-tasks must be MECE (Mutually Exclusive, Collectively Exhaustive)
- every sub-task must have an id, description, depends_on list, is_atomic boolean, and expected_output
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
      "expected_output": "what data this sub-task produces"
    }}
  ]
}}
"""


class Planner:
    def __init__(self, llm: BaseLLM, config: ExecutionConfig):
        self.llm = llm
        self.config = config

    def decompose(self, problem: ProblemSpec) -> TaskDAG:
        dag = self._decompose_level(problem, depth=0)
        self._recursive_expand(dag, depth=0, schemas=problem.all_schemas or [problem.schema])
        return dag

    def _decompose_level(self, problem: ProblemSpec, depth: int = 0) -> TaskDAG:
        system = DECOMPOSE_SYSTEM.format(max_depth=self.config.max_depth)
        user = self._build_user_prompt(problem)
        result = self.llm.generate_json(system, user)
        nodes: dict[str, TaskNode] = {}
        for item in result.get("subtasks", []):
            join_keys = None
            if "join_keys" in item:
                join_keys = [JoinKey(**k) for k in item["join_keys"]]
            node = TaskNode(
                id=item["id"],
                description=item["description"],
                depends_on=item.get("depends_on", []),
                depth=depth,
                is_atomic=item.get("is_atomic", True),
                expected_output_desc=item.get("expected_output", ""),
                join_keys=join_keys,
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

    def _recursive_expand(self, dag: TaskDAG, depth: int, schemas: list[TableSchema]) -> None:
        if not schemas:
            return
        if depth >= self.config.max_depth - 1:
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
        all_schemas = problem.all_schemas or [problem.schema]
        tables_str = ""
        for s in all_schemas:
            cols_str = "\n".join(f"  - {c.name}: {c.type}" for c in s.columns)
            tables_str += f"Table: {s.name}\nColumns:\n{cols_str}\n\n"
        return f"""Database tables:
{tables_str.strip()}

Question: {problem.question}

Decompose this into sub-tasks."""

    def _resolve_root(self, nodes: dict[str, TaskNode]) -> str:
        for nid, node in nodes.items():
            if not node.depends_on:
                return nid
        return list(nodes.keys())[0]

    def _compute_layers(self, nodes: dict[str, TaskNode]) -> list[list[str]]:
        indeg: dict[str, int] = {}
        for nid, node in nodes.items():
            indeg.setdefault(nid, 0)
            for dep in node.depends_on:
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
