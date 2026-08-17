import pandas as pd

from syrch.api import _extract_tables_from_tree
from syrch.core.models import NodeResult


def _node(node_id: str, sql: str, status: str = "SOLVED") -> NodeResult:
    return NodeResult(
        node_id=node_id,
        data=pd.DataFrame(),
        sql=sql,
        confidence=0.7,
        status=status,
    )


def test_extract_tables_from_tree_excludes_task_context():
    tree = [
        _node("A", "SELECT SUM(total_revenue) AS total_revenue FROM mart_sales_daily"),
        _node(
            "B",
            "SELECT AVG(total_revenue) AS avg_daily_revenue FROM _task_context_A "
            "JOIN dim_date ON _task_context_A.sale_date = dim_date.date_key "
            "WHERE dim_date.year = 2024",
        ),
    ]
    tables = _extract_tables_from_tree(tree)
    assert "mart_sales_daily" in tables
    assert "dim_date" in tables
    assert not any(t.startswith("_task_context_") for t in tables)


def test_extract_tables_from_tree_excludes_ctes():
    tree = [
        _node(
            "A",
            "WITH base AS (SELECT * FROM dw_sales_order) "
            "SELECT SUM(total_amount) FROM base",
        ),
    ]
    tables = _extract_tables_from_tree(tree)
    assert tables == ["dw_sales_order"]


def test_extract_tables_from_tree_empty_sql():
    tree = [
        _node("A", "", status="FAILED"),
        _node("B", "", status="SOLVED"),
    ]
    assert _extract_tables_from_tree(tree) == []