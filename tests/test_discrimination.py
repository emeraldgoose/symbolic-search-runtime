"""Discrimination-signal tests (S1/S5/S15): grain/dimension/time match and the
native-grain inference used to separate native vs derived tables."""

from syrch.core.models import (
    RequirementSpec,
    TableSchema,
    ColumnSchema,
    TaskNode,
    infer_layer,
)
from syrch.search.path_evaluator import PathEvaluator, infer_native_grain, _canonical_grain


def _table(name, columns=("total",)):
    return TableSchema(name=name, columns=[ColumnSchema(name=c, type="REAL") for c in columns])


def _node(description="q", grain=None, dimensions=None, time_range=None):
    return TaskNode(
        id="A",
        description=description,
        is_atomic=True,
        requirements=RequirementSpec(
            metrics=["total"],
            aggregation="sum",
            grain=grain,
            dimensions=dimensions or [],
            time_range=time_range,
        ),
    )


def test_infer_native_grain_from_name():
    assert infer_native_grain("mart_sales_daily") == "day"
    assert infer_native_grain("mart_sales_monthly") == "month"
    assert infer_native_grain("mart_sales_weekly") == "week"
    assert infer_native_grain("mart_sales_yearly") == "year"
    assert infer_native_grain("dw_sales_order") is None
    assert infer_native_grain("dim_customer") is None
    assert infer_native_grain("stg_raw") is None


def test_infer_layer_matches_fully_qualified_names():
    assert infer_layer("dw_sales_order") == "dw"
    assert infer_layer("syrch_benchmark.enterprise.dw_sales_order") == "dw"
    assert infer_layer("catalog.schema.mart_sales_monthly") == "mart"
    assert infer_layer("catalog.schema.dim_date") == "dim"
    assert infer_layer("some_schema.fact_events") == "fact"
    assert infer_layer("catalog.schema.orders") == "unknown"


def test_infer_native_grain_matches_fully_qualified_names():
    assert infer_native_grain("catalog.schema.mart_sales_monthly") == "month"
    assert infer_native_grain("catalog.schema.mart_sales_daily") == "day"
    assert infer_native_grain("catalog.schema.dw_sales_order") is None


def test_canonical_grain_maps_variants():
    assert _canonical_grain("monthly") == "month"
    assert _canonical_grain("month") == "month"
    assert _canonical_grain("daily") == "day"
    assert _canonical_grain("per_order") is None
    assert _canonical_grain("total") is None


def test_grain_match_separates_native_from_derived():
    """A table natively built for the requested grain outranks a transactional
    table that could only derive it (S5 native-vs-derived discrimination)."""
    ev = PathEvaluator()
    monthly_req = _node(grain="monthly")
    assert ev.grain_match(monthly_req, _table("mart_sales_monthly")) == 1.0
    assert ev.grain_match(monthly_req, _table("dw_sales_order")) == 0.5

    # "total" requirement accepts any table (no discrimination needed)
    total_req = _node(grain="total")
    assert ev.grain_match(total_req, _table("dw_sales_order")) == 1.0
    assert ev.grain_match(total_req, _table("mart_sales_monthly")) == 1.0


def test_dimension_match_uses_requirement_dimensions():
    ev = PathEvaluator()
    req = _node(dimensions=["customer", "region"])
    assert ev.dimension_match(req, _table("dw_sales_order", ("customer_id", "region"))) == 1.0
    assert ev.dimension_match(req, _table("mart_sales", ("total",))) == 0.0
    # no dimensions → neutral
    assert ev.dimension_match(_node(), _table("x")) == 1.0


def test_time_match_requires_date_column():
    ev = PathEvaluator()
    req = _node(time_range=("2024-01-01", "2024-12-31"))
    assert ev.time_match(req, _table("dw_sales_order", ("order_date", "total"))) == 1.0
    assert ev.time_match(req, _table("mart_sales", ("total",))) == 0.0
    assert ev.time_match(_node(), _table("x")) == 1.0
