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
    """Tier 1 (schema): a candidate without any date-like column can never
    serve a time-ranged requirement."""
    ev = PathEvaluator()
    req = _node(time_range=("2024-01-01", "2024-12-31"))
    # No coverage evidence → unknown, honest neutral score (was fake 1.0)
    assert ev.time_match(req, _table("dw_sales_order", ("order_date", "total"))) == 0.5
    # No date-like column at all → cannot serve the window
    assert ev.time_match(req, _table("mart_sales", ("total",))) == 0.0
    # No time_range requirement → neutral
    assert ev.time_match(_node(), _table("x")) == 1.0


def test_time_match_coverage_separates_archive_from_fact():
    """Tier 2 (COVERAGE probe): a table whose data ends before the requested
    window scores 0.0; one covering it scores 1.0 (S-run: archive_orders_2021
    passed the old name-only check for a 2024 question)."""
    from syrch.search.data_probe import TimeCoverage

    ev = PathEvaluator()
    req = _node(time_range=("2024-01-01", "2024-12-31"))
    schema = TableSchema(
        name="dw_sales_order",
        columns=[ColumnSchema(name="order_date", type="DATE")],
    )
    covering = TimeCoverage(
        table="dw_sales_order", column="order_date",
        min_value="2023-01-01", max_value="2024-06-30",
    )
    assert ev.time_match(req, schema, covering) == 1.0

    archive = TableSchema(
        name="archive_orders_2021",
        columns=[ColumnSchema(name="order_date", type="DATE")],
    )
    stale = TimeCoverage(
        table="archive_orders_2021", column="order_date",
        min_value="2021-01-01", max_value="2021-12-31",
    )
    assert ev.time_match(req, archive, stale) == 0.0


def test_time_match_coverage_unknown_is_neutral_not_punished():
    """A failed/absent coverage probe must not fabricate discrimination."""
    from syrch.search.data_probe import TimeCoverage

    ev = PathEvaluator()
    req = _node(time_range=("2024-01-01", "2024-12-31"))
    schema = TableSchema(
        name="t", columns=[ColumnSchema(name="event_ts", type="TIMESTAMP")]
    )
    assert ev.time_match(req, schema, None) == 0.5
    empty_cov = TimeCoverage(table="t", column="event_ts")
    assert ev.time_match(req, schema, empty_cov) == 0.5


def test_metric_feasible_gate_excludes_dimension_tables():
    """Capability gate: dim_date holds integer calendar keys but no measure —
    it must not enter ranking for a revenue requirement (S-run: dim_date was
    one alphabetical tie-break away from 'winning' a revenue task)."""
    ev = PathEvaluator()
    req = _node()  # metrics=["total"], aggregation=sum

    dim_date = TableSchema(name="dim_date", columns=[
        ColumnSchema(name="date_key", type="INTEGER"),
        ColumnSchema(name="year", type="INTEGER"),
        ColumnSchema(name="month", type="INTEGER"),
    ])
    assert ev.metric_feasible(req, dim_date) is False

    fact = TableSchema(name="dw_sales_order", columns=[
        ColumnSchema(name="customer_id", type="INTEGER"),
        ColumnSchema(name="total_amount", type="DOUBLE"),
    ])
    assert ev.metric_feasible(req, fact) is True


def test_metric_feasible_true_when_no_metrics_required():
    """Without metric requirements the gate never fires (lookup tasks)."""
    ev = PathEvaluator()
    node = TaskNode(id="A", description="list dates", is_atomic=True)
    dim_date = TableSchema(name="dim_date", columns=[
        ColumnSchema(name="date_key", type="INTEGER"),
    ])
    assert ev.metric_feasible(node, dim_date) is True
