from syrch.core.models import RequirementSpec
from syrch.search.validator import Validator


def _req(
    metrics=None,
    time_range=None,
    aggregation=None,
    grain=None,
):
    return RequirementSpec(
        metrics=metrics or [],
        time_range=time_range,
        aggregation=aggregation,
        grain=grain,
    )


def test_uppercase_sum_aggregation_passes():
    validator = Validator()
    sql = "SELECT SUM(total_revenue) AS total_revenue FROM mart_sales_daily WHERE sale_date >= '2024-01-01'"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], time_range=("2024-01-01", "2024-12-31"), aggregation="sum"),
        valid_columns={"total_revenue", "sale_date"},
    )
    assert result.passed is True
    assert result.aggregation_ok is True


def test_lowercase_avg_aggregation_passes():
    validator = Validator()
    sql = "select avg(amount) as amount from dw_sales_order where order_date >= '2024-01-01'"
    result = validator.validate(
        sql,
        _req(metrics=["amount"], time_range=("2024-01-01", "2024-12-31"), aggregation="avg"),
        valid_columns={"amount", "order_date"},
    )
    assert result.passed is True


def test_count_distinct_aggregation_passes():
    validator = Validator()
    sql = "SELECT COUNT(DISTINCT customer_id) AS cnt FROM dw_customer"
    result = validator.validate(
        sql,
        _req(metrics=["customer_id"], aggregation="count"),
        valid_columns={"customer_id"},
    )
    assert result.passed is True


def test_missing_aggregation_still_fails():
    validator = Validator()
    sql = "SELECT total_revenue FROM mart_sales_daily"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], aggregation="sum"),
        valid_columns={"total_revenue"},
    )
    assert result.passed is False
    assert result.aggregation_ok is False


def test_metric_fallback_case_insensitive():
    validator = Validator()
    sql = "SELECT TOTAL_REVENUE FROM mart_sales_daily"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], aggregation="sum"),
        valid_columns={"total_revenue"},
    )
    assert "revenue" not in result.missing_metrics


def test_metric_token_subset_matches_physical_column():
    validator = Validator()
    sql = "SELECT SUM(precipitation_lwe_total) AS precipitation_lwe_total FROM samples.accuweather.forecast_daily_calendar_metric"
    result = validator.validate(
        sql,
        _req(metrics=["precipitation_total"], aggregation="sum"),
        valid_columns={"precipitation_lwe_total"},
    )
    assert result.passed is True
    assert "precipitation_total" not in result.missing_metrics


def test_metric_token_subset_matches_alias_with_extra_tokens():
    validator = Validator()
    sql = "SELECT AVG(humidity_relative_avg) AS humidity_relative_avg FROM samples.accuweather.forecast_daily_calendar_metric"
    result = validator.validate(
        sql,
        _req(metrics=["humidity_avg"], aggregation="avg"),
        valid_columns={"humidity_relative_avg"},
    )
    assert result.passed is True
    assert "humidity_avg" not in result.missing_metrics


def test_metric_token_missing_still_fails():
    validator = Validator()
    sql = "SELECT SUM(amount) FROM mart_sales_daily"
    result = validator.validate(
        sql,
        _req(metrics=["precipitation_total"], aggregation="sum"),
        valid_columns={"amount"},
    )
    assert result.passed is False
    assert "precipitation_total" in result.missing_metrics


def test_date_filter_column_case_insensitive():
    validator = Validator()
    sql = "SELECT SUM(amount) FROM dw_sales_order WHERE SALE_DATE >= '2024-01-01'"
    result = validator.validate(
        sql,
        _req(metrics=["amount"], time_range=("2024-01-01", "2024-12-31"), aggregation="sum"),
        valid_columns={"amount", "sale_date"},
    )
    assert result.passed is True
    assert result.missing_filters == []


def test_preaggregated_monthly_grain_allows_direct_metric_select():
    validator = Validator()
    sql = "SELECT sale_month, total_revenue FROM mart_sales_monthly WHERE sale_month LIKE '2024%' ORDER BY sale_month"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], time_range=("2024-01-01", "2024-12-31"),
             aggregation="sum", grain="monthly"),
        valid_columns={"sale_month", "total_revenue"},
    )
    assert result.passed is True
    assert result.aggregation_ok is True
    assert result.grain_mismatch is False


def test_transactional_table_without_groupby_is_grain_mismatch():
    validator = Validator()
    sql = "SELECT total_revenue FROM mart_sales_daily WHERE sale_date >= '2024-01-01'"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], time_range=("2024-01-01", "2024-12-31"),
             aggregation="sum", grain="monthly"),
        valid_columns={"sale_date", "total_revenue"},
    )
    assert result.grain_mismatch is True


def test_collapsed_aggregate_on_native_grain_table_is_grain_mismatch():
    validator = Validator()
    sql = "SELECT SUM(total_revenue) FROM mart_sales_monthly WHERE sale_month LIKE '2024%'"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], time_range=("2024-01-01", "2024-12-31"),
             aggregation="sum", grain="monthly"),
        valid_columns={"sale_month", "total_revenue"},
    )
    assert result.grain_mismatch is True


def test_non_grain_requirement_on_transactional_table_is_grain_mismatch():
    validator = Validator()
    sql = "SELECT total_revenue FROM mart_sales_daily WHERE sale_date >= '2024-01-01'"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], aggregation="sum", grain="per_order"),
        valid_columns={"sale_date", "total_revenue"},
    )
    assert result.grain_mismatch is True


def test_preaggregated_relaxation_does_not_mask_total_grain():
    validator = Validator()
    sql = "SELECT total_revenue FROM mart_sales_monthly WHERE sale_month LIKE '2024%'"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], aggregation="sum", grain="total"),
        valid_columns={"sale_month", "total_revenue"},
    )
    assert result.passed is False
    assert result.aggregation_ok is False


def test_preaggregated_relaxation_still_requires_metric():
    validator = Validator()
    sql = "SELECT sale_month FROM mart_sales_monthly"
    result = validator.validate(
        sql,
        _req(metrics=["revenue"], aggregation="sum", grain="monthly"),
        valid_columns={"sale_month"},
    )
    assert result.passed is False
    assert "revenue" in result.missing_metrics


def test_supporting_relation_required_in_sql():
    from syrch.core.models import SupportingRelation

    validator = Validator()
    req = RequirementSpec(
        metrics=["revenue"],
        supporting_relations=[
            SupportingRelation(table="dim_date", purpose="exclude holidays"),
        ],
    )
    sql = "SELECT SUM(amount) FROM mart_sales_daily WHERE is_holiday = false"
    result = validator.validate(sql, req, valid_columns={"amount", "is_holiday"})
    assert result.passed is False
    assert any("dim_date" in f for f in result.missing_filters)


def test_supporting_relation_present_passes():
    from syrch.core.models import SupportingRelation

    validator = Validator()
    req = RequirementSpec(
        metrics=["revenue"],
        supporting_relations=[
            SupportingRelation(table="dim_date", purpose="exclude holidays"),
        ],
    )
    sql = (
        "SELECT SUM(amount) AS revenue FROM mart_sales_daily d "
        "JOIN dim_date dd ON d.sale_date = dd.date_key WHERE dd.is_holiday = false"
    )
    result = validator.validate(sql, req, valid_columns={"amount", "sale_date", "is_holiday"})
    assert result.passed is True


def test_value_constraint_enforced():
    from syrch.core.models import ValueConstraint

    validator = Validator()
    req = RequirementSpec(
        metrics=["refund_amount"],
        value_constraints=[ValueConstraint(column="status", values=["refunded"])],
    )
    sql = "SELECT SUM(amount) AS refund_amount FROM dw_sales_order"
    result = validator.validate(sql, req, valid_columns={"amount"})
    assert result.passed is False
    assert any("refunded" in f for f in result.missing_filters)


def test_value_constraint_satisfied_passes():
    from syrch.core.models import ValueConstraint

    validator = Validator()
    req = RequirementSpec(
        metrics=["refund_amount"],
        value_constraints=[ValueConstraint(column="status", values=["refunded"])],
    )
    sql = (
        "SELECT SUM(amount) AS refund_amount FROM dw_sales_order "
        "WHERE status = 'refunded'"
    )
    result = validator.validate(sql, req, valid_columns={"amount", "status"})
    assert result.passed is True


def test_requirement_render_surfaces_semantics_and_relations():
    from syrch.core.models import (
        FilterRequirement,
        MetricRequirement,
        SupportingRelation,
    )

    req = RequirementSpec(
        metrics=["delivery_time"],
        metric_details=[
            MetricRequirement(
                name="delivery_time",
                aggregation="avg",
                expression_semantics="time between payment and shipment",
            )
        ],
        grain="month",
        filters=[FilterRequirement(semantic="holidays", predicate="is_holiday = false",
                                   relation="dim_date")],
        supporting_relations=[SupportingRelation(table="dim_date", purpose="exclude holidays")],
    )
    rendered = req.render()
    assert "delivery_time" in rendered
    assert "time between payment and shipment" in rendered
    assert "is_holiday = false" in rendered
    assert "dim_date" in rendered
    assert "grain: month" in rendered

