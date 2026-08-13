from syrch.core.models import CandidateEvaluation, PathScore, ScoredTable, TableSchema
from syrch.search.search_policy import (
    BeamSearchPolicy,
    ExhaustivePolicy,
    build_policy,
)


def _table(name):
    return ScoredTable(schema=TableSchema(name=name, columns=[]), score=0.0)


def _eval(table, ok, path_score, semantic=0.0, result_quality=0.0, requirement_pass=True,
          structural=0.0):
    return CandidateEvaluation(
        table=table.schema.name,
        ok=ok,
        execution_valid=ok,
        requirement_pass=requirement_pass and ok,
        semantic_match=semantic,
        result_quality=result_quality,
        structural_match=structural,
        has_data=ok and path_score is not None and path_score.total > 0.0,
        candidate_id=table.schema.name,
        path_score=path_score,
    )


def _pscore(requirement=1.0, execution=1.0, retriever=0.0):
    return PathScore(
        total=0.4 * requirement + 0.3 * execution + 0.3 * retriever,
        requirement_coverage=requirement,
        execution_signal=execution,
        retriever_evidence=retriever,
    )


def _run(policy, score_fn):
    """Simulate the RLM loop: consume next candidate, evaluate, update."""
    explored = []
    while policy.has_next():
        cand = policy.next()
        explored.append(cand.schema.name)
        policy.update(score_fn(cand))
    return explored


def test_beam_explores_beam_floor_even_with_clear_winner():
    """Early stop is not allowed before the beam floor is explored."""
    pool = [_table("a"), _table("b"), _table("c"), _table("d")]

    def score(cand):
        # first candidate is a clear winner; the rest are weak
        if cand.schema.name == "a":
            return _eval(cand, True, _pscore(requirement=1.0, execution=1.0),
                         result_quality=1.0)
        return _eval(cand, True, _pscore(requirement=0.5, execution=1.0),
                     result_quality=0.5)

    policy = BeamSearchPolicy(pool, beam_width=3, max_candidates=8, stop_margin=0.1)
    assert _run(policy, score) == ["a", "b", "c"]

    # Without the beam floor, the same margin stops one candidate earlier.
    policy2 = BeamSearchPolicy(pool, beam_width=1, max_candidates=8, stop_margin=0.1)
    assert _run(policy2, score) == ["a", "b"]


def test_beam_single_viable_does_not_stop():
    """One viable candidate among explored set is not dominance: keep exploring."""
    pool = [_table(f"t{i}") for i in range(5)]

    def score(cand):
        if cand.schema.name == "t0":
            return _eval(cand, True, _pscore(requirement=1.0, execution=1.0),
                         semantic=1.0, result_quality=1.0)
        return _eval(cand, False, None)

    policy = BeamSearchPolicy(pool, beam_width=3, max_candidates=8, stop_margin=0.5)
    # single viable after the beam floor → continue up to the pool end
    assert _run(policy, score) == [f"t{i}" for i in range(5)]


def test_beam_budget_caps_exploration():
    pool = [_table(f"t{i}") for i in range(6)]

    def score(cand):
        return _eval(cand, True, _pscore(requirement=1.0, execution=1.0),
                     semantic=1.0, result_quality=1.0)

    policy = BeamSearchPolicy(pool, beam_width=1, max_candidates=3, stop_margin=0.5)
    assert _run(policy, score) == ["t0", "t1", "t2"]


def test_beam_margin_uses_posterior_not_retriever():
    """Termination compares execution/semantic posterior; retriever prior alone
    never triggers a stop (S1 regression guard)."""
    # Candidate a has a much higher retriever_evidence in PathScore, but the
    # posterior (semantic + result_quality) is identical. A total-based margin
    # would stop; posterior-based margin must keep exploring.
    pool = [_table("a"), _table("b"), _table("c")]

    def score(cand):
        if cand.schema.name == "a":
            return _eval(cand, True, _pscore(requirement=1.0, execution=1.0, retriever=1.0),
                         semantic=1.0, result_quality=1.0)
        return _eval(cand, True, _pscore(requirement=1.0, execution=1.0, retriever=0.0),
                     semantic=1.0, result_quality=1.0)

    policy = BeamSearchPolicy(pool, beam_width=2, max_candidates=8, stop_margin=0.3)
    # path_score total: a=1.0, b=0.7 → gap 0.3 == margin → total-based would stop
    # posterior: a=(1+1)/2=1.0, b=(1+1)/2=1.0 → gap 0.0 < margin → keep going
    assert _run(policy, score) == ["a", "b", "c"]


def test_exhaustive_tries_all():
    pool = [_table(f"t{i}") for i in range(5)]

    def score(cand):
        return _eval(cand, True, _pscore())

    assert _run(ExhaustivePolicy(pool), score) == [f"t{i}" for i in range(5)]


def test_build_policy_dispatch():
    pool = [_table("a")]
    assert isinstance(build_policy("exhaustive", pool), ExhaustivePolicy)
    assert isinstance(build_policy("beam", pool), BeamSearchPolicy)
    assert isinstance(build_policy("bogus", pool), BeamSearchPolicy)


def test_candidate_evaluation_viability():
    good = _eval(_table("a"), True, _pscore(requirement=1.0), semantic=1.0, result_quality=1.0)
    assert good.viable is True

    empty = _eval(_table("a"), True, _pscore(requirement=1.0), semantic=1.0, result_quality=1.0)
    empty.has_data = False
    assert empty.viable is False

    failed = _eval(_table("a"), False, None)
    assert failed.viable is False

    zero = _eval(_table("a"), True, PathScore(total=0.0))
    assert zero.viable is False

    no_req = _eval(_table("a"), True, _pscore(requirement=1.0), requirement_pass=False)
    assert no_req.viable is False


def test_lexicographic_ranking_prefers_discrimination_over_semantic():
    """Discrimination signals (structural/grain/dimension/time/result_quality)
    outrank lexical relevance. semantic_match and cost_tokens are excluded
    from the selection key: a candidate that merely *names* the metric better
    is not thereby selected (S1 regression guard)."""
    from syrch.search.rlm_engine import RLMAgent

    a = _eval(_table("a"), True, _pscore(), semantic=0.92, result_quality=0.88,
              structural=0.9)
    b = _eval(_table("b"), True, _pscore(), semantic=0.87, result_quality=0.95,
              structural=0.0)
    # structural dominates result_quality even when semantic/execution differ
    assert RLMAgent._rank_key(a) > RLMAgent._rank_key(b)

    # semantic_match alone never breaks a tie on the discrimination signals
    c = _eval(_table("c"), True, _pscore(), semantic=1.0, result_quality=0.9)
    d = _eval(_table("c"), True, _pscore(), semantic=0.1, result_quality=0.9)
    assert c.ranking_signals == d.ranking_signals


def test_ambiguous_when_ranking_signals_equal():
    """Indistinguishable top candidates → the node must report AMBIGUOUS, never
    resolve by execution order."""
    a = _eval(_table("a"), True, _pscore(), semantic=0.92, result_quality=0.88)
    b = _eval(_table("b"), True, _pscore(), semantic=0.92, result_quality=0.88)
    assert a.viable and b.viable
    assert a.ranking_signals == b.ranking_signals


def test_beam_expand_lifts_budget():
    """expand() extends the budget for ambiguity-driven exploration."""
    pool = [_table(f"t{i}") for i in range(6)]

    def score(cand):
        return _eval(cand, True, _pscore(requirement=1.0, execution=1.0),
                     semantic=1.0, result_quality=1.0)

    policy = BeamSearchPolicy(pool, beam_width=1, max_candidates=3, stop_margin=0.5)
    assert _run(policy, score) == ["t0", "t1", "t2"]
    policy.expand(2)
    assert _run(policy, score) == ["t3", "t4"]
