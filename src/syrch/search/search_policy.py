from __future__ import annotations

import logging

from syrch.core.models import CandidateEvaluation, ScoredTable

logger = logging.getLogger(__name__)


class SearchPolicy:
    """Pluggable control of candidate search.

    Separates the concerns that used to be conflated in the confidence
    threshold:

      * Recoverability    -> RLM REPL loop (parse / schema / execution).
      * Requirement       -> Validator inside the REPL loop.
      * Path evaluation   -> CandidateEvaluation, used only for *comparing*
                             explored candidates.
      * Search budget     -> this policy (ordering + termination).

    The policy never uses the retriever prior to terminate the search; the
    prior only decides which candidate is tried next (best-first).
    """

    def __init__(self, candidates: list[ScoredTable]):
        self._candidates: list[ScoredTable] = list(candidates)
        self._index = 0
        self._evaluated: list[CandidateEvaluation] = []

    def has_next(self) -> bool:
        raise NotImplementedError

    def next(self) -> ScoredTable:
        cand = self._candidates[self._index]
        self._index += 1
        return cand

    def update(self, evaluation: CandidateEvaluation) -> None:
        self._evaluated.append(evaluation)

    def remaining(self) -> list[ScoredTable]:
        return self._candidates[self._index:]

    def expand(self, extra: int = 1) -> None:
        """Temporarily lift the search budget (ambiguity-driven expansion).

        Only extends up to the remaining candidate pool; `has_next()` still
        caps at the pool length.
        """
        raise NotImplementedError


class ExhaustivePolicy(SearchPolicy):
    """Try every candidate in retriever-prior order (small pools, eval runs)."""

    def has_next(self) -> bool:
        return self._index < len(self._candidates)

    def expand(self, extra: int = 1) -> None:
        return None


class BeamSearchPolicy(SearchPolicy):
    """Best-first search bounded by a search budget, not a confidence threshold.

    - Always tries candidates in retriever-prior order (best-first).
    - Always explores at least `beam_width` candidates (beam floor) and at most
      `max_candidates` (search budget). The budget is the primary control.
    - After the beam floor, stops early only when the best *viable* posterior
      beats the second-best viable posterior by at least `stop_margin`.
      A single viable candidate is not treated as dominance: with only one
      success among the explored set there is no evidence the others cannot
      beat it, so exploration continues up to the budget.
    """

    def __init__(
        self,
        candidates: list[ScoredTable],
        beam_width: int = 3,
        max_candidates: int = 8,
        stop_margin: float = 0.15,
    ):
        super().__init__(candidates)
        self.beam_width = max(1, int(beam_width))
        self.max_candidates = max(1, int(max_candidates))
        self.stop_margin = max(0.0, float(stop_margin))

    def has_next(self) -> bool:
        if self._index >= len(self._candidates):
            return False
        if self._index >= self.max_candidates:
            return False
        if self._index < self.beam_width:
            return True
        viable = [e for e in self._evaluated if e.viable]
        if len(viable) < 2:
            return True
        scores = sorted((e.posterior for e in viable), reverse=True)
        return (scores[0] - scores[1]) < self.stop_margin

    def expand(self, extra: int = 1) -> None:
        self.max_candidates += max(0, int(extra))


def build_policy(
    name: str,
    candidates: list[ScoredTable],
    *,
    beam_width: int = 3,
    max_candidates: int = 8,
    stop_margin: float = 0.15,
) -> SearchPolicy:
    name = (name or "beam").strip().lower()
    if name == "exhaustive":
        return ExhaustivePolicy(candidates)
    if name == "beam":
        return BeamSearchPolicy(candidates, beam_width, max_candidates, stop_margin)
    logger.warning("Unknown search_policy=%r, falling back to beam", name)
    return BeamSearchPolicy(candidates, beam_width, max_candidates, stop_margin)
