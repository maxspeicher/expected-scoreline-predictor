"""Predict the result of a single knockout match that must have a winner.

Builds on ``scoreline_predictor.predict_scorelines``: the closing odds give the
prediction for the 90 minutes. If those end in a draw, extra time is played;
if extra time ends in a draw too, a penalty shoot-out decides.

Example
-------
>>> from knockout_predictor import predict_knockout
>>> k = predict_knockout(home=2.9, draw=2.75, away=2.95, over=2.6, under=1.5)
>>> k.home_advances, k.away_advances   # probabilities of going through
>>> k.results                          # top final results with probabilities
>>> k.expected_result                  # one final result; never a draw
>>> k.regular_time                     # the 90-minute prediction

For single matches only, not for two-legged ties. Pure standard library,
Python 3.9+.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from scoreline_predictor import ScorelinePrediction, predict_scorelines

__all__ = ["predict_knockout", "KnockoutPrediction", "KnockoutResult"]

REGULAR_TIME = "regular time"
EXTRA_TIME = "extra time"
PENALTIES = "penalties"

# --- Statistical assumptions; sources are listed under "References" in the README ---
# Expected goals in extra time as a share of the expected goals in 90 minutes.
# In 12,898 matches with extra time, 1.02 goals were scored in extra time against
# 2.69 in an average match (Shelopugin and Sirotkin, 2026). This is a little more
# than the 30/90 = 0.33 that equal scoring per minute would give.
EXTRA_TIME_GOAL_SHARE = 0.38
# Probability that the home (first-named) team wins a shoot-out. Shoot-outs show
# no home advantage and only a small effect of team strength (Wunderlich et al.,
# 2020), and no reliable advantage for the team kicking first (Kocher et al., 2012).
SHOOTOUT_HOME_WIN = 0.5

# --- Numerical setting (precision only) ---
_MAX_GOALS = 15  # per team in extra time; the grid is renormalised after truncation


@dataclass(frozen=True)
class KnockoutResult:
    """One possible final result of a knockout match.

    ``home_goals`` and ``away_goals`` are the score after the last period that
    was played. If the match was decided on penalties, they are the (level)
    score after extra time; shoot-out goals are not counted.
    """

    home_goals: int
    away_goals: int
    decided_in: str  # "regular time", "extra time" or "penalties"
    winner: str  # "home" or "away"
    probability: float

    def __str__(self) -> str:
        score = f"{self.home_goals}-{self.away_goals}"
        if self.decided_in == EXTRA_TIME:
            score += " after extra time"
        elif self.decided_in == PENALTIES:
            score += f", {self.winner} wins on penalties"
        return f"{score} ({self.probability:.1%})"


@dataclass(frozen=True)
class KnockoutPrediction:
    home_advances: float
    away_advances: float
    ends_in_regular_time: float
    ends_in_extra_time: float
    ends_on_penalties: float
    results: List[KnockoutResult]  # top N final results, most likely first
    expected_result: KnockoutResult  # one final result; always has a winner
    extra_time_outcomes: Tuple[float, float, float]  # home win, draw, away win within extra time
    regular_time: ScorelinePrediction = field(repr=False)  # the 90-minute prediction

    def __str__(self) -> str:
        top = "; ".join(str(r) for r in self.results)
        return (
            f"Home advances {self.home_advances:.1%} | Away advances {self.away_advances:.1%}\n"
            f"Decided in: regular time {self.ends_in_regular_time:.1%} | "
            f"extra time {self.ends_in_extra_time:.1%} | penalties {self.ends_on_penalties:.1%}\n"
            f"Expected result: {self.expected_result}\n"
            f"Top results: {top}"
        )


def _poisson_matrix(lam_home: float, lam_away: float) -> List[List[float]]:
    def pmf(lam: float) -> List[float]:
        out = [math.exp(-lam)]
        for k in range(1, _MAX_GOALS + 1):
            out.append(out[-1] * lam / k)
        return out

    ph, pa = pmf(lam_home), pmf(lam_away)
    total = sum(ph) * sum(pa)
    return [[h * a / total for a in pa] for h in ph]


def _best_cell(matrix: List[List[float]], outcome: str) -> Tuple[int, int]:
    """Most likely scoreline of one outcome: 'home', 'draw' or 'away'."""
    keep = {"home": lambda h, a: h > a, "draw": lambda h, a: h == a, "away": lambda h, a: h < a}[outcome]
    return max(
        ((p, -h - a, h, a) for h, row in enumerate(matrix) for a, p in enumerate(row) if keep(h, a))
    )[2:]


def predict_knockout(
    home: float,
    draw: float,
    away: float,
    top_n: int = 5,
    extra_time_goal_share: float = EXTRA_TIME_GOAL_SHARE,
    shootout_home_win: float = SHOOTOUT_HOME_WIN,
    **market_odds: float,
) -> KnockoutPrediction:
    """Predict the final result of a knockout match from decimal closing odds.

    Parameters
    ----------
    home, draw, away : decimal 1X2 closing odds for the 90 minutes (required).
        "Home" is simply the first-named team if the venue is neutral.
    top_n : number of most likely final results to return.
    extra_time_goal_share : expected goals in extra time as a share of the
        expected goals in the 90 minutes. The default ``0.38`` is the observed
        ratio in 12,898 matches with extra time; ``1/3`` would mean the same
        scoring rate per minute as in regular time.
    shootout_home_win : probability that the home team wins a penalty
        shoot-out. The default ``0.5`` treats the shoot-out as a coin flip.
    **market_odds : any optional odds that ``predict_scorelines`` accepts
        (``over``, ``under``, ``total_line``, ``ah_home``, ``ah_away``,
        ``ah_line``, ``btts_yes``, ``btts_no``) and ``rho``.

    All odds must be for the result after 90 minutes, not "to qualify" odds.
    """
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if not (math.isfinite(extra_time_goal_share) and extra_time_goal_share > 0):
        raise ValueError(f"extra_time_goal_share must be a positive number, got {extra_time_goal_share}")
    if not 0.0 <= shootout_home_win <= 1.0:
        raise ValueError(f"shootout_home_win must be between 0 and 1, got {shootout_home_win}")

    regular = predict_scorelines(home, draw, away, **market_odds)
    m90 = regular.score_matrix
    extra = _poisson_matrix(
        regular.lambda_home * extra_time_goal_share, regular.lambda_away * extra_time_goal_share
    )
    et_home = sum(p for i, row in enumerate(extra) for j, p in enumerate(row) if i > j)
    et_draw = sum(extra[i][i] for i in range(len(extra)))
    et_away = 1.0 - et_home - et_draw

    # Probability of every final result: (home goals, away goals, decided in, winner).
    table: Dict[Tuple[int, int, str, str], float] = {}

    def add(h: int, a: int, stage: str, winner: str, p: float) -> None:
        if p > 0.0:
            key = (h, a, stage, winner)
            table[key] = table.get(key, 0.0) + p

    for h, row in enumerate(m90):
        for a, p in enumerate(row):
            if h != a:
                add(h, a, REGULAR_TIME, "home" if h > a else "away", p)
                continue
            for i, et_row in enumerate(extra):  # level after 90 minutes: play extra time
                for j, q in enumerate(et_row):
                    if i != j:
                        add(h + i, a + j, EXTRA_TIME, "home" if i > j else "away", p * q)
                    else:
                        add(h + i, a + j, PENALTIES, "home", p * q * shootout_home_win)
                        add(h + i, a + j, PENALTIES, "away", p * q * (1.0 - shootout_home_win))

    def total(index: int, value: str) -> float:
        return sum(p for key, p in table.items() if key[index] == value)

    home_advances = total(3, "home")
    ranked = sorted(
        (KnockoutResult(*key, probability=p) for key, p in table.items()),
        key=lambda r: (-r.probability, r.home_goals + r.away_goals, r.winner),
    )

    # Expected result: follow the match period by period, always taking the
    # most likely outcome of the period and its most likely scoreline.
    h, a = regular.expected_scoreline
    if h != a:
        key = (h, a, REGULAR_TIME, "home" if h > a else "away")
    else:
        et_outcome = max((et_draw, "draw"), (et_home, "home"), (et_away, "away"))[1]
        i, j = _best_cell(extra, et_outcome)
        if et_outcome != "draw":
            key = (h + i, a + j, EXTRA_TIME, et_outcome)
        else:
            if shootout_home_win != 0.5:
                winner = "home" if shootout_home_win > 0.5 else "away"
            else:
                # A coin flip: pick the favourite, since stronger teams win shoot-outs
                # slightly more often (Wunderlich et al., 2020); home if the odds are equal.
                winner = "home" if home <= away else "away"
            key = (h + i, a + j, PENALTIES, winner)

    return KnockoutPrediction(
        home_advances=home_advances,
        away_advances=1.0 - home_advances,
        ends_in_regular_time=total(2, REGULAR_TIME),
        ends_in_extra_time=total(2, EXTRA_TIME),
        ends_on_penalties=total(2, PENALTIES),
        results=ranked[:top_n],
        expected_result=KnockoutResult(*key, probability=table.get(key, 0.0)),
        extra_time_outcomes=(et_home, et_draw, et_away),
        regular_time=regular,
    )
