"""Predict soccer scorelines for a single match from closing odds.

A Dixon-Coles model (two Poisson goal distributions plus a low-score
correction) is fitted to the margin-free closing odds of one match.

Example
-------
>>> from scoreline_predictor import predict_scorelines
>>> p = predict_scorelines(home=1.85, draw=3.70, away=4.40, over=1.80, under=2.05)
>>> p.home_win, p.draw, p.away_win
>>> p.scorelines          # top scorelines with probabilities
>>> p.expected_scoreline  # single most likely scoreline, e.g. (1, 1)
>>> p.expected_goals      # mean goals per team, e.g. (1.62, 1.03)

Pure standard library, Python 3.9+.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

__version__ = "0.1.0"
__all__ = ["predict_scorelines", "ScorelinePrediction", "Scoreline", "remove_margin"]

DEFAULT_RHO = -0.08  # typical Dixon-Coles low-score correction in modern soccer
_RHO_BOUNDS = (-0.25, 0.25)
_RHO_PRIOR_WEIGHT = 0.05  # pulls a fitted rho gently towards DEFAULT_RHO
_MAX_GOALS = 15  # per team; the grid is renormalised after truncation


@dataclass(frozen=True)
class Scoreline:
    home_goals: int
    away_goals: int
    probability: float

    def __str__(self) -> str:
        return f"{self.home_goals}-{self.away_goals} ({self.probability:.1%})"


@dataclass(frozen=True)
class ScorelinePrediction:
    home_win: float
    draw: float
    away_win: float
    scorelines: List[Scoreline]  # top N, most likely first
    expected_scoreline: Tuple[int, int]  # most likely scoreline of the most likely outcome
    expected_goals: Tuple[float, float]  # mean goals (home, away) over the full distribution
    lambda_home: float
    lambda_away: float
    rho: float
    fit_error: float  # largest absolute gap between model and market probabilities
    score_matrix: List[List[float]] = field(repr=False)  # [home_goals][away_goals]

    def __str__(self) -> str:
        top = ", ".join(str(s) for s in self.scorelines)
        return (
            f"Home {self.home_win:.1%} | Draw {self.draw:.1%} | Away {self.away_win:.1%}\n"
            f"Expected scoreline: {self.expected_scoreline[0]}-{self.expected_scoreline[1]} "
            f"(expected goals {self.expected_goals[0]:.2f}-{self.expected_goals[1]:.2f})\n"
            f"Top scorelines: {top}"
        )


def remove_margin(odds: Sequence[float]) -> List[float]:
    """Convert decimal odds of one market into probabilities that sum to 1.

    Uses the power method (p_i = (1/odds_i) ** k), which takes more margin
    out of longshots than plain normalisation does.
    """
    if not all(math.isfinite(o) and o > 1.0 for o in odds):
        raise ValueError(f"Decimal odds must be finite numbers greater than 1, got {list(odds)}")
    raw = [1.0 / o for o in odds]
    if sum(raw) <= 1.0:  # no margin (or an arbitrage): just normalise
        total = sum(raw)
        return [r / total for r in raw]
    lo, hi = 1.0, 2.0
    while sum(r**hi for r in raw) > 1.0:
        hi *= 2.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if sum(r**mid for r in raw) > 1.0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    probs = [r**k for r in raw]
    total = sum(probs)
    return [p / total for p in probs]


def _score_matrix(lam_home: float, lam_away: float, rho: float) -> List[List[float]]:
    def pmf(lam: float) -> List[float]:
        out = [math.exp(-lam)]
        for k in range(1, _MAX_GOALS + 1):
            out.append(out[-1] * lam / k)
        return out

    ph, pa = pmf(lam_home), pmf(lam_away)
    m = [[h * a for a in pa] for h in ph]
    m[0][0] *= max(0.0, 1.0 - lam_home * lam_away * rho)
    m[0][1] *= max(0.0, 1.0 + lam_home * rho)
    m[1][0] *= max(0.0, 1.0 + lam_away * rho)
    m[1][1] *= max(0.0, 1.0 - rho)
    total = sum(map(sum, m))
    return [[v / total for v in row] for row in m]


def _market_probs(m: List[List[float]]) -> Tuple[float, float, float, float]:
    home = draw = away = btts = 0.0
    for h, row in enumerate(m):
        for a, p in enumerate(row):
            if h > a:
                home += p
            elif h == a:
                draw += p
            else:
                away += p
            if h and a:
                btts += p
    return home, draw, away, btts


def _check_line(name: str, line: float) -> None:
    if (line * 4) % 1 != 0:
        raise ValueError(f"{name} must be a multiple of 0.25, got {line}")


def _line_prob(m: List[List[float]], line: float, handicap: bool) -> float:
    """Fair implied probability of the over (or of the home side of a handicap).

    Bets are settled as bookmakers settle them: a whole line refunds the stake
    when it is hit exactly, and a quarter line splits the stake between the two
    neighbouring lines. The result is win / (win + lose) in stake-weighted
    terms, which is the reciprocal of the fair decimal odds. For a half line
    this is simply the probability of winning the bet.
    """
    parts = [line] if (line * 2) % 1 == 0 else [line - 0.25, line + 0.25]
    win = lose = 0.0
    for h, row in enumerate(m):
        for a, p in enumerate(row):
            for part in parts:
                diff = (h - a + part) if handicap else (h + a - part)
                if diff > 0:
                    win += p
                elif diff < 0:
                    lose += p
    return win / (win + lose)


def _nelder_mead(f: Callable[[List[float]], float], x0: List[float], step: float = 0.2,
                 tol: float = 1e-16, max_iter: int = 2000) -> List[float]:
    n = len(x0)
    simplex = [list(x0)] + [[x + (step if i == j else 0.0) for j, x in enumerate(x0)] for i in range(n)]
    values = [f(x) for x in simplex]
    for _ in range(max_iter):
        order = sorted(range(n + 1), key=values.__getitem__)
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if values[-1] - values[0] < tol:
            break
        centroid = [sum(x[j] for x in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]

        def along(t: float) -> List[float]:
            return [c + t * (c - w) for c, w in zip(centroid, worst)]

        xr = along(1.0)
        fr = f(xr)
        if fr < values[0]:
            xe = along(2.0)
            fe = f(xe)
            simplex[-1], values[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < values[-2]:
            simplex[-1], values[-1] = xr, fr
        else:
            xc = along(0.5) if fr < values[-1] else along(-0.5)
            fc = f(xc)
            if fc < min(fr, values[-1]):
                simplex[-1], values[-1] = xc, fc
            else:  # shrink towards the best point
                best = simplex[0]
                simplex = [best] + [[b + 0.5 * (x - b) for b, x in zip(best, pt)] for pt in simplex[1:]]
                values = [values[0]] + [f(x) for x in simplex[1:]]
    return simplex[min(range(n + 1), key=values.__getitem__)]


def predict_scorelines(
    home: float,
    draw: float,
    away: float,
    over: Optional[float] = None,
    under: Optional[float] = None,
    total_line: float = 2.5,
    ah_home: Optional[float] = None,
    ah_away: Optional[float] = None,
    ah_line: Optional[float] = None,
    btts_yes: Optional[float] = None,
    btts_no: Optional[float] = None,
    top_n: int = 5,
    rho: Optional[float] = None,
) -> ScorelinePrediction:
    """Predict scoreline probabilities for one match from decimal closing odds.

    Parameters
    ----------
    home, draw, away : decimal 1X2 closing odds (required).
    over, under : decimal odds for over/under ``total_line`` goals (optional,
        but strongly recommended: without them the total number of goals is
        only weakly determined by the draw price).
    total_line : the goal line of the over/under market. Any multiple of 0.25
        works, e.g. 2.5, 3 or 2.75.
    ah_home, ah_away, ah_line : decimal Asian handicap odds and the handicap
        applied to the home team (optional). ``ah_line=-0.75`` means the home
        team gives up 0.75 goals; ``ah_line=0.5`` means it receives half a
        goal. Any multiple of 0.25 works.
    btts_yes, btts_no : decimal odds for both teams to score (optional).
    top_n : number of most likely scorelines to return.
    rho : Dixon-Coles low-score correction. ``None`` (default) estimates it
        from the odds when an over/under, Asian handicap or BTTS market
        is given, and otherwise uses ``DEFAULT_RHO``. Pass a number to fix it.

    All odds are for the result after 90 minutes plus stoppage time.
    """
    if (over is None) != (under is None):
        raise ValueError("Provide both 'over' and 'under', or neither")
    if (btts_yes is None) != (btts_no is None):
        raise ValueError("Provide both 'btts_yes' and 'btts_no', or neither")
    if len({ah_home is None, ah_away is None, ah_line is None}) != 1:
        raise ValueError("Provide all of 'ah_home', 'ah_away' and 'ah_line', or none")
    if over is not None:
        _check_line("total_line", total_line)
        if total_line <= 0:
            raise ValueError(f"total_line must be positive, got {total_line}")
    if ah_line is not None:
        _check_line("ah_line", ah_line)
    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if rho is not None and not _RHO_BOUNDS[0] <= rho <= _RHO_BOUNDS[1]:
        raise ValueError(f"rho must be within {_RHO_BOUNDS}, got {rho}")

    target_1x2 = remove_margin([home, draw, away])
    target_over = remove_margin([over, under])[0] if over is not None else None
    target_ah = remove_margin([ah_home, ah_away])[0] if ah_home is not None else None
    target_btts = remove_margin([btts_yes, btts_no])[0] if btts_yes is not None else None
    fit_rho = rho is None and any(t is not None for t in (target_over, target_ah, target_btts))
    fixed_rho = DEFAULT_RHO if rho is None else rho

    def unpack(x: List[float]) -> Tuple[float, float, float]:
        lam_h = math.exp(min(max(x[0], -5.0), 2.3))  # lambda between ~0.007 and ~10
        lam_a = math.exp(min(max(x[1], -5.0), 2.3))
        r = min(max(x[2], _RHO_BOUNDS[0]), _RHO_BOUNDS[1]) if fit_rho else fixed_rho
        return lam_h, lam_a, r

    def gaps(lam_h: float, lam_a: float, r: float) -> List[float]:
        m = _score_matrix(lam_h, lam_a, r)
        mh, md, ma, mb = _market_probs(m)
        out = [mh - target_1x2[0], md - target_1x2[1], ma - target_1x2[2]]
        if target_over is not None:
            out.append(_line_prob(m, total_line, handicap=False) - target_over)
        if target_ah is not None:
            out.append(_line_prob(m, ah_line, handicap=True) - target_ah)
        if target_btts is not None:
            out.append(mb - target_btts)
        return out

    def loss(x: List[float]) -> float:
        lam_h, lam_a, r = unpack(x)
        total = sum(g * g for g in gaps(lam_h, lam_a, r))
        if fit_rho:
            total += (_RHO_PRIOR_WEIGHT * (r - DEFAULT_RHO)) ** 2
        return total

    x = [math.log(1.5), math.log(1.1)] + ([DEFAULT_RHO] if fit_rho else [])
    for step in (0.2, 0.05, 0.01):  # restarts guard against a stalled simplex
        x = _nelder_mead(loss, x, step=step)
    lam_h, lam_a, r = unpack(x + ([] if fit_rho else [fixed_rho]))

    matrix = _score_matrix(lam_h, lam_a, r)
    p_home, p_draw, p_away, _ = _market_probs(matrix)
    ranked = sorted(
        (Scoreline(h, a, p) for h, row in enumerate(matrix) for a, p in enumerate(row)),
        key=lambda s: -s.probability,
    )

    # Expected scoreline: the most likely scoreline within the most likely
    # outcome, so that it never contradicts the 1X2 probabilities.
    outcome = max((p_home, "H"), (p_draw, "D"), (p_away, "A"))[1]
    consistent = {
        "H": lambda s: s.home_goals > s.away_goals,
        "D": lambda s: s.home_goals == s.away_goals,
        "A": lambda s: s.home_goals < s.away_goals,
    }[outcome]
    expected = next(s for s in ranked if consistent(s))

    return ScorelinePrediction(
        home_win=p_home,
        draw=p_draw,
        away_win=p_away,
        scorelines=ranked[:top_n],
        expected_scoreline=(expected.home_goals, expected.away_goals),
        expected_goals=(
            sum(h * p for h, row in enumerate(matrix) for p in row),
            sum(a * p for row in matrix for a, p in enumerate(row)),
        ),
        lambda_home=lam_h,
        lambda_away=lam_a,
        rho=r,
        fit_error=max(abs(g) for g in gaps(lam_h, lam_a, r)),
        score_matrix=matrix,
    )
