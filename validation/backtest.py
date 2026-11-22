"""Backtest of scoreline_predictor against real results.

For every match, the closing odds are turned into a score distribution, which
is then compared with the score that actually occurred.

Usage (from the repository root):

    python3 validation/backtest.py

The match files are downloaded once into validation/data/ and reused. They are
copies of the season files published by football-data.co.uk (results plus
closing odds), taken from a public GitHub repository at a fixed commit. To use
your own copies, put files named <season>_<league>.csv (e.g. 2425_E0.csv) into
validation/data/ before running.

Standard library only. The design (matches, variants, measures) was fixed
before any result was looked at; see "Validation" in the README.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import random
import statistics
import sys
import urllib.request
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(__file__).resolve().parent / "data"
SOURCE_URL = (
    "https://raw.githubusercontent.com/tyee33/bigdata/"
    "cabec6d0a25f7c11dfde1e9730a822d40c046830/hw1/data/raw/{season}_{league}.csv"
)
LEAGUES = {"E0": "Premier League", "D1": "Bundesliga", "I1": "Serie A", "SP1": "La Liga", "F1": "Ligue 1"}
TEST_SEASONS = ["2122", "2223", "2324", "2425", "2526"]
# Earlier seasons are used only for the frequency baseline, never for testing.
HISTORY_SEASONS = ["1112", "1213", "1314", "1415", "1516", "1617", "1718", "1819", "1920", "2021"]

# Odds columns in the football-data.co.uk files. "C" marks closing odds.
PINNACLE = dict(home="PSCH", draw="PSCD", away="PSCA", over="PC>2.5", under="PC<2.5",
                ah_home="PCAHH", ah_away="PCAHA", ah_line="AHCh")
AVERAGE = dict(home="AvgCH", draw="AvgCD", away="AvgCA", over="AvgC>2.5", under="AvgC<2.5",
               ah_home="AvgCAHH", ah_away="AvgCAHA", ah_line="AHCh")

# name -> (odds source, markets passed to the model, fixed rho or None)
VARIANTS = {
    "full": ("pinnacle", ("1x2", "ou", "ah"), None),
    "independent": ("pinnacle", ("1x2", "ou", "ah"), 0.0),
    "1x2_ou": ("pinnacle", ("1x2", "ou"), None),
    "1x2": ("pinnacle", ("1x2",), 0.0),  # independent goals, the default up to version 0.2.0
    "full_average_odds": ("average", ("1x2", "ou", "ah"), None),
}
PRIMARY = "full"
BASELINE = "history"

# Exploratory, added after the first results had been seen: 1X2 odds only, with rho
# fixed at the median value that the odds of the first two test seasons imply
# (variant "1x2_ou"). It is evaluated on the later seasons only. The value comes
# from odds, not from results. Rounded to three decimals, it has been the module's
# DEFAULT_RHO since version 0.3.0.
MARKET_RHO_VARIANT = "1x2_market_rho"
MARKET_RHO_SEASONS = TEST_SEASONS[:2]

CALIBRATION_GRID = 7  # scorelines 0-0 to 6-6 are pooled for the calibration table
CALIBRATION_BINS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 1.0]
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1
HISTORY_GRID = 10  # the frequency baseline counts scores from 0-0 to 9-9


def download(seasons: List[str]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for season in seasons:
        for league in LEAGUES:
            target = DATA_DIR / f"{season}_{league}.csv"
            if not target.exists():
                url = SOURCE_URL.format(season=season, league=league)
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())


def read_matches(season: str, league: str) -> List[dict]:
    text = (DATA_DIR / f"{season}_{league}.csv").read_text(encoding="utf-8-sig", errors="replace")
    matches = []
    for row in csv.DictReader(io.StringIO(text)):
        try:
            goals = (int(row["FTHG"]), int(row["FTAG"]))
        except (KeyError, TypeError, ValueError):
            continue  # blank line or match without a result
        matches.append(dict(season=season, league=league, date=row.get("Date", ""),
                            home_team=row.get("HomeTeam", ""), away_team=row.get("AwayTeam", ""),
                            goals=goals, pinnacle=_odds(row, PINNACLE), average=_odds(row, AVERAGE)))
    return matches


def _odds(row: dict, columns: Dict[str, str]) -> Optional[dict]:
    """All three markets from one source, or None if any price is missing."""
    try:
        odds = {name: float(row[column]) for name, column in columns.items()}
    except (KeyError, TypeError, ValueError):
        return None
    prices = [value for name, value in odds.items() if name != "ah_line"]
    return odds if all(math.isfinite(p) and p > 1.0 for p in prices) else None


def outcome(home_goals: int, away_goals: int) -> str:
    return "H" if home_goals > away_goals else "D" if home_goals == away_goals else "A"


def predict(task: Tuple[int, str, dict, Tuple[int, int]]) -> Tuple[int, str, Optional[dict]]:
    """Run one variant on one match. Executed in worker processes."""
    index, variant, odds, goals, spec = task
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scoreline_predictor import predict_scorelines

    _, markets, rho = spec
    kwargs = dict(home=odds["home"], draw=odds["draw"], away=odds["away"])
    if "ou" in markets:
        kwargs.update(over=odds["over"], under=odds["under"], total_line=2.5)
    if "ah" in markets:
        kwargs.update(ah_home=odds["ah_home"], ah_away=odds["ah_away"], ah_line=odds["ah_line"])
    if rho is not None:
        kwargs.update(rho=rho)
    try:
        p = predict_scorelines(**kwargs)
    except ValueError:
        return index, variant, None
    return index, variant, summarise(p.score_matrix, goals, dict(
        rho=p.rho, fit_error=p.fit_error, expected_goals=p.expected_goals,
        outcome_probs=dict(H=p.home_win, D=p.draw, A=p.away_win),
        expected_scoreline=p.expected_scoreline,
    ))


def summarise(matrix: List[List[float]], goals: Tuple[int, int], extra: dict) -> dict:
    """Everything the evaluation needs from one predicted score distribution."""
    h, a = goals
    ranked = sorted(((p, i, j) for i, row in enumerate(matrix) for j, p in enumerate(row)), reverse=True)
    top = [(i, j) for _, i, j in ranked[:5]]
    probs = dict(H=0.0, D=0.0, A=0.0)
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            probs[outcome(i, j)] += p
    best_outcome = max(probs, key=probs.get)
    expected = extra.get("expected_scoreline") or next((i, j) for _, i, j in ranked if outcome(i, j) == best_outcome)
    in_grid = h < len(matrix) and a < len(matrix[0])
    return dict(
        p_actual=matrix[h][a] if in_grid else 0.0,
        expected_hit=tuple(expected) == (h, a),
        outcome_hit=outcome(*expected) == outcome(h, a),
        mode_hit=top[0] == (h, a),
        mode_outcome_hit=outcome(*top[0]) == outcome(h, a),
        top5_hit=(h, a) in top,
        expected_goals=extra.get("expected_goals") or (
            sum(i * p for i, row in enumerate(matrix) for p in row),
            sum(j * p for row in matrix for j, p in enumerate(row)),
        ),
        outcome_probs=extra.get("outcome_probs") or probs,
        cells=[matrix[i][j] if i < len(matrix) and j < len(matrix[0]) else 0.0
               for i in range(CALIBRATION_GRID) for j in range(CALIBRATION_GRID)],
        rho=extra.get("rho"),
        fit_error=extra.get("fit_error"),
    )


def history_matrix(counts: Counter) -> List[List[float]]:
    """Score frequencies of earlier seasons, with one added to every count (Laplace's rule)."""
    total = sum(counts.values()) + HISTORY_GRID * HISTORY_GRID
    return [[(counts[(i, j)] + 1) / total for j in range(HISTORY_GRID)] for i in range(HISTORY_GRID)]


def evaluate(matches: List[dict], results: Dict[str, List[Optional[dict]]], names: List[str]) -> dict:
    """Compare variants on the matches for which all of them have a prediction."""
    keep = [k for k in range(len(matches)) if all(results[n][k] is not None for n in names)]
    report = dict(matches=len(keep), variants={})
    scores = {}
    for name in names:
        rows = [results[name][k] for k in keep]
        goals = [matches[k]["goals"] for k in keep]
        log_scores = [-math.log(max(r["p_actual"], 1e-12)) for r in rows]
        scores[name] = log_scores
        n = len(rows)
        entry = dict(
            log_score=statistics.fmean(log_scores),
            typical_probability=math.exp(-statistics.fmean(log_scores)),
            expected_scoreline_exact=sum(r["expected_hit"] for r in rows) / n,
            expected_scoreline_outcome=sum(r["outcome_hit"] for r in rows) / n,
            most_likely_scoreline_exact=sum(r["mode_hit"] for r in rows) / n,
            most_likely_scoreline_outcome=sum(r["mode_outcome_hit"] for r in rows) / n,
            actual_in_top5=sum(r["top5_hit"] for r in rows) / n,
            predicted_goals=(statistics.fmean(r["expected_goals"][0] for r in rows),
                             statistics.fmean(r["expected_goals"][1] for r in rows)),
            actual_goals=(statistics.fmean(g[0] for g in goals), statistics.fmean(g[1] for g in goals)),
            scoreline_calibration=calibration(
                [p for r in rows for p in r["cells"]],
                [int((i, j) == g) for g in goals for i in range(CALIBRATION_GRID) for j in range(CALIBRATION_GRID)],
            ),
            outcome_calibration={
                o: dict(predicted=statistics.fmean(r["outcome_probs"][o] for r in rows),
                        observed=sum(outcome(*g) == o for g in goals) / n)
                for o in "HDA"
            },
        )
        rhos = [r["rho"] for r in rows if r["rho"] is not None]
        if rhos:
            q = statistics.quantiles(rhos, n=20)
            entry["rho"] = dict(median=statistics.median(rhos), p5=q[0], p25=q[4], p75=q[14], p95=q[18],
                                share_negative=sum(x < 0 for x in rhos) / len(rhos))
            errors = sorted(r["fit_error"] for r in rows)
            entry["fit_error"] = dict(median=statistics.median(errors), p95=errors[int(0.95 * len(errors))],
                                      max=errors[-1])
        report["variants"][name] = entry
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [rng.choices(range(len(keep)), k=len(keep)) for _ in range(BOOTSTRAP_RESAMPLES)]
    reference = names[0]
    for name in names[1:]:
        diff = [a - b for a, b in zip(scores[name], scores[reference])]
        means = sorted(statistics.fmean(diff[k] for k in sample) for sample in samples)
        report["variants"][name]["log_score_minus_reference"] = dict(
            reference=reference, mean=statistics.fmean(diff),
            ci95=(means[int(0.025 * len(means))], means[int(0.975 * len(means)) - 1]),
        )
    return report


def calibration(probabilities: List[float], hits: List[int]) -> List[dict]:
    table = []
    for low, high in zip(CALIBRATION_BINS, CALIBRATION_BINS[1:]):
        chosen = [(p, hit) for p, hit in zip(probabilities, hits) if low <= p < high]
        if chosen:
            table.append(dict(bin=(low, high), n=len(chosen),
                              predicted=statistics.fmean(p for p, _ in chosen),
                              observed=sum(hit for _, hit in chosen) / len(chosen)))
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workers", type=int, default=None, help="worker processes (default: all cores)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results.json")
    args = parser.parse_args()

    download(HISTORY_SEASONS + TEST_SEASONS)
    matches = [m for s in TEST_SEASONS for league in LEAGUES for m in read_matches(s, league)]

    # Frequency baseline: for each league and season, the scores of all earlier seasons of that league.
    results: Dict[str, List[Optional[dict]]] = {name: [None] * len(matches) for name in list(VARIANTS) + [BASELINE]}
    seasons = HISTORY_SEASONS + TEST_SEASONS
    earlier = {league: {s: Counter() for s in TEST_SEASONS} for league in LEAGUES}
    for league in LEAGUES:
        counts: Counter = Counter()
        for season in seasons:
            if season in earlier[league]:
                earlier[league][season] = Counter(counts)
            counts.update(m["goals"] for m in read_matches(season, league))
    for k, m in enumerate(matches):
        results[BASELINE][k] = summarise(history_matrix(earlier[m["league"]][m["season"]]), m["goals"], {})

    tasks = [(k, name, m[spec[0]], m["goals"], spec) for k, m in enumerate(matches)
             for name, spec in VARIANTS.items() if m[spec[0]] is not None]
    with Pool(args.workers) as pool:
        for done, (k, name, summary) in enumerate(pool.imap_unordered(predict, tasks, chunksize=50), 1):
            results[name][k] = summary
            if done % 5000 == 0:
                print(f"{done}/{len(tasks)} predictions", file=sys.stderr)

        # Exploratory variant: rho implied by the odds of the first two seasons, applied to 1X2 odds only.
        market_rho = statistics.median(
            r["rho"] for r, m in zip(results["1x2_ou"], matches) if r is not None and m["season"] in MARKET_RHO_SEASONS
        )
        spec = ("pinnacle", ("1x2",), market_rho)
        tasks = [(k, MARKET_RHO_VARIANT, m["pinnacle"], m["goals"], spec) for k, m in enumerate(matches)
                 if m["pinnacle"] is not None]
        results[MARKET_RHO_VARIANT] = [None] * len(matches)
        for k, name, summary in pool.imap_unordered(predict, tasks, chunksize=50):
            results[name][k] = summary

    pinnacle = [PRIMARY, "independent", "1x2_ou", "1x2", BASELINE]
    later = [k for k, m in enumerate(matches) if m["season"] not in MARKET_RHO_SEASONS]
    report = dict(
        exploratory_market_rho=dict(
            rho=market_rho, derived_from_seasons=MARKET_RHO_SEASONS,
            evaluation=evaluate([matches[k] for k in later], {n: [results[n][k] for k in later] for n in results},
                                [MARKET_RHO_VARIANT, "1x2", "1x2_ou"]),
        ),
        source=SOURCE_URL, leagues=LEAGUES, test_seasons=TEST_SEASONS, matches_with_result=len(matches),
        primary=evaluate(matches, results, pinnacle),
        average_odds=evaluate(matches, results, ["full_average_odds", BASELINE]),
        both_sources=evaluate(matches, results, [PRIMARY, "full_average_odds"]),
        by_league={LEAGUES[code]: evaluate([m for m in matches if m["league"] == code],
                                           {n: [r for r, m in zip(results[n], matches) if m["league"] == code]
                                            for n in results}, [PRIMARY, BASELINE])
                   for code in LEAGUES},
    )
    args.output.write_text(json.dumps(report, indent=1))
    print_report(report)


def print_report(report: dict) -> None:
    explore = report["exploratory_market_rho"]
    print(f"\nExploratory: rho = {explore['rho']:.4f}, the median implied by the odds of seasons "
          f"{explore['derived_from_seasons']}; evaluated on the later seasons")
    for title, part in (("Pinnacle closing odds", report["primary"]),
                        ("Market-average closing odds", report["average_odds"]),
                        ("Same matches, both odds sources", report["both_sources"]),
                        ("Exploratory: 1X2 odds only, later seasons", explore["evaluation"])):
        print(f"\n## {title}: {part['matches']} matches\n")
        print("| Variant | Log score | Typical prob. | Exp. score exact | Exp. score outcome | Mode exact | Mode outcome "
              "| In top 5 | Goals pred. | Goals actual | Log score vs first (95% CI) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, v in part["variants"].items():
            d = v.get("log_score_minus_reference")
            diff = f"{d['mean']:+.4f} ({d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})" if d else "reference"
            print(f"| {name} | {v['log_score']:.4f} | {v['typical_probability']:.2%} "
                  f"| {v['expected_scoreline_exact']:.2%} | {v['expected_scoreline_outcome']:.2%} "
                  f"| {v['most_likely_scoreline_exact']:.2%} | {v['most_likely_scoreline_outcome']:.2%} "
                  f"| {v['actual_in_top5']:.2%} | {sum(v['predicted_goals']):.3f} | {sum(v['actual_goals']):.3f} | {diff} |")
    first = report["primary"]["variants"][PRIMARY]
    print(f"\n## Calibration of '{PRIMARY}'\n")
    print("| Predicted probability of a scoreline | Scorelines | Mean predicted | Observed |\n|---|---|---|---|")
    for row in first["scoreline_calibration"]:
        print(f"| {row['bin'][0]:.0%} to {row['bin'][1]:.0%} | {row['n']} | {row['predicted']:.2%} | {row['observed']:.2%} |")
    print("\n| | Predicted | Observed |\n|---|---|---|")
    for o, label in (("H", "Home win"), ("D", "Draw"), ("A", "Away win")):
        c = first["outcome_calibration"][o]
        print(f"| {label} | {c['predicted']:.2%} | {c['observed']:.2%} |")
    print(f"| Home goals | {first['predicted_goals'][0]:.3f} | {first['actual_goals'][0]:.3f} |")
    print(f"| Away goals | {first['predicted_goals'][1]:.3f} | {first['actual_goals'][1]:.3f} |")
    print(f"\nrho: {first['rho']}\nfit_error: {first['fit_error']}")
    for name in ("independent", "1x2_ou", "1x2"):
        v = report["primary"]["variants"][name]
        print(f"{name}: predicted goals {v['predicted_goals'][0]:.3f}-{v['predicted_goals'][1]:.3f}, "
              f"draw predicted {v['outcome_calibration']['D']['predicted']:.2%}, rho {v.get('rho')}")
    print("\n## By league (full model against frequency baseline)\n")
    for league, part in report["by_league"].items():
        v, b = part["variants"][PRIMARY], part["variants"][BASELINE]
        d = b["log_score_minus_reference"]
        print(f"{league}: {part['matches']} matches, log score {v['log_score']:.4f}, baseline {b['log_score']:.4f} "
              f"(difference {d['mean']:+.4f}, CI {d['ci95'][0]:+.4f} to {d['ci95'][1]:+.4f})")


if __name__ == "__main__":
    main()
