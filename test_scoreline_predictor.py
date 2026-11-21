"""Tests for scoreline_predictor.

Run with:  python3 -m unittest -v        (or: pytest)

Each test class specifies one piece of behaviour. The strongest checks are the
round-trip tests: fair odds are generated from known model parameters, and the
predictor has to recover those parameters from the odds alone.
"""

import dataclasses
import math
import unittest

import scoreline_predictor as sp
from scoreline_predictor import Scoreline, ScorelinePrediction, predict_scorelines, remove_margin

# A plausible set of closing odds for one match, reused across tests.
ODDS_1X2 = dict(home=1.85, draw=3.70, away=4.40)
ODDS_TOTALS = dict(over=1.80, under=2.05)
ODDS_FULL = dict(**ODDS_1X2, **ODDS_TOTALS)


def poisson(lam, k):
    return math.exp(-lam) * lam**k / math.factorial(k)


def cells(matrix):
    return [(h, a, p) for h, row in enumerate(matrix) for a, p in enumerate(row)]


def outcome_probs(matrix):
    home = sum(p for h, a, p in cells(matrix) if h > a)
    draw = sum(p for h, a, p in cells(matrix) if h == a)
    away = sum(p for h, a, p in cells(matrix) if h < a)
    return home, draw, away


def fair_odds(lam_home, lam_away, rho, total_line=None, ah_line=None, btts=False):
    """Margin-free odds implied by known model parameters."""
    m = sp._score_matrix(lam_home, lam_away, rho)
    home, draw, away = outcome_probs(m)
    odds = dict(home=1 / home, draw=1 / draw, away=1 / away)
    if total_line is not None:
        p = sp._line_prob(m, total_line, handicap=False)
        odds.update(over=1 / p, under=1 / (1 - p), total_line=total_line)
    if ah_line is not None:
        p = sp._line_prob(m, ah_line, handicap=True)
        odds.update(ah_home=1 / p, ah_away=1 / (1 - p), ah_line=ah_line)
    if btts:
        p = sum(p for h, a, p in cells(m) if h and a)
        odds.update(btts_yes=1 / p, btts_no=1 / (1 - p))
    return odds


class RemoveMarginTest(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for odds in ([1.85, 3.70, 4.40], [1.12, 10.0, 21.0], [1.90, 1.90], [1.01, 25.0, 60.0]):
            with self.subTest(odds=odds):
                self.assertAlmostEqual(sum(remove_margin(odds)), 1.0, places=12)

    def test_all_probabilities_are_strictly_between_zero_and_one(self):
        for p in remove_margin([1.01, 25.0, 60.0]):
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_fair_odds_are_returned_unchanged(self):
        probs = remove_margin([2.0, 4.0, 4.0])
        for got, want in zip(probs, [0.5, 0.25, 0.25]):
            self.assertAlmostEqual(got, want, places=12)

    def test_equal_odds_give_equal_probabilities(self):
        self.assertEqual(remove_margin([1.90, 1.90]), [0.5, 0.5])
        for p in remove_margin([2.8, 2.8, 2.8]):
            self.assertAlmostEqual(p, 1 / 3, places=12)

    def test_shorter_odds_get_higher_probability(self):
        probs = remove_margin([1.85, 3.70, 4.40])
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])

    def test_output_order_matches_input_order(self):
        forward = remove_margin([1.85, 3.70, 4.40])
        backward = remove_margin([4.40, 3.70, 1.85])
        for f, b in zip(forward, reversed(backward)):
            self.assertAlmostEqual(f, b, places=12)

    def test_longshots_lose_more_margin_than_with_plain_normalisation(self):
        odds = [1.30, 5.50, 11.0]
        raw = [1 / o for o in odds]
        proportional = [r / sum(raw) for r in raw]
        power = remove_margin(odds)
        self.assertLess(power[2], proportional[2])  # longshot shrinks more
        self.assertGreater(power[0], proportional[0])  # favourite shrinks less

    def test_odds_without_margin_or_below_it_are_normalised(self):
        probs = remove_margin([2.2, 4.4, 4.4])  # implied probabilities sum to < 1
        self.assertAlmostEqual(sum(probs), 1.0, places=12)
        self.assertAlmostEqual(probs[0], 0.5, places=12)

    def test_accepts_any_sequence(self):
        self.assertEqual(remove_margin((1.90, 1.90)), [0.5, 0.5])

    def test_rejects_odds_of_one_or_less(self):
        for bad in (1.0, 0.5, 0.0, -2.0):
            with self.subTest(odds=bad), self.assertRaises(ValueError):
                remove_margin([bad, 3.0, 4.0])

    def test_rejects_non_finite_odds(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(odds=bad), self.assertRaises(ValueError):
                remove_margin([2.0, bad, 4.0])


class ScoreMatrixTest(unittest.TestCase):
    def test_sums_to_one(self):
        for params in ((1.5, 1.1, -0.08), (3.3, 0.5, 0.0), (0.4, 0.4, -0.2), (6.0, 2.0, 0.1)):
            with self.subTest(params=params):
                self.assertAlmostEqual(sum(map(sum, sp._score_matrix(*params))), 1.0, places=12)

    def test_has_no_negative_probabilities(self):
        for params in ((1.5, 1.1, -0.25), (4.0, 3.0, 0.25), (8.0, 6.0, 0.25)):
            with self.subTest(params=params):
                self.assertTrue(all(p >= 0.0 for _, _, p in cells(sp._score_matrix(*params))))

    def test_is_square_and_indexed_home_goals_first(self):
        m = sp._score_matrix(2.5, 0.6, 0.0)
        self.assertTrue(all(len(row) == len(m) for row in m))
        self.assertGreater(m[3][0], m[0][3])  # the stronger home team wins 3-0 more often

    def test_zero_rho_is_independent_poisson(self):
        lam_h, lam_a = 1.6, 1.2
        m = sp._score_matrix(lam_h, lam_a, 0.0)
        for h in range(6):
            for a in range(6):
                self.assertAlmostEqual(m[h][a], poisson(lam_h, h) * poisson(lam_a, a), places=9)

    def test_negative_rho_moves_probability_towards_low_scoring_draws(self):
        base = sp._score_matrix(1.6, 1.2, 0.0)
        adj = sp._score_matrix(1.6, 1.2, -0.1)
        self.assertGreater(adj[0][0], base[0][0])
        self.assertGreater(adj[1][1], base[1][1])
        self.assertLess(adj[1][0], base[1][0])
        self.assertLess(adj[0][1], base[0][1])

    def test_correction_follows_the_dixon_coles_formula(self):
        lam_h, lam_a, rho = 1.6, 1.2, -0.1
        base = sp._score_matrix(lam_h, lam_a, 0.0)
        adj = sp._score_matrix(lam_h, lam_a, rho)
        self.assertAlmostEqual(adj[0][0] / base[0][0], 1 - lam_h * lam_a * rho, places=9)
        self.assertAlmostEqual(adj[1][1] / base[1][1], 1 - rho, places=9)
        self.assertAlmostEqual(adj[1][0] / base[1][0], 1 + lam_a * rho, places=9)
        self.assertAlmostEqual(adj[0][1] / base[0][1], 1 + lam_h * rho, places=9)

    def test_correction_leaves_all_other_scorelines_alone(self):
        base = sp._score_matrix(1.6, 1.2, 0.0)
        adj = sp._score_matrix(1.6, 1.2, -0.1)
        for h, a, p in cells(adj):
            if (h, a) not in {(0, 0), (0, 1), (1, 0), (1, 1)}:
                self.assertAlmostEqual(p, base[h][a], places=9)

    def test_correction_preserves_expected_goals(self):
        m = sp._score_matrix(1.6, 1.2, -0.1)
        self.assertAlmostEqual(sum(h * p for h, _, p in cells(m)), 1.6, places=6)
        self.assertAlmostEqual(sum(a * p for _, a, p in cells(m)), 1.2, places=6)


class LineSettlementTest(unittest.TestCase):
    def setUp(self):
        self.m = sp._score_matrix(1.7, 1.0, -0.08)
        self.home, self.draw, self.away = outcome_probs(self.m)

    def total(self, k):
        return sum(p for h, a, p in cells(self.m) if h + a == k)

    def over(self, k):  # probability of more than k goals
        return sum(p for h, a, p in cells(self.m) if h + a > k)

    def margin(self, k):  # probability the home team wins by exactly k
        return sum(p for h, a, p in cells(self.m) if h - a == k)

    def test_half_total_line_is_the_probability_of_going_over(self):
        for line in (0.5, 1.5, 2.5, 3.5):
            with self.subTest(line=line):
                self.assertAlmostEqual(sp._line_prob(self.m, line, False), self.over(line), places=12)

    def test_whole_total_line_ignores_the_refunded_scorelines(self):
        want = self.over(3) / (1 - self.total(3))
        self.assertAlmostEqual(sp._line_prob(self.m, 3.0, False), want, places=12)

    def test_quarter_total_line_splits_the_stake(self):
        # Over 2.75 = half the stake on over 2.5, half on over 3.
        win = (self.over(2) + self.over(3)) / 2
        lose = ((1 - self.over(2)) + (1 - self.over(3) - self.total(3))) / 2
        self.assertAlmostEqual(sp._line_prob(self.m, 2.75, False), win / (win + lose), places=12)

    def test_quarter_line_lies_between_its_neighbouring_lines(self):
        lower = sp._line_prob(self.m, 2.5, False)
        quarter = sp._line_prob(self.m, 2.75, False)
        upper = sp._line_prob(self.m, 3.0, False)
        self.assertGreater(lower, quarter)
        self.assertGreater(quarter, upper)

    def test_higher_total_line_is_harder_to_go_over(self):
        probs = [sp._line_prob(self.m, line / 4, False) for line in range(2, 22)]
        self.assertEqual(probs, sorted(probs, reverse=True))
        self.assertEqual(len(set(probs)), len(probs))

    def test_handicap_minus_half_is_a_home_win(self):
        self.assertAlmostEqual(sp._line_prob(self.m, -0.5, True), self.home, places=12)

    def test_handicap_plus_half_is_home_win_or_draw(self):
        self.assertAlmostEqual(sp._line_prob(self.m, 0.5, True), self.home + self.draw, places=12)

    def test_handicap_zero_is_draw_no_bet(self):
        self.assertAlmostEqual(sp._line_prob(self.m, 0.0, True), self.home / (1 - self.draw), places=12)

    def test_handicap_minus_one_refunds_a_one_goal_win(self):
        win = self.home - self.margin(1)
        lose = self.draw + self.away
        self.assertAlmostEqual(sp._line_prob(self.m, -1.0, True), win / (win + lose), places=12)

    def test_quarter_handicap_splits_the_stake(self):
        # Home -0.75 = half the stake on -0.5, half on -1.
        win = (self.home + (self.home - self.margin(1))) / 2
        lose = ((self.draw + self.away) + (self.draw + self.away)) / 2
        self.assertAlmostEqual(sp._line_prob(self.m, -0.75, True), win / (win + lose), places=12)

    def test_more_generous_handicap_helps_the_home_side(self):
        probs = [sp._line_prob(self.m, line / 4, True) for line in range(-12, 13)]
        self.assertEqual(probs, sorted(probs))

    def test_handicap_is_mirrored_for_the_away_side(self):
        mirrored = sp._score_matrix(1.0, 1.7, -0.08)
        for line in (-1.25, -0.75, -0.5, 0.0, 0.25, 1.0):
            with self.subTest(line=line):
                home_side = sp._line_prob(self.m, line, True)
                away_side = 1 - sp._line_prob(mirrored, -line, True)
                self.assertAlmostEqual(home_side, away_side, places=12)


class OutcomeProbabilitiesTest(unittest.TestCase):
    def test_sum_to_one(self):
        for odds in (ODDS_1X2, ODDS_FULL, dict(home=1.12, draw=10.0, away=21.0, over=1.33, under=3.4)):
            with self.subTest(odds=odds):
                p = predict_scorelines(**odds)
                self.assertAlmostEqual(p.home_win + p.draw + p.away_win, 1.0, places=12)

    def test_match_the_margin_free_market(self):
        for odds in (ODDS_1X2, ODDS_FULL):
            with self.subTest(odds=odds):
                p = predict_scorelines(**odds)
                want = remove_margin([odds["home"], odds["draw"], odds["away"]])
                for got, target in zip((p.home_win, p.draw, p.away_win), want):
                    self.assertAlmostEqual(got, target, delta=0.002)

    def test_match_the_score_matrix(self):
        p = predict_scorelines(**ODDS_FULL)
        home, draw, away = outcome_probs(p.score_matrix)
        self.assertAlmostEqual(p.home_win, home, places=12)
        self.assertAlmostEqual(p.draw, draw, places=12)
        self.assertAlmostEqual(p.away_win, away, places=12)

    def test_favourite_is_the_most_likely_winner(self):
        self.assertGreater(predict_scorelines(home=1.5, draw=4.2, away=7.0).home_win, 0.5)
        self.assertGreater(predict_scorelines(home=7.0, draw=4.2, away=1.5).away_win, 0.5)

    def test_over_under_market_is_matched(self):
        p = predict_scorelines(**ODDS_FULL)
        want = remove_margin([ODDS_FULL["over"], ODDS_FULL["under"]])[0]
        got = sum(prob for h, a, prob in cells(p.score_matrix) if h + a > 2.5)
        self.assertAlmostEqual(got, want, delta=0.002)


class TopScorelinesTest(unittest.TestCase):
    def test_returns_five_scorelines_by_default(self):
        self.assertEqual(len(predict_scorelines(**ODDS_FULL).scorelines), 5)

    def test_returns_the_requested_number(self):
        for n in (1, 3, 10, 25):
            with self.subTest(top_n=n):
                self.assertEqual(len(predict_scorelines(**ODDS_FULL, top_n=n).scorelines), n)

    def test_are_scoreline_objects_with_whole_non_negative_goals(self):
        for s in predict_scorelines(**ODDS_FULL, top_n=10).scorelines:
            self.assertIsInstance(s, Scoreline)
            self.assertIsInstance(s.home_goals, int)
            self.assertIsInstance(s.away_goals, int)
            self.assertGreaterEqual(s.home_goals, 0)
            self.assertGreaterEqual(s.away_goals, 0)

    def test_are_sorted_from_most_to_least_likely(self):
        probs = [s.probability for s in predict_scorelines(**ODDS_FULL, top_n=30).scorelines]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_are_all_different(self):
        scores = [(s.home_goals, s.away_goals) for s in predict_scorelines(**ODDS_FULL, top_n=30).scorelines]
        self.assertEqual(len(set(scores)), len(scores))

    def test_first_one_is_the_most_likely_cell_of_the_matrix(self):
        p = predict_scorelines(**ODDS_FULL)
        self.assertEqual(p.scorelines[0].probability, max(prob for _, _, prob in cells(p.score_matrix)))

    def test_probabilities_match_the_matrix(self):
        p = predict_scorelines(**ODDS_FULL, top_n=10)
        for s in p.scorelines:
            self.assertEqual(s.probability, p.score_matrix[s.home_goals][s.away_goals])

    def test_nothing_outside_the_list_is_more_likely(self):
        p = predict_scorelines(**ODDS_FULL, top_n=5)
        listed = {(s.home_goals, s.away_goals) for s in p.scorelines}
        cutoff = p.scorelines[-1].probability
        for h, a, prob in cells(p.score_matrix):
            if (h, a) not in listed:
                self.assertLessEqual(prob, cutoff)

    def test_larger_top_n_extends_the_same_list(self):
        short = predict_scorelines(**ODDS_FULL, top_n=3).scorelines
        long = predict_scorelines(**ODDS_FULL, top_n=8).scorelines
        self.assertEqual(long[:3], short)

    def test_include_draws(self):
        scores = {(s.home_goals, s.away_goals) for s in predict_scorelines(**ODDS_FULL).scorelines}
        self.assertIn((1, 1), scores)

    def test_requesting_every_scoreline_covers_the_whole_distribution(self):
        p = predict_scorelines(**ODDS_FULL, top_n=10_000)
        self.assertEqual(len(p.scorelines), len(cells(p.score_matrix)))
        self.assertAlmostEqual(sum(s.probability for s in p.scorelines), 1.0, places=12)


class ExpectedScorelineTest(unittest.TestCase):
    def test_is_a_pair_of_whole_numbers(self):
        home_goals, away_goals = predict_scorelines(**ODDS_FULL).expected_scoreline
        self.assertIsInstance(home_goals, int)
        self.assertIsInstance(away_goals, int)

    def test_agrees_with_the_most_likely_outcome(self):
        home_fav = predict_scorelines(home=1.5, draw=4.2, away=7.0, over=1.9, under=1.9)
        self.assertGreater(home_fav.expected_scoreline[0], home_fav.expected_scoreline[1])
        away_fav = predict_scorelines(home=7.0, draw=4.2, away=1.5, over=1.9, under=1.9)
        self.assertLess(away_fav.expected_scoreline[0], away_fav.expected_scoreline[1])

    def test_is_a_draw_when_the_draw_is_the_most_likely_outcome(self):
        p = predict_scorelines(home=2.9, draw=2.75, away=2.95, over=2.6, under=1.5)
        self.assertGreater(p.draw, max(p.home_win, p.away_win))
        self.assertEqual(p.expected_scoreline[0], p.expected_scoreline[1])

    def test_can_differ_from_the_single_most_likely_scoreline(self):
        # A moderate favourite: 1-1 is the most likely score, a home win the most likely outcome.
        p = predict_scorelines(**ODDS_FULL)
        top = p.scorelines[0]
        self.assertEqual((top.home_goals, top.away_goals), (1, 1))
        self.assertEqual(p.expected_scoreline, (2, 1))

    def test_is_the_most_likely_scoreline_of_that_outcome(self):
        p = predict_scorelines(**ODDS_FULL)
        best_home_win = max((prob, (h, a)) for h, a, prob in cells(p.score_matrix) if h > a)[1]
        self.assertEqual(p.expected_scoreline, best_home_win)

    def test_does_not_depend_on_top_n(self):
        self.assertEqual(
            predict_scorelines(**ODDS_FULL, top_n=1).expected_scoreline,
            predict_scorelines(**ODDS_FULL, top_n=20).expected_scoreline,
        )


class ExpectedGoalsTest(unittest.TestCase):
    def test_is_the_mean_of_the_full_distribution(self):
        p = predict_scorelines(**ODDS_FULL)
        self.assertAlmostEqual(p.expected_goals[0], sum(h * prob for h, _, prob in cells(p.score_matrix)), places=12)
        self.assertAlmostEqual(p.expected_goals[1], sum(a * prob for _, a, prob in cells(p.score_matrix)), places=12)

    def test_is_close_to_the_fitted_goal_rates(self):
        p = predict_scorelines(**ODDS_FULL)
        self.assertAlmostEqual(p.expected_goals[0], p.lambda_home, delta=0.01)
        self.assertAlmostEqual(p.expected_goals[1], p.lambda_away, delta=0.01)

    def test_favourite_is_expected_to_score_more(self):
        p = predict_scorelines(home=1.5, draw=4.2, away=7.0)
        self.assertGreater(p.expected_goals[0], p.expected_goals[1])

    def test_shorter_over_odds_mean_more_goals(self):
        low = predict_scorelines(**ODDS_1X2, over=2.40, under=1.57)
        high = predict_scorelines(**ODDS_1X2, over=1.50, under=2.60)
        self.assertGreater(sum(high.expected_goals), sum(low.expected_goals) + 0.5)

    def test_does_not_depend_on_top_n(self):
        self.assertEqual(
            predict_scorelines(**ODDS_FULL, top_n=1).expected_goals,
            predict_scorelines(**ODDS_FULL, top_n=20).expected_goals,
        )


class SymmetryTest(unittest.TestCase):
    def test_swapping_the_teams_mirrors_the_prediction(self):
        p = predict_scorelines(home=1.85, draw=3.70, away=4.40, over=1.80, under=2.05,
                               ah_home=2.08, ah_away=1.84, ah_line=-0.75)
        q = predict_scorelines(home=4.40, draw=3.70, away=1.85, over=1.80, under=2.05,
                               ah_home=1.84, ah_away=2.08, ah_line=0.75)
        self.assertAlmostEqual(p.home_win, q.away_win, places=4)
        self.assertAlmostEqual(p.draw, q.draw, places=4)
        self.assertAlmostEqual(p.lambda_home, q.lambda_away, places=3)
        self.assertAlmostEqual(p.lambda_away, q.lambda_home, places=3)
        self.assertEqual(p.expected_scoreline, q.expected_scoreline[::-1])
        self.assertEqual(
            [(s.home_goals, s.away_goals) for s in p.scorelines],
            [(s.away_goals, s.home_goals) for s in q.scorelines],
        )

    def test_evenly_matched_teams_get_equal_goal_rates(self):
        p = predict_scorelines(home=2.7, draw=3.3, away=2.7, over=1.9, under=1.9)
        self.assertAlmostEqual(p.home_win, p.away_win, places=4)
        self.assertAlmostEqual(p.lambda_home, p.lambda_away, places=3)


class ParameterRecoveryTest(unittest.TestCase):
    """Fair odds generated from known parameters must lead back to those parameters."""

    MATCHES = [(1.76, 1.05), (1.20, 1.20), (0.90, 2.10), (2.60, 0.70), (0.80, 0.70), (3.30, 0.55)]

    def assert_recovers(self, prediction, lam_home, lam_away, rho, tolerance=0.01):
        self.assertAlmostEqual(prediction.lambda_home, lam_home, delta=tolerance)
        self.assertAlmostEqual(prediction.lambda_away, lam_away, delta=tolerance)
        self.assertAlmostEqual(prediction.rho, rho, delta=tolerance)
        self.assertLess(prediction.fit_error, 1e-4)

    def test_from_1x2_and_half_line_totals(self):
        for lam_h, lam_a in self.MATCHES:
            with self.subTest(lam_home=lam_h, lam_away=lam_a):
                odds = fair_odds(lam_h, lam_a, sp.DEFAULT_RHO, total_line=2.5)
                self.assert_recovers(predict_scorelines(**odds), lam_h, lam_a, sp.DEFAULT_RHO)

    def test_from_1x2_alone(self):
        for lam_h, lam_a in self.MATCHES:
            with self.subTest(lam_home=lam_h, lam_away=lam_a):
                odds = fair_odds(lam_h, lam_a, sp.DEFAULT_RHO)
                self.assert_recovers(predict_scorelines(**odds), lam_h, lam_a, sp.DEFAULT_RHO)

    def test_from_whole_and_quarter_total_lines(self):
        for line in (2.0, 2.25, 2.75, 3.0, 3.25):
            with self.subTest(total_line=line):
                odds = fair_odds(1.76, 1.05, sp.DEFAULT_RHO, total_line=line)
                self.assert_recovers(predict_scorelines(**odds), 1.76, 1.05, sp.DEFAULT_RHO)

    def test_from_asian_handicap_lines(self):
        for line in (-1.5, -1.0, -0.75, -0.25, 0.0, 0.25, 1.0):
            with self.subTest(ah_line=line):
                odds = fair_odds(1.76, 1.05, sp.DEFAULT_RHO, total_line=2.5, ah_line=line)
                self.assert_recovers(predict_scorelines(**odds), 1.76, 1.05, sp.DEFAULT_RHO)

    def test_from_all_markets_together(self):
        odds = fair_odds(1.76, 1.05, sp.DEFAULT_RHO, total_line=2.75, ah_line=-0.75, btts=True)
        self.assert_recovers(predict_scorelines(**odds), 1.76, 1.05, sp.DEFAULT_RHO)

    def test_with_a_fixed_rho(self):
        for rho in (-0.2, -0.12, 0.0, 0.05):
            with self.subTest(rho=rho):
                odds = fair_odds(1.76, 1.05, rho, total_line=2.5)
                self.assert_recovers(predict_scorelines(**odds, rho=rho), 1.76, 1.05, rho)

    def test_fitted_rho_moves_towards_the_true_value(self):
        strong = predict_scorelines(**fair_odds(1.5, 1.2, -0.18, total_line=2.5))
        weak = predict_scorelines(**fair_odds(1.5, 1.2, 0.02, total_line=2.5))
        self.assertLess(strong.rho, sp.DEFAULT_RHO)
        self.assertGreater(weak.rho, sp.DEFAULT_RHO)
        self.assertLess(strong.rho, weak.rho)

    def test_recovers_the_full_score_matrix(self):
        truth = sp._score_matrix(1.76, 1.05, sp.DEFAULT_RHO)
        p = predict_scorelines(**fair_odds(1.76, 1.05, sp.DEFAULT_RHO, total_line=2.5))
        for h, a, prob in cells(truth):
            self.assertAlmostEqual(p.score_matrix[h][a], prob, delta=0.001)


class RhoTest(unittest.TestCase):
    def test_default_is_used_when_only_1x2_odds_are_given(self):
        self.assertEqual(predict_scorelines(**ODDS_1X2).rho, sp.DEFAULT_RHO)

    def test_is_estimated_when_another_market_is_given(self):
        for extra in (ODDS_TOTALS, dict(btts_yes=1.75, btts_no=2.05), dict(ah_home=2.08, ah_away=1.84, ah_line=-0.75)):
            with self.subTest(extra=extra):
                self.assertNotEqual(predict_scorelines(**ODDS_1X2, **extra).rho, sp.DEFAULT_RHO)

    def test_explicit_value_is_kept(self):
        for rho in (-0.25, -0.15, 0.0, 0.1, 0.25):
            with self.subTest(rho=rho):
                self.assertEqual(predict_scorelines(**ODDS_FULL, rho=rho).rho, rho)

    def test_estimate_stays_within_bounds(self):
        # Odds that pull hard in both directions: a very cheap and a very dear draw.
        for draw in (2.2, 9.0):
            with self.subTest(draw=draw):
                p = predict_scorelines(home=2.5, draw=draw, away=2.9, over=1.9, under=1.9)
                self.assertGreaterEqual(p.rho, -0.25)
                self.assertLessEqual(p.rho, 0.25)

    def test_more_negative_rho_makes_low_scoring_draws_more_likely(self):
        low = predict_scorelines(**ODDS_1X2, rho=-0.2).score_matrix
        high = predict_scorelines(**ODDS_1X2, rho=0.0).score_matrix
        self.assertGreater(low[0][0] / low[1][0], high[0][0] / high[1][0])


class FitErrorTest(unittest.TestCase):
    def test_is_small_for_consistent_odds(self):
        self.assertLess(predict_scorelines(**ODDS_FULL).fit_error, 0.005)
        self.assertLess(predict_scorelines(**ODDS_1X2).fit_error, 1e-5)

    def test_is_never_negative(self):
        self.assertGreaterEqual(predict_scorelines(**ODDS_FULL).fit_error, 0.0)

    def test_flags_a_handicap_entered_with_the_wrong_sign(self):
        good = fair_odds(1.76, 1.05, sp.DEFAULT_RHO, total_line=2.5, ah_line=-0.75)
        bad = dict(good, ah_line=0.75)
        self.assertLess(predict_scorelines(**good).fit_error, 1e-4)
        self.assertGreater(predict_scorelines(**bad).fit_error, 0.05)

    def test_flags_markets_that_contradict_each_other(self):
        # A draw at 2.20 does not go together with a high-scoring game.
        p = predict_scorelines(home=3.6, draw=2.2, away=3.6, over=1.25, under=4.0)
        self.assertGreater(p.fit_error, 0.02)


class OptionalMarketsTest(unittest.TestCase):
    def test_only_1x2_odds_are_required(self):
        self.assertIsInstance(predict_scorelines(1.85, 3.70, 4.40), ScorelinePrediction)

    def test_total_line_is_ignored_without_over_under_odds(self):
        a = predict_scorelines(**ODDS_1X2)
        b = predict_scorelines(**ODDS_1X2, total_line=3.5)
        self.assertEqual(a.score_matrix, b.score_matrix)

    def test_total_line_defaults_to_two_and_a_half(self):
        a = predict_scorelines(**ODDS_FULL)
        b = predict_scorelines(**ODDS_FULL, total_line=2.5)
        self.assertEqual(a.score_matrix, b.score_matrix)

    def test_each_optional_market_changes_the_prediction(self):
        base = predict_scorelines(**ODDS_1X2).score_matrix
        for extra in (
            dict(over=1.50, under=2.60),
            dict(btts_yes=1.50, btts_no=2.50),
            dict(ah_home=1.60, ah_away=2.40, ah_line=-0.25),
        ):
            with self.subTest(extra=extra):
                self.assertNotEqual(predict_scorelines(**ODDS_1X2, **extra).score_matrix, base)

    def test_shorter_btts_odds_mean_both_teams_score_more_often(self):
        def btts(prediction):
            return sum(p for h, a, p in cells(prediction.score_matrix) if h and a)

        likely = predict_scorelines(**ODDS_1X2, btts_yes=1.55, btts_no=2.40)
        unlikely = predict_scorelines(**ODDS_1X2, btts_yes=2.40, btts_no=1.55)
        self.assertGreater(btts(likely), btts(unlikely))

    def test_integer_lines_are_accepted(self):
        p = predict_scorelines(**ODDS_1X2, over=2.20, under=1.70, total_line=3, ah_home=2.9, ah_away=1.42, ah_line=-1)
        self.assertAlmostEqual(p.home_win + p.draw + p.away_win, 1.0, places=12)


class ExtremeMatchesTest(unittest.TestCase):
    CASES = {
        "huge home favourite": dict(home=1.03, draw=19.0, away=51.0, over=1.12, under=6.5),
        "huge away favourite": dict(home=51.0, draw=19.0, away=1.03, over=1.12, under=6.5),
        "very low scoring": dict(home=3.1, draw=2.4, away=3.3, over=3.4, under=1.30),
        "very high scoring": dict(home=2.3, draw=4.4, away=2.5, over=1.22, under=4.2, total_line=3.5),
        "huge favourite, 1X2 only": dict(home=1.03, draw=19.0, away=51.0),
    }

    def test_predictions_stay_valid(self):
        for name, odds in self.CASES.items():
            with self.subTest(match=name):
                p = predict_scorelines(**odds)
                values = [p.home_win, p.draw, p.away_win, p.lambda_home, p.lambda_away, p.rho, p.fit_error]
                values += list(p.expected_goals) + [s.probability for s in p.scorelines]
                self.assertTrue(all(math.isfinite(v) for v in values))
                self.assertAlmostEqual(sum(map(sum, p.score_matrix)), 1.0, places=9)
                self.assertTrue(all(prob >= 0.0 for _, _, prob in cells(p.score_matrix)))
                self.assertGreater(p.lambda_home, 0.0)
                self.assertGreater(p.lambda_away, 0.0)

    def test_huge_favourite_is_predicted_to_win_clearly(self):
        p = predict_scorelines(**self.CASES["huge home favourite"])
        self.assertGreater(p.home_win, 0.9)
        self.assertGreaterEqual(p.expected_scoreline[0] - p.expected_scoreline[1], 2)
        self.assertLess(p.fit_error, 0.01)

    def test_low_scoring_match_expects_few_goals(self):
        p = predict_scorelines(**self.CASES["very low scoring"])
        self.assertLess(sum(p.expected_goals), 2.0)
        self.assertIn((0, 0), {(s.home_goals, s.away_goals) for s in p.scorelines[:3]})


class ResultObjectTest(unittest.TestCase):
    def setUp(self):
        self.p = predict_scorelines(**ODDS_FULL)

    def test_is_immutable(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.p.home_win = 0.9
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.p.scorelines[0].probability = 0.9

    def test_same_input_gives_the_same_output(self):
        self.assertEqual(self.p, predict_scorelines(**ODDS_FULL))

    def test_score_matrix_is_indexed_by_home_then_away_goals(self):
        self.assertGreater(self.p.score_matrix[2][0], self.p.score_matrix[0][2])  # home is the favourite

    def test_goal_rates_are_positive(self):
        self.assertGreater(self.p.lambda_home, 0.0)
        self.assertGreater(self.p.lambda_away, 0.0)

    def test_str_is_a_readable_summary(self):
        text = str(self.p)
        self.assertIn("Home 52.8%", text)
        self.assertIn("Draw 25.7%", text)
        self.assertIn("Away 21.5%", text)
        self.assertIn("Expected scoreline: 2-1", text)
        self.assertIn("1-1 (12.2%)", text)

    def test_repr_leaves_out_the_score_matrix(self):
        self.assertNotIn("score_matrix", repr(self.p))
        self.assertIn("home_win", repr(self.p))

    def test_scoreline_str(self):
        self.assertEqual(str(Scoreline(2, 1, 0.0984)), "2-1 (9.8%)")

    def test_module_exports(self):
        self.assertEqual(
            set(sp.__all__), {"predict_scorelines", "ScorelinePrediction", "Scoreline", "remove_margin"}
        )


class InputValidationTest(unittest.TestCase):
    def assert_rejected(self, **kwargs):
        with self.assertRaises(ValueError):
            predict_scorelines(**{**ODDS_1X2, **kwargs})

    def test_rejects_odds_of_one_or_less(self):
        for name in ("home", "draw", "away"):
            for bad in (1.0, 0.9, 0.0, -1.5):
                with self.subTest(field=name, odds=bad):
                    self.assert_rejected(**{name: bad})

    def test_rejects_non_finite_odds(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(odds=bad):
                self.assert_rejected(home=bad)
                self.assert_rejected(over=bad, under=1.9)
                self.assert_rejected(ah_home=bad, ah_away=1.9, ah_line=-0.5)
                self.assert_rejected(btts_yes=bad, btts_no=1.9)

    def test_rejects_invalid_odds_in_optional_markets(self):
        self.assert_rejected(over=1.0, under=1.9)
        self.assert_rejected(over=1.9, under=0.5)
        self.assert_rejected(btts_yes=1.0, btts_no=1.9)
        self.assert_rejected(ah_home=1.9, ah_away=1.0, ah_line=-0.5)

    def test_rejects_half_an_over_under_market(self):
        self.assert_rejected(over=1.9)
        self.assert_rejected(under=1.9)

    def test_rejects_half_a_btts_market(self):
        self.assert_rejected(btts_yes=1.9)
        self.assert_rejected(btts_no=1.9)

    def test_rejects_an_incomplete_asian_handicap(self):
        self.assert_rejected(ah_home=1.9)
        self.assert_rejected(ah_home=1.9, ah_away=1.9)
        self.assert_rejected(ah_home=1.9, ah_line=-0.5)
        self.assert_rejected(ah_line=-0.5)

    def test_rejects_lines_that_are_not_multiples_of_a_quarter(self):
        for line in (2.6, 2.1, 0.3):
            with self.subTest(line=line):
                self.assert_rejected(over=1.9, under=1.9, total_line=line)
                self.assert_rejected(ah_home=1.9, ah_away=1.9, ah_line=-line)

    def test_rejects_non_finite_lines(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(line=bad):
                self.assert_rejected(over=1.9, under=1.9, total_line=bad)
                self.assert_rejected(ah_home=1.9, ah_away=1.9, ah_line=bad)

    def test_rejects_a_total_line_of_zero_or_less(self):
        for line in (0, -0.5, -2.5):
            with self.subTest(line=line):
                self.assert_rejected(over=1.9, under=1.9, total_line=line)

    def test_rejects_top_n_below_one(self):
        for n in (0, -1):
            with self.subTest(top_n=n):
                self.assert_rejected(top_n=n)

    def test_rejects_rho_outside_its_bounds(self):
        for rho in (-0.3, 0.3, 1.0, float("nan")):
            with self.subTest(rho=rho):
                self.assert_rejected(rho=rho)


if __name__ == "__main__":
    unittest.main()
