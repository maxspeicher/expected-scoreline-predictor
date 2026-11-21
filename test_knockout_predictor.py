"""Tests for knockout_predictor.

Run with:  python3 -m unittest -v        (or: pytest)
"""

import dataclasses
import math
import unittest

import knockout_predictor as kp
from knockout_predictor import KnockoutPrediction, KnockoutResult, predict_knockout
from scoreline_predictor import predict_scorelines

# An even, low-scoring match: the draw is the most likely result after 90 minutes.
EVEN = dict(home=2.9, draw=2.75, away=2.95, over=2.6, under=1.5)
# A match with a clear favourite: a home win is the most likely result after 90 minutes.
FAVOURITE = dict(home=1.85, draw=3.70, away=4.40, over=1.80, under=2.05)
ALL = 100_000  # a top_n large enough to return every possible result


def poisson(lam, k):
    return math.exp(-lam) * lam**k / math.factorial(k)


class AdvanceProbabilitiesTest(unittest.TestCase):
    def test_sum_to_one(self):
        for odds in (EVEN, FAVOURITE):
            with self.subTest(odds=odds):
                k = predict_knockout(**odds)
                self.assertAlmostEqual(k.home_advances + k.away_advances, 1.0, places=12)

    def test_follow_from_the_three_periods(self):
        for shootout in (0.5, 0.3, 0.8):
            with self.subTest(shootout_home_win=shootout):
                k = predict_knockout(**FAVOURITE, shootout_home_win=shootout)
                r = k.regular_time
                et_home, et_draw, _ = k.extra_time_outcomes
                want = r.home_win + r.draw * (et_home + et_draw * shootout)
                self.assertAlmostEqual(k.home_advances, want, places=9)

    def test_lie_between_winning_in_90_minutes_and_not_losing_in_90_minutes(self):
        k = predict_knockout(**FAVOURITE)
        self.assertGreater(k.home_advances, k.regular_time.home_win)
        self.assertLess(k.home_advances, k.regular_time.home_win + k.regular_time.draw)

    def test_favourite_is_more_likely_to_advance(self):
        self.assertGreater(predict_knockout(**FAVOURITE).home_advances, 0.6)
        mirrored = dict(FAVOURITE, home=FAVOURITE["away"], away=FAVOURITE["home"])
        self.assertGreater(predict_knockout(**mirrored).away_advances, 0.6)

    def test_evenly_matched_teams_have_equal_chances(self):
        k = predict_knockout(home=2.9, draw=2.75, away=2.9, over=2.6, under=1.5)
        self.assertAlmostEqual(k.home_advances, 0.5, places=4)

    def test_swapping_the_teams_swaps_the_chances(self):
        k = predict_knockout(**FAVOURITE, shootout_home_win=0.6)
        mirrored = dict(FAVOURITE, home=FAVOURITE["away"], away=FAVOURITE["home"])
        q = predict_knockout(**mirrored, shootout_home_win=0.4)
        self.assertAlmostEqual(k.home_advances, q.away_advances, places=4)

    def test_better_shootout_chances_help(self):
        low = predict_knockout(**EVEN, shootout_home_win=0.3)
        high = predict_knockout(**EVEN, shootout_home_win=0.7)
        self.assertGreater(high.home_advances, low.home_advances)
        self.assertAlmostEqual(high.home_advances - low.home_advances, 0.4 * high.ends_on_penalties, places=9)


class MatchLengthTest(unittest.TestCase):
    def test_probabilities_sum_to_one(self):
        for odds in (EVEN, FAVOURITE):
            with self.subTest(odds=odds):
                k = predict_knockout(**odds)
                total = k.ends_in_regular_time + k.ends_in_extra_time + k.ends_on_penalties
                self.assertAlmostEqual(total, 1.0, places=12)

    def test_regular_time_decides_unless_it_is_a_draw(self):
        k = predict_knockout(**EVEN)
        self.assertAlmostEqual(k.ends_in_regular_time, 1 - k.regular_time.draw, places=9)

    def test_penalties_need_a_draw_in_both_periods(self):
        k = predict_knockout(**EVEN)
        self.assertAlmostEqual(k.ends_on_penalties, k.regular_time.draw * k.extra_time_outcomes[1], places=9)

    def test_does_not_depend_on_the_shootout_chances(self):
        a = predict_knockout(**EVEN, shootout_home_win=0.2)
        b = predict_knockout(**EVEN, shootout_home_win=0.9)
        self.assertAlmostEqual(a.ends_on_penalties, b.ends_on_penalties, places=12)
        self.assertAlmostEqual(a.ends_in_extra_time, b.ends_in_extra_time, places=12)

    def test_even_match_goes_beyond_90_minutes_more_often(self):
        self.assertLess(predict_knockout(**EVEN).ends_in_regular_time, predict_knockout(**FAVOURITE).ends_in_regular_time)


class ExtraTimeTest(unittest.TestCase):
    def test_outcome_probabilities_sum_to_one(self):
        self.assertAlmostEqual(sum(predict_knockout(**FAVOURITE).extra_time_outcomes), 1.0, places=12)

    def test_default_goal_share_is_the_published_ratio(self):
        # 1.02 goals in extra time against 2.69 in an average match (Shelopugin and Sirotkin, 2026).
        self.assertEqual(kp.EXTRA_TIME_GOAL_SHARE, 0.38)
        self.assertAlmostEqual(1.02 / 2.69, kp.EXTRA_TIME_GOAL_SHARE, places=2)

    def test_default_shootout_is_a_coin_flip(self):
        self.assertEqual(kp.SHOOTOUT_HOME_WIN, 0.5)

    def test_goal_rates_are_the_regular_ones_times_the_goal_share(self):
        k = predict_knockout(**FAVOURITE)
        lam_h = k.regular_time.lambda_home * kp.EXTRA_TIME_GOAL_SHARE
        lam_a = k.regular_time.lambda_away * kp.EXTRA_TIME_GOAL_SHARE
        draw = sum(poisson(lam_h, i) * poisson(lam_a, i) for i in range(12))
        home = sum(poisson(lam_h, i) * poisson(lam_a, j) for i in range(12) for j in range(i))
        self.assertAlmostEqual(k.extra_time_outcomes[1], draw, places=9)
        self.assertAlmostEqual(k.extra_time_outcomes[0], home, places=9)

    def test_draw_is_more_likely_than_over_90_minutes(self):
        k = predict_knockout(**FAVOURITE)
        self.assertGreater(k.extra_time_outcomes[1], k.regular_time.draw)

    def test_favourite_is_more_likely_to_win_it(self):
        home, _, away = predict_knockout(**FAVOURITE).extra_time_outcomes
        self.assertGreater(home, away)

    def test_more_extra_time_goals_mean_fewer_shootouts(self):
        calm = predict_knockout(**EVEN, extra_time_goal_share=0.2)
        normal = predict_knockout(**EVEN)
        wild = predict_knockout(**EVEN, extra_time_goal_share=0.8)
        self.assertGreater(calm.ends_on_penalties, normal.ends_on_penalties)
        self.assertGreater(normal.ends_on_penalties, wild.ends_on_penalties)

    def test_goal_share_scales_the_goal_rates(self):
        k = predict_knockout(**FAVOURITE, extra_time_goal_share=0.25)
        rate = (k.regular_time.lambda_home + k.regular_time.lambda_away) * 0.25
        every_result = predict_knockout(**FAVOURITE, extra_time_goal_share=0.25, top_n=ALL).results
        goalless = next(r for r in every_result if (r.home_goals, r.away_goals, r.decided_in) == (0, 0, "penalties"))
        # 0-0 after 90 minutes, then no goal in extra time; both shoot-out winners together.
        nil_nil = k.regular_time.score_matrix[0][0]
        self.assertAlmostEqual(goalless.probability * 2, nil_nil * math.exp(-rate), places=9)

    def test_almost_no_extra_time_goals_sends_every_draw_to_penalties(self):
        k = predict_knockout(**EVEN, extra_time_goal_share=1e-6)
        self.assertAlmostEqual(k.ends_on_penalties, k.regular_time.draw, places=5)

    def test_does_not_change_the_90_minute_prediction(self):
        a = predict_knockout(**EVEN, extra_time_goal_share=0.2).regular_time
        b = predict_knockout(**EVEN, extra_time_goal_share=0.8).regular_time
        self.assertEqual(a, b)


class ResultsTest(unittest.TestCase):
    def setUp(self):
        self.k = predict_knockout(**EVEN, top_n=ALL, shootout_home_win=0.55)
        self.results = self.k.results

    def test_returns_five_by_default(self):
        self.assertEqual(len(predict_knockout(**EVEN).results), 5)

    def test_returns_the_requested_number(self):
        for n in (1, 3, 12):
            with self.subTest(top_n=n):
                self.assertEqual(len(predict_knockout(**EVEN, top_n=n).results), n)

    def test_are_sorted_from_most_to_least_likely(self):
        probs = [r.probability for r in self.results]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_are_all_different(self):
        keys = [(r.home_goals, r.away_goals, r.decided_in, r.winner) for r in self.results]
        self.assertEqual(len(set(keys)), len(keys))

    def test_all_results_together_have_probability_one(self):
        self.assertAlmostEqual(sum(r.probability for r in self.results), 1.0, places=9)

    def test_larger_top_n_extends_the_same_list(self):
        self.assertEqual(predict_knockout(**EVEN, top_n=8).results[:3], predict_knockout(**EVEN, top_n=3).results)

    def test_every_result_has_a_winner(self):
        self.assertEqual({r.winner for r in self.results}, {"home", "away"})

    def test_use_the_three_ways_a_match_can_be_decided(self):
        self.assertEqual({r.decided_in for r in self.results}, {"regular time", "extra time", "penalties"})

    def test_score_is_level_only_when_penalties_decide(self):
        for r in self.results:
            self.assertEqual(r.home_goals == r.away_goals, r.decided_in == "penalties")

    def test_winner_has_more_goals_unless_penalties_decide(self):
        for r in self.results:
            if r.decided_in != "penalties":
                self.assertEqual(r.winner, "home" if r.home_goals > r.away_goals else "away")

    def test_regular_time_results_match_the_90_minute_prediction(self):
        matrix = self.k.regular_time.score_matrix
        for r in self.results:
            if r.decided_in == "regular time":
                self.assertEqual(r.probability, matrix[r.home_goals][r.away_goals])

    def test_extra_time_win_combines_every_path_to_that_score(self):
        # 1-0 after extra time can only come from 0-0 after 90 minutes plus 1-0 in extra time.
        k = self.k
        lam_h = k.regular_time.lambda_home * kp.EXTRA_TIME_GOAL_SHARE
        lam_a = k.regular_time.lambda_away * kp.EXTRA_TIME_GOAL_SHARE
        one_nil = next(r for r in self.results if (r.home_goals, r.away_goals, r.decided_in) == (1, 0, "extra time"))
        want = k.regular_time.score_matrix[0][0] * poisson(lam_h, 1) * poisson(lam_a, 0)
        self.assertAlmostEqual(one_nil.probability, want, places=9)
        # 2-1 after extra time: 0-0 then 2-1, or 1-1 then 1-0.
        two_one = next(r for r in self.results if (r.home_goals, r.away_goals, r.decided_in) == (2, 1, "extra time"))
        want = (
            k.regular_time.score_matrix[0][0] * poisson(lam_h, 2) * poisson(lam_a, 1)
            + k.regular_time.score_matrix[1][1] * poisson(lam_h, 1) * poisson(lam_a, 0)
        )
        self.assertAlmostEqual(two_one.probability, want, places=9)

    def test_shootout_results_are_split_by_the_shootout_chances(self):
        for score in (0, 1, 2):
            with self.subTest(score=score):
                pens = {r.winner: r.probability for r in self.results
                        if (r.home_goals, r.away_goals, r.decided_in) == (score, score, "penalties")}
                self.assertAlmostEqual(pens["home"] / (pens["home"] + pens["away"]), 0.55, places=9)

    def test_probabilities_add_up_to_the_summary_values(self):
        k = self.k

        def total(**where):
            return sum(r.probability for r in self.results if all(getattr(r, f) == v for f, v in where.items()))

        self.assertAlmostEqual(total(winner="home"), k.home_advances, places=9)
        self.assertAlmostEqual(total(winner="away"), k.away_advances, places=9)
        self.assertAlmostEqual(total(decided_in="regular time"), k.ends_in_regular_time, places=9)
        self.assertAlmostEqual(total(decided_in="extra time"), k.ends_in_extra_time, places=9)
        self.assertAlmostEqual(total(decided_in="penalties"), k.ends_on_penalties, places=9)

    def test_certain_shootout_winner_leaves_no_result_for_the_other_team(self):
        k = predict_knockout(**EVEN, top_n=ALL, shootout_home_win=1.0)
        self.assertFalse([r for r in k.results if r.decided_in == "penalties" and r.winner == "away"])
        self.assertTrue(all(r.probability > 0 for r in k.results))


class ExpectedResultTest(unittest.TestCase):
    def test_always_has_a_winner(self):
        for odds in (EVEN, FAVOURITE, dict(home=2.9, draw=2.75, away=2.9)):
            with self.subTest(odds=odds):
                self.assertIn(predict_knockout(**odds).expected_result.winner, ("home", "away"))

    def test_is_the_90_minute_prediction_when_that_has_a_winner(self):
        k = predict_knockout(**FAVOURITE)
        e = k.expected_result
        self.assertEqual((e.home_goals, e.away_goals), k.regular_time.expected_scoreline)
        self.assertEqual((e.decided_in, e.winner), ("regular time", "home"))

    def test_away_favourite_wins_in_regular_time(self):
        mirrored = dict(FAVOURITE, home=FAVOURITE["away"], away=FAVOURITE["home"])
        e = predict_knockout(**mirrored).expected_result
        self.assertEqual((e.decided_in, e.winner), ("regular time", "away"))
        self.assertLess(e.home_goals, e.away_goals)

    def test_goes_to_penalties_when_a_draw_is_most_likely_in_both_periods(self):
        k = predict_knockout(**EVEN)
        self.assertEqual(k.regular_time.expected_scoreline, (1, 1))
        e = k.expected_result
        self.assertEqual((e.home_goals, e.away_goals, e.decided_in), (1, 1, "penalties"))

    def test_shootout_goes_to_the_team_with_the_better_shootout_chances(self):
        self.assertEqual(predict_knockout(**EVEN, shootout_home_win=0.6).expected_result.winner, "home")
        self.assertEqual(predict_knockout(**EVEN, shootout_home_win=0.4).expected_result.winner, "away")

    def test_even_shootout_goes_to_the_favourite(self):
        self.assertEqual(predict_knockout(**EVEN).expected_result.winner, "home")  # 2.90 against 2.95
        swapped = dict(EVEN, home=EVEN["away"], away=EVEN["home"])
        self.assertEqual(predict_knockout(**swapped).expected_result.winner, "away")

    def test_even_shootout_between_equal_teams_goes_to_the_home_team(self):
        e = predict_knockout(home=2.9, draw=2.75, away=2.9, over=2.6, under=1.5).expected_result
        self.assertEqual((e.decided_in, e.winner), ("penalties", "home"))

    def test_is_decided_in_extra_time_when_a_win_there_is_most_likely(self):
        k = predict_knockout(**EVEN, extra_time_goal_share=2.0)
        self.assertGreater(k.extra_time_outcomes[0], k.extra_time_outcomes[1])
        e = k.expected_result
        self.assertEqual((e.decided_in, e.winner), ("extra time", "home"))
        # Level at 1-1 after 90 minutes, so the home team must be ahead from there.
        self.assertGreater(e.home_goals, e.away_goals)
        self.assertGreaterEqual(e.away_goals, 1)

    def test_extra_time_can_be_won_by_the_away_team(self):
        swapped = dict(EVEN, home=EVEN["away"], away=EVEN["home"])
        e = predict_knockout(**swapped, extra_time_goal_share=2.0).expected_result
        self.assertEqual((e.decided_in, e.winner), ("extra time", "away"))

    def test_carries_the_probability_of_that_exact_result(self):
        for odds in (EVEN, FAVOURITE):
            with self.subTest(odds=odds):
                k = predict_knockout(**odds, top_n=ALL)
                self.assertIn(k.expected_result, k.results)

    def test_does_not_depend_on_top_n(self):
        self.assertEqual(
            predict_knockout(**EVEN, top_n=1).expected_result,
            predict_knockout(**EVEN, top_n=20).expected_result,
        )


class RegularTimeTest(unittest.TestCase):
    def test_is_exactly_the_scoreline_prediction_for_the_same_odds(self):
        for odds in (EVEN, FAVOURITE, dict(home=1.85, draw=3.70, away=4.40)):
            with self.subTest(odds=odds):
                self.assertEqual(predict_knockout(**odds).regular_time, predict_scorelines(**odds))

    def test_optional_markets_are_passed_on(self):
        odds = dict(home=1.85, draw=3.70, away=4.40, over=2.02, under=1.86, total_line=2.75,
                    ah_home=2.08, ah_away=1.84, ah_line=-0.75, btts_yes=1.8, btts_no=2.0, rho=-0.1)
        self.assertEqual(predict_knockout(**odds).regular_time, predict_scorelines(**odds))

    def test_only_1x2_odds_are_required(self):
        self.assertIsInstance(predict_knockout(2.9, 2.75, 2.95), KnockoutPrediction)

    def test_top_n_applies_to_the_final_results_only(self):
        k = predict_knockout(**EVEN, top_n=2)
        self.assertEqual(len(k.results), 2)
        self.assertEqual(len(k.regular_time.scorelines), 5)


class ResultObjectTest(unittest.TestCase):
    def test_is_immutable(self):
        k = predict_knockout(**EVEN)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            k.home_advances = 0.9
        with self.assertRaises(dataclasses.FrozenInstanceError):
            k.expected_result.winner = "away"

    def test_same_input_gives_the_same_output(self):
        self.assertEqual(predict_knockout(**EVEN), predict_knockout(**EVEN))

    def test_result_str_shows_how_the_match_was_decided(self):
        self.assertEqual(str(KnockoutResult(2, 1, "regular time", "home", 0.098)), "2-1 (9.8%)")
        self.assertEqual(str(KnockoutResult(1, 2, "extra time", "away", 0.031)), "1-2 after extra time (3.1%)")
        self.assertEqual(str(KnockoutResult(1, 1, "penalties", "away", 0.044)), "1-1, away wins on penalties (4.4%)")

    def test_str_is_a_readable_summary(self):
        text = str(predict_knockout(**EVEN))
        self.assertIn("Home advances 50.3%", text)
        self.assertIn("Away advances 49.7%", text)
        self.assertIn("penalties 18.2%", text)
        self.assertIn("Expected result: 1-1, home wins on penalties", text)

    def test_repr_leaves_out_the_90_minute_prediction(self):
        text = repr(predict_knockout(**EVEN))
        self.assertNotIn(" regular_time=", text)
        self.assertNotIn("score_matrix", text)
        self.assertIn("home_advances=", text)


class InputValidationTest(unittest.TestCase):
    def assert_rejected(self, error=ValueError, **kwargs):
        with self.assertRaises(error):
            predict_knockout(**{**EVEN, **kwargs})

    def test_rejects_invalid_odds(self):
        self.assert_rejected(home=1.0)
        self.assert_rejected(draw=float("nan"))
        self.assert_rejected(over=1.9, under=None)
        self.assert_rejected(total_line=2.6)

    def test_rejects_unknown_arguments(self):
        self.assert_rejected(TypeError, qualify_home=1.5)

    def test_rejects_top_n_below_one(self):
        self.assert_rejected(top_n=0)
        self.assert_rejected(top_n=-3)

    def test_rejects_shootout_chances_that_are_not_probabilities(self):
        for bad in (-0.1, 1.1, float("nan")):
            with self.subTest(shootout_home_win=bad):
                self.assert_rejected(shootout_home_win=bad)

    def test_accepts_certain_shootout_outcomes(self):
        self.assertGreater(predict_knockout(**EVEN, shootout_home_win=1.0).home_advances, 0.5)
        self.assertLess(predict_knockout(**EVEN, shootout_home_win=0.0).home_advances, 0.5)

    def test_rejects_an_extra_time_goal_share_that_is_not_positive(self):
        for bad in (0, -1.0, float("nan"), float("inf")):
            with self.subTest(extra_time_goal_share=bad):
                self.assert_rejected(extra_time_goal_share=bad)


if __name__ == "__main__":
    unittest.main()
