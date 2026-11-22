# expected-scoreline-predictor

Turns the closing odds of a single soccer match into scoreline probabilities.

You give it the bookmaker's closing odds. It gives you:

- the probabilities of a home win, a draw and an away win,
- the most likely scorelines with their probabilities,
- one expected scoreline, plus the expected goals per team.

For knockout matches that must have a winner, it also predicts extra time and the penalty shoot-out; see [Knockout matches](#knockout-matches).

It works for any competition, because everything specific to the match (team strength, home advantage, neutral venue) is already contained in the odds.

How well it predicts real results is measured in [Validation](#validation). The statistical reasoning behind every step, with references, is in [Methodology](#methodology).

## Setup

You need Python 3.9 or newer. No other packages are required.

**Option 1: install with pip.**

```
pip install git+https://github.com/maxspeicher/expected-scoreline-predictor@v0.3.0
```

The part after `@` is the version. The available versions are listed under [Releases](https://github.com/maxspeicher/expected-scoreline-predictor/releases).

**Option 2: copy the files.** Copy `scoreline_predictor.py` into your project folder, or next to your notebook. For knockout matches, copy `knockout_predictor.py` as well; it needs the first file.

To see which version you have:

```python
import scoreline_predictor
scoreline_predictor.__version__   # '0.3.0'
```

## Quick start

```python
from scoreline_predictor import predict_scorelines

p = predict_scorelines(home=1.85, draw=3.70, away=4.40, over=1.80, under=2.05)
print(p)
```

```
Home 52.8% | Draw 25.7% | Away 21.5%
Expected scoreline: 2-1 (expected goals 1.76-1.06)
Top scorelines: 1-1 (12.2%), 2-1 (9.8%), 1-0 (9.4%), 2-0 (9.3%), 0-0 (7.1%)
```

The individual values are available as attributes:

```python
p.home_win, p.draw, p.away_win    # 0.528, 0.257, 0.215
p.scorelines[0].home_goals        # 1
p.scorelines[0].away_goals        # 1
p.scorelines[0].probability       # 0.122
p.expected_scoreline              # (2, 1)
p.expected_goals                  # (1.76, 1.06)
```

## Input

All odds are decimal odds (e.g. `1.85`), for the result after 90 minutes plus stoppage time.

| Argument | Required | Meaning |
|---|---|---|
| `home`, `draw`, `away` | yes | 1X2 odds |
| `over`, `under` | no | Over/under odds for `total_line` goals |
| `total_line` | no | Goal line of the over/under market. Default `2.5`; any multiple of 0.25 works (e.g. `3` or `2.75`) |
| `ah_home`, `ah_away`, `ah_line` | no | Asian handicap odds and the handicap applied to the home team. `ah_line=-0.75` means the home team gives up 0.75 goals; `ah_line=0.5` means it receives half a goal |
| `btts_yes`, `btts_no` | no | Both-teams-to-score odds |
| `top_n` | no | How many scorelines to return. Default `5` |
| `rho` | no | Fixes the dependence between the two teams' goals (see [The goal model](#3-the-goal-model)). Leave it out unless you know you need it |

Odds of one market always go together: `over` needs `under`, and the three Asian handicap arguments need each other.

**Pass the over/under odds whenever you have them.** The 1X2 odds say who is likely to win, but they do not determine how many goals to expect. For the example above, the 1X2 odds alone are compatible with anything from about 2.1 to 3.0 expected goals, depending on an assumption the odds cannot settle (see [Calibration](#4-calibration-to-the-odds)). The module then uses a typical value taken from market odds, which is right on average but not for every match. With the over/under odds, the model reproduces all prices exactly and no such assumption is needed, and its predictions of real matches were measurably better; see [Validation](#validation).

Use closing odds from a sharp source (e.g. Pinnacle, or a market average) if you can, and take all odds for a match from the same bookmaker and the same point in time.

## Output

`predict_scorelines` returns a `ScorelinePrediction` with these attributes:

| Attribute | Meaning |
|---|---|
| `home_win`, `draw`, `away_win` | Outcome probabilities; they sum to 1 |
| `scorelines` | The `top_n` most likely scorelines, most likely first. Each has `home_goals`, `away_goals` and `probability` |
| `expected_scoreline` | One scoreline as `(home_goals, away_goals)`; see below |
| `expected_goals` | Average goals per team as `(home, away)`, e.g. `(1.76, 1.06)` |
| `fit_error` | Largest gap between the model and the odds; see [Checking your input](#checking-your-input) |
| `score_matrix` | The full table of probabilities: `score_matrix[home_goals][away_goals]`, for 0 to 15 goals per team |
| `lambda_home`, `lambda_away`, `rho` | The fitted model parameters |

### The expected scoreline

`expected_scoreline` is the most likely scoreline *within the most likely outcome*. In the example above, a home win is the most likely outcome (52.8%), so the expected scoreline is the most likely home win, 2-1, although 1-1 is the single most likely score.

This keeps the one-line prediction in agreement with the outcome probabilities. If you want the single most likely score instead, use `p.scorelines[0]`.

## Checking your input

`fit_error` is the largest gap between a model probability and the matching market probability.

| Odds you passed | What to expect |
|---|---|
| 1X2 only, or 1X2 plus one more market | The model has as many parameters as there are prices, so it reproduces them exactly: `fit_error` is practically 0 |
| 1X2 plus two or more markets | There are more prices than parameters, so a small gap is normal. With all three markets from Pinnacle, the median gap over 7,988 real matches was `0.002`, and 95% of matches were below `0.011` |

A gap well above `0.01` means the odds do not fit together. The usual causes are a handicap with the wrong sign, odds from different bookmakers or different points in time, or a typo. A gap can also appear with a single extra market when the prices are so extreme that no admissible model reproduces them.

Invalid input (odds of 1 or less, half a market, a line that is not a multiple of 0.25) raises a `ValueError`.

## Knockout matches

A knockout match must have a winner. `predict_knockout` takes the same odds as `predict_scorelines` and follows the match to its end: 90 minutes, then extra time if the score is level, then a penalty shoot-out if it is still level.

```python
from knockout_predictor import predict_knockout

k = predict_knockout(home=2.9, draw=2.75, away=2.95, over=2.6, under=1.5)
print(k)
```

```
Home advances 50.3% | Away advances 49.7%
Decided in: regular time 65.2% | extra time 16.6% | penalties 18.2%
Expected result: 1-1, home wins on penalties (4.1%)
Top results: 1-0 (10.3%); 0-1 (10.2%); 2-1 (7.2%); 1-2 (7.1%); 2-0 (6.8%)
```

Pass the normal 1X2 odds for the 90 minutes, not "to qualify" or "to lift the trophy" odds. At a neutral venue, "home" is simply the first-named team.

### Input

`predict_knockout` accepts everything `predict_scorelines` accepts, plus two settings:

| Argument | Default | Meaning |
|---|---|---|
| `extra_time_goal_share` | `0.38` | Expected goals in extra time as a share of the expected goals in the 90 minutes. `1/3` would mean the same scoring rate per minute as in regular time |
| `shootout_home_win` | `0.5` | Probability that the home team wins a penalty shoot-out. `0.5` is a coin flip |

`top_n` sets how many final results are returned. The reasons for both defaults are given in [Extra time and penalties](#6-extra-time-and-penalties).

### Output

| Attribute | Meaning |
|---|---|
| `home_advances`, `away_advances` | Probability of each team going through; they sum to 1 |
| `ends_in_regular_time`, `ends_in_extra_time`, `ends_on_penalties` | Probability of the match being decided in each period; they sum to 1 |
| `results` | The `top_n` most likely final results, most likely first. Each has `home_goals`, `away_goals`, `decided_in`, `winner` and `probability` |
| `expected_result` | One final result with the same fields; it always has a winner |
| `extra_time_outcomes` | Probabilities of a home win, a draw and an away win within extra time, as a tuple |
| `regular_time` | The full 90-minute prediction, exactly what `predict_scorelines` returns |

`decided_in` is `"regular time"`, `"extra time"` or `"penalties"`, and `winner` is `"home"` or `"away"`. The goals are the score after the last period played. For a match decided on penalties, that is the level score after extra time; the shoot-out score itself is not predicted.

### The expected result

`expected_result` follows the match period by period and always takes the most likely outcome:

1. **90 minutes.** If the expected scoreline has a winner, that is the result.
2. **Extra time.** If it is a draw, the most likely outcome of extra time decides. A win there gives a result "after extra time".
3. **Penalties.** If a draw is also the most likely outcome of extra time, the team with the better shoot-out chances wins. With the default coin flip, that is the favourite according to the 1X2 odds.

Extra time is short, so a draw is nearly always its most likely outcome. In practice, a match whose most likely 90-minute result is a draw is therefore predicted to go to penalties.

## Using it on a table of matches

```python
import pandas as pd
from scoreline_predictor import predict_scorelines

matches = pd.read_csv("matches.csv")  # one row per match, with your odds columns

def predict(row):
    p = predict_scorelines(home=row["home_odds"], draw=row["draw_odds"], away=row["away_odds"],
                           over=row["over_odds"], under=row["under_odds"])
    return pd.Series({
        "home_win": p.home_win,
        "draw": p.draw,
        "away_win": p.away_win,
        "expected_scoreline": f"{p.expected_scoreline[0]}-{p.expected_scoreline[1]}",
        "fit_error": p.fit_error,
    })

matches = matches.join(matches.apply(predict, axis=1))
```

One prediction takes less than a tenth of a second, so a full league season (a few hundred matches) takes well under a minute.

## Limits

- **Validated on league matches only.** The 90-minute predictions were checked against 7,988 matches of five European leagues; see [Validation](#validation). Cup matches, international matches and the knockout module have not been checked.
- **`predict_scorelines` covers 90 minutes only.** Use `predict_knockout` for matches that go to extra time and penalties.
- **Extra time and shoot-outs rest on assumptions, not on odds.** The odds only describe the 90 minutes.
- **Two-legged ties are not supported.** `predict_knockout` is for single matches. For a second leg, the 90-minute prediction is also less reliable: the aggregate score changes how teams play, which the model does not know about.
- **Even the most likely scoreline is unlikely.** It rarely has more than a 10 to 15% chance. Read the predictions as probabilities, not as forecasts of the exact result.

## Validation

The model was tested on five seasons of the five big European leagues. In short:

- **It is well calibrated.** Scorelines predicted with a given probability occurred about that often, and predicted goals matched actual goals to within 0.03 per team.
- **It clearly beats the historical frequencies of scorelines**, in every league.
- **The exact score remains hard to predict.** The expected scoreline was exactly right in 11.7% of matches, and the actual score was among the top five in 50.8%.
- **Over/under odds matter a lot; Asian handicap odds add very little on top.**
- **It exposed a biased default for 1X2 odds alone, which version 0.3.0 corrects.**

### Design

The design was fixed before any result was looked at.

- **Matches.** All matches of the Premier League, Bundesliga, Serie A, La Liga and Ligue 1 from 2021/22 to 2025/26 for which Pinnacle closing odds exist for 1X2, over/under 2.5 and the Asian handicap: 7,988 of 8,908 matches. Most of the missing ones are from the second half of 2025/26.
- **No fitting.** The model has no parameter that is estimated from results, so every match is an out-of-sample test.
- **Main measure.** The log score: the negative logarithm of the probability the model gave to the score that occurred, averaged over matches. Lower is better. It is shown together with the "typical probability" it corresponds to (the geometric mean of those probabilities).
- **Comparisons.** The full model against (a) the same model with independent goals, (b) 1X2 and over/under odds only, (c) 1X2 odds only, with independent goals as the module assumed at the time, and (d) a baseline that ignores the match: the frequencies of scores in all earlier seasons of the same league since 2011/12, with one added to every count.
- **Uncertainty.** 95% confidence intervals for the differences in log score, from 2,000 bootstrap resamples of matches.

### Results

| Input | Log score | Typical probability of the actual score | Expected scoreline exactly right | Its outcome right | Actual score in top 5 | Predicted goals per match (actual: 2.80) |
|---|---|---|---|---|---|---|
| 1X2, over/under and Asian handicap | 2.8815 | 5.61% | 11.7% | 54.1% | 50.8% | 2.79 |
| 1X2 and over/under | 2.8819 | 5.60% | 11.7% | 54.1% | 50.7% | 2.79 |
| 1X2 only, independent goals | 2.9081 | 5.46% | 11.2% | 54.1% | 50.0% | 2.49 |
| Baseline: historical frequencies | 3.0472 | 4.75% | 9.6% | 43.3% | 44.1% | 2.89 |

| Comparison with the first row | Difference in log score | 95% confidence interval | Reading |
|---|---|---|---|
| Baseline | +0.1658 | +0.1543 to +0.1771 | The model is clearly better |
| 1X2 only, independent goals | +0.0267 | +0.0213 to +0.0318 | Leaving out the over/under odds is clearly worse |
| 1X2 and over/under | +0.0005 | +0.0001 to +0.0009 | Leaving out the Asian handicap is very slightly worse |
| Independent goals, same odds | +0.0010 | −0.0007 to +0.0026 | No measurable difference |

A positive difference means the first row is better. The effect of the Asian handicap is about fifty times smaller than that of the over/under odds, and its interval only just excludes zero; as one of several comparisons, it is weak evidence.

The advantage over the baseline holds in each league separately, with intervals from +0.13 to +0.20 that all exclude zero.

**Calibration of the full model.** The 49 scorelines from 0-0 to 6-6 of every match are grouped by their predicted probability.

| Predicted probability | Scorelines | Mean predicted | Observed |
|---|---|---|---|
| 0% to 2% | 279,257 | 0.34% | 0.32% |
| 2% to 4% | 36,003 | 2.90% | 2.92% |
| 4% to 6% | 24,923 | 4.96% | 4.88% |
| 6% to 8% | 18,641 | 6.97% | 6.81% |
| 8% to 10% | 17,331 | 9.03% | 9.09% |
| 10% to 12% | 7,577 | 10.96% | 11.39% |
| 12% to 15% | 7,161 | 13.30% | 14.22% |
| above 15% | 519 | 16.05% | 16.18% |

| | Predicted | Observed |
|---|---|---|
| Home win | 44.0% | 43.3% |
| Draw | 24.9% | 25.5% |
| Away win | 31.1% | 31.2% |
| Home goals per match | 1.56 | 1.54 |
| Away goals per match | 1.24 | 1.26 |

The most likely scorelines (above 10%) occurred somewhat more often than predicted. The model is slightly too cautious about them.

### What the results say about the design choices

- **Calibrating $`\rho`$ per match.** It does not measurably improve the scoreline predictions: the difference to independent goals is not distinguishable from zero. What it does is make the outcome probabilities right. With independent goals and the same odds, the model predicts a draw in 23.4% of matches where 25.5% occurred; with calibration it predicts 24.9%. The calibrated value was negative in 92% of matches, with a median of −0.085 and half of all matches between −0.107 and −0.057.
- **The expected scoreline rule.** Compared with simply taking the most likely score, the rule was exactly right less often (11.7% against 13.3%) but named the right outcome far more often (54.1% against 41.5%). It is the better prediction whenever an exact score is worth less than about nine times a correct outcome, which confirms the reasoning in [section 5](#5-from-the-model-to-the-outputs) on real matches.
- **The source of the odds.** On the same 7,988 matches, there was no measurable difference between Pinnacle's closing odds and market-average closing odds (difference −0.0007, interval −0.0016 to +0.0002).

### A weakness the test exposed, and its fix

The "1X2 only" row above assumes independent goals ($`\rho = 0`$), which was the module's default for 1X2 odds up to version 0.2.0. It underestimates the number of goals: 2.49 predicted against 2.80 actual. The odds of real matches imply a negative $`\rho`$ (previous section), so independence is wrong on average.

An additional check, run after seeing this, tested a fix. The median $`\rho`$ implied by the odds of the first two seasons (−0.086, from 3,633 matches) was applied to 1X2 odds in the three later seasons (4,355 matches):

| Input, seasons 2023/24 to 2025/26 | Log score | Predicted goals per match (actual: 2.82) |
|---|---|---|
| 1X2 only, $`\rho = 0`$ | 2.903 | 2.51 |
| 1X2 only, $`\rho = -0.086`$ | 2.882 | 2.85 |
| 1X2 and over/under | 2.876 | 2.81 |

The difference between the first two rows is 0.021 (interval 0.014 to 0.027). The value −0.086 removes the bias in goals and closes most of the gap to using over/under odds; a gap of 0.007 (interval 0.003 to 0.010) remains.

Since version 0.3.0, −0.086 is the default for 1X2 odds alone. Two caveats apply. This check was not part of the original design, so it carries less weight than the results above. And the value was derived from odds of earlier seasons, not from the results it was tested on, but from the same five leagues; whether it suits other competitions is untested.

### What this does not show

- **Other competitions.** Only five top leagues were tested. Cups, lower divisions and international tournaments may behave differently.
- **The knockout module.** Extra time and shoot-outs are not tested; too few matches reach them, and no data set with odds was available.
- **A comparison with the market's own correct-score odds**, which would be the hardest benchmark.
- **Betting value.** The test measures how well probabilities match results, not whether they could be used to bet profitably.

### Reproducing it

```
python3 validation/backtest.py
```

The script downloads the season files, runs all comparisons and writes `validation/results.json`. It takes about six minutes on a laptop.

The files are copies of the season files published by football-data.co.uk, taken from a public GitHub repository at a fixed commit. For the Premier League they were compared with a second, independent copy: the scores of all 1,900 matches and all 1,730 available Pinnacle closing 1X2 odds were identical.

## Methodology

This section gives the statistical reasoning behind each step, the design choices and their alternatives, and the evidence for every assumed value. Author–year citations refer to the [References](#references).

**Prior work.** This module was inspired by Lindstrøm's (2019) blog post "Expected goals from bookmaker odds". The post takes the 1X2 odds of a match, removes the margin with the power method, and searches for the pair of expected goals whose Poisson model reproduces the three outcome probabilities, by least squares. It uses the result to fit team ratings from odds instead of from goals. This module follows the same idea and extends it in four ways:

- it also uses over/under, Asian handicap and both-teams-to-score odds;
- it models the dependence between the two teams' goals, where the post uses plain Poisson goals;
- it returns the full score distribution and a single expected scoreline, not only the expected goals;
- it adds extra time and penalties, and a test against real results.

The post's setup corresponds to the "1X2 only, independent goals" variant in [Validation](#validation), which underestimated goals there; this comparison is with the method as described, not with a run of the post's code.

The pipeline has four steps: (1) take closing odds, (2) remove the bookmaker's margin, (3) calibrate a three-parameter goal model so that it reproduces the resulting probabilities, and (4) read all outputs off the model's joint distribution of the two teams' goals. The knockout module adds a model for extra time and the shoot-out on top.

### 1. Why closing odds

**Odds are strong forecasts.** Forrest, Goddard and Simmons (2005) compared published bookmaker odds with a benchmark statistical model on nearly 10,000 English matches and found that the odds became more accurate as forecasts over time, as competition raised the cost of imprecise odds-setting. Štrumbelj and Robnik Šikonja (2010), using 10,699 matches from six European leagues and 10 online bookmakers, likewise found that bookmaker forecasts became more effective over time, and that some bookmakers' odds predict better than others'.

**Later prices contain more information.** In the point-spread market for professional basketball, Gandar, Dare, Brown and Zuber (1998) showed that the line changes between opening and closing significantly improve the accuracy of the lines as forecasts of game outcomes. This is evidence from another sport; it is the reason for preferring the closing price here, not a soccer-specific result.

**Consequence for the design.** The module does not estimate team strength at all. It treats the closing prices as the best available summary of all match-specific information and only translates them into a full score distribution. Because bookmakers differ in accuracy, the source of the odds can matter; in this module's own test, though, there was no measurable difference between market-average closing odds and those of a single sharp bookmaker (see [Validation](#validation)).

### 2. Removing the margin

Decimal odds $`o_i`$ for the outcomes of one market imply raw probabilities $`q_i = 1/o_i`$ whose sum exceeds 1. The excess is the bookmaker's margin, and it has to be removed before the prices can be read as probabilities.

**The margin is not spread evenly.** Fixed-odds soccer betting shows the favourite–longshot bias known from horse racing: longshots are overpriced relative to favourites (Cain, Law and Peel, 2000). Dividing each $`q_i`$ by the sum (plain normalisation) ignores this and leaves longshot probabilities too high.

**Methods that account for it.** Shin (1993) modelled odds set by bookmakers who face bettors with inside information, which produces this bias, and Štrumbelj (2014) found that probabilities obtained with Shin's model are more accurate than those from plain normalisation. Clarke, Kovalchik and Ingram (2017) compared additive, multiplicative, Shin and power adjustments on three large bookmaker data sets from different sports. They report that the power method "universally outperforms the multiplicative method and outperforms or is comparable to the Shin method", and that it "never produces bookmaker or fair probabilities outside the 0-1 range", which the additive method can. They trace the power transformation to Vovk and Zhdanov (2009).

**Choice.** The module uses the power method:

```math
p_i = q_i^{\,k}, \qquad k \text{ chosen such that } \sum_i p_i = 1 .
```

Since every $`q_i < 1`$, the sum is strictly decreasing in $`k`$, so the solution is unique; with a positive margin it lies above 1 and is found by bisection. The method has no tuning parameter. It is applied to each market separately.

**Caveat.** The evidence for preferring the power method over Shin's rests on one comparative study. For low-margin odds from a sharp bookmaker the two differ little, because there is little margin to distribute.

### 3. The goal model

Let $`X`$ and $`Y`$ be the goals of the home and the away team in the 90 minutes.

**Poisson goals.** Maher (1982) showed that independent Poisson distributions for the two teams give "a reasonably accurate description of football scores", with small systematic differences. Fedrizzi, Canal and Micciolo (2022) found the same for the 51 matches of the UEFA EURO 2020 final phase: the Poisson model fits the number of goals, and an exponential distribution fits the waiting times between them. The model therefore starts from

```math
X \sim \text{Poisson}(\lambda), \qquad Y \sim \text{Poisson}(\mu),
```

where $`\lambda`$ and $`\mu`$ are the expected goals of the home and the away team (`lambda_home`, `lambda_away`).

**Dependence between the two scores.** Dixon and Coles (1997) observed that low-scoring draws were more frequent in their English data than independence implies, and proposed a correction that moves probability between the four lowest scores:

```math
P(X = x, Y = y) = \tau_{\lambda,\mu}(x, y)\; \frac{\lambda^x e^{-\lambda}}{x!}\; \frac{\mu^y e^{-\mu}}{y!},
\qquad
\tau_{\lambda,\mu}(x, y) =
\begin{cases}
1 - \lambda\mu\rho & (x,y) = (0,0) \\
1 + \lambda\rho & (x,y) = (0,1) \\
1 + \mu\rho & (x,y) = (1,0) \\
1 - \rho & (x,y) = (1,1) \\
1 & \text{otherwise}
\end{cases}
```

subject to

```math
\max\!\left(-\frac{1}{\lambda},\, -\frac{1}{\mu}\right) \;\le\; \rho \;\le\; \min\!\left(\frac{1}{\lambda\mu},\, 1\right),
```

which keeps all probabilities non-negative. A negative $`\rho`$ makes 0-0 and 1-1 more likely and 1-0 and 0-1 less likely; a positive $`\rho`$ does the opposite; $`\rho = 0`$ is independence.

Three properties of this correction matter for what follows. All three are verified in the test suite.

1. The marginal distributions stay Poisson with means $`\lambda`$ and $`\mu`$, so `expected_goals` equals the fitted rates.
2. Only the four scores 0-0, 1-0, 0-1 and 1-1 change.
3. The four changes cancel: $`-\lambda\mu\rho\,P_{00} + \lambda\rho\,P_{01} + \mu\rho\,P_{10} - \rho\,P_{11} = 0`$, because $`P_{01} = \mu P_{00}`$, $`P_{10} = \lambda P_{00}`$ and $`P_{11} = \lambda\mu P_{00}`$ under independence. All four scores have at most two goals, so the probability of three or more goals, i.e. of "over 2.5", does not depend on $`\rho`$ (Petretta, Schiavon and Diquigiovanni, 2025, make the same observation).

**Alternatives considered.**

| Model | Why it was not chosen |
|---|---|
| Independent Poisson (Maher, 1982) | It is the special case $`\rho = 0`$. It cannot reproduce a draw price that deviates from independence, and with 1X2 odds alone it underestimated goals in the backtest |
| Bivariate Poisson (Karlis and Ntzoufras, 2003) | Its covariance term is non-negative by construction, so it cannot represent negative dependence, which recent data show (below) |
| Copula models with negative binomial margins (McHale and Scarf, 2011); Sarmanov-family extensions (Michels, Ötting and Karlis, 2025); the Mar-Co model (Petretta et al., 2025) | More flexible and better supported by recent data, but they have more parameters than a single match's odds can identify. They are estimated from thousands of match results, which this module does not use |

**Why Dixon–Coles despite its age.** The module calibrates to a handful of prices per match. The 1X2 and over/under markets together provide three independent probabilities, and the Dixon–Coles model has exactly three parameters, each tied to one of them (next section). It is also, in the words of Petretta et al. (2025), still "a point of reference in the betting industry".

**What the literature says against it.** The structure and size of the dependence are not settled, and a user should know this:

- McHale and Scarf (2011) found that in international matches the dependence is weak when the teams are closely matched and becomes increasingly negative as the match becomes less competitive.
- Petretta et al. (2025) analysed 9,130 matches of the five major European leagues from 2014/15 to 2018/19. They found a Pearson correlation between home and away goals of −0.085 (p < 0.001), and a pattern that differs from the one Dixon and Coles described. Relative to the product of the marginal frequencies, 0-0 occurred 99.5% as often (bootstrap standard error 2.3), 1-1 102.1% (1.7), 1-0 94.5% (1.8) and 0-1 93.0% (2.0), and several higher scores deviated significantly as well. They conclude that the dependence "cannot be simplified to an adjustment of the probabilities of the outcomes 0-0, 1-0, 0-1, 1-1".
- Estimates of $`\rho`$ differ in sign between data sets. Dixon and Coles (1997) introduced the correction for an excess of low-scoring draws, i.e. $`\rho < 0`$. Winkelmann, Michels and Deutscher (2025) estimate $`\hat\rho = 0.102`$ (standard error 0.096) for the 2024/25 UEFA Champions League and $`\hat\rho = 0.005`$ (0.111) for the Europa League, from 144 matches each; neither differs significantly from zero. Michels et al. (2025) show formally that the correction can move probability in either direction.

The module's answer to this uncertainty is not to assume a value of $`\rho`$ where it can be avoided, and to take it from market prices where it cannot, as the next section explains.

### 4. Calibration to the odds

The three parameters $`(\lambda, \mu, \rho)`$ are chosen so that the model reproduces the margin-free market probabilities. This is calibration to prices, not statistical estimation from match results: no historical data enter.

**What each market identifies.**

| Market | Independent probabilities | What it determines |
|---|---|---|
| 1X2 | 2 (the three sum to 1) | Mainly the balance between $`\lambda`$ and $`\mu`$, and the draw probability |
| Over/under at 2.5 or higher | 1 | The total $`\lambda + \mu`$; by property 3 above it is unaffected by $`\rho`$ |
| Both teams to score | 1 | A further combination of all three parameters |
| Asian handicap beyond ±0.5 | 1 | A further combination of all three parameters |
| Asian handicap of ±0.5, ±0.25 or 0 | 0 | Nothing new: these bets are settled on the match outcome alone, so their fair price follows from the 1X2 probabilities |

With 1X2 and one further informative market there are three probabilities for three parameters, and the model reproduces them exactly. Given the total from the over/under market and the balance from the home and away prices, the draw price determines $`\rho`$.

**Handling of $`\rho`$.** This is the central design choice.

- *When the odds identify it* (an over/under market, a both-teams-to-score market, or an Asian handicap beyond ±0.5 is given), $`\rho`$ is calibrated to the odds, within the admissible range above. No value is assumed. The market's draw price then decides how much probability sits on low-scoring draws.
- *When they do not* (1X2 only), $`\rho`$ is set to −0.086. The literature does not supply a value: its estimates differ in sign (previous section). The value is instead the median of what market odds imply in matches where they do identify $`\rho`$: 3,633 matches of the five big European leagues in 2021/22 and 2022/23, with Pinnacle closing odds for 1X2 and over/under 2.5. In the three following seasons it predicted goals without bias, whereas independent goals ($`\rho = 0`$, the model of Maher, 1982) underestimated them by 0.3 per match; see [Validation](#a-weakness-the-test-exposed-and-its-fix).

Calibrating $`\rho`$ match by match is this module's own design choice. The reasoning follows.

**Why the literature cannot supply the value.**

- *It answers a different question.* The cited studies estimate $`\rho`$ from thousands of match results, as one constant for a league and a period. This module by design uses no results and is meant to work for any competition, so it has nothing to estimate such a constant from.
- *Its estimates do not transfer.* A constant taken from one study would have to hold for a World Cup match as well as a Bundesliga match. The published values do not even agree in sign (section 3), and McHale and Scarf (2011) find that the dependence changes with how evenly matched the two teams are. A single borrowed number would therefore be an assumption with no support for most matches it is applied to.
- *It does not cover this use.* The cited work fits score models to results. It does not address how to set the dependence when a model is fitted to the prices of one match, so there is no published rule to adopt.

**Why calibrating it to the odds is sound.**

- *Otherwise the model contradicts the prices it is built on.* With $`\rho`$ fixed, two parameters remain for three market probabilities, and one of them cannot be matched. For the quick-start match with $`\rho`$ fixed at 0, the model puts the draw at 23.7% where the market says 25.7%, and shifts the two points to the home win (+0.7) and the away win (+1.4). The module would then report outcome probabilities that differ from the market it claims to translate.
- *The draw price is information, and $`\rho`$ is the only place it can go.* Section 1 gives the reasons for treating closing prices as the best available forecast. That applies to the draw price as much as to the others. Once the total and the balance are set by the other prices, $`\rho`$ is the one remaining parameter that moves the draw probability, so not calibrating it means discarding one of three prices.
- *It lets the dependence vary by match*, which is what McHale and Scarf (2011) observe in results, without the module having to model why.
- *Its effect is contained.* By the three properties in section 3, $`\rho`$ changes neither the expected goals nor the probability of three or more goals. It only redistributes probability among 0-0, 1-0, 0-1 and 1-1.

This is the same logic as quoting an implied parameter in other markets: the parameter is read off observed prices instead of being estimated from history.

**What it costs, and how the cost is limited.** The calibrated $`\rho`$ also absorbs the rounding of the odds and any error of the margin removal, so it is a fitting parameter, not an estimate of a property of the two teams.

- *The noise is small for realistic rounding.* Changing the draw odds of the quick-start match by one step of 0.01 moves $`\rho`$ by 0.002 and each of the four affected scorelines by at most 0.03 percentage points. A change of five steps moves $`\rho`$ by 0.011 and the scorelines by at most 0.12 points. The order of the top scorelines does not change in either case.
- *Impossible values are excluded* by the admissible range of section 3.
- *Inconsistent prices are reported* through `fit_error`.
- *The choice can be overridden* by passing `rho=`.

The backtest in [Validation](#validation) answers the empirical question: per-match calibration does not make the scoreline predictions measurably better or worse than independent goals, and it brings the predicted share of draws from 23.4% to 24.9%, against 25.5% observed.

**How much the 1X2-only assumption matters.** A great deal, which is why a second market is strongly recommended. For the quick-start match (1.85 / 3.70 / 4.40), the 1X2 odds alone give:

| Assumed $`\rho`$ | Implied expected goals in total |
|---|---|
| −0.15 | 3.01 |
| −0.086 (default) | 2.76 |
| 0 | 2.42 |
| +0.1 | 2.08 |

With the over/under odds added, the total is pinned at 2.82 and $`\rho`$ is calibrated to −0.10 for this (illustrative) set of prices. In real matches, half of the calibrated values lay between −0.107 and −0.057, so the default is a typical value, not the right one for every match.

**Bets with refunds and split stakes.** Whole and quarter lines do not have a simple win probability: a whole line refunds the stake when it is hit exactly, and a quarter line splits the stake between the two neighbouring lines. For a bet that wins with stake-weighted probability $`W`$ and loses with $`L`$ (the remainder is refunded), the fair decimal odds $`o`$ solve $`(o - 1)W - L = 0`$, so

```math
\frac{1}{o} = \frac{W}{W + L}.
```

The model's $`W/(W+L)`$ is compared with the margin-free market probability of the same bet. For a half line this reduces to the plain probability of winning.

**Objective.** The parameters minimise the sum of squared differences between model and market probabilities. When the number of probabilities equals the number of free parameters, the minimum is zero and the choice of objective is irrelevant. With more markets, the unweighted sum of squares treats every price as equally reliable; this is a simplification, not a result from the literature.

**Optimiser.** The minimisation uses the downhill simplex method of Nelder and Mead (1965) with its original coefficients, on $`\log\lambda`$ and $`\log\mu`$ so that the rates stay positive. It needs no derivatives, which keeps the module free of dependencies. The search is restarted twice with smaller simplices to guard against stalling. The test suite generates fair odds from known parameters and checks that the calibration returns those parameters for every market type and line type.

### 5. From the model to the outputs

All outputs are computed from the calibrated joint distribution, on a grid of 0 to 15 goals per team.

- **Outcome probabilities** are the sums over the cells with $`x > y`$, $`x = y`$ and $`x < y`$. When the 1X2 odds are reproduced exactly, they equal the margin-free market probabilities.
- **Top scorelines** are the cells with the highest probability.
- **Expected goals** are the means $`E[X]`$ and $`E[Y]`$ of the full distribution.
- **Expected scoreline** is the most likely score within the most likely outcome. This is the module's second design choice; the reasoning follows.

**Why the literature cannot supply the rule.** A single "expected scoreline" is not a quantity that a distribution defines by itself. Which single value best summarises a distribution depends on how a wrong prediction is penalised: the optimal point forecast is the one that minimises the expected loss, and different loss functions give different answers (Gneiting, 2011). The statistical literature can therefore justify the distribution, but the choice of one score from it is a decision about purpose. The candidates are:

| Summary | Optimal when | Problem as a one-line prediction |
|---|---|---|
| Mean, e.g. 1.76-1.06 | Squared error on each team's goals | Not a possible score. It is reported separately as `expected_goals` |
| Most likely score (mode) | Only the exact score counts | It can contradict the outcome probabilities. In the quick-start match it is 1-1 although a home win is twice as likely as a draw |
| Most likely score of the most likely outcome (used here) | Naming the right outcome counts, and the exact score counts on top | None for typical uses; see the condition below |

**Why the chosen rule is sound.** People who predict a score usually care first about the outcome and then about the exact score, and prediction games are scored accordingly. Suppose a prediction earns $`a`$ for the exact score and $`b`$ for the right outcome with the wrong score, with $`a > b > 0`$. Predicting score $`s`$, which belongs to outcome $`O(s)`$, then earns on average

```math
R(s) = a\,P(s) + b\,\big(P(O(s)) - P(s)\big) = b\,P(O(s)) + (a - b)\,P(s).
```

Two things follow.

- *Within an outcome, the best prediction is always its most likely score*, whatever $`a`$ and $`b`$ are.
- *Between outcomes, the most likely outcome wins unless the exact score is rewarded far more than the outcome.* Outcome probabilities differ by tens of percentage points, top score probabilities by one or two. In the quick-start match, 2-1 (9.8%, home win 52.8%) beats 1-1 (12.2%, draw 25.7%) as long as $`a/b < 12`$, i.e. unless the exact score is worth more than twelve times the outcome.

The rule is therefore the best single prediction for any such reward up to a match-specific ratio, and the mode is the best one only in the special case $`b = 0`$. It also guarantees that the one-line prediction never contradicts the outcome probabilities printed next to it.

**Where it can be wrong.** When two outcomes are almost equally likely, the threshold ratio is low, and a specific reward scheme may favour a score from the second outcome. Anyone with a fixed scheme can evaluate $`R(s)`$ directly from `score_matrix`. The plain mode is available as `scorelines[0]`.

The knockout module's `expected_result` applies the same rule period by period: the most likely outcome of the 90 minutes, then of extra time, then of the shoot-out.

### 6. Extra time and penalties

The odds describe the 90 minutes only, so this part is a model with assumed values. With $`p_H, p_D, p_A`$ the outcome probabilities after 90 minutes, $`e_H, e_D, e_A`$ those of extra time and $`s`$ the probability that the home team wins a shoot-out,

```math
P(\text{home advances}) = p_H + p_D\,(e_H + e_D\, s).
```

**Goals in extra time.** Extra-time goals are modelled as independent Poisson with means $`c\lambda`$ and $`c\mu`$, where $`c`$ is `extra_time_goal_share`.

- *Value.* $`c = 0.38`$. Shelopugin and Sirotkin (2026) report, for 12,898 matches with extra time and known scores by period, 1.02 goals in extra time against 2.69 in an average match, a ratio of 0.38. Per 90 minutes that is 3.05 against 2.69, so scoring per minute is somewhat higher in extra time than on average.
- *Plausibility.* A higher rate late on is in line with Dixon and Robinson (1998), who found that scoring rates generally increase during a match. The same per-minute rate as in regular time would give $`c = 1/3`$; in the eight extra-time periods of EURO 2020 the rate was 0.0292 goals per minute against 0.0294 in normal time (Fedrizzi et al., 2022), which matches $`1/3`$ but rests on seven goals.
- *Sensitivity.* The choice between 1/3 and 0.38 moves the probability of a shoot-out by about one percentage point in the examples of this README (18.2% against 19.4% for the even match above) and the probability of advancing by 0.2 points or less.

Three simplifications should be kept in mind.

1. *The ratio compares different sets of matches.* The 0.38 relates extra-time goals to the goals of an average match, not to the 90 minutes of the same matches. It is also taken from a preprint that has not been peer-reviewed.
2. *The favourite's edge is probably overstated in extra time.* The module uses the same $`\lambda`$ and $`\mu`$ as for the 90 minutes. Shelopugin and Sirotkin (2026) find that differences in strength matter less in extra time and attribute this partly to selection: a level score after 90 minutes is itself evidence that the teams were closer on the day than expected. The module does not update the rates on that evidence.
3. *Goals are not exactly Poisson in extra time.* The same source reports that 40% of extra-time periods end without a goal, while a Poisson distribution with mean 1.02 implies 36%. Matches differ in their scoring rates, which a single average does not capture; the module uses each match's own rates, which addresses part of this.

**The shoot-out.** The default is $`s = 0.5`$.

- *Kicking order.* Apesteguia and Palacios-Huerta (2010) reported that the team kicking first wins 60.5% of shoot-outs. Kocher, Lenz and Sutter (2012) extended the sample to 540 shoot-outs and found no significant first-mover advantage. Because the order is decided by a coin toss just before the shoot-out, it could not be used for a pre-match prediction in any case.
- *Home advantage and team strength.* Wunderlich, Berge, Memmert and Rein (2020) analysed 1,067 shoot-outs from 14 cup competitions, with team strength measured from betting odds. They found no home advantage. Stronger teams did win significantly more often, but the effect was small: even a very weak team keeps a winning probability of about 40% against a very strong one. Their title summarises it: "almost a lottery".
- *Consequence.* A coin flip is the default. It understates the favourite's chance in a shoot-out by a few percentage points at most. Raising $`s`$ from 0.5 to 0.6, the upper end of what Wunderlich et al. report, changes the probability of advancing by 1.1 points for the quick-start favourite and 1.8 points for the even match above. Set `shootout_home_win` if you want to include the effect.
- *Tie-break in the expected result.* When the expected result goes to penalties with $`s = 0.5`$, the module names the favourite by the 1X2 odds as the winner, consistent with the direction of the effect in Wunderlich et al. (2020).

### 7. All constants at a glance

The code contains no unexplained numbers. Statistical assumptions are public, named constants with a source; numerical settings only control precision.

| Constant | Value | Kind | Basis |
|---|---|---|---|
| `DEFAULT_RHO` | −0.086 | Assumption, used only with 1X2 odds alone | Median of the values implied by Pinnacle closing odds in 3,633 matches (this module's backtest); published estimates differ in sign and give no usable value (Dixon and Coles, 1997; Winkelmann et al., 2025; Petretta et al., 2025) |
| `EXTRA_TIME_GOAL_SHARE` | 0.38 | Assumption | 1.02 against 2.69 goals in 12,898 matches (Shelopugin and Sirotkin, 2026) |
| `SHOOTOUT_HOME_WIN` | 0.5 | Assumption | Kocher et al. (2012); Wunderlich et al. (2020) |
| Admissible range of $`\rho`$ | formula | Mathematical constraint | Dixon and Coles (1997) |
| Margin exponent $`k`$ | solved per market | No free parameter | Clarke et al. (2017) |
| Nelder–Mead coefficients | 1, 2, 0.5, 0.5 | Numerical | Original values of Nelder and Mead (1965) |
| Grid size | 0 to 15 goals per team | Numerical | For an expected 5 goals, far above any real match, the probability of more than 15 is below 0.0001; the grid is renormalised |
| Search range for $`\lambda, \mu`$ | 0.01 to 10 | Numerical | Guard rails that no real match approaches |
| Starting point of the search | 2.69 goals, split evenly | Numerical | Average goals per match (Shelopugin and Sirotkin, 2026); the result does not depend on it |
| Simplex sizes, tolerance, iteration limits | see code | Numerical | Chosen so that known parameters are recovered to at least two decimal places in the tests |

### 8. Limitations

- **Validation covers league matches only.** The backtest in [Validation](#validation) covers 7,988 matches of five top European leagues. Other competitions and the knockout module are untested.
- **The default for 1X2 odds alone is a five-league average.** The value −0.086 comes from the odds of five top European leagues and may not suit other competitions.
- **The dependence structure is contested.** Recent data do not support the specific low-score pattern of the Dixon–Coles correction (Petretta et al., 2025; McHale and Scarf, 2011). The correction is used here because it is the richest structure that a single match's prices can identify, not because it is the best description of the data.
- **Per-match $`\rho`$ absorbs noise.** Rounding of the odds and errors of the margin removal flow into the calibrated $`\rho`$ and from there into the probabilities of 0-0, 1-0, 0-1 and 1-1. Section 4 quantifies this for rounding; the effect of the margin removal is not quantified.
- **1X2 odds alone leave the total open.** The default fills the gap with a typical value; see the sensitivity table in section 4.
- **Margin removal rests on limited comparative evidence**, and is applied to each market separately.
- **Extra time and shoot-outs are modelled, not priced.** The extra-time value comes from a preprint; the favourite's edge in extra time is probably overstated; the shoot-out default ignores a small effect of team strength.
- **Equal weighting of markets** when there are more prices than parameters is a simplification.

### References

Bibliographic details were checked against publisher or index records in November 2026. The findings quoted from Fedrizzi et al., Petretta et al., Shelopugin and Sirotkin, and Winkelmann et al. were taken from the full text of the cited arXiv versions, and the description of Lindstrøm's method from the post itself, which is not peer-reviewed. For the other works they rest on the abstract or, for Dixon and Robinson (1998) and Štrumbelj (2014), on descriptions in later papers; no numerical estimate is quoted from Dixon and Coles (1997), whose full text was not available.

- Apesteguia, J., & Palacios-Huerta, I. (2010). Psychological pressure in competitive environments: Evidence from a randomized natural experiment. *American Economic Review*, 100(5), 2548–2564.
- Cain, M., Law, D., & Peel, D. (2000). The favourite-longshot bias and market efficiency in UK football betting. *Scottish Journal of Political Economy*, 47(1), 25–36. https://doi.org/10.1111/1467-9485.00151
- Clarke, S., Kovalchik, S., & Ingram, M. (2017). Adjusting bookmaker's odds to allow for overround. *American Journal of Sports Science*, 5(6), 45–49. https://doi.org/10.11648/j.ajss.20170506.12
- Dixon, M. J., & Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 46(2), 265–280.
- Dixon, M. J., & Robinson, M. E. (1998). A birth process model for association football matches. *Journal of the Royal Statistical Society: Series D (The Statistician)*, 47(3), 523–538.
- Fedrizzi, G., Canal, L., & Micciolo, R. (2022). UEFA EURO 2020: a "pure game of chance"? Preprint, arXiv:2203.07531.
- Forrest, D., Goddard, J., & Simmons, R. (2005). Odds-setters as forecasters: The case of English football. *International Journal of Forecasting*, 21(3), 551–564. https://doi.org/10.1016/j.ijforecast.2005.03.003
- Gandar, J. M., Dare, W. H., Brown, C. R., & Zuber, R. A. (1998). Informed traders and price variations in the betting market for professional basketball games. *Journal of Finance*, 53(1), 385–401.
- Gneiting, T. (2011). Making and evaluating point forecasts. *Journal of the American Statistical Association*, 106(494), 746–762. https://doi.org/10.1198/jasa.2011.r10138
- Karlis, D., & Ntzoufras, I. (2003). Analysis of sports data by using bivariate Poisson models. *Journal of the Royal Statistical Society: Series D (The Statistician)*, 52(3), 381–393.
- Kocher, M. G., Lenz, M. V., & Sutter, M. (2012). Psychological pressure in competitive environments: New evidence from randomized natural experiments. *Management Science*, 58(8), 1585–1591. https://doi.org/10.1287/mnsc.1120.1516
- Lindstrøm, J. C. (2019). Expected goals from bookmaker odds. Blog post, 29 January 2019. https://opisthokonta.net/?p=1760
- Maher, M. J. (1982). Modelling association football scores. *Statistica Neerlandica*, 36(3), 109–118.
- McHale, I., & Scarf, P. (2011). Modelling the dependence of goals scored by opposing teams in international soccer matches. *Statistical Modelling*, 11(3), 219–236.
- Michels, R., Ötting, M., & Karlis, D. (2025). Extending the Dixon and Coles model: an application to women's football data. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 74(1), 167–186. https://doi.org/10.1093/jrsssc/qlae050
- Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313.
- Petretta, M., Schiavon, L., & Diquigiovanni, J. (2025). On the dependence in football match outcomes: traditional model assumptions and an alternative proposal. *Statistical Modelling*. Cited from the preprint, arXiv:2103.07272 (version of December 2022).
- Shelopugin, A., & Sirotkin, A. (2026). A modified Glicko model for rating soccer clubs and national teams worldwide. Preprint, arXiv:2611.15378.
- Shin, H. S. (1993). Measuring the incidence of insider trading in a market for state-contingent claims. *The Economic Journal*, 103(420), 1141–1153.
- Štrumbelj, E. (2014). On determining probability forecasts from betting odds. *International Journal of Forecasting*, 30(4), 934–943.
- Štrumbelj, E., & Robnik Šikonja, M. (2010). Online bookmakers' odds as forecasts: The case of European soccer leagues. *International Journal of Forecasting*, 26(3), 482–488.
- Vovk, V., & Zhdanov, F. (2009). Prediction with expert advice for the Brier game. *Journal of Machine Learning Research*, 10, 2445–2471.
- Winkelmann, D., Michels, R., & Deutscher, C. (2025). Predicting qualification thresholds in the UEFA Champions League and Europa League under the new league phase format. Preprint, arXiv:2508.20075 (version of August 2026).
- Wunderlich, F., Berge, F., Memmert, D., & Rein, R. (2020). Almost a lottery: the influence of team strength on success in penalty shootouts. *International Journal of Performance Analysis in Sport*, 20(5), 857–869. https://doi.org/10.1080/24748668.2020.1799171

## Tests

```
python3 -m unittest -v
```

The 180 tests in `test_scoreline_predictor.py` and `test_knockout_predictor.py` need no extra packages.

## Licence

MIT, see [LICENSE](LICENSE).
