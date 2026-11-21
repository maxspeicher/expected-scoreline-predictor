# expected-scoreline-predictor

Turns the closing odds of a single soccer match into scoreline probabilities.

You give it the bookmaker's closing odds. It gives you:

- the probabilities of a home win, a draw and an away win,
- the most likely scorelines with their probabilities,
- one expected scoreline, plus the expected goals per team.

It works for any competition, because everything specific to the match (team strength, home advantage, neutral venue) is already contained in the odds.

## Setup

You need Python 3.9 or newer. No other packages are required.

**Option 1: install with pip.**

```
pip install git+https://github.com/maxspeicher/expected-scoreline-predictor@v0.1.0
```

The part after `@` is the version. The available versions are listed under [Releases](https://github.com/maxspeicher/expected-scoreline-predictor/releases).

**Option 2: copy the file.** Copy `scoreline_predictor.py` into your project folder, or next to your notebook.

To see which version you have:

```python
import scoreline_predictor
scoreline_predictor.__version__   # '0.1.0'
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
| `rho` | no | Fixes the draw correction (see [How it works](#how-it-works)). Leave it out unless you know you need it |

Odds of one market always go together: `over` needs `under`, and the three Asian handicap arguments need each other.

**Pass the over/under odds whenever you have them.** The 1X2 odds alone say who is likely to win, but only little about how many goals to expect, so the scoreline probabilities are less reliable without a second market.

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

This keeps the one-line prediction in agreement with the outcome probabilities. The single most likely score is 1-1 for a large share of all matches, including many with a clear favourite.

If you want the single most likely score instead, use `p.scorelines[0]`.

## Checking your input

`fit_error` tells you how well the model could reproduce the odds you passed in. It is the largest gap between a model probability and the matching market probability.

| `fit_error` | Meaning |
|---|---|
| below `0.005` | Normal |
| above `0.01` | The odds do not fit together. Check them before trusting the result |

The usual causes of a high value are a handicap with the wrong sign, odds from different bookmakers or different points in time, or a typo.

Invalid input (odds of 1 or less, half a market, a line that is not a multiple of 0.25) raises a `ValueError`.

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

One prediction takes about 30 to 80 milliseconds, so a full league season (a few hundred matches) takes well under a minute.

## Limits

- **90 minutes only.** The odds, and therefore the predictions, do not cover extra time or penalty shoot-outs.
- **Second legs of two-legged ties** are less reliable: the aggregate score changes how teams play, which the model does not know about.
- **Not checked against real results yet.** The tests show that the model reproduces the odds accurately. How well the resulting scorelines match actual results has not been measured on historical data.
- **Even the most likely scoreline is unlikely.** It rarely has more than a 10 to 15% chance. Read the predictions as probabilities, not as forecasts of the exact result.

## How it works

1. **Remove the bookmaker's margin.** Raw odds imply probabilities that sum to more than 100%. The surplus is removed with the power method, which takes more of it from longshots than from favourites.
2. **Fit a goal model to the odds.** Each team's goals follow a Poisson distribution with its own expected number of goals (`lambda_home`, `lambda_away`). A correction factor (`rho`, after Dixon and Coles) makes 0-0 and 1-1 slightly more likely and 1-0 and 0-1 slightly less likely than plain Poisson, which matches real soccer results better. The three values are chosen so that the model reproduces the odds as closely as possible.
3. **Read off the results.** The fitted model gives a probability for every scoreline; all outputs come from that table.

With 1X2 odds only, `rho` is fixed at −0.08, a typical value. With an extra market it is estimated from the odds and kept close to that value.

## Tests

```
python3 -m unittest -v
```

The 105 tests in `test_scoreline_predictor.py` need no extra packages.

## Licence

MIT, see [LICENSE](LICENSE).
