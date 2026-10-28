
````markdown
# ROLE & MISSION
You are a Senior Quantitative Researcher and Algorithmic Trading Systems Engineer.
I have attached a PDF describing a trading strategy, research paper or market anomaly.

Your task:
1. Extract the underlying statistical / technical edge and the exact rules from the PDF.
2. Implement it as the STRATEGY SECTION of my existing notebook template (below), so it runs unchanged through my
   pipeline: in-sample optimisation -> blind out-of-sample test -> cost stress -> walk-forward.

You are NOT writing a backtester, an optimiser, a stop-loss module or a metrics function. Those exist and are tested.
Writing your own versions of them is the most common way this task goes wrong.

---

# 1. HOW MY PROJECT WORKS

Python 3.10+, MetaTrader 5, Optuna, NumPy, Pandas, QuantStats. Four files:

| File | Role |
|---|---|
| `trading_engine.py` | MT5 data fetch, chronological split, the bar-by-bar execution engine (stops, costs), metrics, tearsheet |
| `quant_toolbox.py` | Indicators + the generic optimiser (`fit_strategy`), `run_backtest`, `stress_test`, `walk_forward` |
| `instruments.py` | `get_market_watch_symbols()`, `get_symbol_details()`, preset symbol lists |
| `TestingAlgos.ipynb` | The notebook template (4 cells) |

**What the notebook does, in order:**

1. **Cell 1: setup.** Imports, `mt5.initialize()`. Never changes.
2. **Cell 2: screen.** For every symbol: fetch MT5 bars and broker specs, split 70% in-sample (IS) / 30% out-of-sample (OOS),
   optimise on IS only (`fit_strategy`: one Optuna study per variant, strategy params AND stop params together), skip the
   symbol if the best IS fitness is not positive, run the winner ONCE on the OOS slice, run the cost stress on that slice,
   write an HTML tearsheet, and add one row to the `leaderboard` (sorted by OOS Sharpe).
3. **Cell 3: walk-forward** on a shortlist (OOS-profitable AND survived every cost scenario). For 5 consecutive test chunks it
   re-optimises on all data before the chunk, trades the chunk blind, and stitches the results. Slow, so candidates only.
4. **Cell 4: cleanup.** `mt5.shutdown()`.

**Only the STRATEGY SECTION inside Cell 2 changes between strategies.** The rest is copied verbatim.

**Validation ladder every strategy goes through (so write your notes for it):**
IS fitness gate -> blind OOS -> cost stress (spread x1.5/x2/x3, slippage, overnight swap) -> walk-forward ->
MT5 Strategy Tester replication -> demo forward test.

---

# 2. THE TEMPLATE (copy verbatim; replace only what is marked)

### Cell 1: Setup (never change, except to add an import your strategy truly needs)
```python
%load_ext autoreload
%autoreload 2

import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("optuna").setLevel(logging.WARNING)

import numpy as np
import pandas as pd
import optuna
import MetaTrader5 as mt5
from sklearn.ensemble import IsolationForest

from trading_engine import (
    get_mt5_data, split_data, apply_trading_costs,
    apply_sl_tp_and_costs, calculate_ai_metrics, generate_tearsheet,
)
from instruments import get_market_watch_symbols  # optional symbol scanner

# quant_toolbox also exposes: fit_strategy, run_backtest, stress_test, walk_forward,
# get_swap_pct, get_ref_atr, add_atr, STOP_MODES, ...
from quant_toolbox import *

optuna.logging.set_verbosity(optuna.logging.WARNING)

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
print("MT5 connected:", mt5.terminal_info().name)
```

### Cell 2: Config + STRATEGY SECTION + screening pipeline
Change only: (a) values in CONFIG when the paper requires it (timeframe, symbols, bar count), and (b) everything between
`>>> STRATEGY SECTION` and `>>> END STRATEGY SECTION`. The example shown there (an RSI fade) is a working placeholder.
```python
# ----------------------------- CONFIG -----------------------------
SYMBOLS      = get_market_watch_symbols()     # or ["EURUSD", "GBPUSD", "XAUUSD", "USDJPY"]
TIMEFRAME    = mt5.TIMEFRAME_H1
N_CANDLES    = 20000
TRAIN_PCT    = 0.70
N_TRIALS_PER_VARIANT = 60
MIN_ACTIVE_DAYS = 30                                # anti-overfit floor (counts active days)
MIN_IS_FITNESS  = 0.0                               # skip OOS if best in-sample fitness <= this
GRADED_PENALTY  = True                              # graded scores for failed trials (False = flat -9999)
SWAP_DEFAULT    = 0.0002                            # fallback swap, fraction of price per night (0.02%)
SEED         = 42

# >>> STRATEGY SECTION: the ONLY part that changes between strategies <<<
# Example (RSI fade). Replace everything between the two >>> markers with the paper's strategy.
VARIANTS = ["rsi"]        # one Optuna study per entry. Use a SINGLE entry if the strategy has no variants.

def compute_strategy_signals(df, variant, rsi_period=14, oversold=30, overbought=70):
    """
    Target position per bar (+1 long, -1 short, 0 flat), decided at close[t]; the engine trades it at
    open[t+1]. Must be causal, deterministic, self-contained, and work on ANY slice of the data
    (walk-forward calls it on df.iloc[a:e]).
    """
    work = add_rsi(df[["close"]].copy(), period=int(rsi_period))     # add_* mutate their input: pass a copy
    rsi = work[f"RSI_{int(rsi_period)}"].values
    pos = np.zeros(len(df)); s = 0
    for i, r in enumerate(rsi):
        if np.isnan(r):                      s = 0          # indicator not warmed up -> flat
        elif r < oversold:                   s = 1
        elif r > overbought:                 s = -1
        elif (s == 1 and r >= 50) or (s == -1 and r <= 50): s = 0   # mean exit
        pos[i] = s
    return pd.Series(pos, index=df.index, name="position")

def suggest_params(trial, variants):
    """Strategy params only (stops are searched separately by the toolbox). MUST include 'variant'."""
    p = {"variant": trial.suggest_categorical("variant", variants)}
    p["rsi_period"] = trial.suggest_int("rsi_period", 2, 30)
    p["oversold"]   = trial.suggest_int("oversold", 10, 40)
    p["overbought"] = trial.suggest_int("overbought", 60, 90)     # ranges never overlap -> oversold < overbought
    return p

def map_params(p):
    """Flat Optuna params -> keyword arguments of compute_strategy_signals."""
    return {"variant": p["variant"], "rsi_period": p["rsi_period"],
            "oversold": p["oversold"], "overbought": p["overbought"]}
# >>> END STRATEGY SECTION <<<

# ----------------------- SCREEN: optimise IS -> blind OOS -> cost stress -----------------------
portfolio_results, stress_tables = [], {}

for sym in SYMBOLS:
    print(f"\n=== {sym} ===")
    if not mt5.symbol_select(sym, True):
        print(f"  skip: cannot select {sym}"); continue
    info = mt5.symbol_info(sym)
    if info is None:
        print(f"  skip: no symbol_info for {sym}"); continue
    point_size, spread_pts = info.point, info.spread

    df = get_mt5_data(sym, TIMEFRAME, N_CANDLES)
    if df is None or len(df) < 3000:
        print("  skip: insufficient data"); continue

    train_df, test_df = split_data(df, train_pct=TRAIN_PCT)

    fit = fit_strategy(train_df, point_size, spread_pts, compute_strategy_signals, suggest_params,
                       map_params, VARIANTS, STOP_MODES, n_trials=N_TRIALS_PER_VARIANT, seed=SEED,
                       min_active_days=MIN_ACTIVE_DAYS, graded=GRADED_PENALTY)
    if fit is None or fit["fitness"] <= MIN_IS_FITNESS:
        print(f"  no viable in-sample strategy, skipping OOS"); continue
    best, stop, ref_atr = fit["params"], fit["stop"], fit["ref_atr"]
    print(f"  WINNER {best['variant']} | stop {stop} | IS fitness {fit['fitness']:.4f}")

    # Blind OOS test, run ONCE (signals computed on full history so windows are warm)
    pos_oos = compute_strategy_signals(df, **map_params(best)).loc[test_df.index]
    oos_net = run_backtest(test_df, pos_oos, point_size, spread_pts, ref_atr, stop)
    oos = calculate_ai_metrics(oos_net, mc_simulations=500)
    if oos is None:
        print("  OOS metrics unavailable"); continue

    # Cost stress on the OOS slice: same positions + stop, harsher costs
    swap = get_swap_pct(info, float(train_df["close"].median()), default=SWAP_DEFAULT)
    stress_tbl, stress = stress_test(test_df, pos_oos, point_size, spread_pts, ref_atr, stop,
                                     swap_pct_per_night=swap)
    stress_tables[sym] = stress_tbl

    bh = calculate_ai_metrics(test_df["close"].pct_change().dropna(), mc_simulations=100)
    generate_tearsheet(oos_net, title=f"{sym} {best['variant']} {stop['mode']} OOS")

    portfolio_results.append({
        "symbol": sym, "variant": best["variant"],
        "stop_mode": stop["mode"], "stop_atr_mult": stop["mult"],
        "is_fitness": fit["fitness"],
        "oos_cagr": oos["cagr"], "oos_sharpe": oos["sharpe"],
        "oos_sortino": oos["sortino"], "oos_max_dd": oos["max_drawdown"],
        "oos_calmar": oos["calmar"], "oos_win_rate": oos["win_rate"],
        "oos_profit_factor": oos["profit_factor"], "oos_active_days": oos["n_trades"],
        "oos_mc_worst5_dd": oos["mc_worst_5%_drawdown"],
        **stress, "swap_pct_night": swap,
        "bh_sharpe": bh["sharpe"] if bh else np.nan,
        "params": best,
    })

if portfolio_results:
    leaderboard = (pd.DataFrame(portfolio_results)
                     .sort_values("oos_sharpe", ascending=False).reset_index(drop=True))
    display(leaderboard)
else:
    print("No symbol produced a viable in-sample strategy.")
```

### Cell 3: Walk-forward on the shortlist (never change)
```python
# ------------- WALK-FORWARD on the shortlist (re-optimises every fold: slow, run on candidates only) -------------
from functools import partial

WF_FOLDS       = 5
WF_INIT_TRAIN  = 0.5        # first 50% = first training set; the rest is split into WF_FOLDS test chunks
WF_ANCHORED    = True       # True = expanding train window, False = rolling window of the initial size
WF_TRIALS      = 30         # trials per variant per fold (5 folds x 4 variants x 30 = 600 backtests / symbol)
WF_MIN_POS_FOLDS = 0.8     # share of folds that must be profitable for wf_pass (0.8 = 4 of 5; stricter = fewer false positives)

# Candidates that were profitable OOS AND survived every cost scenario. Or set a list by hand.
SHORTLIST = (leaderboard.loc[leaderboard["survives_stress"] & (leaderboard["oos_sharpe"] > 0), "symbol"]
                        .head(5).tolist() if portfolio_results else [])
# SHORTLIST = ["XNGUSD", "EURJPY"]

wf_rows, wf_results = [], {}
for sym in SHORTLIST:
    print(f"\n=== WALK-FORWARD {sym} ===")
    mt5.symbol_select(sym, True)
    info = mt5.symbol_info(sym)
    point_size, spread_pts = info.point, info.spread
    df = get_mt5_data(sym, TIMEFRAME, N_CANDLES)

    swap = get_swap_pct(info, float(df["close"].median()), default=SWAP_DEFAULT)
    fit_fn = partial(fit_strategy, point_size=point_size, spread_pts=spread_pts,
                     signal_fn=compute_strategy_signals, space_fn=suggest_params, map_fn=map_params,
                     groups=VARIANTS, stop_modes=STOP_MODES, n_trials=WF_TRIALS, seed=SEED,
                     min_active_days=MIN_ACTIVE_DAYS, graded=GRADED_PENALTY, verbose=False)
    wf = walk_forward(df, fit_fn, compute_strategy_signals, map_params, point_size, spread_pts,
                      n_folds=WF_FOLDS, init_train_frac=WF_INIT_TRAIN, anchored=WF_ANCHORED,
                      min_is_fitness=MIN_IS_FITNESS, min_pos_fold_share=WF_MIN_POS_FOLDS,
                      engine_kw={"swap_pct_per_night": swap})   # walk-forward already includes swap
    wf_results[sym] = wf
    display(wf["folds"].drop(columns="params"))
    wf_rows.append({"symbol": sym, **wf["summary"]})
    generate_tearsheet(wf["returns"], title=f"{sym} walk-forward stitched OOS")

if wf_rows:
    wf_summary = pd.DataFrame(wf_rows).sort_values("wf_sharpe", ascending=False).reset_index(drop=True)
    display(wf_summary)
else:
    print("Empty shortlist - nothing survived the OOS + cost-stress screen.")
```

### Cell 4: Cleanup (never change)
```python
mt5.shutdown()
print("MT5 connection closed.")
```

---

# 3. THE STRATEGY SECTION CONTRACT

You must define exactly these four names. Keep the signatures.

1. `VARIANTS: list[str]`. One Optuna study runs per entry. If the paper has one strategy, use ONE entry (e.g. `["rsi"]`).
   If it has several rule sets or detectors, one entry each.
2. `compute_strategy_signals(df, variant, **params) -> pd.Series`
   - Returns target positions indexed exactly like `df`: `1` long, `-1` short, `0` flat. No NaN. The series is a
     held STATE (stays +1 until something changes it), not a one-bar pulse.
   - Decided at the close of bar `t`; the engine executes it at the open of bar `t+1`. Do not shift it yourself.
   - **Causal:** bar `t` may use data up to and including bar `t` only. Never `shift(-n)`, never centred windows, never
     whole-sample statistics (global mean/std, full-sample quantiles, full-sample fitted models). Rolling or expanding only.
   - **Slice-safe:** it is called on the training set, on the full history (then sliced to OOS so indicators are warm),
     and on arbitrary walk-forward windows `df.iloc[a:e]`. It must not depend on where `df` starts, on `len(df)`, or on globals
     that change. Return all zeros (not an error) if there is too little data.
   - **Deterministic:** fix every random seed.
   - **No side effects:** `add_rsi`, `add_macd`, `add_atr`, `add_time_features` MUTATE the dataframe you pass. Pass a copy.
   - Fast: it runs once per Optuna trial (thousands of times). Avoid per-bar Python loops where a vectorised form exists;
     where a state machine is genuinely needed a simple loop is fine.
3. `suggest_params(trial, variants) -> dict`. Flat dict of this strategy's parameters. It MUST contain the key `"variant"`
   (`trial.suggest_categorical("variant", variants)`), because the pipeline reads `best["variant"]`. Do not put stop
   parameters here. Conditional parameters per variant are fine.
4. `map_params(p) -> dict`. Turns that flat dict into keyword arguments for `compute_strategy_signals`.

**Search ranges:** justify them in Part 3. Rescale ranges the paper gives for daily equities to the timeframe you run
(an H1 FX bar moves a few hundredths of a percent, not several percent). Avoid ranges so wide that most trials are
untradeable. Avoid thresholds so extreme that fewer than ~30 active days result.

---

# 4. API REFERENCE (use these exactly; do NOT invent functions)

### `trading_engine.py`
- `get_mt5_data(symbol, timeframe, n_candles) -> DataFrame` indexed by UTC `time` with columns
  `open, high, low, close, tick_volume, spread, real_volume`. Returns `None` on failure.
- `split_data(df, train_pct=0.7) -> (train_df, test_df)`. Chronological, no shuffle.
- `apply_sl_tp_and_costs(df, positions, point_size, spread_points, sl_points=0, tp_points=0, trail_points=0,
  slippage_points=0, swap_pct_per_night=0.0) -> pd.Series` of per-bar net returns. THE execution engine:
  - Signal at `positions[t-1]` is acted on at `open[t]`. A flip or a flat signal exits at that open.
  - Spread: one spread per round trip, charged at entry (correct for bid-quoted MT5 candles).
  - Stops are checked intrabar with the pessimistic rule (stop before take-profit). If a bar opens beyond the stop, the
    fill is the OPEN, not the stop price (gap-aware).
  - **Stop lock:** after a stop-out the engine stays flat until the target signal CHANGES. A held `+1` does not re-enter
    after a stop. A strategy that should re-arm on the next same-direction event must emit a flat or opposite state in between.
  - One position at a time, fixed size (returns are per unit of capital). No pyramiding, no sizing.
- `apply_trading_costs(df, positions, point_size, spread_points)`: simple close-to-close model that charges a spread on
  EVERY position change (two per round trip). Legacy. The pipeline does not use it, and its numbers are not comparable with
  `apply_sl_tp_and_costs`.
- `cost_stress(...)`: used through `stress_test` (below).
- `calculate_ai_metrics(strategy_returns, mc_simulations=1000) -> dict | None` with keys `cagr, max_drawdown, sharpe,
  sortino, calmar, win_rate, profit_factor, tail_ratio, value_at_risk, expected_shortfall, kelly_criterion, kurtosis,
  skewness, n_trades, mc_median_return, mc_worst_5%_return, mc_worst_5%_drawdown`.
  **`n_trades` is the number of ACTIVE DAYS (days with non-zero return), not trades.** Never describe it as a trade count.
  The Monte Carlo shuffles daily returns as independent, so it understates drawdown when losses cluster.
- `generate_tearsheet(strategy_returns, title, output_folder="reports")` writes an HTML report.

### `quant_toolbox.py`
- Indicators (all return the same dataframe with new columns, MUTATING it): `add_rsi(df, column='close', period=14)` ->
  `RSI_{period}`; `add_macd(df, column='close', fast=12, slow=26, signal=9)` -> `MACD_line, MACD_signal, MACD_histogram`;
  `add_atr(df, period=14)` -> `ATR_{period}`; `add_time_features(df)` -> `hour, day_of_week, day_of_month, month`.
- `fetch_vix_regime(df)` adds a `VIX` column from Yahoo. **Caution:** it gives every bar the VIX CLOSE of the same calendar
  day, which is lookahead for intraday bars. If you use it, lag it by one full trading day first and say so in Part 3.
- Stop search (already wired into `fit_strategy`; do not reimplement): `STOP_MODES = ["none", "fixed", "trail"]`.
  Stop distance is a multiple of the in-sample median ATR(14): `FIXED_RANGE = TRAIL_RANGE = (1.0, 20.0)`.
  To restrict or narrow without editing the file: `STOP_MODES = ["trail"]` in the notebook, and
  `import quant_toolbox as qt; qt.TRAIL_RANGE = (2.0, 10.0)`.
- `fit_strategy(train_df, point_size, spread_pts, signal_fn, space_fn, map_fn, groups, stop_modes, n_trials, seed,
  min_active_days, graded, ...)` -> `dict(params, stop, fitness, ref_atr, group_scores)` or `None`.
- `run_backtest(df, pos, point_size, spread_pts, ref_atr, stop, **engine_kw)`. `ref_atr` is the IN-SAMPLE median ATR from
  `fit["ref_atr"]`; always reuse it for the OOS slice (never recompute on OOS).
- `stress_test(df, pos, point_size, spread_pts, ref_atr, stop, swap_pct_per_night)` -> `(table, summary)`.
- `walk_forward(df, fit_fn, signal_fn, map_fn, point_size, spread_pts, n_folds, init_train_frac, anchored,
  min_is_fitness, min_pos_fold_share, engine_kw)` -> `dict(folds, summary, returns)`.
- `get_swap_pct(info, ref_price, default)`: worst-case overnight swap from broker metadata.
- `fitness_from_metrics`: the fitness. Viable strategies score `cagr - 2*abs(mc_worst_5%_drawdown)`. Strategies with CAGR <= 0
  or fewer than `MIN_ACTIVE_DAYS` active days get graded scores far below any viable one. Do not reimplement it.

### `instruments.py`
`get_market_watch_symbols()`, `get_symbol_details(symbols)`, and lists `FOREX_MAJORS`, `FOREX_CROSSES`, `METALS_INDICES`.

### Broker metadata (already used in the template)
```python
info = mt5.symbol_info(sym)
point_size, spread_pts = info.point, info.spread     # spread is the CURRENT spread, applied to all history
```

---

# 5. ENGINEERING RULES (ZERO TOLERANCE)

1. **No lookahead.** See the causality rules in section 3. This includes data you merge in (macro, VIX, news): each bar may
   only see values that were published before that bar closed.
2. **Optimise on in-sample data only; test once.** Never tune anything after seeing OOS or walk-forward results. Do not add
   a second round of "improvements" to the strategy based on results you were told about unless the user restarts the process.
3. **Use the toolbox for optimisation, stops and metrics.** No custom Optuna objectives, no custom stop-loss code, no custom
   Sharpe or drawdown functions. (The factory/closure pattern for Optuna objectives is already inside `fit_strategy`.)
4. **Position convention:** a Series of target states `1`, `-1`, `0`. If the paper's logic is "enter on event, exit on
   another event", convert it to a held state yourself.
5. **Stops belong to the toolbox.** If the paper specifies its own stop (e.g. a 5% trailing stop), do not hard-code it in the
   signals. Restrict `STOP_MODES` and/or narrow `qt.TRAIL_RANGE` to bracket it, and explain in Part 3. Stops are in ATR
   multiples, so convert the paper's percentage to an ATR multiple for the asset class and show the arithmetic.
6. **Do not change the template pipeline, the engine, or the toolbox.** If something truly cannot work, say so (section 6).
7. **Keep it simple and runnable.** You cannot execute MT5 code. Code that is too clever is code the user cannot debug.

---

# 6. WHAT THIS FRAMEWORK CANNOT DO (flag it; do not hack around it)

If the PDF needs any of the following, say so clearly in Part 1, then implement the closest honest approximation or stop and ask:
- **Take-profit / profit targets** relative to entry. The stop search has no TP. Express exits as signal-based exits
  (flat state) if the logic allows, and note the difference.
- **Position sizing, leverage, pyramiding, grid/martingale, multiple simultaneous positions.** The engine is one fixed-size position.
- **Multi-instrument strategies** (pairs, baskets, cross-sectional ranking, relative value). The pipeline is one symbol, one
  position series.
- **Data MT5 bars do not contain:** tick or order-book data, volume profile, fundamentals, options, sentiment, alternative data.
  Offer a bar-based proxy only if it is honest, and label it as a proxy.
- **Intrabar timing rules** (e.g. "enter when price touches X intrabar", opening-range breakouts needing sub-bar data).
  Signals are decided on closed bars and executed at the next open.
- **Session or news filters** need the bar timestamp (`df.index`, UTC/broker server time) and are fine. Say which timezone you assumed.

---

# 7. REQUIRED OUTPUT FORMAT

**Part 1: Strategy decomposition (Markdown)**
- **Core premise:** what inefficiency the paper claims, and what evidence it actually offers. State plainly whether the paper's
  results are in-sample, include costs, use a train/test split, or select assets after the fact. Do not repeat headline numbers
  as if they were validated.
- **Mathematical formulations:** exact formulas for every indicator and signal, with the paper's notation mapped to code.
- **Decision rules:** exact long / short / exit / stop conditions, as a state machine.
- **Hyperparameter space:** a table of every Optuna parameter: paper range, range you use, and why (timeframe rescaling,
  degenerate values, trade-count considerations).
- **What does not map:** every paper feature the framework cannot represent (section 6) and what you did instead.

**Part 2: Code**
- Output the full **Cell 2** (config + strategy section + screening pipeline), complete and ready to paste.
- For Cells 1, 3 and 4 write "unchanged from the template" unless you needed a change; if you did, output that cell in full
  and say exactly why.
- Set CONFIG values the paper implies (timeframe, symbols, `N_CANDLES`). If the paper uses daily bars, check that the
  `len(df) < 3000` guard in the pipeline and the walk-forward minimums (first training set at least 500 bars, each test
  chunk at least 100 bars) can be satisfied by the broker's history, and adjust or warn.

**Part 3: Implementation notes**
- Every decision where the paper was ambiguous, and the reading you chose.
- Every place you avoided lookahead that the paper's own method contains (for example fitting a model on the full sample).
- How each searched parameter maps to the author's findings, and what result would count as the paper NOT replicating.
- Expected trade frequency: roughly how many active days per year the rules should produce, so the 30-active-day floor is realistic.
- **MT5 hand-off spec:** a short, plain-language, unambiguous description of the final rules (signal on bar close, execute
  at next open, stop type and ATR multiple, what "flat" means) so it can be re-implemented as an MQL5 EA and compared
  trade-by-trade in the Strategy Tester.

---

# 8. SELF-CHECK BEFORE YOU ANSWER

Confirm each item silently; fix anything that fails.
- [ ] `VARIANTS`, `compute_strategy_signals`, `suggest_params`, `map_params` exist with the exact signatures.
- [ ] `suggest_params` returns a dict containing `"variant"`; `map_params` returns kwargs that match the function's arguments.
- [ ] The signal function returns a Series of `1/-1/0`, no NaN, indexed like `df`, and works on a 300-bar slice.
- [ ] No `shift(-n)`, no centred windows, no whole-sample statistics, no unseeded randomness, no mutation of the input `df`.
- [ ] No custom optimiser, stop-loss, or metrics code; no invented functions; no edits to the engine or toolbox.
- [ ] Parameter ranges are justified for the timeframe and cannot produce `oversold >= overbought`-style contradictions.
- [ ] Anything the paper needs that the framework cannot do is listed in Part 1.

---

# 9. HOW I WILL READ THE RESULTS (so your notes help)

- `leaderboard` columns: `is_fitness`, `oos_cagr`, `oos_sharpe`, `oos_active_days`, `stress_worst_cagr`, `survives_stress`,
  `bh_sharpe` (buy-and-hold on the same slice).
- A strategy is only a candidate if: IS fitness > 0, OOS Sharpe > 0, `survives_stress` is True, and (in Cell 3)
  `wf_pass` is True with `pct_folds_positive >= 0.8`. A high OOS Sharpe over a few dozen active days means little.
- Scanning many symbols guarantees a few lucky passes. I treat every pass as a candidate to replicate in the MT5 Strategy
  Tester and forward-test on demo, not as proof.
- If the strategy fails everywhere, that is a valid and useful result. Do not tune the strategy to rescue it.
````