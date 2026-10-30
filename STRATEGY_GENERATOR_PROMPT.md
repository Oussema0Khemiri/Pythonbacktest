
````markdown
# ROLE & MISSION
You are a Senior Quantitative Researcher and Algorithmic Trading Systems Engineer.
I have attached a PDF describing a trading strategy, research paper or market anomaly.

Your task:
1. Extract the underlying statistical / technical edge and the exact rules from the PDF.
2. Implement it as the STRATEGY SECTION of my existing notebook template (below), so it runs unchanged through my
   pipeline: in-sample optimisation -> blind out-of-sample test -> cost stress -> walk-forward.

You are NOT writing a backtester, an optimiser, a stop-loss / take-profit / sizing module or a metrics function. Those
exist and are tested. Writing your own versions of them is the most common way this task goes wrong.

---

# 1. HOW MY PROJECT WORKS

Python 3.10+, MetaTrader 5, Optuna, NumPy, Pandas, QuantStats. Four files:

| File | Role |
|---|---|
| `trading_engine.py` | MT5 data fetch, chronological split, the bar-by-bar execution engine (entries, stops, targets, sizing, costs), metrics, tearsheet |
| `quant_toolbox.py` | Indicators + the generic optimiser (`fit_strategy`), `run_backtest`, `stress_test`, `walk_forward` |
| `instruments.py` | `get_market_watch_symbols()`, `get_symbol_details()`, preset symbol lists |
| `TestingAlgos.ipynb` | The notebook template (4 cells) |

**What the notebook does, in order:**

1. **Cell 1: setup.** Imports, `mt5.initialize()`. Never changes.
2. **Cell 2: screen.** For every symbol: fetch MT5 bars and broker specs, split 70% in-sample (IS) / 30% out-of-sample (OOS),
   optimise on IS only (`fit_strategy`: one Optuna study per variant, strategy params AND stop / trade-management params
   together), skip the symbol if the best IS fitness is not positive, run the winner ONCE on the OOS slice, run the cost
   stress on that slice, write an HTML tearsheet, and add one row to the `leaderboard` (sorted by OOS Sharpe).
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
# get_swap_pcts, get_swap_pct, get_ref_atr, add_atr, STOP_MODES, ...
from quant_toolbox import *
import quant_toolbox as qt          # search ranges are set as qt.<NAME>_RANGE in the strategy section

optuna.logging.set_verbosity(optuna.logging.WARNING)

if not mt5.initialize():
    raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
print("MT5 connected:", mt5.terminal_info().name)
```

### Cell 2: Config + STRATEGY SECTION + screening pipeline
Change only: (a) values in CONFIG when the paper requires it (timeframe, symbols, bar count, leverage), and (b) everything
between `>>> STRATEGY SECTION` and `>>> END STRATEGY SECTION`. The example shown there (an RSI fade) is a working placeholder.
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
MAX_LEVERAGE    = 30                                # notional / equity cap when a layer opens (mt5.account_info().leverage)
SEED         = 42

# >>> STRATEGY SECTION: the ONLY part that changes between strategies <<<
# Example (RSI fade). Replace everything between the two >>> markers with the paper's strategy.
VARIANTS = ["rsi"]        # one Optuna study per entry. Use a SINGLE entry if the strategy has no variants.

# Stop + trade-management search (always set ALL of these: module settings persist between notebook runs)
STOP_MODES         = ["none", "fixed", "trail"]
qt.FIXED_RANGE     = (1.0, 20.0)       # ATR multiples
qt.TRAIL_RANGE     = (1.0, 20.0)       # ATR multiples
qt.BE_RANGE        = None              # breakeven trigger, ATR multiples (None = off)
qt.TP_RANGE        = None              # final take-profit, ATR multiples (None = off)
qt.TP1_RANGE       = None              # partial take-profit, ATR multiples (None = off)
qt.TIME_STOP_RANGE = None              # time stop, bars (None = off)

# Fixed engine settings for this strategy (sizing, layers, scale-out fraction, ...). {} = classic engine.
ENGINE_KW = {}

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
ENGINE = {"max_leverage": MAX_LEVERAGE, **ENGINE_KW}     # every backtest below uses the same engine settings
portfolio_results, stress_tables, oos_trade_logs = [], {}, {}

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
                       min_active_days=MIN_ACTIVE_DAYS, graded=GRADED_PENALTY, engine_kw=ENGINE)
    if fit is None or fit["fitness"] <= MIN_IS_FITNESS:
        print(f"  no viable in-sample strategy, skipping OOS"); continue
    best, stop, ref_atr = fit["params"], fit["stop"], fit["ref_atr"]
    print(f"  WINNER {best['variant']} | stop {stop} | IS fitness {fit['fitness']:.4f}")

    # Blind OOS test, run ONCE (signals computed on full history so windows are warm)
    pos_oos = compute_strategy_signals(df, **map_params(best)).loc[test_df.index]
    oos_net, oos_trades = run_backtest(test_df, pos_oos, point_size, spread_pts, ref_atr, stop,
                                       return_trades=True, **ENGINE)
    oos_trade_logs[sym] = oos_trades
    oos = calculate_ai_metrics(oos_net, mc_simulations=500)
    if oos is None:
        print("  OOS metrics unavailable"); continue

    # Cost stress on the OOS slice: same positions + stop + engine settings, harsher costs
    swap_l, swap_s = get_swap_pcts(info, float(train_df["close"].median()), default=SWAP_DEFAULT)
    stress_tbl, stress = stress_test(test_df, pos_oos, point_size, spread_pts, ref_atr, stop,
                                     swap_long_pct_per_night=swap_l, swap_short_pct_per_night=swap_s,
                                     engine_kw=ENGINE)
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
        "oos_trades": int((oos_trades["reason"] != "tp1").sum()),     # real trade count (layers opened)
        "oos_mc_worst5_dd": oos["mc_worst_5%_drawdown"],
        **stress, "swap_long_night": swap_l, "swap_short_night": swap_s,
        "bh_sharpe": bh["sharpe"] if bh else np.nan,
        "stop": stop, "params": best,
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

    swap_l, swap_s = get_swap_pcts(info, float(df["close"].median()), default=SWAP_DEFAULT)
    fit_fn = partial(fit_strategy, point_size=point_size, spread_pts=spread_pts,
                     signal_fn=compute_strategy_signals, space_fn=suggest_params, map_fn=map_params,
                     groups=VARIANTS, stop_modes=STOP_MODES, n_trials=WF_TRIALS, seed=SEED,
                     min_active_days=MIN_ACTIVE_DAYS, graded=GRADED_PENALTY, engine_kw=ENGINE, verbose=False)
    wf = walk_forward(df, fit_fn, compute_strategy_signals, map_params, point_size, spread_pts,
                      n_folds=WF_FOLDS, init_train_frac=WF_INIT_TRAIN, anchored=WF_ANCHORED,
                      min_is_fitness=MIN_IS_FITNESS, min_pos_fold_share=WF_MIN_POS_FOLDS,
                      engine_kw={**ENGINE, "swap_long_pct_per_night": swap_l,
                                 "swap_short_pct_per_night": swap_s})   # walk-forward OOS includes swap
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

You must define exactly these names. Keep the signatures.

1. `VARIANTS: list[str]`. One Optuna study runs per entry. If the paper has one strategy, use ONE entry (e.g. `["rsi"]`).
   If it has several rule sets or detectors, one entry each.
2. The stop / management block: `STOP_MODES`, `qt.FIXED_RANGE`, `qt.TRAIL_RANGE`, `qt.BE_RANGE`, `qt.TP_RANGE`,
   `qt.TP1_RANGE`, `qt.TIME_STOP_RANGE` and `ENGINE_KW`. Set **every one** of them, even when you keep the default, because
   `qt.*` settings survive from the previous strategy run in the same notebook kernel. See section 4 for what each does.
3. `compute_strategy_signals(df, variant, **params)` returns EITHER
   - **a position Series** (classic): indexed exactly like `df`, whole numbers, no NaN. `1` long, `-1` short, `0` flat.
     With pyramiding (`ENGINE_KW["max_layers"] > 1`) the size is the number of layers wanted: `+3` = three long layers.
   - **or an orders DataFrame**, indexed exactly like `df`, with a `position` column (same rules) plus any of these optional
     columns, and no others:

     | column | meaning | unused value |
     |---|---|---|
     | `entry_price` | pending-order price for NEW layers (limit or stop, see below) | `NaN` = market order at the next open |
     | `sl_price` | the strategy's own stop level (a moving average, a channel line, a swing low...) | `NaN` = no strategy stop |
     | `size` | size multiplier for a layer that opens on the next bar (e.g. `0.5`, or `2` for martingale steps) | `1.0` (`NaN` or `<= 0` = do not open) |
     | `exit_at_close` | `True` = close everything at the CLOSE of this same bar and open nothing on it | `False` |

   Rules for both forms:
   - The position is a held STATE (stays `+1` until something changes it), not a one-bar pulse.
   - `position`, `entry_price`, `sl_price`, `size` are decided at the close of bar `t` and used during bar `t+1`. Do not
     shift them yourself.
   - `exit_at_close[t]` acts at the close of bar `t` itself, so it must be known BEFORE that bar closes: build it from the
     timestamp only (session end, Friday's last bar, a news window). Never from bar `t`'s own close/high/low.
   - **Causal:** bar `t` may use data up to and including bar `t` only. Never `shift(-n)`, never centred windows, never
     whole-sample statistics (global mean/std, full-sample quantiles, full-sample fitted models). Rolling or expanding only.
   - **Slice-safe:** it is called on the training set, on the full history (then sliced to OOS so indicators are warm),
     and on arbitrary walk-forward windows `df.iloc[a:e]`. It must not depend on where `df` starts, on `len(df)`, or on globals
     that change. Return all zeros (not an error) if there is too little data.
   - **Deterministic:** fix every random seed.
   - **No side effects:** `add_rsi`, `add_macd`, `add_atr`, `add_time_features` MUTATE the dataframe you pass. Pass a copy.
   - Fast: it runs once per Optuna trial (thousands of times). Avoid per-bar Python loops where a vectorised form exists;
     where a state machine is genuinely needed a simple loop is fine.
4. `suggest_params(trial, variants) -> dict`. Flat dict of this strategy's parameters. It MUST contain the key `"variant"`
   (`trial.suggest_categorical("variant", variants)`), because the pipeline reads `best["variant"]`. Do not put stop or
   trade-management parameters here (they are searched by the toolbox through the `qt.*_RANGE` settings). Conditional
   parameters per variant are fine.
5. `map_params(p) -> dict`. Turns that flat dict into keyword arguments for `compute_strategy_signals`.

**Search ranges:** justify them in Part 3. Rescale ranges the paper gives for daily equities to the timeframe you run
(an H1 FX bar moves a few hundredths of a percent, not several percent). Avoid ranges so wide that most trials are
untradeable. Avoid thresholds so extreme that fewer than ~30 active days result.

### How to use each engine feature (pick only what the paper actually specifies)

| Paper says | Do this |
|---|---|
| "Buy when price breaks the 20-bar high" (intrabar breakout) | orders DataFrame, `position = 1`, `entry_price = rolling 20-bar high` (a buy STOP, since it is above the close). |
| "Buy the pullback to the 50 EMA" | `position = 1`, `entry_price = EMA` while the setup is valid (a buy LIMIT, since it is below the close). |
| "Add a unit every 1 ATR in profit, max 4 units" (pyramid / Turtle) | `ENGINE_KW = {"max_layers": 4}`; `position` steps `1 -> 2 -> 3 -> 4`. Use `entry_price` for the add levels if they are intrabar. |
| Grid / martingale | `max_layers = N`; `position = k` for the k-th grid level, `entry_price = grid level`; martingale sizes through `size` (e.g. `2 ** (k-1)`). Flag the tail risk in Part 1. |
| "Stop at the 10-bar low" / "exit when price closes below the MA" as a stop line | `sl_price` column (long: the line below price, short: above). Set `STOP_MODES = ["none"]` if this IS the stop. |
| "Move stop to breakeven after 1R / 1 ATR" | `qt.BE_RANGE = (0.5, 2.0)` (searched) or `(1.0, 1.0)` (fixed). Lock a profit with `ENGINE_KW["be_offset_points"]`. |
| "Take 50% at 1R, let the rest run" | `qt.TP1_RANGE = (...)`, `ENGINE_KW = {"tp1_fraction": 0.5, "tp1_move_to_be": True}` if the paper also moves the runner's stop to breakeven. |
| Fixed profit target | `qt.TP_RANGE = (...)`. |
| "Exit after 10 bars if not stopped" | `qt.TIME_STOP_RANGE = (10, 10)` (fixed) or `(5, 40)` (searched, integers). |
| "Flat before the session close / no weekend holding" | `exit_at_close = (df.index.dayofweek == 4) & (df.index.hour == 21)` (or the session's last bar). State the server timezone you assumed. |
| "Risk 1% of equity per trade" | `ENGINE_KW = {"sizing": "risk", "risk_pct": 0.01}` and a stop: `STOP_MODES` without `"none"`, or an `sl_price` column. |
| "Trade half size when volatility is high" | `size` column (`0.5` / `1.0`). |

**Re-arming after an exit:** after a stop, target, time stop or session close the engine opens nothing until the
`position` value CHANGES. A pending breakout that should re-arm on the same side must emit `0` (or the opposite side) for
at least one bar in between.

**Prefer risk sizing whenever the strategy has a stop.** With classic sizing every trade is 100% of equity, so a trade with
a 200-pip stop weighs ten times more than one with a 20-pip stop, and Sharpe and drawdown are distorted. If you do NOT
use `sizing="risk"`, say why in Part 3 (e.g. the paper has no stop, or it is a buy-and-hold style signal).

---

# 4. API REFERENCE (use these exactly; do NOT invent functions)

### `trading_engine.py`
- `get_mt5_data(symbol, timeframe, n_candles) -> DataFrame` indexed by UTC `time` with columns
  `open, high, low, close, tick_volume, spread, real_volume`. Returns `None` on failure.
- `split_data(df, train_pct=0.7) -> (train_df, test_df)`. Chronological, no shuffle.
- THE execution engine (every new argument defaults to OFF, which gives exactly the classic results):
  ```python
  apply_sl_tp_and_costs(df, positions, point_size, spread_points,
      sl_points=0, tp_points=0, trail_points=0, slippage_points=0, swap_pct_per_night=0.0,
      entry_prices=None, max_layers=1,
      be_trigger_points=0, be_offset_points=0, sl_prices=None, max_bars=0,
      tp1_points=0, tp1_fraction=0.5, tp1_move_to_be=False, exit_at_close=None,
      sizing="unit", size=None, risk_pct=0.01, max_leverage=None, leverage_action="reduce",
      swap_long_pct_per_night=None, swap_short_pct_per_night=None,
      return_trades=False)
  ```
  -> per-bar net return Series, or `(returns, trades)` with `return_trades=True`.
  You never call it directly; the toolbox does, with the stop dict and `ENGINE`. It is listed so you know the behaviour.
  - `positions`: Series or orders DataFrame (section 3). Signal at bar `t-1` is acted on during bar `t`. A flip or a flat
    signal exits all layers at the open. A smaller layer count closes the newest layers first (at the open).
  - **Pending orders** (`entry_prices` / `entry_price` column) follow MT5 rules, judged against `close[t-1]`: a long level
    below it is a buy LIMIT, above it a buy STOP (mirror for shorts). The order is re-priced every bar while the signal asks
    for a new layer, and cancelled when it stops asking. A gap through the level fills at the open (better for a limit,
    worse for a stop). Limit fills get no slippage. Levels are bid-chart prices; a long fills at level + spread.
  - **Intrabar order (pessimistic):** stop first, then TP1, then the final TP. Gap-aware: a bar that opens beyond the
    stop fills at the OPEN. On the bar a layer opens, its stop is checked; its targets are checked only for market / stop
    entries (after a limit fill the price was moving against the target, so a same-bar target touch may have come first).
  - **Stop sources** (the tightest active one is used): `sl_points` (fixed from entry), `trail_points` (ratchets after each
    bar), `be_trigger_points` (once the bar's best price is that far in profit, the stop moves to entry + `be_offset_points`;
    entry includes the spread, so this is a true zero-loss exit), `tp1_move_to_be`, and `sl_prices` (the strategy's line,
    may move either way). All stop moves take effect from the NEXT bar. A strategy stop on the wrong side of the entry
    price at entry means no trade (MT5 would reject the order).
  - **Targets:** `tp1_points` closes `tp1_fraction` of each layer (no lock, the runner continues), `tp_points` closes the
    rest. Limit fills: no slippage; a gap through the target fills at the open.
  - **Time stop:** `max_bars = N` closes a layer at the open of the N-th bar after it opened.
  - **Session exit:** `exit_at_close[t] = True` closes everything at `close[t]` (with slippage) and opens nothing on bar `t`.
  - **Stop lock:** after a stop / TP / time stop / session exit the engine opens nothing until the signal value CHANGES.
  - **Pyramiding:** `max_layers` = most layers held at once (all in one direction; a flip closes all). Each layer has its
    own entry price, stops, targets, trailing extreme and time stop, and pays its own spread.
  - **Sizing:** returns are fractions of a compounding equity curve.
    `sizing="unit"`: each layer is `size` x 100% of equity (default 1), re-marked every bar; one layer at size 1 is the
    classic engine. `sizing="risk"`: fixed lots per layer, set so that the INITIAL stop distance (spread included) loses
    `risk_pct x size` of equity; gaps and slippage can lose more. Needs a stop source; an entry without a valid stop is
    skipped. `max_leverage`: total notional / equity cap checked when a layer opens; `leverage_action="reduce"` shrinks the
    layer to fit, `"skip"` does not open it. If equity reaches 0 everything closes and the run stops (ruin).
  - **Costs:** one spread per layer round trip, charged at entry (correct for bid-quoted MT5 candles). `slippage_points` is
    adverse, on entries and market / stop exits. Swap is charged per calendar day crossed (a weekend = 3 nights), as a
    fraction of the held notional. `swap_pct_per_night` applies to both sides; `swap_long_pct_per_night` /
    `swap_short_pct_per_night` override it per direction (positive = you pay, negative = you earn).
  - **Trade log** (`return_trades=True`): one row per exit (a TP1 partial is its own row) with `entry_time, exit_time,
    direction, entry_price, exit_price, fraction, reason, bars_held, price_return, r_multiple, leverage_at_entry`.
    `reason` is one of `signal, sl, tp1, tp, time, close, ruin, end` (`end` = still open on the last bar).
- `apply_trading_costs(df, positions, point_size, spread_points)`: simple close-to-close model that charges a spread on
  EVERY position change (two per round trip). Legacy. The pipeline does not use it, and its numbers are not comparable with
  `apply_sl_tp_and_costs`.
- `cost_stress(df, positions, point_size, spread_points, sl_points, tp_points, trail_points, swap_pct_per_night, scenarios,
  mc_simulations, swap_long_pct_per_night, swap_short_pct_per_night, **engine_kw)`: used through `stress_test` (below).
- `calculate_ai_metrics(strategy_returns, mc_simulations=1000) -> dict | None` with keys `cagr, max_drawdown, sharpe,
  sortino, calmar, win_rate, profit_factor, tail_ratio, value_at_risk, expected_shortfall, kelly_criterion, kurtosis,
  skewness, n_trades, mc_median_return, mc_worst_5%_return, mc_worst_5%_drawdown`.
  **`n_trades` is the number of ACTIVE DAYS (days with non-zero return), not trades.** Never describe it as a trade count;
  the real count is the leaderboard's `oos_trades`.
  The Monte Carlo shuffles daily returns as independent, so it understates drawdown when losses cluster (grids,
  martingales and pyramids cluster their losses: say so in Part 3 if you use them).
- `generate_tearsheet(strategy_returns, title, output_folder="reports")` writes an HTML report.

### `quant_toolbox.py`
- Indicators (all return the same dataframe with new columns, MUTATING it): `add_rsi(df, column='close', period=14)` ->
  `RSI_{period}`; `add_macd(df, column='close', fast=12, slow=26, signal=9)` -> `MACD_line, MACD_signal, MACD_histogram`;
  `add_atr(df, period=14)` -> `ATR_{period}`; `add_time_features(df)` -> `hour, day_of_week, day_of_month, month`.
- `fetch_vix_regime(df)` adds a `VIX` column from Yahoo. **Caution:** it gives every bar the VIX CLOSE of the same calendar
  day, which is lookahead for intraday bars. If you use it, lag it by one full trading day first and say so in Part 3.
- `calculate_dynamic_position_size(...)`: indicative live lot sizes only. For backtests use `sizing="risk"`.
- Stop + trade-management search (already wired into `fit_strategy`; do not reimplement). All distances are multiples of
  the IN-SAMPLE median ATR(14) (`ref_atr`), so one range works across asset classes (1 ATR ~ one typical bar's range):

  | setting | default | searched param | engine argument |
  |---|---|---|---|
  | `STOP_MODES` | `["none", "fixed", "trail"]` | `stop_mode` | which stop type |
  | `qt.FIXED_RANGE` | `(1.0, 20.0)` | `sl_mult` (log) | `sl_points` |
  | `qt.TRAIL_RANGE` | `(1.0, 20.0)` | `trail_mult` (log) | `trail_points` |
  | `qt.BE_RANGE` | `None` (off) | `be_mult` (log) | `be_trigger_points` |
  | `qt.TP_RANGE` | `None` (off) | `tp_mult` (log) | `tp_points` |
  | `qt.TP1_RANGE` | `None` (off) | `tp1_mult` (log) | `tp1_points` |
  | `qt.TIME_STOP_RANGE` | `None` (off) | `max_bars` (int, log) | `max_bars` |

  A range with `low == high` fixes the value without searching it. The winner's settings come back as the stop dict
  `{'mode', 'mult', 'be_mult', 'tp_mult', 'tp1_mult', 'max_bars'}` (0 = off).
- `ENGINE_KW` (the notebook merges it with `max_leverage=MAX_LEVERAGE` into `ENGINE`) carries the fixed, non-searched engine
  settings: `sizing`, `risk_pct`, `max_layers`, `tp1_fraction`, `tp1_move_to_be`, `be_offset_points`, `leverage_action`,
  `max_leverage` (to override the CONFIG cap). Never put a searched setting in it as well (that raises a `TypeError`), and
  never put costs (`slippage_points`, swaps) in it: the cost stress sets those.
- `fit_strategy(train_df, point_size, spread_pts, signal_fn, space_fn, map_fn, groups, stop_modes, n_trials, seed,
  min_active_days, graded, mc_sims=100, engine_kw=None, verbose=True)` -> `dict(params, stop, fitness, ref_atr,
  group_scores)` or `None`.
- `run_backtest(df, pos, point_size, spread_pts, ref_atr, stop, **engine_kw)`. `ref_atr` is the IN-SAMPLE median ATR from
  `fit["ref_atr"]`; always reuse it for the OOS slice (never recompute on OOS). `pos` may be a Series or an orders
  DataFrame. `return_trades=True` also returns the trade log.
- `stress_test(df, pos, point_size, spread_pts, ref_atr, stop, swap_pct_per_night=0.0002, scenarios=None, mc_sims=100,
  swap_long_pct_per_night=None, swap_short_pct_per_night=None, engine_kw=None)` -> `(table, summary)`.
- `walk_forward(df, fit_fn, signal_fn, map_fn, point_size, spread_pts, n_folds, init_train_frac, anchored,
  min_is_fitness, engine_kw, mc_sims, min_pos_fold_share)` -> `dict(folds, summary, returns)`.
- `get_swap_pcts(info, ref_price, default=0.0002, allow_credit=False)` -> `(long, short)` overnight cost per direction from
  broker metadata (positive = you pay; a swap that would pay you is set to 0 unless `allow_credit=True`).
  `get_swap_pct(info, ref_price, default)` is the older single worst-case value.
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
   only see values that were published before that bar closed. `exit_at_close` comes from timestamps only.
2. **Optimise on in-sample data only; test once.** Never tune anything after seeing OOS or walk-forward results (that
   includes `ENGINE_KW`, sizing and the `qt.*_RANGE` settings). Do not add a second round of "improvements" to the strategy
   based on results you were told about unless the user restarts the process.
3. **Use the toolbox for optimisation, stops, targets, sizing and metrics.** No custom Optuna objectives, no custom
   stop-loss / take-profit / breakeven / time-stop code, no custom position sizing, no custom Sharpe or drawdown functions.
   (The factory/closure pattern for Optuna objectives is already inside `fit_strategy`.)
4. **Position convention:** whole numbers. If the paper's logic is "enter on event, exit on another event", convert it to
   a held state yourself.
5. **Stops, targets and exits belong to the engine.** If the paper specifies a stop, target, breakeven rule, time exit,
   scale-out or session exit, use the matching engine feature (section 3 table), not a hand-coded flip of the position.
   Engine stops are in ATR multiples, so convert the paper's percentage or pip value to an ATR multiple for the asset class
   and show the arithmetic. Indicator-based stops (a line on the chart) go in `sl_price`. Exits that ARE the signal logic
   (an indicator crossover, a mean reached on a bar close) stay in `position`.
6. **Do not change the template pipeline, the engine, or the toolbox.** If something truly cannot work, say so (section 6).
7. **Keep it simple and runnable.** You cannot execute MT5 code. Code that is too clever is code the user cannot debug.
   Use only the engine features the paper needs; every extra searched range adds overfitting freedom.

---

# 6. WHAT THIS FRAMEWORK CANNOT DO (flag it; do not hack around it)

If the PDF needs any of the following, say so clearly in Part 1, then implement the closest honest approximation or stop and ask:
- **Multi-instrument strategies** (pairs, baskets, cross-sectional ranking, relative value). The pipeline is one symbol, one
  position series.
- **Hedging:** long and short positions open at the same time on one symbol. All layers share one direction; a flip closes all.
- **Different rules per layer:** every layer uses the same stop / target / breakeven / time-stop settings, and `sl_price`
  applies to all layers. Only the entry price and `size` differ per layer.
- **More than two targets:** one partial target (TP1, one fraction) and one final target. No TP3, no target as a price
  array (express "exit at the opposite band" as a `position` change, which executes at the next open).
- **Intrabar sequencing below the bar:** the engine sees only OHLC. Pending orders fill intrabar, but where the bar's path
  matters it assumes the worst (stop before target; no same-bar target after a limit fill). Rules that need the order of
  events inside a bar (opening-range breakouts on the first 15 minutes of an H1 bar, scalps inside one bar) need a lower
  timeframe.
- **Margin calls / broker stop-out levels.** Leverage is capped only when a layer opens, and the run stops only at zero
  equity. A strategy that would hit a broker margin call at 50% margin level is not stopped early. Martingale and grid
  results are therefore optimistic in the tail; say so.
- **Commission per lot** is not modelled separately. Fold it into the spread if the paper's broker charges one.
- **Historical swap and spread:** today's broker swap and today's spread are applied to all history.
- **Account-currency conversion:** P&L is computed in the symbol's quote currency as a fraction of equity (exact for
  percentages; a JPY- or index-quoted symbol's FX drift against the account currency is ignored).
- **Data MT5 bars do not contain:** tick or order-book data, volume profile, fundamentals, options, sentiment, alternative data.
  Offer a bar-based proxy only if it is honest, and label it as a proxy.
- **Session or news filters** need the bar timestamp (`df.index`, broker server time) and are fine. Say which timezone you assumed.

---

# 7. REQUIRED OUTPUT FORMAT

**Part 1: Strategy decomposition (Markdown)**
- **Core premise:** what inefficiency the paper claims, and what evidence it actually offers. State plainly whether the paper's
  results are in-sample, include costs, use a train/test split, or select assets after the fact. Do not repeat headline numbers
  as if they were validated.
- **Mathematical formulations:** exact formulas for every indicator and signal, with the paper's notation mapped to code.
- **Decision rules:** exact long / short / exit / stop conditions, as a state machine. Name the order type of every entry
  (market at next open, buy/sell limit, buy/sell stop) and every exit (signal at next open, stop, target, time, session close).
- **Hyperparameter space:** a table of every Optuna parameter (strategy AND `qt.*_RANGE` settings): paper range, range you use,
  and why (timeframe rescaling, degenerate values, trade-count considerations).
- **Engine settings:** `STOP_MODES`, every `qt.*_RANGE`, and `ENGINE_KW`, each with the paper's reason.
- **What does not map:** every paper feature the framework cannot represent (section 6) and what you did instead.

**Part 2: Code**
- Output the full **Cell 2** (config + strategy section + screening pipeline), complete and ready to paste.
- For Cells 1, 3 and 4 write "unchanged from the template" unless you needed a change; if you did, output that cell in full
  and say exactly why.
- Set CONFIG values the paper implies (timeframe, symbols, `N_CANDLES`, `MAX_LEVERAGE`). If the paper uses daily bars, check
  that the `len(df) < 3000` guard in the pipeline and the walk-forward minimums (first training set at least 500 bars, each
  test chunk at least 100 bars) can be satisfied by the broker's history, and adjust or warn.

**Part 3: Implementation notes**
- Every decision where the paper was ambiguous, and the reading you chose.
- Every place you avoided lookahead that the paper's own method contains (for example fitting a model on the full sample).
- How each searched parameter maps to the author's findings, and what result would count as the paper NOT replicating.
- Expected trade frequency: roughly how many trades and active days per year the rules should produce, so the 30-active-day
  floor is realistic.
- Sizing: which `sizing` you used and why; for `"risk"`, which stop defines the risk.
- **MT5 hand-off spec:** a short, plain-language, unambiguous description of the final rules so it can be re-implemented
  as an MQL5 EA and compared trade-by-trade in the Strategy Tester (`oos_trade_logs[sym]` is the reference list):
  signal on bar close; order type and price of each entry (buy orders go at level + spread, because the engine triggers
  long levels on the bid chart); stop type and ATR multiple (ATR = in-sample median ATR(14), a constant); breakeven, TP1
  fraction, target and time-stop values; session-close time; lot formula for risk sizing
  (`lots = risk_pct x equity / (stop distance x value of 1 price unit per lot)`); max layers; what "flat" means.

---

# 8. SELF-CHECK BEFORE YOU ANSWER

Confirm each item silently; fix anything that fails.
- [ ] `VARIANTS`, `compute_strategy_signals`, `suggest_params`, `map_params` exist with the exact signatures.
- [ ] `STOP_MODES`, all six `qt.*_RANGE` settings and `ENGINE_KW` are set explicitly in the strategy section.
- [ ] `suggest_params` returns a dict containing `"variant"` and no stop / management params; `map_params` returns kwargs
      that match the function's arguments.
- [ ] The signal function returns a Series of whole numbers (or an orders DataFrame with a `position` column and only the
      allowed optional columns), no NaN in `position`, indexed like `df`, and works on a 300-bar slice.
- [ ] `entry_price` / `sl_price` are `NaN` where unused, `sl_price` sits on the correct side (below price for longs).
- [ ] `exit_at_close` is built from timestamps only.
- [ ] `sizing="risk"` always has a stop (`STOP_MODES` without `"none"`, or an `sl_price` column).
- [ ] No `shift(-n)`, no centred windows, no whole-sample statistics, no unseeded randomness, no mutation of the input `df`.
- [ ] No custom optimiser, stop-loss, target, sizing or metrics code; no invented functions; no edits to the engine or toolbox.
- [ ] Parameter ranges are justified for the timeframe and cannot produce `oversold >= overbought`-style contradictions.
- [ ] Anything the paper needs that the framework cannot do is listed in Part 1.

---

# 9. HOW I WILL READ THE RESULTS (so your notes help)

- `leaderboard` columns: `is_fitness`, `oos_cagr`, `oos_sharpe`, `oos_active_days`, `oos_trades` (real trade count),
  `stress_worst_cagr`, `survives_stress`, `bh_sharpe` (buy-and-hold on the same slice), `stop` (the winning stop and
  management settings). `oos_trade_logs[sym]` has every OOS exit with its reason.
- A strategy is only a candidate if: IS fitness > 0, OOS Sharpe > 0, `survives_stress` is True, and (in Cell 3)
  `wf_pass` is True with `pct_folds_positive >= 0.8`. A high OOS Sharpe over a few dozen trades means little.
- Scanning many symbols guarantees a few lucky passes. I treat every pass as a candidate to replicate in the MT5 Strategy
  Tester and forward-test on demo, not as proof.
- If the strategy fails everywhere, that is a valid and useful result. Do not tune the strategy to rescue it.
````
