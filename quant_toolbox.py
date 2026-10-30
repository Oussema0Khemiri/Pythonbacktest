import numpy as np
import pandas as pd
import yfinance as yf
import requests
import optuna
from trading_engine import (apply_sl_tp_and_costs, calculate_ai_metrics,
                            cost_stress as _engine_cost_stress)
# =====================================================================
# MODULE 1: ADVANCED TECHNICALS & MOMENTUM
# =====================================================================
def add_rsi(df, column='close', period=14):
    """Calculates the Relative Strength Index (Momentum / Overbought/Oversold)."""
    delta = df[column].diff()
    # Wilder's EMA (industry standard RSI, matches TradingView / MT5 built-in)
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = gain / loss
    df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df, column='close', fast=12, slow=26, signal=9):
    """Calculates MACD (Trend following & Momentum)."""
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    df['MACD_line'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD_line'].ewm(span=signal, adjust=False).mean()
    df['MACD_histogram'] = df['MACD_line'] - df['MACD_signal']
    return df

def add_atr(df, period=14):
    """
    Average True Range (ATR). 
    CRITICAL tool: Measures exact market volatility. Use this for Stop Losses!
    """
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df[f'ATR_{period}'] = true_range.rolling(period).mean()
    return df

# =====================================================================
# MODULE 2: SEASONALITY & TIME FEATURES (Crucial for Machine Learning)
# =====================================================================
def add_time_features(df):
    """
    Extracts time cycles. 
    e.g., The AI might learn that buying EURUSD is profitable at 8 AM London time, 
    but terrible at 4 PM on Fridays.
    """
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek # 0=Monday, 4=Friday
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    return df

# =====================================================================
# MODULE 3: MACRO REGIME & EXTERNAL DATA
# =====================================================================
def fetch_vix_regime(df_target):
    """
    Pulls the CBOE Volatility Index (VIX) from Yahoo Finance.
    If the VIX is > 30, the stock market is crashing (Panic mode).
    This allows your Forex bot to halt trading during global panics.
    """
    start_date = df_target.index.min().strftime('%Y-%m-%d')
    end_date = df_target.index.max().strftime('%Y-%m-%d')
    
    # Download VIX data
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
    
    # Handle both old and new yfinance column formats (MultiIndex vs flat)
    vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.squeeze()
    
    # Strip timezone if present for clean alignment
    if hasattr(vix_close.index, 'tz') and vix_close.index.tz is not None:
        vix_close = vix_close.tz_localize(None)
    
    # Build a complete daily series (forward-fill weekends/holidays)
    all_dates = pd.date_range(df_target.index.min().normalize(),
                               df_target.index.max().normalize(), freq='D')
    vix_filled = vix_close.reindex(all_dates, method='ffill')
    
    # Map each hourly candle to its corresponding date's VIX value
    # .values prevents index alignment issues between daily VIX and hourly forex
    df_target['VIX'] = vix_filled.reindex(df_target.index.normalize()).values
    return df_target

# =====================================================================
# MODULE 4: NEWS & FUNDAMENTALS (API Template)
# =====================================================================
def fetch_news_sentiment(api_key, symbol, start_date, end_date):
    """
    TEMPLATE: Connects to a News API (like Finnhub, AlphaVantage, or NewsAPI).
    In the future, you will use this to get a Sentiment Score (-1 to 1) 
    based on news headlines to block your bot from trading before Powell speaks.
    """
    # Example endpoint (you will replace this with your chosen provider later)
    url = f"https://finnhub.io/api/v1/news?category=forex&token={api_key}"
    # response = requests.get(url).json()
    
    # Mock return for architecture purposes
    print(f"Would fetch news for {symbol} here.")
    return pd.DataFrame() 

# =====================================================================
# MODULE 5: RISK MANAGEMENT (Smart Position Sizing)
# =====================================================================
def calculate_dynamic_position_size(df, account_balance=10000, risk_pct=0.01, atr_col='ATR_14', contract_size=100000):
    """
    Instead of trading 1 lot every time, this calculates exact lot sizes.
    If volatility (ATR) is HIGH, it buys LESS to protect your account.
    If volatility (ATR) is LOW, it buys MORE to maximize profit.
    
    contract_size should be pulled from MT5: mt5.symbol_info(sym).trade_contract_size
    Examples: Forex = 100000, Gold = 100, Indices = 1

    For BACKTESTS use the engine's sizing="risk" instead (it sizes off the real stop distance and the
    compounding equity); this helper only gives indicative live lot sizes.
    """
    # Example: Risking 1% of $10,000 = $100 per trade.
    risk_amount = account_balance * risk_pct
    
    # Calculate position size based on ATR (assuming ATR is the Stop Loss distance)
    df['dynamic_position_size'] = risk_amount / (df[atr_col] * contract_size)
    return df

# =====================================================================
# MODULE 6: STOP-LOSS SEARCH + GENERIC STRATEGY OPTIMISER (strategy-agnostic)
# =====================================================================
# Stop distances are MULTIPLES OF ATR (in-sample median ATR(14) of the asset), so one range works
# for natural gas, EURUSD, indices, etc. 1.0 ATR ~ one typical bar's range.
STOP_MODES  = ["none", "fixed", "trail"]     # restrict per run, e.g. ["trail"] or ["none", "fixed"]
FIXED_RANGE = (1.0, 20.0)                    # fixed stop distance, in ATR multiples
TRAIL_RANGE = (1.0, 20.0)                    # trailing stop distance, in ATR multiples
# Optional trade-management searches. None = OFF (not searched, engine default). Set a (low, high)
# tuple to search it together with the stop, e.g. qt.BE_RANGE = (0.5, 3.0). low == high fixes it.
BE_RANGE        = None                       # move stop to breakeven after this profit, in ATR multiples
TP_RANGE        = None                       # final take-profit distance, in ATR multiples
TP1_RANGE       = None                       # partial take-profit (scale-out) distance, in ATR multiples
TIME_STOP_RANGE = None                       # time stop: close a layer after this many bars (ints)

def get_ref_atr(train_df, period=14):
    """IN-SAMPLE median ATR in price units. Compute once on train_df and reuse for IS and OOS."""
    atr = add_atr(train_df.copy(), period)[f'ATR_{period}']
    return float(atr.median())

_MGMT_KEYS = ("be_mult", "tp_mult", "tp1_mult", "max_bars")

def _suggest_range(trial, name, rng, integer=False):
    lo, hi = rng
    if lo == hi:
        return int(lo) if integer else float(lo)
    if integer:
        return trial.suggest_int(name, int(lo), int(hi), log=True)
    return trial.suggest_float(name, lo, hi, log=True)

def suggest_stop_params(trial, modes=STOP_MODES):
    """
    Returns {'mode', 'mult', 'be_mult', 'tp_mult', 'tp1_mult', 'max_bars'}. mult / *_mult are ATR
    multiples (0 = off), max_bars is bars (0 = off). The management keys are only searched when their
    module-level *_RANGE is set (read at call time, so qt.BE_RANGE = (...) in the notebook works).
    """
    mode = trial.suggest_categorical("stop_mode", modes)
    mult = 0.0
    if mode == "fixed":
        mult = trial.suggest_float("sl_mult", *FIXED_RANGE, log=True)
    elif mode == "trail":
        mult = trial.suggest_float("trail_mult", *TRAIL_RANGE, log=True)
    stop = {"mode": mode, "mult": mult, "be_mult": 0.0, "tp_mult": 0.0, "tp1_mult": 0.0, "max_bars": 0}
    if BE_RANGE is not None:
        stop["be_mult"] = _suggest_range(trial, "be_mult", BE_RANGE)
    if TP_RANGE is not None:
        stop["tp_mult"] = _suggest_range(trial, "tp_mult", TP_RANGE)
    if TP1_RANGE is not None:
        stop["tp1_mult"] = _suggest_range(trial, "tp1_mult", TP1_RANGE)
    if TIME_STOP_RANGE is not None:
        stop["max_bars"] = _suggest_range(trial, "max_bars", TIME_STOP_RANGE, integer=True)
    return stop

def stop_from_params(best):
    """
    Rebuild the stop dict from study.best_params (flat dict). A range fixed with low == high is not
    an Optuna param, so it is read back from the current *_RANGE setting.
    """
    mode = best["stop_mode"]
    if mode == "fixed":
        mult = best.get("sl_mult", 0.0)
    elif mode == "trail":
        mult = best.get("trail_mult", 0.0)
    else:
        mult = 0.0
    stop = {"mode": mode, "mult": mult}
    for key, rng in zip(_MGMT_KEYS, (BE_RANGE, TP_RANGE, TP1_RANGE, TIME_STOP_RANGE)):
        default = (rng[0] if rng is not None and rng[0] == rng[1] else 0)
        stop[key] = best.get(key, default)
    stop["max_bars"] = int(stop["max_bars"])
    return stop

def stop_to_points(stop, ref_atr, point_size):
    """
    Stop dict -> engine kwargs. sl_points / trail_points are always returned; the management kwargs
    (tp_points, be_trigger_points, tp1_points, max_bars) only when switched on, so the same setting
    can instead be fixed through engine_kw (giving it both ways raises a TypeError).
    """
    pts = stop["mult"] * ref_atr / point_size if stop["mult"] > 0 else 0.0
    kw = {"sl_points": pts if stop["mode"] == "fixed" else 0,
          "trail_points": pts if stop["mode"] == "trail" else 0}
    to_pts = lambda m: m * ref_atr / point_size
    if stop.get("tp_mult", 0) > 0:  kw["tp_points"] = to_pts(stop["tp_mult"])
    if stop.get("be_mult", 0) > 0:  kw["be_trigger_points"] = to_pts(stop["be_mult"])
    if stop.get("tp1_mult", 0) > 0: kw["tp1_points"] = to_pts(stop["tp1_mult"])
    if stop.get("max_bars", 0) > 0: kw["max_bars"] = int(stop["max_bars"])
    return kw

def run_backtest(df, pos, point_size, spread_pts, ref_atr, stop, **engine_kw):
    """
    Every mode goes through apply_sl_tp_and_costs, so none / fixed / trail share identical
    entry, spread and gap conventions and can be compared fairly.
    ref_atr must be the IN-SAMPLE median ATR (same value for IS and OOS) -> no leakage.
    engine_kw passes any other engine setting straight through (slippage_points, swap, sizing,
    risk_pct, max_leverage, max_layers, tp1_fraction, ..., return_trades=True for a trade log).
    """
    return apply_sl_tp_and_costs(df, pos, point_size, spread_pts,
                                 **stop_to_points(stop, ref_atr, point_size), **engine_kw)

def fitness_from_metrics(metrics, min_active_days=30, graded=True):
    """
    Anti-overfit fitness. Viable strategies score cagr - 2 x |MC worst-5% drawdown|.
    Unviable ones get graded scores far below any viable one (so TPE keeps a gradient), or a
    flat -9999 when graded=False.
    """
    if metrics is None:
        return -9999.0
    if metrics["n_trades"] < min_active_days:
        return -100.0 + metrics["n_trades"] / max(min_active_days, 1) if graded else -9999.0
    if metrics["cagr"] <= 0:
        return -10.0 + max(metrics["cagr"], -1.0) if graded else -9999.0
    return metrics["cagr"] - abs(metrics["mc_worst_5%_drawdown"]) * 2.0

def _make_objective(train_data, pt_size, spread, signal_fn, space_fn, map_fn, group, stop_modes,
                    ref_atr, min_active_days, graded, mc_sims, engine_kw):
    """Factory: binds loop-scoped data so objective() never touches outer variables."""
    def objective(trial):
        try:
            p = space_fn(trial, [group])
            stop = suggest_stop_params(trial, stop_modes)
            pos = signal_fn(train_data, **map_fn(p))
            net = run_backtest(train_data, pos, pt_size, spread, ref_atr, stop, **engine_kw)
            metrics = calculate_ai_metrics(net, mc_simulations=mc_sims)
        except Exception as e:
            trial.set_user_attr("error", repr(e))
            return -9999.0
        return fitness_from_metrics(metrics, min_active_days, graded)
    return objective

def fit_strategy(train_df, point_size, spread_pts, signal_fn, space_fn, map_fn, groups,
                 stop_modes=STOP_MODES, n_trials=60, seed=42, min_active_days=30, graded=True,
                 mc_sims=100, engine_kw=None, verbose=True):
    """
    Optimise one strategy on IN-SAMPLE data only: one Optuna study per group (e.g. per variant),
    keep the best in-sample winner. Strategy-specific pieces are passed in:
      signal_fn(df, **map_fn(params)) -> position Series (+1/-1/0)
      space_fn(trial, [group])        -> flat params dict
    Returns dict(params, stop, fitness, ref_atr, group_scores) or None if nothing could run.
    """
    engine_kw = engine_kw or {}
    if verbose and engine_kw.get("sizing") == "risk" and "none" in stop_modes:
        print("  note: sizing='risk' needs a stop; 'none' trials only run if the strategy supplies sl_price")
    ref_atr = get_ref_atr(train_df)
    if not np.isfinite(ref_atr) or ref_atr <= 0:
        if verbose: print("  skip: invalid ATR")
        return None
    best_study, scores = None, {}
    for g in groups:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(_make_objective(train_df, point_size, spread_pts, signal_fn, space_fn, map_fn,
                                       g, stop_modes, ref_atr, min_active_days, graded, mc_sims, engine_kw),
                       n_trials=n_trials, show_progress_bar=False)
        scores[g] = study.best_value
        errs = [t.user_attrs["error"] for t in study.trials if "error" in t.user_attrs]
        if errs and verbose and study.best_value <= -9999.0:
            print(f"  [{g}] every trial raised: {errs[0]}")
        if best_study is None or study.best_value > best_study.best_value:
            best_study = study
    if verbose:
        print("  IS fitness by group:", {k: round(v, 4) for k, v in scores.items()})
    best = best_study.best_params
    return {"params": best, "stop": stop_from_params(best), "fitness": best_study.best_value,
            "ref_atr": ref_atr, "group_scores": scores}

# =====================================================================
# MODULE 7: COST STRESS (same positions + stops, harsher costs)
# =====================================================================
def get_swap_pcts(info, ref_price, default=0.0002, allow_credit=False):
    """
    Per-direction overnight swap as fractions of price per night: (long, short). Positive = you pay.
    Same conversion rules as get_swap_pct. allow_credit=False (default, conservative) turns a swap
    that would PAY you into 0; True keeps it as a negative cost (today's rate applied to all history).
    Pass them to the engine as swap_long_pct_per_night / swap_short_pct_per_night.
    """
    mode = getattr(info, "swap_mode", None)
    if mode == 0:
        return 0.0, 0.0
    if mode == 1:
        conv = float(info.point) / float(ref_price)
        long_cost, short_cost = -float(info.swap_long) * conv, -float(info.swap_short) * conv
        if not allow_credit:
            long_cost, short_cost = max(0.0, long_cost), max(0.0, short_cost)
        return long_cost, short_cost
    return default, default

def get_swap_pct(info, ref_price, default=0.0002):
    """
    Worst-case overnight swap as a fraction of price per night, from MT5 symbol_info.
    Only the 'points' swap mode (swap_mode == 1) can be converted reliably; if swaps are disabled
    it returns 0.0; for other modes (money / interest) it returns `default`. Always sanity-check
    against your broker's contract specification.
    """
    mode = getattr(info, "swap_mode", None)
    if mode == 0:
        return 0.0
    if mode == 1:
        worst = min(float(info.swap_long), float(info.swap_short))     # negative = you pay
        return max(0.0, -worst) * float(info.point) / float(ref_price)
    return default

def stress_test(df, pos, point_size, spread_pts, ref_atr, stop, swap_pct_per_night=0.0002,
                scenarios=None, mc_sims=100, swap_long_pct_per_night=None, swap_short_pct_per_night=None,
                engine_kw=None):
    """
    Re-run the SAME positions and stop under harsher costs (spread x1.5/x2/x3, slippage, overnight
    swap, and a combined 'harsh' case). Returns (table, summary). summary['survives_stress'] is True
    only if CAGR stays positive in EVERY scenario.
    Directional swaps (from get_swap_pcts) replace swap_pct_per_night when given. engine_kw must be the
    same non-cost engine settings used for the backtest (sizing, layers, ...), without swap/slippage.
    """
    table = _engine_cost_stress(df, pos, point_size, spread_pts,
                                swap_pct_per_night=swap_pct_per_night, scenarios=scenarios,
                                mc_simulations=mc_sims, swap_long_pct_per_night=swap_long_pct_per_night,
                                swap_short_pct_per_night=swap_short_pct_per_night,
                                **stop_to_points(stop, ref_atr, point_size), **(engine_kw or {}))
    summary = {"stress_worst_cagr": float(table["cagr"].min()),
               "stress_cagr_harsh": float(table["cagr"].iloc[-1]),
               "survives_stress": bool(table["survives"].all())}
    return table, summary

# =====================================================================
# MODULE 8: WALK-FORWARD VALIDATION (re-optimise every fold, test on the next unseen chunk)
# =====================================================================
def walk_forward(df, fit_fn, signal_fn, map_fn, point_size, spread_pts, n_folds=5, init_train_frac=0.5,
                 anchored=True, min_is_fitness=0.0, engine_kw=None, mc_sims=100, min_pos_fold_share=0.6):
    """
    The first `init_train_frac` of the data is the first training set; the remainder is cut into
    `n_folds` consecutive test chunks. For each fold the strategy is RE-OPTIMISED on data before the
    chunk only (anchored = expanding window, else a rolling window of the initial size), then traded
    blind on the chunk with that fold's parameters and ATR reference. Folds whose in-sample fitness is
    <= min_is_fitness trade nothing (flat) - the honest outcome when no viable strategy exists.

    fit_fn(train_df) -> dict from fit_strategy (or None); signal_fn / map_fn as in fit_strategy.
    Returns dict(folds=DataFrame, summary=dict, returns=stitched OOS return Series).
    """
    engine_kw = engine_kw or {}
    n = len(df)
    init = int(n * init_train_frac)
    fold_len = (n - init) // n_folds
    if init < 500 or fold_len < 100:
        raise ValueError(f"Not enough data for {n_folds} folds: {n} bars")

    rows, nets = [], []
    for f in range(n_folds):
        s = init + f * fold_len
        e = n if f == n_folds - 1 else s + fold_len
        a = 0 if anchored else s - init
        train, test = df.iloc[a:s], df.iloc[s:e]

        res = fit_fn(train)
        viable = res is not None and res["fitness"] > min_is_fitness
        if viable:
            sig = signal_fn(df.iloc[a:e], **map_fn(res["params"]))   # causal: nothing after bar e is seen
            net = run_backtest(test, sig.loc[test.index], point_size, spread_pts,
                               res["ref_atr"], res["stop"], **engine_kw)
        else:
            net = pd.Series(0.0, index=test.index)
        nets.append(net)

        m = calculate_ai_metrics(net, mc_simulations=mc_sims) if viable else None
        active = int(((1 + net).resample("D").prod() - 1).ne(0).sum())
        rows.append({
            "fold": f + 1, "train_bars": len(train), "test_start": test.index[0], "test_end": test.index[-1],
            "viable": viable,
            "variant": res["params"].get("variant") if viable else None,
            "stop_mode": res["stop"]["mode"] if viable else None,
            "stop_mult": res["stop"]["mult"] if viable else np.nan,
            "stop": res["stop"] if viable else None,
            "is_fitness": res["fitness"] if res is not None else np.nan,
            "oos_return": float((1 + net).prod() - 1),
            "oos_sharpe": m["sharpe"] if m else np.nan,
            "oos_max_dd": m["max_drawdown"] if m else np.nan,
            "oos_active_days": active,
            "params": res["params"] if viable else None,
        })

    folds = pd.DataFrame(rows)
    stitched = pd.concat(nets)
    allm = calculate_ai_metrics(stitched, mc_simulations=mc_sims)
    n_viable = int(folds["viable"].sum())
    n_pos = int((folds["oos_return"] > 0).sum())
    variants = folds.loc[folds["viable"], "variant"]
    summary = {
        "wf_total_return": float((1 + stitched).prod() - 1),
        "wf_cagr": allm["cagr"] if allm else np.nan,
        "wf_sharpe": allm["sharpe"] if allm else np.nan,
        "wf_max_dd": allm["max_drawdown"] if allm else np.nan,
        "folds_viable": n_viable,
        "folds_positive": n_pos,
        "pct_folds_positive": n_pos / n_folds,
        "median_fold_return": float(folds["oos_return"].median()),
        "worst_fold_return": float(folds["oos_return"].min()),
        "variant_stability": float(variants.value_counts(normalize=True).iloc[0]) if len(variants) else np.nan,
    }
    summary["wf_pass"] = bool(summary["wf_total_return"] > 0
                              and summary["pct_folds_positive"] >= min_pos_fold_share
                              and n_viable >= int(np.ceil(n_folds / 2)))
    return {"folds": folds, "summary": summary, "returns": stitched}