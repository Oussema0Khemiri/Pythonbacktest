


### 1. Advanced Entry Mechanics
*   **Pending Orders (Limit and Stop Entries):**
    *   **What:** The ability to pass a specific target entry price to the engine, rather than just entering at the next open.
    *   **Why:** Captures Breakout strategies (buy when price breaks a high *intra-bar*) and Pullback strategies (buy when price drops to a moving average *intra-bar*). Right now, waiting for the bar to close misses the exact moment of the edge.
*   **Pyramiding / Scaling In:**
    *   **What:** Allowing the engine to hold multiple "layers" or tranches of a position simultaneously, rather than being strictly limited to a state of `1` or `-1`.
    *   **Why:** Required to test Grid strategies, Martingale systems, or aggressive trend-following strategies that "add to winners" as the trend continues.

### 2. Trade Management (Stops & Trailing)
*   **Move-to-Breakeven (BE) Logic:**
    *   **What:** A parameter that automatically moves the Stop Loss to the Entry Price once the trade reaches a certain profit threshold (e.g., 1 ATR).
    *   **Why:** This is the most common risk-management technique in discretionary and retail trading. Without it, your AI will reject perfectly good strategies simply because winning trades eventually reversed and hit the full stop loss.
*   **Dynamic / Path-Dependent Stop Losses:**
    *   **What:** Allowing the strategy to pass a continuous array of Stop Loss values per bar (e.g., the exact value of a Moving Average or a Donchian Channel line), rather than relying on the engine's built-in fixed/trailing ATR logic.
    *   **Why:** You encountered this exact issue with your C++ MT4 EA. If the stop-loss *is* the strategy (like a stepped trail based on an indicator), the engine needs to accept an external stop-loss array.
*   **Time Stops (Maximum Hold Duration):**
    *   **What:** A rule that forces an exit if a position has been open for `X` bars.
    *   **Why:** In quantitative finance, "opportunity cost" matters. If a momentum trade doesn't move immediately, the premise is wrong, and it should be cut before it hits the price stop.

### 3. Advanced Exits & Targets
*   **Partial Take Profits (Scaling Out):**
    *   **What:** The ability to close 50% of the position at Target 1, and let the remaining 50% run to Target 2 (or a trailing stop).
    *   **Why:** Very common in Prop-Firm trading and mean-reversion strategies to lock in guaranteed profit while preserving a "runner" for fat-tail outsized wins.
*   **Session-Close / Same-Bar Forced Exits:**
    *   **What:** The ability to force the engine to close a trade exactly at the *Close* of the current bar (e.g., Friday at 21:59 UTC), rather than the *Open* of the next bar.
    *   **Why:** Day-trading strategies on indices (like US30 or NAS100) must close before the session ends. Your current engine executes exits on the next bar's open, meaning a Friday exit signal actually executes on Monday's open, forcing you to eat the weekend gap risk.

### 4. Capital & Risk Modeling
*   **Dynamic Position Sizing / Fractional Compounding:**
    *   **What:** Integrating real lot sizing inside the engine’s return calculation based on the Stop Loss distance (e.g., risking exactly 1% of the compounding equity curve per trade).
    *   **Why:** Currently, your engine assumes 1 unit per trade. A trade with a 200-pip stop has massive weight on your equity curve compared to a trade with a 20-pip stop. Standardizing risk per trade is required to calculate accurate Sharpe ratios and Drawdowns.
*   **Margin / Leverage Caps:**
    *   **What:** A built-in check that prevents the system from taking a trade (or reduces its size) if the required margin exceeds the simulated account equity.
    *   **Why:** Prevents the AI from discovering "miracle" strategies that look highly profitable in percentages but would actually require 5000:1 leverage to execute in the real world.

### 5. Cost Realism
*   **Directional Asymmetry (Short Borrow Costs):**
    *   **What:** Applying different overnight costs depending on whether the trade is long or short.
    *   **Why:** In equities and some commodities, shorting costs significantly more than going long (borrow fees). Even in Forex, broker swap rates are highly asymmetrical (e.g., holding Long USDTRY pays you, holding Short USDTRY drains your account instantly). If you trade daily charts, this will destroy theoretical edges.







UPDATE STRATEGY_GENERATOR_PROMPT.md to be aligned with the new engine .