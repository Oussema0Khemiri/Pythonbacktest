import MetaTrader5 as mt5
import pandas as pd

def get_market_watch_symbols():
    """
    Scans MT5 and returns a list of all symbol names 
    currently visible in your MT5 Market Watch window.
    """
    if not mt5.initialize():
        print("MT5 initialization failed:", mt5.last_error())
        return []
    
    # Fetch all symbols from MT5
    symbols = mt5.symbols_get()
    mt5.shutdown()
    
    if symbols is None:
        print("No symbols found.")
        return []
    
    # Filter only for symbols where 'visible' is True (displayed in Market Watch)
    market_watch_list = [s.name for s in symbols if s.visible]
    
    print(f"✅ Found {len(market_watch_list)} instruments in MT5 Market Watch.")
    return market_watch_list

def get_symbol_details(symbols_list):
    """
    Returns a clean DataFrame showing spread, digits, pip value, and margin currency 
    for all symbols in your list. Great for inspecting broker conditions.
    """
    if not mt5.initialize():
        return None
    
    details = []
    for symbol in symbols_list:
        info = mt5.symbol_info(symbol)
        if info is not None:
            details.append({
                "Symbol": info.name,
                "Base Currency": info.currency_base,
                "Profit Currency": info.currency_profit,
                "Spread (points)": info.spread,
                "Digits": info.digits,
                "Trade Mode": "Enabled" if info.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL else "Restricted"
            })
            
    mt5.shutdown()
    return pd.DataFrame(details)

# =====================================================================
# CURATED PRESETS (Fallback if Market Watch is not customized)
# =====================================================================
FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]

FOREX_CROSSES = [
    "EURGBP", "EURJPY", "GBPJPY", "EURCAD", "AUDJPY", 
    "GBPAUD", "AUDCAD", "EURNZD", "GBPNZD", "NZDJPY"
]

METALS_INDICES = ["XAUUSD", "XAGUSD", "US30", "US500", "NAS100", "GER40"]