# TR_mappings.py -- Data dictionaries and configuration constants for Borsa Istanbul FIX data.
# Moved from TR_Borsa_Parser.py to keep parsing logic clean.

# spec Side(54) valid values are 1=Buy, 2=Sell only in market data feed
SIDE_MAP = {
    "1": "BUY",
    "2": "SELL",
}

# spec ExecType(150) valid values are 4=Cancelled and F=Trade only
EXECTYPE_MAP = {
    "4": "CANCELLED",
    "F": "TRADE",
}

# separate MDStreamID maps for Snapshot (35=W) vs Tick (UA201/UA202)
SNAPSHOT_SEGMENT_MAP = {
    "010": "REG",
    "020": "BILLS_BOND",
    "030": "STOCK_DEL_FUT",
    "040": "STOCK_CS_FUT",
    "050": "STOCK_DEL_OPT",
    "060": "INDEX_OPT",
    "070": "STOCK_IDX_FUT",
    "080": "ODD_LOT",
    "100": "EQ_SQUARE_UP",
    "120": "FUT_SQUARE_UP",
    "900": "INDEX",
}

TICK_SEGMENT_MAP = {
    "011": "REG",
    "031": "STOCK_DEL_FUT",
    "041": "STOCK_CS_FUT",
    "051": "STOCK_DEL_OPT",
    "061": "INDEX_OPT",
    "071": "STOCK_IDX_FUT",
    "081": "ODD_LOT",
    "091": "NDM",
}

# Fix 11: MDEntryType dictionary corrected against spec text verbatim
MDENTRY_MAP = {
    "0":  "BID",
    "1":  "OFFER",
    "2":  "LAST_TRADE",
    "3":  "INDEX_VALUE",
    "4":  "OPENING_PRICE",
    "5":  "CLOSING_PRICE",          # Fix 11
    "6":  "SETTLEMENT_PRICE",       # Fix 11
    "7":  "SESSION_HIGH",
    "8":  "SESSION_LOW",
    "x1": "NET_CHANGE_1",           # latest px minus prev close
    "x2": "NET_CHANGE_2",           # latest px minus last latest px
    "x3": "AGG_BID",                # VWAP px / total qty within auction range
    "x4": "AGG_OFFER",              # VWAP px / total qty within auction range
    "x5": "PE_RATIO_1",             # Fix 11: reserved, not released
    "x6": "PE_RATIO_2",             # Fix 11: reserved, not released
    "x7": "FUND_PREV_NAV",          # Fix 11: fund prev NAV incl. ETF
    "x8": "ETF_INAV",               # Fix 11: ETF intraday NAV
    "xa": "PREV_CLOSE_INDEX",       # Fix 11: corrected (was INDEX_OPEN)
    "xb": "OPEN_INDEX",             # Fix 11: corrected
    "xc": "HIGH_INDEX",             # Fix 11: corrected (was INDEX_LOW_?)
    "xd": "LOW_INDEX",              # Fix 11: corrected
    "xe": "UPPER_CIRCUIT_BREAKER",
    "xf": "LOWER_CIRCUIT_BREAKER",
    "xg": "OPEN_INTEREST",          # position qty of derivative contract
    "xl": "CLOSE_INDEX",
}

# Fix 12: sentinel value for "no limit" on xe (up circuit breaker) ONLY.
# xf (down circuit breaker) has no universal sentinel — its no-limit value
# equals the market's minimum price tick (e.g. 0.01 for Regular Market),
# which varies by market/segment and must not be hard-coded/nulled blindly.
XE_NO_LIMIT_SENTINEL = 999999999.9999

# Fix 13: TradingPhaseCode(8538) 0th-digit phase code -> human label
PHASE_MAP = {
    "S": "STARTING",
    "O": "OPEN_CALL_AUCTION",
    "T": "CONTINUOUS_AUCTION",
    "B": "TRADING_BREAK",
    "N": "NORMAL_CALL_AUCTION_PM",
    "H": "TEMPORARY_SUSPENSION",
    "V": "NORMAL_CALL_AUCTION_RESUME",
    "C": "CLOSE_CALL_AUCTION",
    "A": "AFTER_HOUR_TRADING",
    "E": "MARKET_CLOSED",
}

# Fix 13: break-reason 2nd digit (only meaningful when phase == B)
BREAK_REASON_MAP = {
    "1": "AFTER_PRE_OPEN",
    "2": "FRIDAY_LUNCH_BREAK",
    "3": "AFTER_PRE_OPEN_PM_FRIDAY",
    "4": "BEFORE_POST_CLOSE",
}
