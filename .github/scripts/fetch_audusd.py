# /.github/scripts/fetch_audusd.py
import json, os, time
import pandas as pd
import yfinance as yf

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _invert_ohlc(o, h, l, c):
    def inv(v):
        return (1.0 / v) if (v is not None and v != 0) else None
    io, ih, il, ic = inv(o), inv(h), inv(l), inv(c)
    return io, (il if il is not None else None), (ih if ih is not None else None), ic

def _build_payload(df, symbol, interval):
    candles = []
    sess_open = sess_high = sess_low = sess_close = None
    vol_sum = 0.0
    for ts, row in df.iterrows():
        o = _safe_float(row.get("Open"))
        h = _safe_float(row.get("High"))
        l = _safe_float(row.get("Low"))
        c = _safe_float(row.get("Close"))
        v = _safe_float(row.get("Volume")) or 0.0
        if c is None:
            continue
        epoch_ms = int(ts.to_pydatetime().timestamp() * 1000)
        candles.append({"ts": epoch_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
        if sess_open is None and o is not None:
            sess_open = o
        if h is not None:
            sess_high = h if sess_high is None else max(sess_high, h)
        if l is not None:
            sess_low = l if sess_low is None else min(sess_low, l)
        if c is not None:
            sess_close = c
        vol_sum += v
    return {
        "symbol": symbol,
        "interval": interval,
        "updated": int(time.time() * 1000),
        "tz": "UTC",
        "meta": {
            "records": len(candles),
            "first_ts": candles[0]["ts"] if candles else None,
            "last_ts": candles[-1]["ts"] if candles else None,
            "session_ohlc": { "open": candles[0]["open"] if candles else None,
                               "high": max([c["high"] for c in candles if c["high"] is not None], default=None),
                               "low": min([c["low"] for c in candles if c["low"] is not None], default=None),
                               "close": candles[-1]["close"] if candles else None },
            "volume_sum": sum(c["volume"] for c in candles if c["volume"] is not None)
        },
        "candles": candles
    }

def _invert_payload(payload, out_symbol):
    inv_candles = []
    sess = payload.get("meta", {}).get("session_ohlc", {}) or {}
    io, ih, il, ic = _invert_ohlc(sess.get("open"), sess.get("high"), sess.get("low"), sess.get("close"))
    for c in payload["candles"]:
        o, h, l, cl = c.get("open"), c.get("high"), c.get("low"), c.get("close")
        io_c, ih_c, il_c, ic_c = _invert_ohlc(o, h, l, cl)
        inv_candles.append({ "ts": c["ts"], "open": io_c, "high": ih_c, "low": il_c, "close": ic_c, "volume": c.get("volume", 0.0) })
    return {
        "symbol": out_symbol,
        "interval": payload.get("interval"),
        "updated": payload.get("updated"),
        "tz": payload.get("tz"),
        "meta": {
            "records": len(inv_candles),
            "first_ts": inv_candles[0]["ts"] if inv_candles else None,
            "last_ts": inv_candles[-1]["ts"] if inv_candles else None,
            "session_ohlc": { "open": io, "high": ih, "low": il, "close": ic },
            "volume_sum": sum(c.get("volume", 0.0) for c in inv_candles)
        },
        "candles": inv_candles
    }

def main():
    # Intraday 15m (last 1 day)
    tkr = yf.Ticker("AUDUSD=X")
    df_15m = tkr.history(period="1d", interval="15m", auto_adjust=False)
    if df_15m is None or df_15m.empty:
        print("WARN: No 15m data returned from Yahoo Finance (AUDUSD=X, 1d/15m).")
        df_15m = None
    else:
        aud_15m = _build_payload(df_15m, "AUDUSD=X", "15m")
        usd_15m = _invert_payload(aud_15m, "USDAUD (inverted)")
        os.makedirs("data", exist_ok=True)
        with open("data/audusd_15m.json", "w") as f: json.dump(aud_15m, f)
        with open("data/usdaud_15m.json", "w") as f: json.dump(usd_15m, f)
        print(f"Wrote {len(aud_15m['candles'])} candles → data/audusd_15m.json")
        print(f"Wrote {len(usd_15m['candles'])} candles → data/usdaud_15m.json")

    # Daily 1d (last ~2y, covers 1Y range solidly)
    df_1d = tkr.history(period="2y", interval="1d", auto_adjust=False)
    if df_1d is None or df_1d.empty:
        raise SystemExit("No daily data returned from Yahoo Finance (AUDUSD=X, 2y/1d).")
    aud_1d = _build_payload(df_1d, "AUDUSD=X", "1d")
    usd_1d = _invert_payload(aud_1d, "USDAUD (inverted)")

    with open("data/audusd_daily.json", "w") as f: json.dump(aud_1d, f)
    with open("data/usdaud_daily.json", "w") as f: json.dump(usd_1d, f)
    print(f"Wrote {len(aud_1d['candles'])} candles → data/audusd_daily.json")
    print(f"Wrote {len(usd_1d['candles'])} candles → data/usdaud_daily.json")

if __name__ == "__main__":
    main()
