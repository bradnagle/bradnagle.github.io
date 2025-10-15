# /.github/scripts/fetch_audusd.py
import json, os, time, math, datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_SM = True
except Exception:
    HAS_SM = False

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
    for ts, row in df.iterrows():
        o = _safe_float(row.get("Open"))
        h = _safe_float(row.get("High"))
        l = _safe_float(row.get("Low"))
        c = _safe_float(row.get("Close"))
        v = _safe_float(row.get("Volume")) or 0.0
        if c is None:
            continue
        epoch_ms = int(pd.Timestamp(ts).to_pydatetime().timestamp() * 1000)
        candles.append({"ts": epoch_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})
    sess_open = candles[0]["open"] if candles else None
    sess_close = candles[-1]["close"] if candles else None
    sess_high = max([c["high"] for c in candles if c["high"] is not None], default=None)
    sess_low  = min([c["low"] for c in candles if c["low"] is not None], default=None)
    vol_sum = sum([c["volume"] for c in candles if c["volume"] is not None])
    return {"symbol": symbol, "interval": interval, "updated": int(time.time()*1000), "tz":"UTC",
            "meta":{"records":len(candles), "first_ts": candles[0]["ts"] if candles else None, "last_ts": candles[-1]["ts"] if candles else None,
                    "session_ohlc":{"open":sess_open,"high":sess_high,"low":sess_low,"close":sess_close},"volume_sum":vol_sum},
            "candles": candles}

def _invert_payload(payload, out_symbol):
    inv_candles = []
    sess = payload.get("meta", {}).get("session_ohlc", {}) or {}
    io, ih, il, ic = _invert_ohlc(sess.get("open"), sess.get("high"), sess.get("low"), sess.get("close"))
    for c in payload["candles"]:
        o, h, l, cl = c.get("open"), c.get("high"), c.get("low"), c.get("close")
        io_c, ih_c, il_c, ic_c = _invert_ohlc(o, h, l, cl)
        inv_candles.append({ "ts": c["ts"], "open": io_c, "high": ih_c, "low": il_c, "close": ic_c, "volume": c.get("volume", 0.0) })
    return {"symbol": out_symbol, "interval": payload.get("interval"), "updated": payload.get("updated"), "tz": payload.get("tz"),
            "meta":{"records":len(inv_candles), "first_ts": inv_candles[0]["ts"] if inv_candles else None, "last_ts": inv_candles[-1]["ts"] if inv_candles else None,
                    "session_ohlc":{"open":io,"high":ih,"low":il,"close":ic}, "volume_sum": sum(c.get("volume",0.0) for c in inv_candles)},
            "candles": inv_candles}

def _business_days(start_ts, periods):
    start_date = dt.datetime.utcfromtimestamp(start_ts/1000.0).date()
    dates = []
    d = start_date + dt.timedelta(days=1)
    while len(dates) < periods:
        if d.weekday() < 5:
            dates.append(int(dt.datetime(d.year, d.month, d.day).timestamp()*1000))
        d += dt.timedelta(days=1)
    return dates

def _forecast_daily(df_close: pd.Series, horizon=5):
    model_used = "linreg"; yhat = []; yhat_lo = []; yhat_hi = []
    try:
        if HAS_SM and len(df_close) > 30:
            model = ARIMA(df_close, order=(1,1,1))
            fit = model.fit()
            f = fit.get_forecast(steps=horizon)
            yhat = f.predicted_mean.tolist()
            ci = f.conf_int(alpha=0.05)
            yhat_lo = ci.iloc[:,0].tolist()
            yhat_hi = ci.iloc[:,1].tolist()
            model_used = "arima(1,1,1)"
        else:
            raise RuntimeError
    except Exception:
        x = np.arange(len(df_close))
        coeffs = np.polyfit(x, df_close.values, deg=1)
        trend = np.poly1d(coeffs)
        resid = df_close.values - trend(x)
        se = np.std(resid)
        future_x = np.arange(len(df_close), len(df_close)+horizon)
        yhat = trend(future_x).tolist()
        yhat_lo = (trend(future_x) - 1.96*se).tolist()
        yhat_hi = (trend(future_x) + 1.96*se).tolist()
        model_used = "linreg"
    last_ts = int(pd.Timestamp(df_close.index[-1]).to_pydatetime().timestamp()*1000)
    future_ts = _business_days(last_ts, horizon)
    return model_used, [{"ts": int(t), "yhat": float(y), "yhat_lower": float(lo), "yhat_upper": float(hi)} for t,y,lo,hi in zip(future_ts, yhat, yhat_lo, yhat_hi)]

def main():
    tkr = yf.Ticker("AUDUSD=X")
    df_15m = tkr.history(period="1d", interval="15m", auto_adjust=False)
    if df_15m is not None and not df_15m.empty:
        aud_15m = _build_payload(df_15m, "AUDUSD=X", "15m")
        usd_15m = _invert_payload(aud_15m, "USDAUD (inverted)")
        os.makedirs("data", exist_ok=True)
        with open("data/audusd_15m.json", "w") as f: json.dump(aud_15m, f)
        with open("data/usdaud_15m.json", "w") as f: json.dump(usd_15m, f)
        print(f"Wrote {len(aud_15m['candles'])} candles → data/audusd_15m.json")
        print(f"Wrote {len(usd_15m['candles'])} candles → data/usdaud_15m.json")
    else:
        print("WARN: No 15m data returned")

    df_1d = tkr.history(period="2y", interval="1d", auto_adjust=False)
    if df_1d is None or df_1d.empty:
        raise SystemExit("No daily data returned from Yahoo Finance (AUDUSD=X, 2y/1d).")
    aud_1d = _build_payload(df_1d, "AUDUSD=X", "1d")
    usd_1d = _invert_payload(aud_1d, "USDAUD (inverted)")
    with open("data/audusd_daily.json", "w") as f: json.dump(aud_1d, f)
    with open("data/usdaud_daily.json", "w") as f: json.dump(usd_1d, f)
    print(f"Wrote {len(aud_1d['candles'])} candles → data/audusd_daily.json")
    print(f"Wrote {len(usd_1d['candles'])} candles → data/usdaud_daily.json")

    closes = df_1d["Close"].dropna()
    model, points = _forecast_daily(closes, horizon=5)
    with open("data/audusd_forecast.json","w") as f: json.dump({"symbol":"AUDUSD=X","generated":int(time.time()*1000),"horizon_days":5,"model":model,"points":points}, f)

    inv_points = []
    for p in points:
        y, lo, hi = p["yhat"], p["yhat_lower"], p["yhat_upper"]
        inv_points.append({"ts": p["ts"], "yhat": (1/y if y else None), "yhat_lower": (1/hi if hi else None), "yhat_upper": (1/lo if lo else None)})
    with open("data/usdaud_forecast.json","w") as f: json.dump({"symbol":"USDAUD (inverted)","generated":int(time.time()*1000),"horizon_days":5,"model":model,"points":inv_points}, f)
    print("Wrote forecasts → data/audusd_forecast.json and data/usdaud_forecast.json")

if __name__ == "__main__":
    main()
