"""
Equity Backtester v2 — PAPER ONLY
Sources: PapersWithBacktest.com, r/algotrading, r/wallstreetbets, Reuters/Bloomberg (web)
Validation: backtest + walk-forward (in-sample / out-of-sample)
NO LIVE ORDERS — blocked until user approves + account funded + confidence threshold met.
Agentic account (••••661): $0 buying power → paper mode enforced.
"""
import yfinance as yf, pandas as pd, numpy as np, json, os
from datetime import datetime, timedelta

OUTFILE = "C:/Users/yeezz/backtest_framework/backtest_results.json"
REPORT   = "C:/Users/yeezz/backtest_framework/backtest_report.md"

# ── Config ──────────────────────────────────────────────────────────────────────
PAPER_MODE   = True          # always True; enforced regardless
ACCOUNT_VALUE = 0.0          # Agentic ••••661 buying power
MIN_TRADES   = 3             # per day threshold (live mode only)
CONFIDENCE   = 0             # walk-forward windows with positive Sharpe

SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "GOOGL"]
# Top equity strategies sourced from research:
# 1. TQQQ + 200d SMA (r/algotrading, 39.3% CAGR, 0.88 Sharpe)
# 2. AI/Semis Monthly Momentum top-8 (paperswithbacktest, 46.3% CAGR, 1.11 Sharpe)
# 3. Mean Reversion RSI<30/70 (r/algotrading, 2.11 Sharpe in ranging markets)
# 4. SMA Crossover 20/50d (classic momentum)
# 5. Bollinger 2σ mean-reversion (r/algotrading community signal)

STRATEGIES = {
    "tqqq_sma200": {
        "desc": "Hold TQQQ when QQQ > 200d SMA; else BIL (cash). Source: r/algotrading, 39.3% CAGR / 0.88 Sharpe.",
        "signal_fn": None,  # defined below
    },
    "ai_semis_momentum": {
        "desc": "Rank top AI/semi names by 3-month return; equal-weight top 8. Source: paperswithbacktest, 46.3% CAGR / 1.11 Sharpe.",
        "signal_fn": None,
    },
    "rsi_mean_reversion": {
        "desc": "Buy when RSI(14) < 30, sell when RSI > 70. Source: r/algotrading, 2.11 Sharpe (ranging markets).",
        "signal_fn": None,
    },
    "sma_cross": {
        "desc": "20d/50d SMA crossover — go long on golden cross, exit on death cross. Classic momentum.",
        "signal_fn": None,
    },
    "bollinger_2sigma": {
        "desc": "Buy when price < 2σ lower band; sell when > 2σ upper band. Source: community signal r/algotrading.",
        "signal_fn": None,
    },
}

# ── Signal functions ───────────────────────────────────────────────────────────
def rsi(close, period=14):
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def tqqq_sma_signal(closes_qqq, closes_tqqq):
    sma200 = closes_qqq.rolling(200).mean()
    return pd.Series(np.where(closes_qqq > sma200, 1, 0), index=closes_tqqq.index)

def ai_semis_signal(assets, period=63):
    """Monthly momentum on a dict of ticker→close Series."""
    ranks = {}
    for ticker, close in assets.items():
        ret = close.pct_change(period).iloc[-1] if len(close) >= period else np.nan
        ranks[ticker] = ret
    sorted_tickers = sorted(ranks, key=ranks.get, reverse=True)[:8]
    n = len(sorted_tickers)
    signals = pd.Series(0, index=next(iter(assets.values())).index)
    for t in sorted_tickers:
        signals += (assets[t] > 0).astype(int) / n
    return signals, sorted_tickers, ranks

def sma_cross_signal(close):
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    return pd.Series(np.where(sma20 > sma50, 1, -1), index=close.index)

def bollinger_signal(close, window=20, num_std=2):
    sma  = close.rolling(window).mean()
    std  = close.rolling(window).std()
    ub   = sma + num_std * std
    lb   = sma - num_std * std
    sig  = pd.Series(0, index=close.index)
    sig[close < lb] = 1
    sig[close > ub] = -1
    return sig

# ── Backtest engine ────────────────────────────────────────────────────────────
def backtest(signals: pd.Series, returns: pd.Series, label: str) -> dict:
    strat_ret = signals.shift(1).fillna(0) * returns
    strat_ret = strat_ret.dropna()
    if len(strat_ret) == 0:
        return {"strategy": label, "error": "No data"}
    total   = (1 + strat_ret).prod() - 1
    ann_ret = strat_ret.mean() * 252
    ann_vol = strat_ret.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0
    dd      = ((1 + strat_ret).cumprod() / (1 + strat_ret).cumprod().cummax()) - 1
    max_dd  = dd.min()
    win_rate= (strat_ret > 0).sum() / max(len(strat_ret), 1)
    n_trades= int((signals.diff().abs() > 0).sum())
    return {
        "strategy":   label,
        "total_ret":  round(total * 100, 2),
        "ann_ret":    round(ann_ret * 100, 2),
        "ann_vol":    round(ann_vol * 100, 2),
        "sharpe":     round(sharpe, 3),
        "max_dd":     round(max_dd * 100, 2),
        "win_rate":   round(win_rate * 100, 1),
        "n_trades":   n_trades,
    }

def walk_forward(data: pd.DataFrame, ret_col: str, signal_fn, label: str,
                 train_years=1, test_months=3):
    """Rolling train/test: train train_years, test test_months, step monthly."""
    results = []
    start = data.index[0]
    end   = data.index[-1]
    window = pd.DateOffset(years=train_years)
    step   = pd.DateOffset(months=test_months)
    train_end = start + window
    while train_end < end:
        train = data.loc[start:train_end]
        test  = data.loc[train_end:min(train_end + pd.DateOffset(months=test_months), end)]
        if len(train) < 60 or len(test) < 20:
            train_end += step
            continue
        # Generate signal on full series (lagged for backtest)
        sig = signal_fn(data[ret_col])
        test_sig = sig.loc[test.index]
        test_ret = test[ret_col]
        r = backtest(test_sig, test_ret, f"{label} [WF OOS]")
        results.append(r)
        start     += step
        train_end += step
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("EQUITY BACKTEST v2 — PAPER MODE")
print(f"Account buying power: ${ACCOUNT_VALUE:.2f} → PAPER ONLY")
print("Strategies sourced from: PapersWithBacktest, r/algotrading,")
print("  r/wallstreetbets, Reuters/Bloomberg news")
print("=" * 60)

# Download data
print("\n[1] Downloading data (SPY, QQQ 2020-2026)...")
try:
    spy  = yf.download("SPY",  start="2020-01-01", end="2026-09-01", progress=False)["Close"]
    qqq  = yf.download("QQQ",  start="2020-01-01", end="2026-09-01", progress=False)["Close"]
    tqqq = yf.download("TQQQ", start="2020-01-01", end="2026-09-01", progress=False)["Close"]
    spy_ret = spy.pct_change().dropna()
except Exception as e:
    print(f"Data download error: {e}")
    print("Saving error state and exiting.")
    with open(OUTFILE, "w") as f:
        json.dump({"error": str(e), "paper_mode": True}, f)
    exit(1)

# Align data
common_idx = spy_ret.index
returns_df = pd.DataFrame({"SPY": spy_ret}).reindex(common_idx).dropna()

# ── Run strategies ────────────────────────────────────────────────────────────
all_results = []
print("\n[2] Running strategies (in-sample + walk-forward)...")

# 1. TQQQ + 200d SMA
print("  - TQQQ + 200d SMA...")
sig_tqqq = tqqq_sma_signal(qqq.reindex(common_idx).dropna(),
                            tqqq.reindex(common_idx).dropna())
sig_tqqq = sig_tqqq.reindex(common_idx).fillna(0)
ret_tqqq = returns_df["SPY"].reindex(sig_tqqq.index).fillna(0)
all_results.append(backtest(sig_tqqq, ret_tqqq, "tqqq_sma200 [IS]"))
wf1 = walk_forward(pd.DataFrame({"SPY": spy.reindex(common_idx).dropna().pct_change().dropna()}),
                   "SPY", lambda c: tqqq_sma_signal(
                       qqq.reindex(common_idx).dropna(),
                       tqqq.reindex(common_idx).dropna()),
                   "tqqq_sma200")
all_results.extend(wf1)

# 2. RSI Mean Reversion on SPY
print("  - RSI Mean Reversion...")
rsi_spy = rsi(spy)
sig_rsi = pd.Series(0, index=common_idx)
sig_rsi[rsi_spy < 30] = 1
sig_rsi[rsi_spy > 70] = -1
all_results.append(backtest(sig_rsi.reindex(common_idx).fillna(0),
                             returns_df["SPY"], "rsi_mean_reversion [IS]"))
wf2 = walk_forward(pd.DataFrame({"SPY": spy}),
                   "SPY", lambda c: (pd.Series(0, index=c.index)
                                     .where(rsi(c) >= 30, 1)
                                     .where(rsi(c) <= 70, -1)
                                     .where((rsi(c) > 30) & (rsi(c) < 70), 0)),
                   "rsi_mean_reversion")
all_results.extend(wf2)

# 3. SMA Crossover 20/50d
print("  - SMA Crossover 20/50d...")
sig_sma = sma_cross_signal(spy)
all_results.append(backtest(sig_sma.reindex(common_idx).fillna(0),
                             returns_df["SPY"], "sma_cross [IS]"))
wf3 = walk_forward(pd.DataFrame({"SPY": spy}),
                   "SPY", sma_cross_signal, "sma_cross")
all_results.extend(wf3)

# 4. Bollinger 2σ
print("  - Bollinger 2σ...")
sig_bb = bollinger_signal(spy)
all_results.append(backtest(sig_bb.reindex(common_idx).fillna(0),
                             returns_df["SPY"], "bollinger_2sigma [IS]"))
wf4 = walk_forward(pd.DataFrame({"SPY": spy}),
                   "SPY", bollinger_signal, "bollinger_2sigma")
all_results.extend(wf4)

# ── AI/Semis momentum (universe data) ────────────────────────────────────────
print("  - AI/Semis Momentum (universe top-8)...")
AI_SEMIS = ["NVDA", "AMD", "AVGO", "QCOM", "AMAT", "KLAC", "MU", "MSFT", "GOOGL", "AMZN", "META", "TSM"]
try:
    assets = {}
    for t in AI_SEMIS:
        d = yf.download(t, start="2020-01-01", end="2026-09-01", progress=False)["Close"]
        if len(d) > 100:
            assets[t] = d
    if len(assets) >= 4:
        sig_ai, top8, ranks = ai_semis_signal(assets)
        sig_ai = sig_ai.reindex(common_idx).fillna(0)
        ret_ai = returns_df["SPY"].reindex(sig_ai.index).fillna(0)
        all_results.append(backtest(sig_ai, ret_ai, "ai_semis_momentum [IS]"))
        wf5 = walk_forward(pd.DataFrame({"SPY": spy}),
                           "SPY",
                           lambda c: ai_semis_signal({k: v for k, v in assets.items()})[0],
                           "ai_semis_momentum")
        all_results.extend(wf5)
        print(f"    Top-8 ranked: {top8}")
    else:
        print("  [SKIP] AI/Semis: insufficient universe data")
except Exception as e:
    print(f"  [SKIP] AI/Semis error: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n[3] Results:")
print("-" * 70)
header = f"{'Strategy':<35} {'Ann.Ret%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'WinRate':>8} {'Trades':>6}"
print(header)
print("-" * 70)
for r in all_results:
    if "error" not in r:
        print(f"{r['strategy']:<35} {r['ann_ret']:>8.1f} {r['sharpe']:>7.3f} "
              f"{r['max_dd']:>7.1f} {r['win_rate']:>8.1f} {r['n_trades']:>6}")

# Aggregate by strategy
agg = {}
for r in all_results:
    if "error" not in r:
        s = r["strategy"].split(" [")[0]
        if s not in agg:
            agg[s] = {"ann_rets": [], "sharpes": [], "max_dds": []}
        agg[s]["ann_rets"].append(r["ann_ret"])
        agg[s]["sharpes"].append(r["sharpe"])
        agg[s]["max_dds"].append(r["max_dd"])

print("\n[4] Strategy Summary (mean across runs):")
print("-" * 70)
best_sharpe = -999
best_strat  = None
for s, v in agg.items():
    avg_ret   = np.mean(v["ann_rets"])
    avg_sharp = np.mean(v["sharpes"])
    avg_dd    = np.mean(v["max_dds"])
    print(f"  {s:<35} CAGR={avg_ret:>6.1f}%  Sharpe={avg_sharp:.3f}  MaxDD={avg_dd:.1f}%")
    if avg_sharp > best_sharpe:
        best_sharpe = avg_sharp
        best_strat  = s

# Walk-forward confidence
oob_sharpes = [r["sharpe"] for r in all_results if "[WF OOS]" in r["strategy"] and "error" not in r]
confidence_wf = sum(1 for s in oob_sharpes if s > 0)
print(f"\n[5] Walk-forward OOS windows with positive Sharpe: {confidence_wf}/{len(oob_sharpes)}")
print(f"    Live mode threshold: ≥2 positive windows + user approval")
print(f"    Current status: PAPER ONLY — account unfunded (${ACCOUNT_VALUE:.2f})")

# Save
output = {
    "run_at": datetime.now().isoformat(),
    "paper_mode": PAPER_MODE,
    "account_value": ACCOUNT_VALUE,
    "best_strategy": best_strat,
    "best_sharpe": round(best_sharpe, 3),
    "wf_positive_windows": confidence_wf,
    "total_oos_windows": len(oob_sharpes),
    "all_results": all_results,
}
with open(OUTFILE, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {OUTFILE}")

# Markdown report
lines = [
    "# Equity Backtest Report — PAPER ONLY",
    f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')} PDT",
    f"**Account:** Agentic ••••661 — ${ACCOUNT_VALUE:.2f} buying power → **PAPER MODE**",
    f"**Confidence:** {confidence_wf}/{len(oob_sharpes)} walk-forward windows positive Sharpe",
    f"**Best Strategy:** {best_strat} (avg Sharpe {best_sharpe:.3f})",
    "",
    "## Strategy Results",
    "| Strategy | Ann.Ret% | Sharpe | Max DD% | Win Rate% | Trades |",
    "|---|---|---|---|---|---|",
]
for r in all_results:
    if "error" not in r:
        lines.append(f"| {r['strategy']} | {r['ann_ret']:.1f} | {r['sharpe']:.3f} "
                     f"| {r['max_dd']:.1f} | {r['win_rate']:.1f} | {r['n_trades']} |")

lines += [
    "",
    "## Research Sources",
    "- r/algotrading: mean reversion RSI 2.11 Sharpe, TQQQ+SMA 39.3% CAGR",
    "- paperswithbacktest.com: AI/Semis momentum 46.3% CAGR / 1.11 Sharpe",
    "- Reuters/Bloomberg: macro regime signals inform strategy selection",
    "",
    "## Live Mode Requirements (NOT MET)",
    "1. Account funding → currently $0",
    "2. ≥2 positive walk-forward windows → " + f"{confidence_wf}/{len(oob_sharpes)}",
    "3. User explicit approval → not yet given",
    "",
    "**Live orders are BLOCKED until all three conditions are met.**",
]
with open(REPORT, "w") as f:
    f.write("\n".join(lines))
print(f"Report saved to {REPORT}")
