# Robinhood Backtester

Equity-only backtesting framework. **Paper-trade only** — no live orders until you explicitly approve.

## Status
- Account: Agentic ••••661 | Buying power: $0 → PAPER MODE
- 5 strategies backtested + walk-forward validated
- Live orders BLOCKED until account funded + user approval

## Strategies
| Strategy | Source | Sharpe (lit.) | Notes |
|---|---|---|---|
| TQQQ + 200d SMA | r/algotrading | 0.88 | Trend filter exits to cash in bear regimes |
| AI/Semis Momentum Top-8 | paperswithbacktest | 1.11 | Equal-weight monthly rebalance |
| RSI Mean Reversion | r/algotrading | 2.11* | Buy <30 / sell >70 (*ranging markets) |
| SMA 20/50 Crossover | Classic | — | Golden/death cross signals |
| Bollinger 2σ | r/algotrading | — | Community signal |

## Run
```bash
python equity_backtest.py
```

Requires: `yfinance`, `pandas`, `numpy`
