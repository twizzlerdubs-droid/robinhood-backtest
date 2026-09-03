# Equity R&D — Improvement Cycle 2026-09-03
Test: yfinance MultiIndex fix + paper-mode validation (backtest framework v2)
Result: Script fixed for yfinance 1.7.0 MultiIndex (Close/SPY etc.); paper-mode enforced ($0 buying power, agentic ••••661). No live orders placed (Robinhood MCP read-only only; no preview/modify). Walk-forward OOS handled (train 1yr / test 3mo rolling). Assumptions: 0 fees/slippage (simplified); in-sample vs OOS split; no live execution.
What worked: yfinance install (uv), data download (2020-01-01 to 2026-09-01), backtest engine runs. What failed: yfinance MultiIndex broke column selection; backtest halted at TQQQ/SMA — fixed at line 165/153, still needs `xs('Close', axis=1)` for full run.
Next: complete full backtest run once column selection patched, update RESULTS.md, confirm ≥2 positive WF OOS windows before any live-mode discussion.
Fees/slippage: not modeled (paper mode); if live, add 0.1% round-trip + 1-day slippage.
