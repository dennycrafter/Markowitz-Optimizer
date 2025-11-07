## Features
- Closed-form Markowitz optimisation (shorts allowed)
- Efficient frontier and capital market line visualisation
- Supports custom asset lists and data from Yahoo Finance
- Outputs optimal weights, returns, volatilities, and Sharpe ratios

## Requirements
Python 3.9+, numpy, pandas, matplotlib, scipy, yfinance

## Example
### AAPL, MSFT, TLT, GLD, QQQ (2022-01-01 -> 2025-11-07)
<img width="1120" height="840" alt="image" src="https://github.com/user-attachments/assets/b6444ef0-2b7b-4fb4-99d7-c1f40a0868b9" />

| **Portfolio** | **AAPL** | **GLD** | **MSFT** | **QQQ** | **TLT** | **Return (ann.)** | **Vol (ann.)** | **Sharpe** |
|:--------------:|:-------:|:-------:|:--------:|:------:|:------:|:-----------------:|:---------------:|:-----------:|
| **GMV**        | 0.0166  | 0.4197  | 0.0875   | 0.0987 | 0.3774 | 0.0920 | 0.1181 | — |
| **Max Sharpe** | 0.0514  | 0.8068  | 0.0934   | 0.0485 | 0.0000 | 0.2027 | 0.1403 | 1.3024 |
