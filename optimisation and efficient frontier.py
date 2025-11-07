import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

#raw data
tickers = ["AAPL", "MSFT","TLT","GLD","QQQ"]
start = "2022-01-01"
end = None
risk_free_annual = 0.02
periods_per_year = 252

data = yf.download(
    tickers,
    start=start,
    end=end,
    auto_adjust=True,   # adjusted closes in 'Close'
    actions=False,
    progress=False
)

# 2) Select price panel safely whether 1 or many tickers
if isinstance(data, pd.DataFrame) and "Close" in data.columns:
    prices = data["Close"].copy()
elif isinstance(data, pd.DataFrame) and "Adj Close" in data.columns:
    prices = data["Adj Close"].copy()
else:
    # yfinance can return a Series for single ticker
    if isinstance(data, pd.Series):
        prices = data.to_frame(name=tickers[0])
    else:
        raise ValueError("Could not find Close/Adj Close in downloaded data.")

# Ensure columns even if a single ticker
if isinstance(prices, pd.Series):
    prices = prices.to_frame(name=tickers[0])

# 3) Drop tickers with no data at all; report failures
all_na = prices.columns[prices.isna().all()]
if len(all_na) > 0:
    print("Dropping tickers with no data:", list(all_na))
    prices = prices.drop(columns=all_na)

# 4) Require >= 80% non-NaN rows per column; report drops
min_rows = int(0.8 * len(prices))
keep_mask = prices.notna().sum(axis=0) >= min_rows
dropped_partial = list(prices.columns[~keep_mask])
if dropped_partial:
    print("Dropping tickers for insufficient data:", dropped_partial)
prices = prices.loc[:, keep_mask]

# Optionally allow small gaps
# prices = prices.ffill().bfill()

# 5) Re-derive tickers and n AFTER cleaning
tickers = list(prices.columns)
n = len(tickers)
print("Final tickers:", tickers)
if n < 2:
    raise ValueError(f"Need ≥2 assets after cleaning; got {n}. Remaining: {tickers}")

# 6) Align rows strictly and compute returns
prices = prices.dropna()
rets = prices.pct_change().dropna()

# 7) Convert to NumPy arrays (avoid pandas alignment issues)
mu = (rets.mean() * periods_per_year).to_numpy()     # shape (n,)
Sigma = (rets.cov() * periods_per_year).to_numpy()   # shape (n, n)
rf = risk_free_annual

#helperz
def portfolio_stats(w, mu, Sigma):
    ret = float(w @ mu)
    var = float(w @ Sigma @ w)
    vol = np.sqrt(var)
    return ret, vol, var

def neg_sharpe(w, mu, Sigma, rf):
    ret, vol, _ = portfolio_stats(w, mu, Sigma)
    return - (ret - rf) / (vol + 1e-12) #avoid division by zero

def weight_sum_to_one():
    return {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

def target_return(R_target, mu):
    return {'type': 'eq', 'fun': lambda w: w @ mu - R_target}

def long_only_bounds(n):
    return tuple((0.0, 1.0) for _ in range (n))

bounds = long_only_bounds(n)

#GMV portfolio
def solve_gmv(mu, Sigma, bounds):
    n_local = len(mu)
    x0 = np.full(n_local, 1/n_local)
    cons = [weight_sum_to_one()]
    obj = lambda w: w @ Sigma @ w
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"GMV optimisation failed: {res.message}")
    return res.x

#max sharpe
def solve_max_sharpe(mu, Sigma, rf, bounds):
    n_local = len(mu)
    x0 = np.full(n, 1/n)
    cons = [weight_sum_to_one()]
    obj = lambda w: neg_sharpe(w, mu, Sigma, rf)
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimisation failed: {res.message}")
    return res.x

def efficient_frontier(mu, Sigma, n_pts=60, bounds=None):
    mu = np.asarray(mu)
    n_local = len(mu)
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n_local))

    # GMV first
    def gmv_obj(w): return w @ Sigma @ w
    x0 = np.full(n_local, 1/n_local)
    cons_sum = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    gmv_res = minimize(gmv_obj, x0, method="SLSQP", bounds=bounds, constraints=[cons_sum])
    if not gmv_res.success:
        raise RuntimeError(f"GMV failed: {gmv_res.message}")
    w_gmv = gmv_res.x
    gmv_ret = float(w_gmv @ mu)

    # targets from GMV return up to single-asset max return
    idx_max = int(np.argmax(mu))
    max_ret = float(mu[idx_max])

    targets = np.linspace(gmv_ret, max_ret, n_pts)

    ws, rets_out, vols_out = [], [], []
    for R in targets:
        cons = [
            cons_sum,
            {'type': 'ineq', 'fun': lambda w, R=R: w @ mu - R}  # >= R
        ]
        res = minimize(gmv_obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            w = res.x
            ws.append(w)
            r = float(w @ mu)
            v = float(np.sqrt(w @ Sigma @ w))
            rets_out.append(r); vols_out.append(v)

    # explicitly add the TRUE max-return corner (100% in argmax asset)
    one_hot = np.zeros(n_local); one_hot[idx_max] = 1.0
    ws.append(one_hot)
    rets_out.append(max_ret)
    vols_out.append(float(np.sqrt(Sigma[idx_max, idx_max])))

    ef = pd.DataFrame({"vol": vols_out, "ret": rets_out})
    ef = ef.dropna().sort_values("vol").reset_index(drop=True)
    return ef, np.array(ws)

bounds = long_only_bounds(n)

w_gmv = solve_gmv(mu, Sigma, bounds)
w_tan = solve_max_sharpe(mu, Sigma, rf, bounds)
ef, ef_ws = efficient_frontier(mu, Sigma, n_pts=60, bounds=bounds)

gmv_ret, gmv_vol, _ = portfolio_stats(w_gmv, mu, Sigma)
tan_ret, tan_vol, _ = portfolio_stats(w_tan, mu, Sigma)
tan_sharpe = (tan_ret - rf) / tan_vol

summary = pd.DataFrame({
    "Portfolio": ["GMV","Max Sharpe"],
    "Return (ann.)": [gmv_ret, tan_ret],
    "Vol (ann.)": [gmv_vol, tan_vol],
    "Sharpe": [np.nan, tan_sharpe]
})

def markowitz_closed_form(mu, Sigma, rf, n_pts=300):
    mu = np.asarray(mu, float)
    Sigma = np.asarray(Sigma, float)
    n = len(mu)
    one = np.ones(n)

    # robust inverse (handles near singular Σ)
    try:
        invS = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        invS = np.linalg.pinv(Sigma)

    A = one @ invS @ mu
    B = one @ invS @ one
    C = mu  @ invS @ mu
    D = B*C - A**2
    if D <= 0:
        raise ValueError("Frontier degenerates: D<=0 (check Σ / data window).")

    # GMV (global min var)
    w_gmv = (invS @ one) / B
    ret_gmv = float(mu @ w_gmv)
    vol_gmv = float(np.sqrt(w_gmv @ Sigma @ w_gmv))

    # Tangency (max Sharpe) with respect to rf
    k = invS @ (mu - rf*one)
    w_tan = k / (one @ k)
    ret_tan = float(mu @ w_tan)
    vol_tan = float(np.sqrt(w_tan @ Sigma @ w_tan))

    # Parametric frontier (full hyperbola)
    Rmin = min(mu.min(), ret_gmv) - 0.02*abs(ret_gmv)
    Rmax = max(mu.max(), ret_tan) + 0.02*abs(ret_tan)
    R = np.linspace(Rmin, Rmax, n_pts)
    var = (B*R**2 - 2*A*R + C) / D
    var = np.clip(var, 0, None)
    sig = np.sqrt(var)

    return {
        "R": R, "sigma": sig,
        "w_gmv": w_gmv, "ret_gmv": ret_gmv, "vol_gmv": vol_gmv,
        "w_tan": w_tan, "ret_tan": ret_tan, "vol_tan": vol_tan
    }

print("Weights (GMV):")
print(pd.Series(w_gmv, index=tickers).round(4))
print("\nWeights (Max Sharpe):")
print(pd.Series(w_tan, index=tickers).round(4))
print("\nSummary:")
print(summary.round(4))

def plot_markowitz_schematic(mu, Sigma, rf, out, tickers):
    vols_assets = np.sqrt(np.diag(Sigma))

    fig, ax = plt.subplots(figsize=(8,6), dpi=140)

    # Individual assets (gold dots)
    ax.scatter(vols_assets, mu, s=55, color="#DAA520", zorder=3, label="Individual Assets")
    for i, t in enumerate(tickers):
        ax.annotate(t, (vols_assets[i], mu[i]), xytext=(6,6), textcoords="offset points")

    # Full frontier (light grey), efficient part (dark)
    ax.plot(out["sigma"], out["R"], color="#777777", lw=1.25, zorder=2, label="Frontier (full)")
    eff_mask = out["R"] >= out["ret_gmv"]          # right/up branch is efficient
    ax.plot(out["sigma"][eff_mask], out["R"][eff_mask],
            color="#2D5BFF", lw=2.5, zorder=4, label="Efficient Frontier")

    # GMV and Tangency
    ax.scatter([out["vol_gmv"]],[out["ret_gmv"]], s=110, marker="X",
               color="#F28E2B", zorder=5, label="Tangency Portfolio" if np.isclose(out["ret_gmv"], out["ret_tan"]) else "GMV")
    ax.scatter([out["vol_tan"]],[out["ret_tan"]], s=110, marker="o",
               facecolor="#D62728", edgecolor="white", zorder=6,
               label="Tangency Portfolio")

    # CAL (Capital Market Line) from rf through tangency
    slope = (out["ret_tan"] - rf) / out["vol_tan"] if out["vol_tan"] > 0 else 0.0
    x_max = max(out["sigma"].max(), vols_assets.max(), out["vol_tan"]) * 1.05
    xs = np.linspace(0, x_max, 200)
    ax.plot(xs, rf + slope*xs, ls="-", lw=1.25, color="#B07D62", zorder=1, label="Best possible CAL")

    # Axes, grid, styling
    ax.set_xlim(0, x_max)
    y_min = min(rf, out["R"].min(), mu.min()) - 0.02
    y_max = max(out["R"].max(), mu.max()) + 0.02
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(True, ls=":", alpha=0.6)
    ax.set_xlabel("Standard Deviation")
    ax.set_ylabel("Expected Return")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    plt.show()

out = markowitz_closed_form(mu, Sigma, rf)   # uses shorts-allowed closed form
plot_markowitz_schematic(mu, Sigma, rf, out, tickers)


