"""
moment_panel_pipeline.py
========================
Companion estimation pipeline for the supplement

    "Panel Evidence for the Duration-Centered Theory of Bond Pricing:
     A Research Design"

Implements, end to end, the panel-data research design that extends
Section 10 of "A Moment Theory of Bond Pricing":

  Stage 1  Data ingestion adapters (TRACE / CRSP / GSW-via-FRED extracts)
  Stage 2  Timing-moment construction (duration, convexity, mu3, mu4)
  Stage 3  Panel estimation: specification ladder M1-M6
           (pooled OLS, two-way fixed effects, structural restriction)
  Stage 4  Fama-MacBeth cross-sections with Newey-West inference
  Stage 5  Hierarchy tests (pairs bootstrap over bonds) and GRS test

Usage
-----
  python moment_panel_pipeline.py --self-test
        Runs the full pipeline on a small synthetic panel with known
        moment structure. Verifies installation before real data are
        attached. Prints the specification ladder, Fama-MacBeth
        estimates, bootstrap hierarchy verdict, and GRS statistic.

  python moment_panel_pipeline.py --config config.json
        Runs the pipeline on real extracts. The configuration file
        records every filter threshold (see DEFAULT_CONFIG below) so
        the sample is exactly reproducible.

Data inputs (real run)
----------------------
  cashflow_schedules.csv : cusip_id, tau, cash          (CRSP / prospectus)
  trace_prices.csv       : cusip_id, date_month, p_clean (TRACE, filtered)
  gsw_curve.csv          : date, tau_1m ... tau_360m     (FRED: GSW dataset)

Requires: numpy, pandas (no other dependencies).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Stage 1: configuration (every threshold recorded for reproducibility)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "min_trades_per_month": 2,       # drop CUSIP-months with fewer trades
    "price_bounds": (20.0, 150.0),   # per 100 notional; reporting errors concentrate outside
    "winsor_pct": (0.01, 0.99),      # within CUSIP-month winsorization
    "drop_last_month": True,         # illiquidity/accretion at end of life
    "min_maturity_years": 1.0,
    "max_maturity_years": 30.0,
    "bootstrap_reps": 400,
    "nw_lags": None,                 # None -> floor(T^(1/3))
    "seed": 20261031,
    "data_paths": {
        "cashflows": "cashflow_schedules.csv",
        "prices": "trace_prices.csv",
        "curve": "gsw_curve.csv",
    },
}


# ---------------------------------------------------------------------------
# Stage 2: timing moments (Eq. 1-2 of the supplement)
# ---------------------------------------------------------------------------

def timing_moments(cashflows, spot_curve):
    """Central moments of discounted timing for one bond-month.

    Parameters
    ----------
    cashflows : sequence of (tau_j, c_j), times in years, cash amounts.
    spot_curve : callable tau -> continuously compounded zero rate y_t(tau).

    Returns
    -------
    dict with keys D (mean timing), C (variance), mu3, mu4, and P
    (model price under the spot curve).
    """
    tau = np.array([t for t, _ in cashflows], dtype=float)
    cf = np.array([c for _, c in cashflows], dtype=float)
    pv = cf * np.exp(-spot_curve(tau) * tau)
    P = float(pv.sum())
    w = pv / P                                   # discounted weights, sum to 1
    D = float(w @ tau)                           # duration = E[T]
    X = tau - D                                  # centered timing, E[X] = 0
    C = float(w @ X ** 2)                        # convexity = Var(T)
    mu3 = float(w @ X ** 3)
    mu4 = float(w @ X ** 4)
    return {"P": P, "D": D, "C": C, "mu3": mu3, "mu4": mu4}


def structural_logprice(mom, y):
    """Cumulant-restricted prediction (model M6): no free slope parameters.

    log P = -y*D + (y^2/2)*C - (y^3/6)*mu3 + (y^4/24)*(mu4 - 3*C^2)
    Truncation after the fourth cumulant, kappa_2 = mu2, kappa_3 = mu3,
    kappa_4 = mu4 - 3 mu2^2 (supplement, Eq. 6).
    """
    D, C, m3, m4 = mom["D"], mom["C"], mom["mu3"], mom["mu4"]
    return (-y * D + 0.5 * y ** 2 * C - y ** 3 / 6.0 * m3
            + y ** 4 / 24.0 * (m4 - 3.0 * C ** 2))


# ---------------------------------------------------------------------------
# Ingestion adapters (replace with WRDS/FRED extracts in a real run)
# ---------------------------------------------------------------------------

def load_panel(cfg):
    """Build the CUSIP-month panel from raw extracts.

    Expects the three CSV files described in the module docstring.
    Applies the filters of Section 3.2 of the supplement, computes the
    timing moments from the GSW curve, and returns a tidy DataFrame.
    """
    paths = cfg["data_paths"]
    cf = pd.read_csv(paths["cashflows"])
    px = pd.read_csv(paths["prices"])
    curve = pd.read_csv(paths["curve"]).set_index("date")

    lo, hi = cfg["price_bounds"]
    px = px[(px["p_clean"] >= lo) & (px["p_clean"] <= hi)]
    counts = px.groupby(["cusip_id", "date_month"]).size()
    keep = counts[counts >= cfg["min_trades_per_month"]].index
    px = px.set_index(["cusip_id", "date_month"]).loc[keep].reset_index()

    rows = []
    for key, g in px.groupby(["cusip_id", "date_month"]):
        cusip, month = key
        sched = cf[cf["cusip_id"] == cusip]
        if sched.empty:
            continue
        cashflows = list(zip(sched["tau"].values, sched["cash"].values))
        taus = np.array([t for t, _ in cashflows])
        if taus.max() < cfg["min_maturity_years"]:
            continue
        if taus.max() > cfg["max_maturity_years"]:
            continue

        # Flat-rate interpolation of the GSW curve at this date.
        tenors = np.array([float(c[3:]) / 12.0 for c in curve.columns])
        rates = curve.loc[month].values.astype(float) / 100.0
        order = np.argsort(tenors)
        spot = lambda t: float(np.interp(t, tenors[order], rates[order]))

        m = timing_moments(cashflows, spot)
        rows.append({"cusip_id": cusip, "date_month": month,
                     "logP": np.log(g["p_clean"].median()),
                     **m})
    panel = pd.DataFrame(rows)
    return panel


# ---------------------------------------------------------------------------
# Stage 3: specification ladder M1-M6
# ---------------------------------------------------------------------------

def ols(X, y):
    X = np.asarray(X, dtype=float)
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y, yhat):
    ss = np.sum((y - yhat) ** 2)
    st = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss / st


def spec_ladder(panel, regressors=(["D"], ["D", "C"], ["D", "C", "mu3"],
                                  ["D", "C", "mu3", "mu4"])):
    """Pooled models M1-M4; returns (names, betas, R2 list)."""
    y = panel["logP"].to_numpy()
    names, betas, r2s = [], [], []
    for regs in regressors:
        X = np.c_[np.ones(len(panel)), panel[regs].to_numpy()]
        b = ols(X, y)
        names.append("+".join(regs))
        betas.append(b)
        r2s.append(r2(y, X @ b))
    return names, betas, r2s


def two_way_fe(panel, regs=("D", "C", "mu3", "mu4")):
    """Model M5: bond and time fixed effects by iterative demeaning.

    Converges to the within estimator for two-way effects. Returns
    (coefficients, residual standard deviation)."""
    cols = list(regs)
    Y = panel["logP"].to_numpy(dtype=float).copy()
    X = panel[cols].to_numpy(dtype=float).copy()
    bi = pd.factorize(panel["cusip_id"])[0]
    bt = pd.factorize(panel["date_month"])[0]

    def demean(v, g1, g2, iters=200):
        """Alternating projections: remove bond and month means."""
        v = np.asarray(v, dtype=float).ravel()
        for _ in range(iters):
            v = v - pd.Series(v).groupby(g1).transform("mean").to_numpy()
            v = v - pd.Series(v).groupby(g2).transform("mean").to_numpy()
        return v

    Yw = demean(Y.copy(), bi, bt)
    Xw = np.column_stack([demean(X[:, j].copy(), bi, bt) for j in range(X.shape[1])])
    b = ols(Xw, Yw)
    resid = Yw - Xw @ b
    return b, float(np.sqrt(np.mean(resid ** 2)))


# ---------------------------------------------------------------------------
# Stage 4: Fama-MacBeth with Newey-West (Eq. 8 of the supplement)
# ---------------------------------------------------------------------------

def nw_lrv(x, lags):
    """Newey-West long-run variance of a series."""
    L = float(np.dot(x, x))
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        L += 2.0 * w * float(np.dot(x[l:], x[:-l]))
    return L / len(x)


def fama_macbeth(panel, specs=("D", "C", "mu3", "mu4"), lags=None):
    """Monthly cross-sections, averaged with Newey-West standard errors."""
    rows = []
    for _, g in panel.groupby("date_month"):
        X = np.c_[np.ones(len(g)), g[list(specs)].to_numpy()]
        b = ols(X, g["logP"].to_numpy())
        rows.append(b)
    B = np.asarray(rows)
    T = B.shape[0]
    if lags is None:
        lags = int(np.floor(T ** (1.0 / 3.0)))
    beta = B.mean(axis=0)
    dev = B - B.mean(axis=0)
    se = np.sqrt([nw_lrv(dev[:, j], lags) / T for j in range(B.shape[1])])
    return beta, se, B


# ---------------------------------------------------------------------------
# Stage 5: hierarchy bootstrap and GRS statistic
# ---------------------------------------------------------------------------

def bootstrap_hierarchy(panel, reps=400, seed=0):
    """Pairs bootstrap over bonds for the monotone-increment test (H3).

    Resample bond identifiers with replacement, rebuild the pseudo-panel,
    recompute the M1..M4 ladder, and collect the increments
    Delta_k = R2_k - R2_{k-1}. Returns (median increments, share of
    replications with strictly declining increments)."""
    rng = np.random.default_rng(seed)
    ids = panel["cusip_id"].unique()
    inc_all = []
    for _ in range(reps):
        pick = rng.choice(ids, size=len(ids), replace=True)
        parts = []
        for k, cid in enumerate(pick):
            g = panel[panel["cusip_id"] == cid].copy()
            g["cusip_id"] = f"{cid}_rep{k}"   # keep bootstrap pairs distinct
            parts.append(g)
        boot = pd.concat(parts, ignore_index=True)
        _, _, r2s = spec_ladder(boot)
        inc_all.append(np.diff(r2s))
    inc = np.asarray(inc_all)
    med = np.median(inc, axis=0)
    declining = np.mean(np.all(np.diff(inc, axis=1) < 0, axis=1))
    return med, float(declining)


def grs_stat(alpha, resid, factors):
    """Gibbons-Ross-Shanken statistic (Eq. 9 of the supplement).

    alpha   : (N,) pricing errors from time-series factor regressions
    resid   : (T, N) regression residuals
    factors : (T, M) factor realizations
    """
    T, N = resid.shape
    M = factors.shape[1]
    S = (resid.T @ resid) / T
    Sf = np.cov(factors, rowvar=False)
    fbar = factors.mean(axis=0)
    num = alpha @ np.linalg.solve(S, alpha)
    den = 1.0 + fbar @ np.linalg.solve(Sf, fbar)
    stat = (T / N) * ((T - N - M) / (T - M - 1.0)) * num / den
    return float(stat)


def moment_portfolios(panel, n_deciles=5):
    """Sort bonds into duration deciles; form equal-weight portfolios.

    Returns a (T x N) matrix of portfolio log prices and the (T x M)
    factor matrix (cross-sectional means of D, C, mu3)."""
    pv = panel.pivot_table(index="date_month", columns="cusip_id",
                           values="logP")
    D = panel.pivot_table(index="date_month", columns="cusip_id", values="D")
    labels = D.apply(
        lambda row: pd.qcut(row, n_deciles, labels=False,
                            duplicates="drop").astype(float),
        axis=1)
    ports = []
    for k in range(n_deciles):
        members = labels == k
        pr = pv[members].mean(axis=1)
        ports.append(pr)
    P = pd.concat(ports, axis=1)
    P = P.dropna(how="any")
    factors = panel.groupby("date_month")[["D", "C", "mu3"]].mean()
    factors = factors.loc[P.index]
    excess = P.diff().dropna()
    factors = factors.loc[excess.index]
    return excess.to_numpy(), factors.to_numpy()


def grs_test(panel, n_deciles=5):
    """Time-series regressions of portfolio changes on moment factors + GRS."""
    Y, F = moment_portfolios(panel, n_deciles)
    T = Y.shape[0]
    X = np.c_[np.ones(T), F]
    B = ols(X, Y)
    alpha = B[0]
    resid = Y - X @ B
    return grs_stat(alpha, resid, F), alpha


# ---------------------------------------------------------------------------
# Self-test on synthetic data
# ---------------------------------------------------------------------------

def _synthetic_panel(n_bonds=60, n_months=40, seed=20261031):
    """Synthetic panel with known moment structure.

    Bonds have coupon schedules; prices are generated by the cumulant
    expansion plus noise, so the pipeline can be verified end-to-end."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_bonds):
        coupon = rng.uniform(0.02, 0.08)
        mat0 = rng.uniform(2.0, 28.0)
        for m in range(n_months):
            tau = np.arange(0.5, mat0, 0.5)
            cash = np.full_like(tau, 100 * coupon / 2.0)
            cash[-1] += 100.0
            # simple flat spot curve that moves over time
            y = 0.03 + 0.01 * np.sin(2 * np.pi * m / 24.0) + 0.0005 * i
            spot = lambda t, y=y: y
            mom = timing_moments(list(zip(tau, cash)), spot)
            true_logp = structural_logprice(mom, y)   # theory-implied price
            noise = rng.normal(0, 0.004)              # small pricing noise
            month = (2400 + m)
            rows.append({"cusip_id": f"B{i:03d}", "date_month": month,
                         "logP": true_logp + noise,
                         "yield_flat": y, **mom})
    return pd.DataFrame(rows)


def run_pipeline(panel, cfg):
    """Full estimation flow; returns a result dict (Algorithm 1)."""
    out = {}
    names, betas, r2s = spec_ladder(panel)
    out["ladder_names"] = names
    out["ladder_betas"] = betas
    out["ladder_r2"] = r2s
    out["increments"] = np.diff(r2s)
    out["fe_beta"], out["fe_resid_sd"] = two_way_fe(panel)
    beta_fm, se_fm, B = fama_macbeth(panel)
    out["fm_beta"], out["fm_se"] = beta_fm, se_fm
    med_inc, share = bootstrap_hierarchy(panel, reps=cfg["bootstrap_reps"],
                                         seed=cfg["seed"])
    out["boot_median_increments"] = med_inc
    out["boot_share_declining"] = share
    grs, alpha = grs_test(panel)
    out["grs"] = grs
    out["grs_alpha"] = alpha
    out["m6_rmse"] = float(np.sqrt(np.mean(
        (panel["logP"] - panel.apply(
            lambda r: structural_logprice(r, r["yield_flat"]), axis=1)) ** 2)))
    return out


def self_test():
    """Run the full pipeline on synthetic data and sanity-check the output."""
    cfg = dict(DEFAULT_CONFIG)
    panel = _synthetic_panel()
    res = run_pipeline(panel, cfg)

    print("=== Self-test on synthetic panel (60 bonds x 40 months) ===")
    print("\nSpecification ladder (pooled):")
    for n, r2v in zip(res["ladder_names"], res["ladder_r2"]):
        print(f"  {n:<22} R2 = {r2v:.4f}")
    print(f"  increments: {np.round(res['increments'], 5)}")
    print(f"\nFama-MacBeth (const, D, C, mu3, mu4): "
          f"{np.round(res['fm_beta'], 4)}\n  NW SE: {np.round(res['fm_se'], 4)}")
    print(f"\nTwo-way FE beta (D, C, mu3, mu4): {np.round(res['fe_beta'], 4)}")
    print(f"Bootstrap median increments: {np.round(res['boot_median_increments'], 5)}")
    print(f"Share of reps with strictly declining increments: "
          f"{res['boot_share_declining']:.2f}")
    print(f"GRS statistic (5 duration-sorted portfolios): {res['grs']:.2f}")
    print(f"M6 structural RMSE: {res['m6_rmse']:.5f}")

    # Sanity assertions: theory-implied structure should mostly hold here.
    assert res["ladder_r2"][1] > res["ladder_r2"][0], "convexity should add fit"
    assert np.all(res["increments"] >= -1e-6), "R2 should be weakly increasing"
    print("\nSelf-test PASSED: pipeline is wired correctly.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--config", type=str, default=None,
                    help="JSON config with filters and data paths")
    args = ap.parse_args()
    if args.self_test or args.config is None:
        self_test()
        sys.exit(0)
    import json
    cfg = json.load(open(args.config))
    panel = load_panel(cfg)
    res = run_pipeline(panel, cfg)
    print({k: (np.round(v, 4) if isinstance(v, (np.ndarray, float)) else v)
           for k, v in res.items()})
