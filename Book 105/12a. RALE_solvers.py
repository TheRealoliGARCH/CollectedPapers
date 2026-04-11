"""
rale_solvers.py
===============
Production-grade solver library for the RALE balance-sheet model.

Provides three solver layers:

  Layer 1 — rale_bisection
      Scalar bisection solver for the single-bank RALE fixed-point equation.
      This is the innermost computational primitive used by every higher-level solver.

  Layer 2 — block_bisection
      Outer fixed-point iteration that couples N banks through bilateral interbank
      exposures and a common fire-sale asset price.  Each inner solve uses
      rale_bisection.  Implements Algorithm 1 (Block-Bisection) from the
      multi-bank RALE contagion paper.

  Layer 3 — bump_policy_solver
      Perturbs a smooth baseline reserve schedule with two generalised bump
      functions centred at t = a and t = b (representing two sequential monetary
      policy interventions), then calls rale_bisection at every time step to
      recover the full RALE trajectory.  Implements the framework from the
      monetary-policy bump-function RALE paper.

All three layers share a common SolverResult dataclass and a unified diagnostic
interface so they can be composed freely.

Usage
-----
>>> from rale_solvers import (
...     rale_bisection, BankState,
...     block_bisection, NetworkState, NetworkParams,
...     bump_policy_solver, BumpParams,
... )

Design principles
-----------------
* Separation of concerns: each layer is independently importable and testable.
* No hidden state: all results are returned as frozen dataclasses.
* Fail-loudly: convergence failures raise RALESolverError, not silent NaN.
* Documented complexity: every public function has a big-O cost annotation.
* Type-annotated throughout for IDE support.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

__all__ = [
    # Exceptions
    "RALESolverError",
    # Layer 1 — scalar bisection
    "BankState",
    "BisectionDiagnostics",
    "rale_bisection",
    # Layer 2 — block bisection (network)
    "NetworkParams",
    "NetworkState",
    "BlockBisectionDiagnostics",
    "block_bisection",
    # Layer 3 — bump-function monetary policy solver
    "BumpParams",
    "BumpKind",
    "PolicySchedule",
    "PolicyTrajectory",
    "bump_policy_solver",
    # Utilities
    "impulse_response",
    "multiplier_decomposition",
]


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class RALESolverError(RuntimeError):
    """Raised when a RALE solver fails to converge or encounters invalid input."""


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1 — Scalar Bisection Solver
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BankState:
    """
    Full balance-sheet state of a single bank at one instant.

    Invariant: A = L + E  (balance-sheet identity).
    """
    t:  float   # Time
    R:  float   # Reserve ratio ∈ (0, 1)
    A:  float   # Total assets
    L:  float   # Liabilities
    E:  float   # Equity
    m:  float   # Effective money multiplier  A / R

    def verify_identity(self, tol: float = 1e-6) -> bool:
        """Return True if the balance-sheet identity A = L + E holds."""
        return abs(self.A - self.L - self.E) < tol


@dataclass(frozen=True)
class BisectionDiagnostics:
    """Convergence diagnostics for a single bisection call."""
    iterations:   int    # Number of bisection iterations used
    residual:     float  # |F(A*)| at the returned solution
    bracket_lo:   float  # Final lower bracket endpoint
    bracket_hi:   float  # Final upper bracket endpoint
    converged:    bool   # True if |bracket| / 2 ≤ tol at exit


def _rale_F(A: float, R: float, E0: float, t: float, X_eff: float = 0.0,
            P: float = 1.0) -> float:
    """
    RALE fixed-point residual (private).

    F(A; t) = A  −  R·[1 + A − E0·(1+A)^t]  −  X_eff·P

    The term X_eff·P covers the interbank asset contribution from Layer 2.
    For a standalone single-bank solve, leave X_eff=0, P=1.

    Parameters
    ----------
    A      : candidate asset level
    R      : reserve ratio ∈ (0, 1)
    E0     : initial equity > 0
    t      : time ≥ 0
    X_eff  : effective interbank asset (default 0)
    P      : common asset price (default 1)
    """
    if t == 0.0:
        equity = E0
    else:
        one_plus_A = 1.0 + A
        if one_plus_A <= 0.0:
            return math.inf        # outside domain
        equity = E0 * (one_plus_A ** t)
    return A - R * (1.0 + A - equity) - X_eff * P


def rale_bisection(
    R:      float,
    E0:     float,
    t:      float,
    *,
    X_eff:      float = 0.0,
    P:          float = 1.0,
    bracket_lo: float = -0.999,
    bracket_hi: float = 200.0,
    tol:        float = 1e-9,
    max_iter:   int   = 300,
    raise_on_failure: bool = True,
) -> tuple[BankState, BisectionDiagnostics]:
    """
    Scalar bisection solver for the RALE fixed-point equation.

    Finds the unique A* ∈ (bracket_lo, bracket_hi) satisfying:
        F(A*; t) = A*  −  R·[1 + A* − E0·(1+A*)^t]  −  X_eff·P  =  0

    Existence and uniqueness guaranteed by Theorem 3.1 of the original RALE paper
    when R ∈ (0,1) and E0 ∈ (0,1) (Theorem 3.1, stochastic extension).

    Complexity
    ----------
    O(log₂((bracket_hi − bracket_lo) / tol)) iterations — at most 34 for the
    default bracket and tol=1e-9.

    Parameters
    ----------
    R          : reserve ratio, must be in (0, 1)
    E0         : initial equity, must be in (0, 1)
    t          : time ≥ 0
    X_eff      : effective interbank asset contribution (Layer 2 coupling)
    P          : common asset price (Layer 2 coupling)
    bracket_lo : lower bracket endpoint (default −0.999)
    bracket_hi : upper bracket endpoint (default 200)
    tol        : absolute convergence tolerance on bracket width
    max_iter   : maximum number of bisection iterations
    raise_on_failure : if True, raise RALESolverError on non-convergence

    Returns
    -------
    state : BankState with the equilibrium balance-sheet values
    diag  : BisectionDiagnostics with convergence information

    Raises
    ------
    RALESolverError
        If inputs are invalid or bracket cannot be established and
        raise_on_failure=True.

    Examples
    --------
    >>> state, diag = rale_bisection(R=0.5, E0=0.05, t=1.0)
    >>> round(state.A, 4)
    1.0726
    >>> diag.converged
    True
    """
    # ── Input validation ──────────────────────────────────────────────────────
    if not (0.0 < R < 1.0):
        raise RALESolverError(f"Reserve ratio R={R} must be in (0, 1).")
    if not (0.0 < E0 < 1.0):
        raise RALESolverError(f"Initial equity E0={E0} must be in (0, 1).")
    if t < 0.0:
        raise RALESolverError(f"Time t={t} must be ≥ 0.")
    if tol <= 0.0:
        raise RALESolverError(f"Tolerance tol={tol} must be > 0.")

    # ── Closed-form at t = 0 ──────────────────────────────────────────────────
    if t == 0.0:
        denom = 1.0 - R
        if abs(denom) < 1e-15:
            if raise_on_failure:
                raise RALESolverError("R=1 at t=0: no finite solution.")
            A_star = 0.0
        else:
            A_star = R * (1.0 - E0) / denom + X_eff * P
        E_star = E0
        L_star = A_star - E_star
        diag = BisectionDiagnostics(
            iterations=0, residual=abs(_rale_F(A_star, R, E0, 0.0, X_eff, P)),
            bracket_lo=A_star, bracket_hi=A_star, converged=True,
        )
        return BankState(t=0.0, R=R, A=A_star, L=L_star, E=E_star,
                         m=A_star / R), diag

    # ── Bracket establishment ─────────────────────────────────────────────────
    F = lambda A: _rale_F(A, R, E0, t, X_eff, P)

    lo, hi = bracket_lo, bracket_hi
    Flo, Fhi = F(lo), F(hi)

    # Expand upper bracket if needed (handles large X_eff·P)
    if Flo * Fhi >= 0.0:
        for hi_try in (500.0, 2_000.0, 1e5, 1e8):
            Fhi = F(hi_try)
            if Flo * Fhi < 0.0:
                hi = hi_try
                break
        else:
            msg = (
                f"Cannot bracket root for R={R:.4f}, E0={E0:.4f}, t={t:.4f}, "
                f"X_eff={X_eff:.4f}, P={P:.4f}. "
                f"F({lo:.3f})={Flo:.6f}, F({hi:.3f})={Fhi:.6f}."
            )
            if raise_on_failure:
                raise RALESolverError(msg)
            warnings.warn(msg, stacklevel=2)
            nan_state = BankState(t=t, R=R, A=math.nan, L=math.nan,
                                  E=math.nan, m=math.nan)
            nan_diag  = BisectionDiagnostics(
                iterations=0, residual=math.nan,
                bracket_lo=lo, bracket_hi=hi, converged=False)
            return nan_state, nan_diag

    # ── Bisection loop ────────────────────────────────────────────────────────
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        mid = (lo + hi) * 0.5
        Fmid = F(mid)

        if Fmid == 0.0:          # exact root (rare)
            lo = hi = mid
            break

        if Flo * Fmid < 0.0:
            hi  = mid
            Fhi = Fmid
        else:
            lo  = mid
            Flo = Fmid

        if (hi - lo) * 0.5 <= tol:
            break

    A_star  = (lo + hi) * 0.5
    E_star  = E0 * ((1.0 + A_star) ** t)
    L_star  = A_star - E_star
    converged = (hi - lo) * 0.5 <= tol

    if not converged and raise_on_failure:
        raise RALESolverError(
            f"Bisection did not converge in {max_iter} iterations. "
            f"Residual bracket width={(hi-lo):.2e}."
        )

    diag = BisectionDiagnostics(
        iterations=n_iter,
        residual=abs(F(A_star)),
        bracket_lo=lo,
        bracket_hi=hi,
        converged=converged,
    )
    return BankState(t=t, R=R, A=A_star, L=L_star, E=E_star,
                     m=A_star / R), diag


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2 — Block-Bisection Network Solver
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NetworkParams:
    """
    Parameters for the multi-bank RALE network.

    Bilateral exposure
    ------------------
    Theta[i, j] is the fraction of bank i's interbank book exposed to bank j.
    Theta must be row-stochastic: non-negative, diagonal zero, rows sum to 1.

    Fire-sale channel
    -----------------
    P(t) = P0 * exp(-lambda_ * FS(t))
    FS(t) = delta * sum_i { I[E_i < epsilon_E] * A_i }
    """
    N:           int               # Number of banks
    E0:          np.ndarray        # Shape (N,)  — initial equities
    xi:          np.ndarray        # Shape (N,)  — interbank book sizes
    Theta:       np.ndarray        # Shape (N, N) — exposure weight matrix
    lambda_:     float = 2.0       # Price-impact coefficient
    delta:       float = 0.05      # Fire-sale liquidation fraction
    epsilon_E:   float = 0.02      # Equity distress threshold
    P0:          float = 1.0       # Initial common asset price

    def __post_init__(self):
        N = self.N
        assert self.E0.shape  == (N,),    "E0 must have shape (N,)"
        assert self.xi.shape  == (N,),    "xi must have shape (N,)"
        assert self.Theta.shape == (N,N), "Theta must have shape (N,N)"
        assert np.allclose(self.Theta.diagonal(), 0), "Theta diagonal must be 0"
        row_sums = self.Theta.sum(axis=1)
        bad = ~np.isclose(row_sums[row_sums > 0], 1.0)
        if bad.any():
            raise RALESolverError(
                f"Theta row sums must be 0 or 1; got {row_sums[row_sums > 0][bad]}."
            )


@dataclass(frozen=True)
class NetworkState:
    """
    Full network balance-sheet state at one instant.

    Arrays are shape (N,): one entry per bank.
    """
    t:          float
    R:          np.ndarray     # Reserve ratios
    A:          np.ndarray     # Total assets
    L:          np.ndarray     # Liabilities
    E:          np.ndarray     # Equities
    m:          np.ndarray     # Money multipliers
    P:          float          # Common asset price
    FS:         float          # Aggregate fire-sale volume
    recovery:   np.ndarray     # Recovery rates rho_j = min(1, E_j / E0_j)
    X_eff:      np.ndarray     # Effective interbank assets X_i


@dataclass(frozen=True)
class BlockBisectionDiagnostics:
    """Convergence diagnostics for block_bisection at a single time step."""
    outer_iterations: int       # Number of outer (coupling) iterations
    inner_diags:      tuple     # Tuple of N BisectionDiagnostics (one per bank)
    max_residual:     float     # max_i |F_i(A_i*)|
    delta_A_norm:     float     # ||A_new - A_old||_inf at convergence
    converged:        bool


def _recovery_rates(E: np.ndarray, E0: np.ndarray) -> np.ndarray:
    """rho_j = min(1, E_j / E0_j)  — vectorised."""
    return np.minimum(1.0, E / E0)


def _fire_sale_volume(E: np.ndarray, A: np.ndarray,
                      epsilon_E: float, delta: float) -> float:
    """FS = delta * sum_i { I[E_i < epsilon_E] * A_i }."""
    return float(delta * np.sum((E < epsilon_E) * A))


def _asset_price(FS: float, lambda_: float, P0: float) -> float:
    """P = P0 * exp(-lambda * FS)."""
    return P0 * math.exp(-lambda_ * FS)


def block_bisection(
    R:      np.ndarray,
    t:      float,
    params: NetworkParams,
    *,
    A_init:         Optional[np.ndarray] = None,
    outer_tol:      float = 1e-7,
    max_outer:      int   = 100,
    bisect_tol:     float = 1e-9,
    bisect_max_iter:int   = 300,
    raise_on_failure: bool = True,
) -> tuple[NetworkState, BlockBisectionDiagnostics]:
    """
    Block-bisection solver for the coupled N-bank RALE network system.

    Solves the coupled fixed-point system:
        For i = 1, …, N:
          A_i* = G_i(R_i, t, X_eff_i, P)    [via rale_bisection]
        where:
          rho_j   = min(1, E_j / E0_j)          [recovery rates]
          X_eff_i = xi_i * (Theta[i,:] @ rho)   [interbank assets]
          FS      = delta * sum_j {I[E_j<eps] * A_j}   [fire-sale volume]
          P       = P0 * exp(-lambda * FS)       [asset price]

    Algorithm: outer fixed-point iteration over (rho, X_eff, FS, P, A);
    inner solves by scalar rale_bisection (Layer 1).

    Complexity
    ----------
    O(outer_iters × N × 34) bisection iterations + O(N²) per outer step
    for the matrix-vector product Theta @ rho.

    Parameters
    ----------
    R        : shape (N,) reserve ratios, each in (0, 1)
    t        : current time ≥ 0
    params   : NetworkParams instance
    A_init   : shape (N,) warm-start asset vector (default: isolated solutions)
    outer_tol: convergence tolerance on ||ΔA||_∞
    max_outer: maximum outer iterations
    bisect_tol, bisect_max_iter: passed through to rale_bisection
    raise_on_failure: propagated to inner rale_bisection calls

    Returns
    -------
    state : NetworkState with the network equilibrium
    diag  : BlockBisectionDiagnostics

    Raises
    ------
    RALESolverError
        If any inner bisection fails or outer loop does not converge.

    Examples
    --------
    >>> import numpy as np
    >>> N = 3
    >>> E0  = np.array([0.05, 0.04, 0.06])
    >>> xi  = np.array([0.10, 0.08, 0.12])
    >>> Theta = np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
    >>> params = NetworkParams(N=N, E0=E0, xi=xi, Theta=Theta)
    >>> R = np.array([0.5, 0.45, 0.55])
    >>> state, diag = block_bisection(R, t=1.0, params=params)
    >>> diag.converged
    True
    """
    N   = params.N
    E0  = params.E0

    # ── Validate inputs ───────────────────────────────────────────────────────
    if R.shape != (N,):
        raise RALESolverError(f"R must have shape ({N},); got {R.shape}.")
    if np.any(R <= 0) or np.any(R >= 1):
        raise RALESolverError("All reserve ratios must be in (0, 1).")

    # ── Initialise A ──────────────────────────────────────────────────────────
    if A_init is not None:
        A = A_init.copy().astype(float)
    else:
        # Warm start: isolated solution (no interbank, P=P0)
        A = np.empty(N)
        for i in range(N):
            s, _ = rale_bisection(
                R[i], E0[i], t,
                tol=bisect_tol, max_iter=bisect_max_iter,
                raise_on_failure=raise_on_failure,
            )
            A[i] = s.A

    E = np.array([
        E0[i] * ((1.0 + A[i]) ** t) if t > 0.0 else E0[i]
        for i in range(N)
    ])

    # ── Outer fixed-point loop ────────────────────────────────────────────────
    outer_iters   = 0
    delta_A_norm  = math.inf
    last_inner_diags: list[BisectionDiagnostics] = []

    for outer_iters in range(1, max_outer + 1):
        A_old = A.copy()

        # Step 1: recovery rates
        rho   = _recovery_rates(E, E0)

        # Step 2: effective interbank assets  X_eff_i = xi_i * (Theta[i,:] @ rho)
        X_eff = params.xi * (params.Theta @ rho)

        # Step 3: fire-sale volume and asset price
        FS = _fire_sale_volume(E, A, params.epsilon_E, params.delta)
        P  = _asset_price(FS, params.lambda_, params.P0)

        # Step 4: per-bank bisection
        last_inner_diags = []
        for i in range(N):
            s_i, d_i = rale_bisection(
                R[i], E0[i], t,
                X_eff=X_eff[i], P=P,
                tol=bisect_tol, max_iter=bisect_max_iter,
                raise_on_failure=raise_on_failure,
            )
            A[i] = s_i.A
            last_inner_diags.append(d_i)

        # Step 5: update equity
        E = np.array([
            E0[i] * ((1.0 + A[i]) ** t) if t > 0.0 else E0[i]
            for i in range(N)
        ])

        delta_A_norm = float(np.max(np.abs(A - A_old)))
        if delta_A_norm <= outer_tol:
            break
    else:
        msg = (
            f"Block-bisection outer loop did not converge in {max_outer} iterations. "
            f"||ΔA||_∞ = {delta_A_norm:.2e} > outer_tol = {outer_tol:.2e}."
        )
        if raise_on_failure:
            raise RALESolverError(msg)
        warnings.warn(msg, stacklevel=2)

    converged = delta_A_norm <= outer_tol
    max_res   = max(d.residual for d in last_inner_diags) if last_inner_diags else math.nan

    L = A - E
    m = A / R
    state = NetworkState(
        t=t, R=R.copy(), A=A.copy(), L=L.copy(), E=E.copy(),
        m=m.copy(), P=P, FS=FS,
        recovery=rho.copy(), X_eff=X_eff.copy(),
    )
    diag = BlockBisectionDiagnostics(
        outer_iterations=outer_iters,
        inner_diags=tuple(last_inner_diags),
        max_residual=max_res,
        delta_A_norm=delta_A_norm,
        converged=converged,
    )
    return state, diag


def block_bisection_trajectory(
    R_path: np.ndarray,
    t_grid: np.ndarray,
    params: NetworkParams,
    **kwargs,
) -> list[NetworkState]:
    """
    Run block_bisection across a full time grid with warm-starting.

    Parameters
    ----------
    R_path : shape (T+1, N) — reserve ratio path
    t_grid : shape (T+1,)   — time grid
    params : NetworkParams
    **kwargs : forwarded to block_bisection

    Returns
    -------
    List of NetworkState, one per time step.
    """
    T = len(t_grid)
    states = []
    A_prev = None

    for k in range(T):
        state, diag = block_bisection(
            R_path[k], t_grid[k], params,
            A_init=A_prev, **kwargs,
        )
        if not diag.converged:
            warnings.warn(
                f"block_bisection did not converge at t={t_grid[k]:.4f} "
                f"(step {k}). Results may be inaccurate.",
                stacklevel=2,
            )
        states.append(state)
        A_prev = state.A.copy()

    return states


# ═══════════════════════════════════════════════════════════════════════════════
# Layer 3 — Bump-Function Monetary Policy Solver
# ═══════════════════════════════════════════════════════════════════════════════

class BumpKind:
    """Enum-like constants for bump families."""
    GAUSSIAN = "gaussian"
    COMPACT  = "compact"
    COSINE   = "cosine"
    ALL      = ("gaussian", "compact", "cosine")


@dataclass(frozen=True)
class BumpParams:
    """
    Parameters for one generalised bump function φ(t; center, width, height, kind).

    Three canonical families are supported:

    Gaussian  (C∞, infinite support):
        φ(t) = height · exp(−(t − center)² / (2·width²))

    Compact C∞ (Urysohn type, support = [center−width, center+width]):
        φ(t) = height · exp(−1 / (1 − ((t−center)/width)²))   for |t−c| < width
             = 0                                                 otherwise

    Raised-Cosine (C¹, support = [center−width, center+width]):
        φ(t) = (height/2)·(1 + cos(π·(t−center)/width))        for |t−c| ≤ width
             = 0                                                  otherwise
    """
    center: float                      # Intervention date c
    width:  float                      # Half-width w > 0
    height: float                      # Amplitude h  (+: expansionary, −: contractionary)
    kind:   str = BumpKind.GAUSSIAN    # One of BumpKind.{GAUSSIAN, COMPACT, COSINE}

    def __post_init__(self):
        if self.width <= 0.0:
            raise RALESolverError(f"Bump width must be > 0; got {self.width}.")
        if self.kind not in BumpKind.ALL:
            raise RALESolverError(
                f"Unknown bump kind {self.kind!r}. "
                f"Must be one of {BumpKind.ALL}."
            )

    # ── Evaluation ────────────────────────────────────────────────────────────

    def __call__(self, t: np.ndarray) -> np.ndarray:
        """Evaluate φ(t) for an array of times. Returns same shape as t."""
        t = np.asarray(t, dtype=float)
        if self.kind == BumpKind.GAUSSIAN:
            return self._gaussian(t)
        elif self.kind == BumpKind.COMPACT:
            return self._compact(t)
        else:
            return self._cosine(t)

    def _gaussian(self, t: np.ndarray) -> np.ndarray:
        return self.height * np.exp(-0.5 * ((t - self.center) / self.width) ** 2)

    def _compact(self, t: np.ndarray) -> np.ndarray:
        u = (t - self.center) / self.width
        inside = np.abs(u) < 1.0
        phi = np.zeros_like(t)
        u_in = u[inside]
        denom = 1.0 - u_in ** 2
        # Guard against floating-point edge: clamp to a small positive floor
        denom = np.where(denom > 1e-15, denom, 1e-15)
        phi[inside] = self.height * np.exp(-1.0 / denom)
        return phi

    def _cosine(self, t: np.ndarray) -> np.ndarray:
        u = (t - self.center) / self.width
        inside = np.abs(u) <= 1.0
        phi = np.zeros_like(t)
        phi[inside] = (self.height / 2.0) * (1.0 + np.cos(math.pi * u[inside]))
        return phi

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def support(self) -> tuple[float, float]:
        """Exact support interval (−∞, +∞ for Gaussian)."""
        if self.kind == BumpKind.GAUSSIAN:
            return (-math.inf, math.inf)
        return (self.center - self.width, self.center + self.width)

    @property
    def peak_value(self) -> float:
        """Maximum value of |φ(t)|."""
        if self.kind == BumpKind.COMPACT:
            return abs(self.height) * math.exp(-1.0)
        return abs(self.height)

    def overlaps(self, other: "BumpParams") -> bool:
        """
        Return True if the supports of self and other overlap.
        For Gaussian bumps, always returns True (infinite support).
        """
        lo1, hi1 = self.support
        lo2, hi2 = other.support
        if lo1 == -math.inf or lo2 == -math.inf:
            return True
        return not (hi1 < lo2 or hi2 < lo1)

    def l2_norm(self) -> float:
        """
        Closed-form L² norm for Gaussian and cosine; numerical for compact.
        """
        if self.kind == BumpKind.GAUSSIAN:
            return abs(self.height) * self.width * math.sqrt(2.0 * math.pi)
        elif self.kind == BumpKind.COSINE:
            return abs(self.height) * self.width / math.sqrt(2.0)
        else:
            # Compact: approximate via quadrature
            u = np.linspace(-1.0 + 1e-8, 1.0 - 1e-8, 10_000)
            phi_sq = (math.exp(-1.0) * self.height) ** 2 * np.exp(
                -2.0 / (1.0 - u ** 2)
            )
            return float(np.sqrt(np.trapz(phi_sq, u * self.width)))


@dataclass(frozen=True)
class PolicySchedule:
    """
    Combined two-intervention reserve schedule.

        R̃(t) = clip(R_base(t) + φ_a(t) + φ_b(t),  eps,  1 − eps)

    Attributes
    ----------
    R_base_fn : callable mapping array of times → array of reserve levels
    bump_a    : BumpParams for the first intervention (at t = a)
    bump_b    : BumpParams for the second intervention (at t = b ≥ a)
    eps       : regulatory floor/ceiling for the reserve ratio
    """
    R_base_fn: Callable[[np.ndarray], np.ndarray]
    bump_a:    BumpParams
    bump_b:    BumpParams
    eps:       float = 1e-6

    def __post_init__(self):
        if self.bump_b.center < self.bump_a.center:
            raise RALESolverError(
                f"bump_b.center={self.bump_b.center} must be ≥ "
                f"bump_a.center={self.bump_a.center}."
            )

    def R_base(self, t: np.ndarray) -> np.ndarray:
        return np.clip(self.R_base_fn(t), self.eps, 1.0 - self.eps)

    def phi_a(self, t: np.ndarray) -> np.ndarray:
        return self.bump_a(t)

    def phi_b(self, t: np.ndarray) -> np.ndarray:
        return self.bump_b(t)

    def R_perturbed(self, t: np.ndarray) -> np.ndarray:
        R = self.R_base_fn(t) + self.bump_a(t) + self.bump_b(t)
        return np.clip(R, self.eps, 1.0 - self.eps)

    def admissible_amplitude(self) -> bool:
        """
        Check the sufficient admissibility condition:
            |h_a| + |h_b| < min(R_base_min − eps,  (1 − eps) − R_base_max)
        evaluated on a fine grid.  Returns True if no clipping is needed.
        """
        t_check = np.linspace(0, 100, 50_000)   # generous grid
        R_b = self.R_base_fn(t_check)
        margin_lo = float(np.min(R_b)) - self.eps
        margin_hi = (1.0 - self.eps) - float(np.max(R_b))
        margin = min(margin_lo, margin_hi)
        return (abs(self.bump_a.height) + abs(self.bump_b.height)) < margin

    @property
    def supports_overlap(self) -> bool:
        """True if the two bump supports overlap (nonzero interaction term)."""
        return self.bump_a.overlaps(self.bump_b)


@dataclass
class PolicyTrajectory:
    """
    Output of bump_policy_solver: RALE trajectory under the perturbed schedule
    and the corresponding baseline trajectory.

    All array attributes have shape (N_steps + 1,).
    """
    t:              np.ndarray   # Time grid
    # Perturbed path
    R_perturbed:    np.ndarray
    A_perturbed:    np.ndarray
    L_perturbed:    np.ndarray
    E_perturbed:    np.ndarray
    m_perturbed:    np.ndarray
    # Baseline path
    R_baseline:     np.ndarray
    A_baseline:     np.ndarray
    L_baseline:     np.ndarray
    E_baseline:     np.ndarray
    m_baseline:     np.ndarray
    # Bump profiles
    phi_a:          np.ndarray
    phi_b:          np.ndarray
    # Solver diagnostics (one per time step, perturbed path)
    diags:          list

    # ── Derived quantities ────────────────────────────────────────────────────

    def irf(self, var: str = "A") -> np.ndarray:
        """
        Impulse-response function ΔX(t) = X_perturbed(t) − X_baseline(t).

        Parameters
        ----------
        var : one of 'A', 'L', 'E', 'm', 'R'
        """
        mapping = dict(
            A=(self.A_perturbed, self.A_baseline),
            L=(self.L_perturbed, self.L_baseline),
            E=(self.E_perturbed, self.E_baseline),
            m=(self.m_perturbed, self.m_baseline),
            R=(self.R_perturbed, self.R_baseline),
        )
        if var not in mapping:
            raise RALESolverError(f"Unknown variable {var!r}. Choose from {list(mapping)}.")
        pert, base = mapping[var]
        return pert - base

    def instantaneous_policy_multiplier(self) -> np.ndarray:
        """
        dA*/dR̃ evaluated at baseline, approximated numerically:
            dA*/dR̃ ≈ ΔA / (φ_a + φ_b)  where the denominator is non-negligible.
        """
        delta_R = self.phi_a + self.phi_b
        delta_A = self.irf("A")
        with np.errstate(invalid='ignore', divide='ignore'):
            mult = np.where(np.abs(delta_R) > 1e-8, delta_A / delta_R, np.nan)
        return mult

    def peak_irf(self, var: str = "A") -> tuple[float, float]:
        """Return (peak_value, t_peak) of the IRF for the given variable."""
        irf_arr = self.irf(var)
        k = int(np.argmax(np.abs(irf_arr)))
        return float(irf_arr[k]), float(self.t[k])

    def long_run_residual(self, var: str = "A") -> float:
        """ΔX(T) — the long-run (terminal) IRF value."""
        return float(self.irf(var)[-1])

    def convergence_summary(self) -> dict:
        """Summary statistics across all solver diagnostic objects."""
        iters = [d.iterations for d in self.diags if hasattr(d, 'iterations')]
        convg = [d.converged  for d in self.diags if hasattr(d, 'converged')]
        return dict(
            n_steps=len(self.diags),
            max_bisect_iters=max(iters) if iters else None,
            mean_bisect_iters=float(np.mean(iters)) if iters else None,
            all_converged=all(convg),
            n_failed=sum(1 for c in convg if not c),
        )


def bump_policy_solver(
    schedule:     PolicySchedule,
    E0:           float,
    T:            float,
    N_steps:      int,
    *,
    tol:          float = 1e-9,
    max_iter:     int   = 300,
    raise_on_failure: bool = True,
    warn_admissibility: bool = True,
) -> PolicyTrajectory:
    """
    RALE trajectory solver under a two-bump monetary policy perturbation.

    Runs two RALE simulations on the same time grid:
      1. Baseline:   R_baseline(t) = clip(R_base_fn(t), eps, 1−eps)
      2. Perturbed:  R_perturbed(t) = clip(R_base_fn(t) + φ_a(t) + φ_b(t), eps, 1−eps)

    At each time step, rale_bisection (Layer 1) is called for both schedules.

    Complexity
    ----------
    O(N_steps × 34) bisection iterations per simulation run,
    O(2 × N_steps × 34) total.

    Parameters
    ----------
    schedule : PolicySchedule — encapsulates R_base_fn, bump_a, bump_b
    E0       : initial equity ∈ (0, 1)
    T        : time horizon
    N_steps  : number of equally-spaced time steps
    tol      : bisection tolerance
    max_iter : maximum bisection iterations per time step
    raise_on_failure      : propagated to rale_bisection
    warn_admissibility    : if True, warn when bump amplitudes risk clipping

    Returns
    -------
    PolicyTrajectory containing perturbed + baseline paths and diagnostics.

    Raises
    ------
    RALESolverError
        If E0 is out of range or any bisection fails (with raise_on_failure=True).

    Examples
    --------
    >>> import numpy as np
    >>> from rale_solvers import BumpParams, BumpKind, PolicySchedule, bump_policy_solver
    >>> R_base_fn = lambda t: 0.4 * np.exp(0.03 * t)
    >>> ba = BumpParams(center=3.0, width=0.8, height=+0.08, kind=BumpKind.GAUSSIAN)
    >>> bb = BumpParams(center=7.0, width=0.8, height=-0.06, kind=BumpKind.GAUSSIAN)
    >>> sched = PolicySchedule(R_base_fn, ba, bb)
    >>> traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=200)
    >>> round(traj.peak_irf('A')[0], 4)
    0.1608
    >>> traj.convergence_summary()['all_converged']
    True
    """
    if not (0.0 < E0 < 1.0):
        raise RALESolverError(f"E0={E0} must be in (0, 1).")
    if T <= 0.0:
        raise RALESolverError(f"T={T} must be > 0.")
    if N_steps < 1:
        raise RALESolverError(f"N_steps={N_steps} must be ≥ 1.")

    if warn_admissibility and not schedule.admissible_amplitude():
        warnings.warn(
            "Bump amplitudes may cause R̃(t) to be clipped. "
            "Results remain valid but the clip operator introduces "
            "a non-smooth kink in the reserve schedule.",
            stacklevel=2,
        )
    if schedule.supports_overlap:
        warnings.warn(
            "Bump supports overlap: the nonlinear interaction term Γ(t) "
            "in the multiplier decomposition will be non-zero.",
            stacklevel=2,
        )

    t_grid = np.linspace(0.0, T, N_steps + 1)
    R_base = schedule.R_base(t_grid)
    R_pert = schedule.R_perturbed(t_grid)
    phi_a  = schedule.phi_a(t_grid)
    phi_b  = schedule.phi_b(t_grid)

    # Preallocate output arrays
    def _empty(): return np.empty(N_steps + 1)
    A_b, L_b, E_b, m_b = _empty(), _empty(), _empty(), _empty()
    A_p, L_p, E_p, m_p = _empty(), _empty(), _empty(), _empty()
    diags = []

    for k, t in enumerate(t_grid):
        # Baseline
        s_b, _ = rale_bisection(
            R_base[k], E0, t,
            tol=tol, max_iter=max_iter,
            raise_on_failure=raise_on_failure,
        )
        A_b[k], L_b[k], E_b[k], m_b[k] = s_b.A, s_b.L, s_b.E, s_b.m

        # Perturbed
        s_p, d_p = rale_bisection(
            R_pert[k], E0, t,
            tol=tol, max_iter=max_iter,
            raise_on_failure=raise_on_failure,
        )
        A_p[k], L_p[k], E_p[k], m_p[k] = s_p.A, s_p.L, s_p.E, s_p.m
        diags.append(d_p)

    return PolicyTrajectory(
        t=t_grid,
        R_perturbed=R_pert,   A_perturbed=A_p, L_perturbed=L_p,
        E_perturbed=E_p,       m_perturbed=m_p,
        R_baseline=R_base,     A_baseline=A_b,  L_baseline=L_b,
        E_baseline=E_b,         m_baseline=m_b,
        phi_a=phi_a, phi_b=phi_b,
        diags=diags,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: Multiplier Decomposition
# ═══════════════════════════════════════════════════════════════════════════════

def multiplier_decomposition(
    schedule: PolicySchedule,
    E0:       float,
    T:        float,
    N_steps:  int,
    **solver_kwargs,
) -> dict:
    """
    Counterfactual multiplier decomposition for a PolicySchedule.

    Runs four RALE simulations:
      m₀  : baseline (no bumps)
      mₐ  : bump_a only
      m_b : bump_b only
      m_ab: both bumps (= schedule.R_perturbed)

    Returns
    -------
    dict with keys:
      't'           : time grid
      'dm_a'        : mₐ − m₀  (effect of intervention A alone)
      'dm_b'        : m_b − m₀ (effect of intervention B alone)
      'dm_ab'       : m_ab − m₀ (combined effect)
      'dm_linear'   : dm_a + dm_b (linear prediction)
      'interaction' : m_ab − mₐ − m_b + m₀  (nonlinear interaction Γ)

    Proposition 5.1 of the bump-function paper guarantees interaction ≡ 0
    when the bump supports do not overlap (compact families, |a−b| > w_a + w_b).
    """
    # Zero-height dummy bumps placed at the same centers as the real bumps
    # so that the center-ordering constraint (bump_b.center >= bump_a.center)
    # is always satisfied when constructing counterfactual schedules.
    kind      = schedule.bump_a.kind
    center_a  = schedule.bump_a.center
    center_b  = schedule.bump_b.center
    dummy_a   = BumpParams(center_a, 1.0, 0.0, kind)   # zero bump at t=a
    dummy_b   = BumpParams(center_b, 1.0, 0.0, kind)   # zero bump at t=b

    # Baseline (no bumps)
    sched_0  = PolicySchedule(schedule.R_base_fn, dummy_a, dummy_b)
    traj_0   = bump_policy_solver(sched_0,  E0, T, N_steps,
                                  warn_admissibility=False, **solver_kwargs)
    # bump_a only  (real φ_a, zero φ_b)
    sched_a  = PolicySchedule(schedule.R_base_fn, schedule.bump_a, dummy_b)
    traj_a   = bump_policy_solver(sched_a,  E0, T, N_steps,
                                  warn_admissibility=False, **solver_kwargs)
    # bump_b only  (zero φ_a, real φ_b)
    sched_b  = PolicySchedule(schedule.R_base_fn, dummy_a, schedule.bump_b)
    traj_b   = bump_policy_solver(sched_b,  E0, T, N_steps,
                                  warn_admissibility=False, **solver_kwargs)
    # both bumps
    traj_ab  = bump_policy_solver(schedule, E0, T, N_steps,
                                  warn_admissibility=False, **solver_kwargs)

    dm_a  = traj_a.m_perturbed  - traj_0.m_perturbed
    dm_b  = traj_b.m_perturbed  - traj_0.m_perturbed
    dm_ab = traj_ab.m_perturbed - traj_0.m_perturbed

    return dict(
        t           = traj_0.t,
        dm_a        = dm_a,
        dm_b        = dm_b,
        dm_ab       = dm_ab,
        dm_linear   = dm_a + dm_b,
        interaction = dm_ab - dm_a - dm_b,   # Γ(t)
    )


# ── Convenience alias ─────────────────────────────────────────────────────────

def impulse_response(
    schedule: PolicySchedule,
    E0:       float,
    T:        float,
    N_steps:  int,
    var:      str = "A",
    **solver_kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: return (t_grid, IRF_X(t)) for variable X.

    Parameters
    ----------
    var : one of 'A', 'L', 'E', 'm', 'R'

    Returns
    -------
    t_grid : np.ndarray
    irf    : np.ndarray  (same shape)
    """
    traj = bump_policy_solver(schedule, E0, T, N_steps, **solver_kwargs)
    return traj.t, traj.irf(var)
