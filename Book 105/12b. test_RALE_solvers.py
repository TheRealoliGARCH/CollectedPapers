"""
test_rale_solvers.py
====================
Comprehensive unit and integration tests for rale_solvers.py.

Test organisation
-----------------
  TestBumpParams              — BumpParams construction, evaluation, properties
  TestBumpEvaluation          — numeric values of all three bump families
  TestRaleBisection           — Layer 1: scalar bisection solver
  TestBlockBisection          — Layer 2: network block-bisection solver
  TestBumpPolicySolver        — Layer 3: bump-function policy solver
  TestMultiplierDecomposition — multiplier decomposition and interaction term
  TestImpulseResponse         — impulse-response utility wrapper
  TestTheoreticalProperties   — properties proved in the paper
  TestEdgeCases               — numerical edge cases and error handling

Run with:
    python -m pytest test_rale_solvers.py -v
or:
    python test_rale_solvers.py
"""

import math
import warnings
import numpy as np
import pytest
import sys
sys.path.insert(0, '/home/claude')

from rale_solvers import (
    RALESolverError,
    BankState,
    BisectionDiagnostics,
    rale_bisection,
    NetworkParams,
    NetworkState,
    BlockBisectionDiagnostics,
    block_bisection,
    block_bisection_trajectory,
    BumpKind,
    BumpParams,
    PolicySchedule,
    PolicyTrajectory,
    bump_policy_solver,
    multiplier_decomposition,
    impulse_response,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_params():
    """Standard single-bank parameters used across tests."""
    return dict(R=0.5, E0=0.05, t=1.0)


@pytest.fixture
def simple_network():
    """Minimal 3-bank symmetric network."""
    N = 3
    E0  = np.array([0.05, 0.04, 0.06])
    xi  = np.array([0.10, 0.08, 0.12])
    Theta = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ])
    params = NetworkParams(N=N, E0=E0, xi=xi, Theta=Theta)
    R = np.array([0.5, 0.45, 0.55])
    return params, R


@pytest.fixture
def gaussian_schedule():
    """Standard PolicySchedule with Gaussian bumps for tests."""
    R_base_fn = lambda t: 0.4 * np.exp(0.03 * t)
    ba = BumpParams(center=3.0, width=0.8, height=+0.08, kind=BumpKind.GAUSSIAN)
    bb = BumpParams(center=7.0, width=0.8, height=-0.06, kind=BumpKind.GAUSSIAN)
    return PolicySchedule(R_base_fn, ba, bb)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBumpParams
# ═══════════════════════════════════════════════════════════════════════════════

class TestBumpParams:

    def test_gaussian_construction(self):
        bp = BumpParams(center=2.0, width=1.0, height=0.1, kind=BumpKind.GAUSSIAN)
        assert bp.center == 2.0
        assert bp.width  == 1.0
        assert bp.height == 0.1
        assert bp.kind   == BumpKind.GAUSSIAN

    def test_compact_construction(self):
        bp = BumpParams(center=5.0, width=1.5, height=-0.05, kind=BumpKind.COMPACT)
        assert bp.kind == BumpKind.COMPACT

    def test_cosine_construction(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.07, kind=BumpKind.COSINE)
        assert bp.kind == BumpKind.COSINE

    def test_invalid_width_raises(self):
        with pytest.raises(RALESolverError, match="width must be > 0"):
            BumpParams(center=1.0, width=0.0, height=0.1, kind=BumpKind.GAUSSIAN)

    def test_negative_width_raises(self):
        with pytest.raises(RALESolverError):
            BumpParams(center=1.0, width=-1.0, height=0.1, kind=BumpKind.GAUSSIAN)

    def test_unknown_kind_raises(self):
        with pytest.raises(RALESolverError, match="Unknown bump kind"):
            BumpParams(center=1.0, width=1.0, height=0.1, kind="trapezoid")

    def test_gaussian_peak(self):
        bp = BumpParams(center=2.0, width=1.0, height=0.1, kind=BumpKind.GAUSSIAN)
        assert bp.peak_value == pytest.approx(0.1)

    def test_compact_peak(self):
        bp = BumpParams(center=2.0, width=1.0, height=0.1, kind=BumpKind.COMPACT)
        assert bp.peak_value == pytest.approx(0.1 * math.exp(-1.0))

    def test_cosine_peak(self):
        bp = BumpParams(center=2.0, width=1.0, height=0.1, kind=BumpKind.COSINE)
        assert bp.peak_value == pytest.approx(0.1)

    def test_gaussian_support_infinite(self):
        bp = BumpParams(center=0.0, width=1.0, height=1.0, kind=BumpKind.GAUSSIAN)
        lo, hi = bp.support
        assert lo == -math.inf
        assert hi ==  math.inf

    def test_compact_support_finite(self):
        bp = BumpParams(center=3.0, width=1.2, height=1.0, kind=BumpKind.COMPACT)
        lo, hi = bp.support
        assert lo == pytest.approx(3.0 - 1.2)
        assert hi == pytest.approx(3.0 + 1.2)

    def test_cosine_support_finite(self):
        bp = BumpParams(center=5.0, width=2.0, height=1.0, kind=BumpKind.COSINE)
        lo, hi = bp.support
        assert lo == pytest.approx(3.0)
        assert hi == pytest.approx(7.0)

    def test_overlap_gaussian_always_true(self):
        bp1 = BumpParams(center=0.0, width=1.0, height=1.0, kind=BumpKind.GAUSSIAN)
        bp2 = BumpParams(center=100.0, width=1.0, height=1.0, kind=BumpKind.GAUSSIAN)
        assert bp1.overlaps(bp2) is True

    def test_overlap_compact_non_overlapping(self):
        bp1 = BumpParams(center=2.0, width=0.5, height=1.0, kind=BumpKind.COMPACT)
        bp2 = BumpParams(center=5.0, width=0.5, height=1.0, kind=BumpKind.COMPACT)
        assert bp1.overlaps(bp2) is False

    def test_overlap_compact_overlapping(self):
        bp1 = BumpParams(center=2.0, width=1.5, height=1.0, kind=BumpKind.COMPACT)
        bp2 = BumpParams(center=3.0, width=1.5, height=1.0, kind=BumpKind.COMPACT)
        assert bp1.overlaps(bp2) is True

    def test_gaussian_l2_norm(self):
        h, w = 0.1, 1.0
        bp = BumpParams(center=0.0, width=w, height=h, kind=BumpKind.GAUSSIAN)
        expected = h * w * math.sqrt(2 * math.pi)
        assert bp.l2_norm() == pytest.approx(expected, rel=1e-6)

    def test_cosine_l2_norm(self):
        h, w = 0.1, 2.0
        bp = BumpParams(center=0.0, width=w, height=h, kind=BumpKind.COSINE)
        expected = h * w / math.sqrt(2.0)
        assert bp.l2_norm() == pytest.approx(expected, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBumpEvaluation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBumpEvaluation:

    def test_gaussian_at_center_equals_height(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.08, kind=BumpKind.GAUSSIAN)
        val = bp(np.array([3.0]))
        assert val[0] == pytest.approx(0.08)

    def test_gaussian_symmetric(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.08, kind=BumpKind.GAUSSIAN)
        left  = bp(np.array([2.0]))
        right = bp(np.array([4.0]))
        assert left[0] == pytest.approx(right[0])

    def test_gaussian_tails_nonzero(self):
        # At distance 5σ the value is exp(-12.5) ≈ 3.7e-6, clearly non-zero
        bp = BumpParams(center=0.0, width=1.0, height=1.0, kind=BumpKind.GAUSSIAN)
        far = bp(np.array([5.0]))   # 5 standard deviations: exp(-12.5) > 0
        assert far[0] > 0.0        # Gaussian has infinite support

    def test_compact_zero_outside_support(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.08, kind=BumpKind.COMPACT)
        outside = bp(np.array([1.9, 4.1, -5.0, 20.0]))
        assert np.all(outside == 0.0)

    def test_compact_positive_inside_support(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.08, kind=BumpKind.COMPACT)
        inside = bp(np.array([3.0, 3.5, 2.5]))
        assert np.all(inside > 0.0)

    def test_compact_peak_at_center(self):
        bp = BumpParams(center=3.0, width=1.0, height=0.08, kind=BumpKind.COMPACT)
        val = bp(np.array([3.0]))[0]
        assert val == pytest.approx(0.08 * math.exp(-1.0), rel=1e-6)

    def test_compact_smooth_at_boundary(self):
        """Numerical derivative at boundary should be ~0 (all derivatives vanish)."""
        bp = BumpParams(center=3.0, width=1.0, height=1.0, kind=BumpKind.COMPACT)
        dt = 1e-5
        # One-sided finite difference approaching the left boundary from inside
        t_inner = np.array([3.0 - 1.0 + dt,    # just inside
                            3.0 - 1.0 + 2*dt])  # slightly more inside
        vals = bp(t_inner)
        deriv = abs(float(vals[1] - vals[0]) / dt)
        assert deriv < 1e-3    # should be very small near boundary

    def test_cosine_zero_outside_support(self):
        bp = BumpParams(center=5.0, width=2.0, height=0.06, kind=BumpKind.COSINE)
        outside = bp(np.array([2.9, 7.1, 0.0, 100.0]))
        assert np.all(outside == 0.0)

    def test_cosine_at_center_equals_height(self):
        bp = BumpParams(center=5.0, width=2.0, height=0.06, kind=BumpKind.COSINE)
        val = bp(np.array([5.0]))[0]
        assert val == pytest.approx(0.06)

    def test_cosine_at_boundary_is_zero(self):
        bp = BumpParams(center=5.0, width=2.0, height=0.06, kind=BumpKind.COSINE)
        bd = bp(np.array([3.0, 7.0]))
        assert np.allclose(bd, 0.0, atol=1e-12)

    def test_negative_height_gaussian(self):
        bp = BumpParams(center=0.0, width=1.0, height=-0.05, kind=BumpKind.GAUSSIAN)
        assert bp(np.array([0.0]))[0] == pytest.approx(-0.05)

    def test_negative_height_compact(self):
        bp = BumpParams(center=0.0, width=1.0, height=-0.05, kind=BumpKind.COMPACT)
        val = bp(np.array([0.0]))[0]
        assert val < 0.0

    def test_array_output_shape(self):
        bp = BumpParams(center=0.0, width=1.0, height=1.0, kind=BumpKind.GAUSSIAN)
        t = np.linspace(-3, 3, 200)
        out = bp(t)
        assert out.shape == t.shape


# ═══════════════════════════════════════════════════════════════════════════════
# TestRaleBisection
# ═══════════════════════════════════════════════════════════════════════════════

class TestRaleBisection:

    def test_basic_solve_returns_bankstate(self, default_params):
        state, diag = rale_bisection(**default_params)
        assert isinstance(state, BankState)
        assert isinstance(diag, BisectionDiagnostics)

    def test_balance_sheet_identity(self, default_params):
        state, _ = rale_bisection(**default_params)
        assert state.verify_identity(tol=1e-6)

    def test_fixed_point_residual_small(self, default_params):
        _, diag = rale_bisection(**default_params)
        assert diag.residual < 1e-8

    def test_converged_flag_true(self, default_params):
        _, diag = rale_bisection(**default_params)
        assert diag.converged is True

    def test_money_multiplier_definition(self, default_params):
        state, _ = rale_bisection(**default_params)
        assert state.m == pytest.approx(state.A / state.R, rel=1e-10)

    def test_equity_formula(self, default_params):
        state, _ = rale_bisection(**default_params)
        expected_E = 0.05 * (1.0 + state.A) ** 1.0
        assert state.E == pytest.approx(expected_E, rel=1e-8)

    def test_t_zero_closed_form(self):
        R, E0 = 0.4, 0.05
        state, diag = rale_bisection(R=R, E0=E0, t=0.0)
        expected_A = R * (1 - E0) / (1 - R)
        assert state.A == pytest.approx(expected_A, rel=1e-10)
        assert diag.iterations == 0    # closed-form: no iteration needed

    def test_assets_positive(self, default_params):
        state, _ = rale_bisection(**default_params)
        assert state.A > 0.0

    def test_equity_positive(self, default_params):
        state, _ = rale_bisection(**default_params)
        assert state.E > 0.0

    def test_monotone_in_R(self):
        """Higher reserves → higher assets (Theorem 3.1 implication)."""
        E0, t = 0.05, 2.0
        R_vals = np.linspace(0.2, 0.8, 10)
        A_vals = []
        for R in R_vals:
            state, _ = rale_bisection(R=R, E0=E0, t=t)
            A_vals.append(state.A)
        assert all(A_vals[i] < A_vals[i+1] for i in range(len(A_vals)-1))

    def test_different_tolerances_consistent(self):
        """Tighter tolerance should give same result to within loose tolerance."""
        s_loose, _ = rale_bisection(R=0.5, E0=0.05, t=1.0, tol=1e-4)
        s_tight, _ = rale_bisection(R=0.5, E0=0.05, t=1.0, tol=1e-9)
        assert abs(s_loose.A - s_tight.A) < 1e-4

    def test_invalid_R_below_zero(self):
        with pytest.raises(RALESolverError, match="Reserve ratio"):
            rale_bisection(R=-0.1, E0=0.05, t=1.0)

    def test_invalid_R_above_one(self):
        with pytest.raises(RALESolverError, match="Reserve ratio"):
            rale_bisection(R=1.1, E0=0.05, t=1.0)

    def test_invalid_E0_zero(self):
        with pytest.raises(RALESolverError, match="equity"):
            rale_bisection(R=0.5, E0=0.0, t=1.0)

    def test_invalid_t_negative(self):
        with pytest.raises(RALESolverError, match="Time"):
            rale_bisection(R=0.5, E0=0.05, t=-1.0)

    def test_invalid_tol(self):
        with pytest.raises(RALESolverError, match="Tolerance"):
            rale_bisection(R=0.5, E0=0.05, t=1.0, tol=0.0)

    def test_with_interbank_asset(self):
        """X_eff > 0 should increase assets relative to isolated case."""
        s_iso,  _ = rale_bisection(R=0.5, E0=0.05, t=1.0, X_eff=0.0)
        s_net,  _ = rale_bisection(R=0.5, E0=0.05, t=1.0, X_eff=0.05, P=1.0)
        assert s_net.A > s_iso.A

    def test_asset_price_reduction(self):
        """Lower P (fire-sale) should decrease assets."""
        s_high, _ = rale_bisection(R=0.5, E0=0.05, t=1.0, X_eff=0.1, P=1.0)
        s_low,  _ = rale_bisection(R=0.5, E0=0.05, t=1.0, X_eff=0.1, P=0.8)
        assert s_low.A < s_high.A

    def test_many_parameter_combinations(self):
        """Solver should converge for a grid of (R, E0, t) values."""
        failures = 0
        for R in np.linspace(0.1, 0.9, 6):
            for E0 in [0.01, 0.05, 0.10]:
                for t in [0.0, 0.5, 1.0, 3.0, 5.0]:
                    try:
                        state, diag = rale_bisection(R=R, E0=E0, t=t)
                        assert diag.converged, f"Not converged: R={R}, E0={E0}, t={t}"
                        assert state.verify_identity(), f"Identity fails: R={R}, E0={E0}, t={t}"
                    except RALESolverError:
                        failures += 1
        assert failures == 0, f"{failures} parameter combinations failed."

    def test_iteration_count_bounded(self):
        """Should converge in ≤ 34 iterations for default bracket and tol=1e-9."""
        _, diag = rale_bisection(R=0.5, E0=0.05, t=2.0, tol=1e-9)
        expected_max = math.ceil(math.log2((200.0 - (-0.999)) / 1e-9))
        assert diag.iterations <= expected_max


# ═══════════════════════════════════════════════════════════════════════════════
# TestBlockBisection
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlockBisection:

    def test_basic_returns_network_state(self, simple_network):
        params, R = simple_network
        state, diag = block_bisection(R, t=1.0, params=params)
        assert isinstance(state, NetworkState)
        assert isinstance(diag, BlockBisectionDiagnostics)

    def test_converged(self, simple_network):
        params, R = simple_network
        _, diag = block_bisection(R, t=1.0, params=params)
        assert diag.converged is True

    def test_balance_sheet_identity_all_banks(self, simple_network):
        params, R = simple_network
        state, _ = block_bisection(R, t=1.0, params=params)
        residuals = np.abs(state.A - state.L - state.E)
        assert np.all(residuals < 1e-6)

    def test_output_shapes(self, simple_network):
        params, R = simple_network
        N = params.N
        state, diag = block_bisection(R, t=1.0, params=params)
        assert state.A.shape       == (N,)
        assert state.L.shape       == (N,)
        assert state.E.shape       == (N,)
        assert state.m.shape       == (N,)
        assert state.recovery.shape == (N,)
        assert state.X_eff.shape   == (N,)
        assert len(diag.inner_diags) == N

    def test_recovery_rates_bounded(self, simple_network):
        params, R = simple_network
        state, _ = block_bisection(R, t=1.0, params=params)
        assert np.all(state.recovery >= 0.0)
        assert np.all(state.recovery <= 1.0)

    def test_price_in_0_1(self, simple_network):
        params, R = simple_network
        state, _ = block_bisection(R, t=1.0, params=params)
        assert 0.0 < state.P <= params.P0

    def test_no_interbank_equals_isolated(self):
        """With xi=0 (no interbank book) and lambda_=0, each bank is isolated."""
        N  = 3
        E0 = np.array([0.05, 0.04, 0.06])
        xi = np.zeros(N)
        Theta = np.array([[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0.]])
        params = NetworkParams(N=N, E0=E0, xi=xi, Theta=Theta, lambda_=0.0)
        R = np.array([0.5, 0.45, 0.55])
        t = 2.0

        state_net, _ = block_bisection(R, t=t, params=params)
        for i in range(N):
            state_iso, _ = rale_bisection(R[i], E0[i], t)
            assert state_net.A[i] == pytest.approx(state_iso.A, rel=1e-5)

    def test_invalid_R_shape(self, simple_network):
        params, R = simple_network
        with pytest.raises(RALESolverError, match="shape"):
            block_bisection(R[:2], t=1.0, params=params)

    def test_invalid_R_out_of_range(self, simple_network):
        params, _ = simple_network
        R_bad = np.array([0.5, 0.5, 1.5])
        with pytest.raises(RALESolverError, match="reserve ratios"):
            block_bisection(R_bad, t=1.0, params=params)

    def test_warm_start_gives_same_result(self, simple_network):
        params, R = simple_network
        state_cold, _ = block_bisection(R, t=1.0, params=params, A_init=None)
        state_warm, _ = block_bisection(R, t=1.0, params=params,
                                         A_init=state_cold.A)
        assert np.allclose(state_cold.A, state_warm.A, atol=1e-7)

    def test_trajectory_length(self, simple_network):
        params, _ = simple_network
        N_steps  = 20
        t_grid   = np.linspace(0, 5, N_steps + 1)
        R_path   = np.tile(np.array([0.5, 0.45, 0.55]), (N_steps+1, 1))
        states   = block_bisection_trajectory(R_path, t_grid, params)
        assert len(states) == N_steps + 1

    def test_trajectory_identity_at_all_steps(self, simple_network):
        params, _ = simple_network
        N_steps  = 10
        t_grid   = np.linspace(0, 2, N_steps + 1)
        R_path   = np.tile(np.array([0.5, 0.45, 0.55]), (N_steps+1, 1))
        states   = block_bisection_trajectory(R_path, t_grid, params)
        for s in states:
            residuals = np.abs(s.A - s.L - s.E)
            assert np.all(residuals < 1e-5)

    def test_fire_sale_zero_when_well_capitalised(self, simple_network):
        """Healthy banks (E >> epsilon_E) should not trigger fire sales."""
        params, R = simple_network
        # Initial equity E0 >> epsilon_E=0.02, so no distress at t=0
        state, _ = block_bisection(R, t=0.0, params=params)
        assert state.FS == pytest.approx(0.0, abs=1e-10)


# ═══════════════════════════════════════════════════════════════════════════════
# TestBumpPolicySolver
# ═══════════════════════════════════════════════════════════════════════════════

class TestBumpPolicySolver:

    def test_returns_policy_trajectory(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        assert isinstance(traj, PolicyTrajectory)

    def test_output_array_lengths(self, gaussian_schedule):
        N_steps = 100
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=N_steps)
        expected = N_steps + 1
        for arr in (traj.t, traj.A_perturbed, traj.A_baseline,
                    traj.E_perturbed, traj.E_baseline, traj.phi_a, traj.phi_b):
            assert len(arr) == expected

    def test_all_converged(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        summary = traj.convergence_summary()
        assert summary['all_converged'] is True

    def test_no_bump_no_irf(self):
        """With zero-height bumps, IRF should be exactly zero."""
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=3.0, width=1.0, height=0.0, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=7.0, width=1.0, height=0.0, kind=BumpKind.GAUSSIAN)
        sched = PolicySchedule(R_base_fn, ba, bb, eps=1e-6)
        traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=100,
                                  warn_admissibility=False)
        assert np.allclose(traj.irf('A'), 0.0, atol=1e-8)

    def test_expansionary_bump_raises_assets(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=200)
        irf_A = traj.irf('A')
        assert irf_A.max() > 0.0, "Expansionary bump should raise assets."

    def test_contractionary_bump_lowers_assets(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=200)
        irf_A = traj.irf('A')
        assert irf_A.min() < 0.0, "Contractionary bump should lower assets."

    def test_balance_sheet_identity_perturbed(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        residuals = np.abs(traj.A_perturbed - traj.L_perturbed - traj.E_perturbed)
        assert np.all(residuals < 1e-6)

    def test_balance_sheet_identity_baseline(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        residuals = np.abs(traj.A_baseline - traj.L_baseline - traj.E_baseline)
        assert np.all(residuals < 1e-6)

    def test_money_multiplier_definition_perturbed(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        m_check = traj.A_perturbed / traj.R_perturbed
        assert np.allclose(traj.m_perturbed, m_check, rtol=1e-8)

    def test_phi_a_and_phi_b_shape(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        assert traj.phi_a.shape == traj.t.shape
        assert traj.phi_b.shape == traj.t.shape

    def test_all_three_bump_kinds(self):
        """Solver should succeed for all bump families."""
        R_base_fn = lambda t: 0.4 * np.exp(0.03 * t)
        for kind in BumpKind.ALL:
            ba = BumpParams(center=3.0, width=1.0, height=+0.06, kind=kind)
            bb = BumpParams(center=7.0, width=1.0, height=-0.04, kind=kind)
            sched = PolicySchedule(R_base_fn, ba, bb)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=100)
            assert traj.convergence_summary()['all_converged']

    def test_invalid_E0(self, gaussian_schedule):
        with pytest.raises(RALESolverError, match="E0"):
            bump_policy_solver(gaussian_schedule, E0=1.5, T=10.0, N_steps=100)

    def test_invalid_T(self, gaussian_schedule):
        with pytest.raises(RALESolverError, match="T"):
            bump_policy_solver(gaussian_schedule, E0=0.05, T=-1.0, N_steps=100)

    def test_invalid_N_steps(self, gaussian_schedule):
        with pytest.raises(RALESolverError, match="N_steps"):
            bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=0)

    def test_bump_b_before_bump_a_raises(self):
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=7.0, width=1.0, height=+0.06, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=3.0, width=1.0, height=-0.04, kind=BumpKind.GAUSSIAN)
        with pytest.raises(RALESolverError, match="bump_b.center"):
            PolicySchedule(R_base_fn, ba, bb)

    def test_peak_irf_returns_float(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=200)
        val, t_peak = traj.peak_irf('A')
        assert isinstance(val, float)
        assert isinstance(t_peak, float)

    def test_long_run_residual_near_zero_compact(self):
        """Compact bumps with support well inside [0,T]: Δ A(T) ≈ 0 (Theorem 6.1)."""
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=2.0, width=0.8, height=+0.06, kind=BumpKind.COMPACT)
        bb = BumpParams(center=6.0, width=0.8, height=-0.04, kind=BumpKind.COMPACT)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=500)
        lr = abs(traj.long_run_residual('A'))
        assert lr < 1e-6, f"Long-run residual of A should be ~0; got {lr:.2e}"

    def test_instantaneous_multiplier_positive(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=200)
        mult = traj.instantaneous_policy_multiplier()
        valid = mult[~np.isnan(mult)]
        assert np.all(valid > 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# TestMultiplierDecomposition
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiplierDecomposition:

    def _compact_non_overlapping_schedule(self):
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=2.0, width=0.5, height=+0.06, kind=BumpKind.COMPACT)
        bb = BumpParams(center=7.0, width=0.5, height=-0.04, kind=BumpKind.COMPACT)
        return PolicySchedule(R_base_fn, ba, bb)

    def test_returns_expected_keys(self):
        sched = self._compact_non_overlapping_schedule()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=100)
        expected_keys = {'t', 'dm_a', 'dm_b', 'dm_ab', 'dm_linear', 'interaction'}
        assert expected_keys == set(result.keys())

    def test_interaction_zero_non_overlapping_compact(self):
        """Proposition 5.1: Γ ≡ 0 for non-overlapping compact bumps."""
        sched = self._compact_non_overlapping_schedule()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=500)
        assert np.allclose(dec['interaction'], 0.0, atol=1e-8), \
            f"Max |Γ(t)| = {np.max(np.abs(dec['interaction'])):.2e}"

    def test_combined_equals_linear_plus_interaction(self):
        R_base_fn = lambda t: 0.4 * np.exp(0.02 * t)
        ba = BumpParams(center=3.0, width=0.8, height=+0.05, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=7.0, width=0.8, height=-0.04, kind=BumpKind.GAUSSIAN)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=200)
        reconstructed = dec['dm_linear'] + dec['interaction']
        assert np.allclose(reconstructed, dec['dm_ab'], atol=1e-8)

    def test_interaction_nonzero_gaussian(self):
        """Gaussian bumps overlap → interaction should be nonzero."""
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=5.0, width=2.0, height=+0.06, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=5.5, width=2.0, height=-0.04, kind=BumpKind.GAUSSIAN)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=200)
        assert np.max(np.abs(dec['interaction'])) > 1e-10

    def test_output_array_lengths(self):
        sched = self._compact_non_overlapping_schedule()
        N_steps = 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=N_steps)
        for k, v in dec.items():
            assert len(v) == N_steps + 1, f"Array {k} has wrong length."


# ═══════════════════════════════════════════════════════════════════════════════
# TestImpulseResponse
# ═══════════════════════════════════════════════════════════════════════════════

class TestImpulseResponse:

    def test_returns_two_arrays(self, gaussian_schedule):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, irf = impulse_response(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        assert len(t) == 101
        assert len(irf) == 101

    def test_irf_A_default(self, gaussian_schedule):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t, irf = impulse_response(gaussian_schedule, E0=0.05, T=10.0, N_steps=100)
        assert isinstance(irf, np.ndarray)

    def test_irf_variable_selection(self, gaussian_schedule):
        for var in ('A', 'L', 'E', 'm', 'R'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t, irf = impulse_response(
                    gaussian_schedule, E0=0.05, T=10.0, N_steps=100, var=var
                )
            assert irf.shape == t.shape

    def test_irf_invalid_variable(self, gaussian_schedule):
        with pytest.raises(RALESolverError, match="Unknown variable"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                impulse_response(
                    gaussian_schedule, E0=0.05, T=10.0, N_steps=100, var='X'
                )


# ═══════════════════════════════════════════════════════════════════════════════
# TestTheoreticalProperties  (properties proven in the papers)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTheoreticalProperties:

    def test_theorem31_original_existence_uniqueness(self):
        """
        Theorem 3.1 (original paper): for any R ∈ (0,1) and E0 ∈ (0,1),
        a unique A* exists. Test over a dense grid.
        """
        R_grid  = np.linspace(0.05, 0.95, 20)
        E0_grid = [0.01, 0.05, 0.10, 0.20]
        t_grid  = [0.0, 0.1, 1.0, 5.0]
        n_fail = 0
        for R in R_grid:
            for E0 in E0_grid:
                for t in t_grid:
                    try:
                        state, diag = rale_bisection(R=R, E0=E0, t=t)
                        assert diag.converged
                        assert state.verify_identity(tol=1e-5)
                    except Exception:
                        n_fail += 1
        assert n_fail == 0

    def test_corollary31_smoothness_A_in_t(self):
        """
        Corollary 3.2 (original paper): A*(t) is smooth in t.
        Check numerically via second-order finite differences.
        """
        R, E0 = 0.5, 0.05
        ts = np.linspace(0.5, 5.0, 200)
        A_vals = np.array([rale_bisection(R, E0, t)[0].A for t in ts])
        # Finite-difference second derivative — should not have spikes
        d2A = np.diff(A_vals, 2)
        dt  = (ts[1] - ts[0])
        d2A_normalised = d2A / dt**2
        assert np.all(np.abs(d2A_normalised) < 200), \
            "Large second derivative suggests non-smoothness."

    def test_proposition41_asymmetric_transmission(self):
        """
        Proposition 4.1 (bump paper): |peak IRF under φ_a| > |trough IRF under φ_b|
        for equal and opposite amplitudes.
        """
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        h = 0.07
        # φ_a expansionary, φ_b contractionary, equal amplitude, symmetric timing
        ba = BumpParams(center=3.0, width=1.0, height=+h, kind=BumpKind.COMPACT)
        bb = BumpParams(center=7.0, width=1.0, height=-h, kind=BumpKind.COMPACT)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=500)
        irf_A = traj.irf('A')
        peak   = float(irf_A.max())
        trough = float(irf_A.min())
        assert abs(peak) > abs(trough), (
            f"Expected |peak| {abs(peak):.4f} > |trough| {abs(trough):.4f}. "
            "Asymmetric transmission not confirmed."
        )

    def test_theorem61_long_run_neutrality_assets(self):
        """
        Theorem 6.1(i): compact bump with support in (0, T) → ΔA(T) = 0.
        """
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        ba = BumpParams(center=2.0, width=0.5, height=+0.07, kind=BumpKind.COMPACT)
        bb = BumpParams(center=7.0, width=0.5, height=-0.05, kind=BumpKind.COMPACT)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=1000)
        lr_A = abs(traj.long_run_residual('A'))
        assert lr_A < 1e-6, f"ΔA(T) = {lr_A:.2e} should be ≈ 0."

    def test_proposition51_interaction_zero_non_overlapping(self):
        """
        Proposition 5.1: interaction Γ(t) ≡ 0 for non-overlapping compact bumps.
        """
        R_base_fn = lambda t: 0.4 * np.exp(0.02 * t)
        ba = BumpParams(center=2.0, width=0.6, height=+0.05, kind=BumpKind.COMPACT)
        bb = BumpParams(center=8.0, width=0.6, height=-0.04, kind=BumpKind.COMPACT)
        sched = PolicySchedule(R_base_fn, ba, bb)
        assert not sched.supports_overlap, "Supports should not overlap."
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = multiplier_decomposition(sched, E0=0.05, T=10.0, N_steps=500)
        max_gamma = float(np.max(np.abs(dec['interaction'])))
        assert max_gamma < 1e-8, f"Max |Γ(t)| = {max_gamma:.2e} should be ≈ 0."

    def test_block_bisection_theorem_nonneutral_equity(self):
        """
        Theorem 6.1(ii) analogue: during [a+w, b-w], equity is non-neutral.
        Use Gaussian bumps (infinite support) to ensure visible IRF_E in the window.
        """
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        # Gaussian bumps: φ_a raises reserves around t=2, φ_b lowers around t=7
        # The Gaussian tails ensure E is perturbed between the two centers.
        ba = BumpParams(center=2.0, width=0.8, height=+0.07, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=7.0, width=0.8, height=-0.05, kind=BumpKind.GAUSSIAN)
        sched = PolicySchedule(R_base_fn, ba, bb)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traj = bump_policy_solver(sched, E0=0.05, T=10.0, N_steps=500)
        irf_E = traj.irf('E')
        # In the window (3, 6) between the two interventions, equity should be elevated
        mask = (traj.t > 3.0) & (traj.t < 6.0)
        assert np.any(np.abs(irf_E[mask]) > 1e-6), \
            "Equity should be non-neutral between interventions."


# ═══════════════════════════════════════════════════════════════════════════════
# TestEdgeCases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_very_small_E0(self):
        """Very small equity: solver should still converge."""
        state, diag = rale_bisection(R=0.5, E0=1e-4, t=1.0)
        assert diag.converged
        assert state.verify_identity()

    def test_very_large_t(self):
        """Large t: equity compounding is strong; solver must still converge."""
        state, diag = rale_bisection(R=0.5, E0=0.05, t=10.0)
        assert diag.converged
        assert state.A > 0.0

    def test_R_close_to_zero(self):
        state, diag = rale_bisection(R=0.01, E0=0.05, t=1.0)
        assert diag.converged

    def test_R_close_to_one(self):
        state, diag = rale_bisection(R=0.99, E0=0.05, t=1.0)
        assert diag.converged

    def test_zero_height_bump_evaluates_to_zero(self):
        bp = BumpParams(center=5.0, width=1.0, height=0.0, kind=BumpKind.GAUSSIAN)
        t = np.linspace(0, 10, 100)
        assert np.allclose(bp(t), 0.0)

    def test_compact_bump_boundary_exactly_zero(self):
        bp = BumpParams(center=5.0, width=2.0, height=1.0, kind=BumpKind.COMPACT)
        boundary = np.array([3.0, 7.0])
        assert np.allclose(bp(boundary), 0.0, atol=1e-10)

    def test_policy_schedule_admissibility_check(self):
        R_base_fn = lambda t: 0.4 * np.ones_like(t)
        # Enormous amplitude: clearly inadmissible
        ba = BumpParams(center=3.0, width=1.0, height=+5.0, kind=BumpKind.GAUSSIAN)
        bb = BumpParams(center=7.0, width=1.0, height=-5.0, kind=BumpKind.GAUSSIAN)
        sched = PolicySchedule(R_base_fn, ba, bb)
        assert not sched.admissible_amplitude()

    def test_single_step_simulation(self, gaussian_schedule):
        traj = bump_policy_solver(gaussian_schedule, E0=0.05, T=10.0, N_steps=1)
        assert len(traj.t) == 2

    def test_network_params_diagonal_check(self):
        N = 2
        Theta_bad = np.array([[0.5, 0.5], [0.5, 0.5]])  # diagonal non-zero
        E0 = np.array([0.05, 0.05])
        xi = np.array([0.1, 0.1])
        with pytest.raises(AssertionError):
            NetworkParams(N=N, E0=E0, xi=xi, Theta=Theta_bad)

    def test_block_bisection_t_zero(self, simple_network):
        params, R = simple_network
        state, diag = block_bisection(R, t=0.0, params=params)
        assert diag.converged
        residuals = np.abs(state.A - state.L - state.E)
        assert np.all(residuals < 1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        capture_output=False,
    )
    sys.exit(result.returncode)
