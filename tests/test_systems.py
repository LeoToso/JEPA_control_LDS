import numpy as np
import pytest

from jepa_lds.systems import make_double_mode_system, make_linearized_cartpole_system


@pytest.mark.parametrize("maker", [make_double_mode_system, make_linearized_cartpole_system])
def test_open_loop_unstable(maker):
    s = maker()
    assert s.is_open_loop_unstable()
    assert s.spectral_radius() > 1.0


@pytest.mark.parametrize("maker", [make_double_mode_system, make_linearized_cartpole_system])
def test_stabilizable(maker):
    s = maker()
    assert s.is_stabilizable()


def test_double_mode_fully_controllable():
    s = make_double_mode_system()
    assert s.is_controllable()
    assert s.pbh_uncontrollable_modes() == []


def test_double_mode_observable_through_full_rank_C():
    """A full-row-rank 2D output map should see both modes -- the dual of
    `test_double_mode_fully_controllable`."""
    s = make_double_mode_system(unstable_eig=0.85, stable_eig=0.6)
    rng = np.random.default_rng(0)
    C = rng.standard_normal((3, s.n))  # generic full-column-rank-ish 3x2 output map
    assert s.is_observable(C)
    assert s.pbh_unobservable_modes(C) == []


def test_double_mode_rank_deficient_C_hides_one_mode():
    """A C that only sees ONE modal coordinate should correctly be flagged as
    making the OTHER mode unobservable (regression check for the PBH dual)."""
    s = make_double_mode_system(unstable_eig=0.85, stable_eig=0.6)
    _w, _V, Vinv = s.modal_decomposition()
    idx_dominant = s.unstable_mode_index()
    idx_other = 1 - idx_dominant
    C_bad = Vinv[[idx_dominant]].real  # only sees the dominant mode's modal coordinate
    assert not s.is_observable(C_bad)
    unobservable = s.pbh_unobservable_modes(C_bad)
    assert len(unobservable) == 1
    w, _, _ = s.modal_decomposition()
    assert np.isclose(unobservable[0], w[idx_other])


def test_dlqr_stabilizes_ground_truth():
    for maker in [make_double_mode_system, make_linearized_cartpole_system]:
        s = maker()
        K, _P = s.dlqr()
        rho_cl = s.closed_loop_spectral_radius(K)
        assert rho_cl < 1.0, f"{s.name}: LQR should stabilize the true system, got rho={rho_cl}"


def test_simulate_matches_step():
    s = make_double_mode_system()
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(s.n)
    actions = rng.standard_normal((5, s.m))
    xs = s.simulate(x0, actions)
    assert xs.shape == (6, s.n)
    x_manual = x0
    for t in range(5):
        x_manual = s.step(x_manual, actions[t])
    assert np.allclose(xs[-1], x_manual)


def test_unstable_modal_coordinate_shapes():
    s = make_double_mode_system()
    x = np.random.default_rng(0).standard_normal((100, s.n))
    xi = s.unstable_modal_coordinate(x)
    assert xi.shape == (100,)
