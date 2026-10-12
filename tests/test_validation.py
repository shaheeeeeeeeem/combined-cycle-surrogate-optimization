import numpy as np
import pytest

from src.props.gas import AIR, MOLAR_MASS, R_U, SPECIES, T_REF, cp, h, s0
from src.validation import check1

pytestmark = pytest.mark.filterwarnings("ignore::src.props.gas.A6ExtrapolationWarning")

TEMPS = [300.0, 800.0, 1400.0]


@pytest.mark.parametrize("species", SPECIES)
def test_species_reference_state_is_zero(species):
    assert h(T_REF, species) == 0.0
    assert s0(T_REF, species) == 0.0


def test_air_reference_state_is_zero():
    assert AIR.h(T_REF) == 0.0
    assert AIR.s0(T_REF) == 0.0


@pytest.mark.parametrize("species", SPECIES)
@pytest.mark.parametrize("T", TEMPS)
def test_species_dh_dT_equals_cp(species, T):
    dT = 1e-2
    dh_dT = (h(T + dT, species) - h(T - dT, species)) / (2 * dT)
    assert dh_dT == pytest.approx(cp(T, species), rel=1e-6)


@pytest.mark.parametrize("T", TEMPS)
def test_air_dh_dT_equals_cp(T):
    dT = 1e-2
    dh_dT = (AIR.h(T + dT) - AIR.h(T - dT)) / (2 * dT)
    assert dh_dT == pytest.approx(AIR.cp(T), rel=1e-6)


def test_functions_accept_arrays():
    T = np.array(TEMPS)
    assert AIR.h(T).shape == T.shape
    assert AIR.T_from_h(AIR.h(T)) == pytest.approx(T, abs=1e-6)


@pytest.mark.parametrize("T", TEMPS)
def test_T_from_h_round_trip(T):
    assert AIR.T_from_h(AIR.h(T)) == pytest.approx(T, abs=1e-6)


@pytest.mark.parametrize("T", TEMPS)
def test_T_from_s0_round_trip(T):
    assert AIR.T_from_s0(AIR.s0(T)) == pytest.approx(T, abs=1e-6)


def test_inversion_bracket_keeps_h_monotonic():
    lo, hi = AIR.T_bracket
    T = np.arange(lo, hi + 1.0, 1.0)
    assert np.all(AIR.cp(T) > 0.0)
    assert hi > 1600.0


def test_isentropic_compress_expand_round_trip():
    T1, P1, P2 = 300.0, 100e3, 1500e3
    T2s = AIR.compress(T1, P1, P2, eta=1.0)["T_out"]
    assert AIR.expand(T2s, P2, P1, eta=1.0)["T_out"] == pytest.approx(T1, abs=1e-6)


def test_air_composition():
    assert sum(AIR.mass_fractions.values()) == pytest.approx(1.0, abs=1e-12)
    M_expected = 0.21 * MOLAR_MASS["O2"] + 0.79 * MOLAR_MASS["N2"]
    assert AIR.R_mix == pytest.approx(R_U / M_expected, rel=1e-12)
    assert AIR.R_mix / 1e3 == pytest.approx(0.2882, rel=1e-3)


def test_check1_within_gas_tolerance():
    table = check1(csv_name=None)
    failed = table[table["Pass/Fail"] != "Pass"]
    assert failed.empty, failed.to_string()
