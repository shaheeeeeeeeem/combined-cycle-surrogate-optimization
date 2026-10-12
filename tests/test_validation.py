import itertools
import time

import numpy as np
import pytest

from src.props.gas import AIR, MOLAR_MASS, R_U, SPECIES, T_REF, cp, h, s0
from src.combined_cycle import BASELINE, MARGIN_KEYS, direct_model, direct_model_tau
from src.combustor import ETA_F, F_STOICH, LHV_CH4, product_mixture
from src.gas_turbine import gas_turbine
from src.rankine import rankine
from src.validation import CHECK2_INPUTS, check1, check2, check_textbook_cengel

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


def test_check2_within_steam_tolerance():
    table = check2(csv_name=None)
    failed = table[table["Pass/Fail"] != "Pass"]
    assert failed.empty, failed.to_string()


def test_rankine_energy_closure():
    cycle = rankine(**CHECK2_INPUTS)
    assert cycle["q_in"] - cycle["q_out"] - cycle["w_net"] == pytest.approx(0.0, abs=1e-6)


def test_rankine_ideal_components_are_isentropic():
    inputs = {**CHECK2_INPUTS, "eta_st": 1.0, "eta_pump": 1.0}
    st = rankine(**inputs)["states"]
    assert st["6"]["s"] == pytest.approx(st["5"]["s"], rel=1e-6)
    assert st["8"]["s"] == pytest.approx(st["7"]["s"], rel=1e-6)


def test_rankine_rejects_Ts_at_or_below_saturation():
    with pytest.raises(ValueError):
        rankine(Ps=30e5, Ts=450.0, Pcond=7.5e3)


def test_rankine_rejects_Pcond_at_or_above_Ps():
    with pytest.raises(ValueError):
        rankine(Ps=30e5, Ts=723.15, Pcond=30e5)


def test_rankine_higher_superheat_raises_quality_and_efficiency():
    low = rankine(Ps=30e5, Ts=673.15, Pcond=7.5e3)
    high = rankine(Ps=30e5, Ts=773.15, Pcond=7.5e3)
    assert high["x6"] > low["x6"]
    assert high["eta"] > low["eta"]


def test_f_stoich():
    assert F_STOICH == pytest.approx(0.05839, rel=1e-3)


def test_combustor_energy_closure():
    gt = gas_turbine(rp=15.0, TIT=1500.0)
    st = gt["states"]
    supplied = gt["m_fuel"] * LHV_CH4 * ETA_F + gt["W_c"]
    absorbed = gt["m_gas"] * st["3"]["h"] - gt["m_air"] * st["1"]["h"]
    assert supplied == pytest.approx(absorbed, rel=1e-9)


@pytest.mark.parametrize("f", [0.0, 0.02, F_STOICH])
def test_product_mixture_composition(f):
    products = product_mixture(f)
    assert sum(products.mass_fractions.values()) == pytest.approx(1.0, abs=1e-12)
    assert products.mass_fractions["O2"] >= -1e-15


def test_product_mixture_rejects_rich_mixture():
    with pytest.raises(ValueError):
        product_mixture(1.01 * F_STOICH)


def test_fuel_air_ratio_increases_with_TIT():
    f = [gas_turbine(rp=15.0, TIT=T)["f"] for T in (1200.0, 1300.0, 1400.0, 1500.0)]
    assert np.all(np.diff(f) > 0)


def test_equivalence_ratio_lean_over_design_range():
    phi = [gas_turbine(rp, TIT)["phi"] for rp in (8.0, 15.0, 30.0) for TIT in (1200.0, 1500.0)]
    assert max(phi) < 1.0


def test_gt_efficiency_has_interior_maximum_in_rp():
    eta = [gas_turbine(rp, 1200.0)["eta_GT"] for rp in (6.0, 19.0, 45.0)]
    assert eta[1] > eta[0] and eta[1] > eta[2]


@pytest.mark.parametrize("key, expected", [
    ("f", 0.0210), ("phi", 0.359), ("T4_C", 595.0), ("w_net_kJ", 382.0), ("eta_pct", 36.4),
])
def test_gt_reference_point_tit1500_rp15(key, expected):
    gt = gas_turbine(rp=15.0, TIT=1500.0)
    values = {"f": gt["f"], "phi": gt["phi"], "T4_C": gt["T4"] - 273.15,
              "w_net_kJ": gt["w_net"] / 1e3, "eta_pct": 100 * gt["eta_GT"]}
    assert values[key] == pytest.approx(expected, rel=5e-3)


DESIGN_BOUNDS = {"rp": (8.0, 30.0), "TIT": (1200.0, 1500.0), "Ps": (10e5, 60e5), "tau": (0.0, 1.0),
                 "Pcond": (0.05e5, 0.15e5)}


@pytest.fixture(scope="module")
def baseline():
    return direct_model(**BASELINE)


def test_check3_closure_at_baseline(baseline):
    assert abs(baseline["energy"]["residual_rel"]) < 1e-6


def test_check3_closure_at_random_design_points():
    rng = np.random.default_rng(42)
    lo = np.array([b[0] for b in DESIGN_BOUNDS.values()])
    hi = np.array([b[1] for b in DESIGN_BOUNDS.values()])
    failed, checked = 0, 0
    for x in lo + rng.random((30, 5)) * (hi - lo):
        result = direct_model_tau(*x)
        if "energy" not in result:
            failed += 1
            continue
        checked += 1
        assert abs(result["energy"]["residual_rel"]) < 1e-6, result["reason"]
    print(f"\nCheck-3 random points: {checked} closed, {failed} physics failures")
    assert checked > 0


def test_pinch_verified_over_full_profile(baseline):
    assert baseline["min_dT"] >= 10.0 - 1e-6
    assert baseline["min_dT_location"] == "4b"


def test_direct_model_never_raises_at_design_corners():
    start = time.perf_counter()
    corners = list(itertools.product(*DESIGN_BOUNDS.values()))
    results = [direct_model_tau(*c) for c in corners]
    elapsed = (time.perf_counter() - start) / len(corners)
    print(f"\nmean direct_model_tau wall time: {1e3 * elapsed:.1f} ms per call")
    for result in results:
        assert set(MARGIN_KEYS) <= set(result["margins"])
        assert isinstance(result["feasible"], bool) and isinstance(result["reason"], str)


def test_textbook_cengel_within_tolerance():
    table = check_textbook_cengel(csv_name=None)
    failed = table[table["Pass/Fail"] != "Pass"]
    assert failed.empty, failed.to_string()


@pytest.mark.parametrize("key, expected, scale", [
    ("f", 0.01821, 1.0), ("T4", 533.4, "C"), ("m_s", 14.13, 1.0), ("T4c", 135.5, "C"), ("x6", 0.9191, 1.0),
    ("W_GT_net", 32.53, 1e-6), ("W_ST_net", 13.59, 1e-6), ("eta_cc", 50.66, 100.0),
    ("work_ratio", 0.418, 1.0), ("eps_HRSG", 0.815, 1.0),
])
def test_baseline_sanity_values(baseline, key, expected, scale):
    value = baseline[key] - 273.15 if scale == "C" else baseline[key] * scale
    assert value == pytest.approx(expected, rel=5e-3)


def test_baseline_is_feasible(baseline):
    assert baseline["feasible"], baseline["reason"]


def test_hrsg_limit_only_moves_its_own_margin(baseline):
    relaxed = direct_model(**BASELINE, T_hrsg_max=700.0 + 273.15)
    tight = direct_model(**BASELINE, T_hrsg_max=500.0 + 273.15)
    for other in (relaxed, tight):
        assert other["eta_cc"] == baseline["eta_cc"]
        for k in MARGIN_KEYS:
            if k != "m_hrsg_inlet":
                assert other["margins"][k] == baseline["margins"][k]
    assert relaxed["margins"]["m_hrsg_inlet"] - baseline["margins"]["m_hrsg_inlet"] == pytest.approx(50.0)
    assert relaxed["feasible"] and not tight["feasible"]
    assert tight["reason"] == "m_hrsg_inlet"
