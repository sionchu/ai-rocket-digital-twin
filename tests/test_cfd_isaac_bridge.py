"""CFD tooling (case generator, GCI, CFD map contract) and Isaac wrench bridge, without OpenFOAM/Isaac."""
import json

import numpy as np
import pytest

from rocket_dt.cfd import post
from rocket_dt.cfd.case_generator import level_cells, write_case
from rocket_dt.cfd.cfd_map import build_cfd_map
from rocket_dt.dynamics.wrench import PointForce, total_wrench_body
from rocket_dt.isaac_env.wrench_bridge import batched_wrench
from rocket_dt.propulsion import nozzle1d


def test_case_generator_files_and_levels(tmp_path, engines):
    case = write_case(engines["bell_B"], tmp_path, level=2)
    for f in ("system/blockMeshDict", "system/controlDict", "system/fvSchemes", "system/fvSolution",
              "constant/thermophysicalProperties", "0/p", "0/T", "0/U", "case_meta.json"):
        assert (case / f).exists(), f
    meta = json.loads((case / "case_meta.json").read_text())
    assert meta["exit_area_m2"] == pytest.approx(engines["bell_B"].exit_area_m2)
    bm = (case / "system/blockMeshDict").read_text()
    assert "wedge" in bm and f"({level_cells(2)['nx_div']} {level_cells(2)['nr']} 1)" in bm
    assert level_cells(3)["nr"] / level_cells(2)["nr"] == pytest.approx(1.5, rel=0.03)


def test_gci_recovers_known_order():
    # f(h) = f0 + C h^2 sampled at h, 1.5h, 2.25h
    f0, C, h = 100.0, 3.0, 0.1
    f1, f2, f3 = (f0 + C * (h * s) ** 2 for s in (1.0, 1.5, 2.25))
    g = post.gci(f1, f2, f3, 1.5, 1.5)
    assert g["p"] == pytest.approx(2.0, rel=1e-6)
    assert g["f_extrap"] == pytest.approx(f0, rel=1e-9)
    assert g["asymptotic_ratio"] == pytest.approx(1.0, rel=1e-3)  # ~1 in the asymptotic range


def test_cfd_map_matches_1d_when_fed_1d_numbers(engines):
    e = engines["conical_A"]
    p = nozzle1d.evaluate(e, 1.0, 0.0)  # vacuum point -> stream thrust (with lambda)
    finest = {"history": [{"stream_thrust_n": p.thrust_n, "mdot_out": p.mass_flow_kg_s,
                           "exit_p_avg_pa": p.exit_pressure_pa, "exit_area_m2": e.exit_area_m2}],
              "case": "synthetic", "level": 0, "openfoam_version": "none", "qa": {"status": "PASS"}}
    m = build_cfd_map(e, finest, {})
    for thr, pa in ((1.0, 101325.0), (0.6, 80000.0)):
        f, mdot, isp, ok = m.lookup(thr, pa)
        ref = nozzle1d.evaluate(e, thr, pa)
        assert f == pytest.approx(ref.thrust_n, rel=1e-9)
        assert mdot == pytest.approx(ref.mass_flow_kg_s, rel=1e-9)
        assert ok == ref.valid


def test_batched_wrench_matches_reference():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(5, 3, 3)) * 100
    P = rng.normal(size=(5, 3, 3))
    com = rng.normal(size=(5, 3)) * 0.1
    f, tau = batched_wrench(F, P, com)
    for i in range(5):
        fr, tr = total_wrench_body([PointForce(str(k), F[i, k], P[i, k]) for k in range(3)], com[i])
        assert np.allclose(f[i, 0], fr) and np.allclose(tau[i, 0], tr)
