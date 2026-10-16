"""G1/G2-lite: compare CFD vs ideal-1D engine performance and inject CFD maps into the flight loop.

Reads runs/cfd/cfd_summary.json + runs/cfd/maps/*_cfd.json (from run_cfd_batch.py).
Writes runs/cfd_experiments/{results.json, summary.md}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rocket_dt.evaluation import experiments as ex  # noqa: E402
from rocket_dt.evaluation.artifacts import _default, software_versions  # noqa: E402
from rocket_dt.propulsion import nozzle1d  # noqa: E402
from rocket_dt.propulsion.performance_map import EnginePerformanceMap  # noqa: E402
from rocket_dt.schemas import load_engine  # noqa: E402


def main(n_suite: int = 30):
    out = REPO / "runs/cfd_experiments"
    out.mkdir(parents=True, exist_ok=True)
    cs = json.loads((REPO / "runs/cfd/cfd_summary.json").read_text())
    perf = []
    for name, info in cs["maps"].items():
        e = load_engine(REPO / f"configs/engines/{name}.yaml")
        m = EnginePerformanceMap.load(info["path"])
        nom = m.provenance["nominal"]
        for thr, pa in ((1.0, 101325.0), (0.6, 101325.0), (1.0, 50000.0)):
            f_c, md_c, isp_c, ok_c = m.lookup(thr, pa)
            p1 = nozzle1d.evaluate(e, thr, pa)
            perf.append({"engine": name, "throttle": thr, "pa": pa, "F_cfd": f_c, "F_1d": p1.thrust_n,
                         "dF_rel": (f_c - p1.thrust_n) / p1.thrust_n, "mdot_cfd": md_c, "mdot_1d": p1.mass_flow_kg_s,
                         "dmdot_rel": (md_c - p1.mass_flow_kg_s) / p1.mass_flow_kg_s, "isp_cfd": isp_c,
                         "isp_1d": p1.isp_s, "valid_cfd": ok_c, "valid_1d": p1.valid})
        perf.append({"engine": name, "pe_cfd_nominal": nom["exit_p_avg_pa"],
                     "pe_1d_nominal": nozzle1d.evaluate(e, 1.0, 101325.0).exit_pressure_pa,
                     "gci_stream_thrust": cs["gci"].get(name, {}).get("stream_thrust")})
    rows, suites = ex.cfd_vs_1d(REPO, 4, n_suite)
    res = {"software": software_versions(), "cfd_summary": cs, "performance": perf, "flight": rows, "suites": suites}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=_default))
    L = ["# CFD vs ideal-1D (inviscid wedge CFD, OpenFOAM version in cfd_summary.json)\n",
         "| engine | thr | Pa [Pa] | F CFD [N] | F 1D [N] | dF | mdot CFD | mdot 1D | dmdot | Isp CFD | Isp 1D | valid CFD/1D |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in perf:
        if "F_cfd" in p:
            L.append(f"| {p['engine']} | {p['throttle']} | {p['pa']:.0f} | {p['F_cfd']:.0f} | {p['F_1d']:.0f} | {100 * p['dF_rel']:+.2f}% | "
                     f"{p['mdot_cfd']:.3f} | {p['mdot_1d']:.3f} | {100 * p['dmdot_rel']:+.2f}% | {p['isp_cfd']:.1f} | {p['isp_1d']:.1f} | "
                     f"{p['valid_cfd']}/{p['valid_1d']} |")
    L += ["\n| engine | Pe CFD (area avg) | Pe 1D | GCI stream thrust (fine) | observed order p | asymptotic ratio |", "|---|---|---|---|---|---|"]
    for p in perf:
        if "pe_cfd_nominal" in p:
            g = p["gci_stream_thrust"] or {}
            L.append(f"| {p['engine']} | {p['pe_cfd_nominal']:.0f} | {p['pe_1d_nominal']:.0f} | "
                     f"{100 * g.get('gci_fine', float('nan')):.3f}% | {g.get('p', float('nan')):.2f} | {g.get('asymptotic_ratio', float('nan')):.3f} |")
    L += ["\n## Flight with 1D map vs CFD map (PD baseline, Python reference integrator)\n",
          "| map | engine | scenario | result | physics valid | fuel [kg] | mean throttle | pos RMS / td pos err [m] |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r["metrics"]
        td = m.get("touchdown") or {}
        L.append(f"| {r['experiment']} | {r['engine']} | {r['scenario']} | {m['termination_reason']} | {m['physics_valid']} | "
                 f"{m['fuel_used_kg']:.2f} | {m['mean_throttle_engine_on']:.4f} | {m.get('pos_rms_m', td.get('pos_err', float('nan'))):.3f} |")
    L += ["\n## Landing suite (seeds 1000+)\n", "| map | engine | success | success & valid | fuel mean [kg] | td pos err P95 [m] |", "|---|---|---|---|---|---|"]
    for src, s in suites.items():
        for e, v in s.items():
            L.append(f"| {src} | {e} | {v['n_success']}/{v['n']} | {v['n_success_and_physics_valid']}/{v['n']} | {v['fuel_mean_kg']:.2f} | {v['td_pos_err_p95']:.2f} |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
