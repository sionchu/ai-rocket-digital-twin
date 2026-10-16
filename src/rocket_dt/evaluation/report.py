"""Run the P0 experiment set and write machine + human readable summaries."""
from __future__ import annotations

import json
import time
from pathlib import Path

from rocket_dt.evaluation import experiments as ex
from rocket_dt.evaluation.artifacts import _default, run_id, software_versions, write_run
from rocket_dt.schemas import RunManifest, load_vehicle

LABEL = "ideal-1D map (not CFD-verified), Python RK4 reference integrator, PD baseline"


def run_all(root: Path, out: Path, n_suite: int = 30, workers: int = 4):
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    veh = load_vehicle(root / "configs/vehicle_v0.yaml")
    sw = software_versions()
    e0 = ex.e0_sensitivity(root)
    e12 = ex.e1_e2(root, workers)
    for row, log, ep in e12:
        man = RunManifest(run_id(row["experiment"], row["scenario"], row["engine"]), "experiments",
                          [row["engine"]], veh.content_hash(), {row["engine"]: row["map_hash"]}, sw,
                          extra={"mass_engine_id": row["mass_engine"], "label": LABEL})
        write_run(out / "runs", man, row["metrics"], log, ep)
    suites = {
        "S3_nominal": ex.landing_suite(root, n_suite, workers)[1],
        "S3_rcs_01_failed_known": ex.landing_suite(root, n_suite, workers, rcs_failed=("rcs_01",))[1],
        "S3_rcs_01_failed_unknown": ex.landing_suite(root, n_suite, workers, rcs_failed=("rcs_01",), fault_known=False)[1],
        "S3_rcs_01_03_failed_known": ex.landing_suite(root, n_suite, workers, rcs_failed=("rcs_01", "rcs_03"))[1],
    }
    result = {"label": LABEL, "software": sw, "vehicle_hash": veh.content_hash(),
              "E0": e0, "E1_E2": [r[0] for r in e12], **suites,
              "suite_seeds": [1000, 1000 + n_suite - 1], "wall_time_s": time.time() - t0}
    (out / "results.json").write_text(json.dumps(result, indent=1, default=_default))
    (out / "summary.md").write_text(_markdown(result))
    print((out / "summary.md").read_text())
    return result


def _f(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def _markdown(r) -> str:
    L = [f"# P0 experiment summary\n\nLabel: **{r['label']}**. Commit `{r['software']['rocket_dt_commit']}`.\n",
         "## E0 open-loop sensitivity (throttle 1.0, gimbal_y 0.035 rad, sea level, full tanks)\n",
         "| engine | F [N] | mdot [kg/s] | engine mass [kg] | CoM z [m] | Ixx [kg m2] | a_z [m/s2] | alpha_y [rad/s2] | map valid |",
         "|---|---|---|---|---|---|---|---|---|"]
    for e in r["E0"]:
        L.append(f"| {e['engine']} | {e['thrust_n']:.0f} | {e['mdot_kg_s']:.3f} | {e['engine_mass_kg']:.2f} | "
                 f"{e['com_z_m']:.4f} | {e['Ixx_kg_m2']:.1f} | {e['acc_z_m_s2']:.4f} | {e['ang_acc_y_rad_s2']:.5f} | {e['map_valid']} |")
    L += ["\n## E1 (engine-A hardware mass, map swapped) and E2 (full engine swap)\n",
          "| exp | engine | scenario | result | physics valid | fuel [kg] | mean throttle | pos RMS / td pos err [m] | td vz [m/s] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for row in r["E1_E2"]:
        m = row["metrics"]
        td = m.get("touchdown") or {}
        pos = m.get("pos_rms_m", td.get("pos_err"))
        L.append(f"| {row['experiment']} | {row['engine']} | {row['scenario']} | {m['termination_reason']} | "
                 f"{m['physics_valid']} ({m['map_invalid_steps']} invalid steps) | {m['fuel_used_kg']:.2f} | "
                 f"{m['mean_throttle_engine_on']:.4f} | {_f(pos, 3)} | {_f(td.get('vz'), 3)} |")
    for key, title in (("S3_nominal", "nominal"), ("S3_rcs_01_failed_known", "rcs_01 failed, fault known to allocator"),
                       ("S3_rcs_01_failed_unknown", "rcs_01 failed silently (allocator unaware)"),
                       ("S3_rcs_01_03_failed_known", "rcs_01 + rcs_03 failed (no +roll authority)")):
        L += [f"\n## S3 randomized landing suite, PD baseline, {title} (seeds {r['suite_seeds'][0]}–{r['suite_seeds'][1]})\n",
              "| engine | success | Wilson 95% | success & physics-valid | td pos err median/P95/max [m] | td |vz| P95 | td vh P95 | max td tilt [deg] | fuel mean/P95 [kg] | failure reasons |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for e, s in r[key].items():
            lo, hi = s["wilson95"]
            L.append(f"| {e} | {s['n_success']}/{s['n']} | [{lo:.2f}, {hi:.2f}] | {s['n_success_and_physics_valid']}/{s['n']} | "
                     f"{_f(s['td_pos_err_median'])}/{_f(s['td_pos_err_p95'])}/{_f(s['td_pos_err_max'])} | "
                     f"{_f(s['td_vz_abs_p95'])} | {_f(s['td_vh_p95'])} | {_f(s['td_tilt_deg_max'])} | "
                     f"{s['fuel_mean_kg']:.1f}/{s['fuel_p95_kg']:.1f} | {', '.join(s['failure_reasons']) or '-'} |")
    return "\n".join(L) + "\n"
