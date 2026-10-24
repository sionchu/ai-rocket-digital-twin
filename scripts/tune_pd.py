"""Re-tune the CascadedPD per scenario (ADR-0011) on the Python reference integrator, then check on held-out seeds.

Cross-entropy search in log-gain space over PDGains (attitude, position, descent profile, limits). Tuning seeds
5000+ (training distribution of isaac_env.scenarios.eval_scenarios), never the evaluation seeds 1000/2000.
Physics v1: quasi-steady aero (wind as a velocity), true variable mass (the reference integrator), and the
engine's CFD map (cloud-v1912 maps exist for conical_A and bell_B). Label: cloud-v1912 (reference integrator,
not Isaac; the Isaac numbers come from scripts/ppo_eval.py --controller pd --pd_gains on the PC).

  python scripts/tune_pd.py --scenario wind            -> runs/pd_tuning/wind.json
  python scripts/tune_pd.py --scenario wind --check 60 -> adds default-vs-tuned on seeds 1000..1059
"""
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import yaml

from rocket_dt.control.baseline import PDGains
from rocket_dt.evaluation.experiments import load_all
from rocket_dt.evaluation.simulate import Scenario, run_scenario
from rocket_dt.isaac_env.scenarios import eval_scenarios

ROOT = Path(__file__).resolve().parents[1]
# name: (low, high) search bounds; max_tilt in degrees here, radians in PDGains
BOUNDS = {"kp_pos": (0.1, 2.0), "kd_pos": (0.3, 4.0), "kp_vz": (0.5, 4.0), "kp_att": (1.0, 16.0),
          "kd_att": (1.0, 8.0), "max_tilt_deg": (5.0, 20.0), "max_horiz_acc": (0.5, 4.0),
          "vz_slope": (0.1, 0.6), "vz_min": (0.5, 1.5), "vz_max": (2.0, 8.0)}
# the "noise" scenario settings of VtvlEnvCfg.apply_scenario
NOISE = dict(noise_pos_m=0.3, noise_vel_m_s=0.15, noise_att_rad=0.009, noise_rate_rad_s=0.02, obs_delay_steps=1)


def gains_from(vec: dict) -> PDGains:
    d = dict(vec)
    d["max_tilt_rad"] = math.radians(d.pop("max_tilt_deg"))
    return PDGains(**d)


def default_vec() -> dict:
    g = asdict(PDGains())
    g["max_tilt_deg"] = math.degrees(g.pop("max_tilt_rad"))
    return {k: g[k] for k in BOUNDS}


def make_scenario(name: str, d: dict, physics: str, veh) -> Scenario:
    base = yaml.safe_load(open(ROOT / "configs/scenarios/landing.yaml"))
    keep = ("com_position_w", "velocity_w", "tilt_axis", "tilt_rad", "omega_b", "disturbance_force_w")
    kw = {**base, **{k: d[k] for k in keep}, "seed": d["seed"]}
    if "gust_period_s" in d:
        kw.update(gust_amp=0.4, gust_period_s=d["gust_period_s"], gust_phase=d["gust_phase"])
    if "rcs_failed" in d:
        kw.update(rcs_failed=[veh.layout.rcs[d["rcs_failed"]].id], fault_known_to_controller=False)
    if name == "noise":
        kw.update(NOISE)
    kw["aero"] = physics == "v1"
    return Scenario(**kw)


def _episode(args):
    gvec, sc_name, d, physics, map_source = args
    veh, eng, maps = load_all(ROOT, map_source)
    sc = make_scenario(sc_name, d, physics, veh)
    _, m, _ = run_scenario(sc, veh, eng["conical_A"], eng["conical_A"], maps["conical_A"], gains_from(gvec))
    td = m.get("touchdown") or {}
    return {"seed": d["seed"], "success": bool(m["success"]), "reason": m["termination_reason"],
            "fuel_kg": m["fuel_used_kg"], "td_vz": td.get("vz"), "td_pos_err": td.get("pos_err")}


def score(rows) -> float:
    s = sum(r["success"] for r in rows)
    soft = sum(math.exp(-(r["td_pos_err"] or 9.0) / 2) * math.exp(-abs(r["td_vz"] or 9.0) / 2) for r in rows)
    return s + 0.1 * soft / len(rows)  # tie-break towards accurate, soft touchdowns


def evaluate(pool, gvecs, sc_name, scen, physics, map_source):
    jobs = [(g, sc_name, d, physics, map_source) for g in gvecs for d in scen]
    out = list(pool.map(_episode, jobs, chunksize=1))
    n = len(scen)
    return [out[i * n:(i + 1) * n] for i in range(len(gvecs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["nominal", "wind", "rcs_fail", "noise", "lateral"])
    ap.add_argument("--physics", default="v1", choices=["v0", "v1"])
    ap.add_argument("--map_source", default="cfd", choices=["ideal1d", "cfd"])
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--elite", type=int, default=4)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--check", type=int, default=0, help="held-out episodes on seeds 1000.. (0 = skip)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "runs/pd_tuning"))
    a = ap.parse_args()
    rng = np.random.default_rng(7)
    names = list(BOUNDS)
    lo = np.log([BOUNDS[k][0] for k in names])
    hi = np.log([BOUNDS[k][1] for k in names])
    mu = np.log([default_vec()[k] for k in names])
    sd = np.full(len(names), 0.35)
    train = eval_scenarios(a.scenario, a.n_seeds, 5000)
    hist, best = [], (-1.0, default_vec(), None)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        for it in range(a.iters):
            pts = [mu] + [np.clip(mu + sd * rng.standard_normal(len(names)), lo, hi) for _ in range(a.pop - 1)]
            vecs = [dict(zip(names, map(float, np.exp(p)))) for p in pts]
            res = evaluate(pool, vecs, a.scenario, train, a.physics, a.map_source)
            sc = np.array([score(r) for r in res])
            order = np.argsort(-sc)
            elite = np.array([pts[i] for i in order[:a.elite]])
            mu, sd = elite.mean(0), np.maximum(elite.std(0), 0.05)
            if sc[order[0]] > best[0]:
                best = (float(sc[order[0]]), vecs[order[0]], res[order[0]])
            hist.append({"iter": it, "best_score": float(sc[order[0]]), "mean_score": float(sc.mean()),
                         "best_success": sum(r["success"] for r in res[order[0]]), "elapsed_s": time.time() - t0})
            print(json.dumps(hist[-1]), flush=True)
        result = {"label": "cloud-v1912 (reference integrator)", "scenario": a.scenario, "physics": a.physics,
                  "map_source": a.map_source, "tuning_seeds": [5000, 5000 + a.n_seeds - 1],
                  "default_gains": default_vec(), "tuned_gains": best[1],
                  "tuned_train_success": f"{sum(r['success'] for r in best[2])}/{a.n_seeds}", "history": hist}
        if a.check:
            test = eval_scenarios(a.scenario, a.check // 2, 1000) + eval_scenarios(a.scenario, a.check - a.check // 2, 2000)
            dflt, tuned = evaluate(pool, [default_vec(), best[1]], a.scenario, test, a.physics, a.map_source)
            result["check"] = {"seeds": "1000.. and 2000..", "n": len(test),
                               "default_success": sum(r["success"] for r in dflt),
                               "tuned_success": sum(r["success"] for r in tuned),
                               "default_reasons": _count(dflt), "tuned_reasons": _count(tuned)}
            print(json.dumps(result["check"]), flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if (a.physics, a.map_source) == ("v1", "cfd") else f"_{a.physics}_{a.map_source}"
    (out / f"{a.scenario}{tag}.json").write_text(json.dumps(result, indent=1))
    assert set(result["tuned_gains"]) <= {f.name for f in fields(PDGains)} | {"max_tilt_deg"}


def _count(rows):
    c = {}
    for r in rows:
        c[r["reason"]] = c.get(r["reason"], 0) + 1
    return c


if __name__ == "__main__":
    main()
