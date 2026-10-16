"""Command-line entry points (`python -m rocket_dt <command>`).

Implemented in P0: validate-physics, build-map, simulate, experiments, cfd-case.
Planned (not implemented): train, finetune, evaluate (RL), visualize.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

from rocket_dt.evaluation.artifacts import REPO, run_id, software_versions, write_run
from rocket_dt.schemas import RunManifest, load_engine, load_vehicle

RUNS = REPO / "runs"


def cmd_validate_physics(a):
    out = RUNS / "validate_physics"
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", str(REPO / "tests"),
                        f"--junitxml={out / 'junit.xml'}"], capture_output=True, text=True)
    (out / "pytest.log").write_text(r.stdout + r.stderr)
    print(r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr)
    (out / "manifest.json").write_text(json.dumps({"command": "validate-physics", "returncode": r.returncode,
                                                   "software": software_versions()}, indent=1))
    return r.returncode


def cmd_build_map(a):
    from rocket_dt.propulsion.performance_map import build_ideal_map
    e = load_engine(a.engine)
    m = build_ideal_map(e)
    path = RUNS / "maps" / f"{e.engine_design_id}.json"
    h = m.save(path)
    print(f"{path} map_hash={h} invalid_cells={int((~m.valid).sum())}/{m.valid.size}")
    return 0


def cmd_simulate(a):
    from rocket_dt.evaluation.simulate import Scenario, run_scenario
    from rocket_dt.propulsion.performance_map import build_ideal_map
    veh = load_vehicle(a.vehicle)
    e = load_engine(a.engine)
    me = load_engine(a.mass_engine) if a.mass_engine else e
    sc = Scenario(**yaml.safe_load(open(a.scenario)))
    m = build_ideal_map(e)
    log, met, ep = run_scenario(sc, veh, me, e, m)
    man = RunManifest(run_id("sim", sc.name, e.engine_design_id, payload=[asdict(sc), m.content_hash(), me.content_hash()]),
                      "simulate", [e.engine_design_id], veh.content_hash(), {e.engine_design_id: m.content_hash()},
                      software_versions(), sc.seed, extra={"mass_engine_id": me.engine_design_id,
                                                           "map_source": m.provenance["source"]})
    d = write_run(RUNS, man, met, log, ep)
    print(d, json.dumps({k: met[k] for k in ("termination_reason", "success", "physics_valid", "fuel_used_kg")}))
    return 0


def cmd_experiments(a):
    from rocket_dt.evaluation import report
    report.run_all(REPO, RUNS / "p0_experiments", n_suite=a.n_suite, workers=a.workers)
    return 0


def cmd_cfd_case(a):
    from rocket_dt.cfd.case_generator import write_case
    e = load_engine(a.engine)
    d = write_case(e, Path(a.out), level=a.level, ambient_pressure_pa=a.pa, throttle=a.throttle)
    print(d)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="rocket_dt")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate-physics").set_defaults(fn=cmd_validate_physics)
    b = sub.add_parser("build-map"); b.add_argument("--engine", required=True); b.set_defaults(fn=cmd_build_map)
    s = sub.add_parser("simulate")
    s.add_argument("--engine", required=True); s.add_argument("--scenario", required=True)
    s.add_argument("--vehicle", default=str(REPO / "configs/vehicle_v0.yaml"))
    s.add_argument("--mass-engine", default=None, help="engine whose hardware mass is used (E1)")
    s.set_defaults(fn=cmd_simulate)
    x = sub.add_parser("experiments"); x.add_argument("--n-suite", type=int, default=30)
    x.add_argument("--workers", type=int, default=4); x.set_defaults(fn=cmd_experiments)
    c = sub.add_parser("cfd-case")
    c.add_argument("--engine", required=True); c.add_argument("--out", required=True)
    c.add_argument("--level", type=int, default=1); c.add_argument("--pa", type=float, default=101325.0)
    c.add_argument("--throttle", type=float, default=1.0); c.set_defaults(fn=cmd_cfd_case)
    for name in ("train", "finetune", "evaluate", "visualize"):
        sub.add_parser(name).set_defaults(fn=lambda a, n=name: _not_impl(n))
    a = p.parse_args(argv)
    return a.fn(a)


def _not_impl(name):
    print(f"'{name}' is planned for a later stage (needs Isaac Lab / GPU). Status: NOT_RUN", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
