"""Generate, mesh, initialise and run the P2 CFD matrix; then QA, GCI and CFD-derived maps.

usage: python scripts/run_cfd_batch.py --engines conical_A bell_B --levels 1 2 3 --jobs 4 --out runs/cfd
Requires an OpenFOAM installation providing blockMesh/checkMesh/rhoCentralFoam
(sourced from $FOAM_BASHRC, default /usr/share/openfoam/etc/bashrc).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rocket_dt.cfd import post  # noqa: E402
from rocket_dt.cfd.case_generator import REFINEMENT_RATIO, write_case  # noqa: E402
from rocket_dt.cfd.cfd_map import build_cfd_map  # noqa: E402
from rocket_dt.schemas import load_engine  # noqa: E402

BASHRC = os.environ.get("FOAM_BASHRC", "/usr/share/openfoam/etc/bashrc")


def sh(case: Path, cmd: str, log: str) -> int:
    full = f"source {BASHRC} >/dev/null 2>&1; cd '{case}' && {cmd} > {log} 2>&1"
    return subprocess.run(["bash", "-c", full]).returncode


def run_case(args):
    eng_name, level, out = args
    e = load_engine(REPO / f"configs/engines/{eng_name}.yaml")
    case = write_case(e, Path(out), level=level)
    t0 = time.time()
    rc = {"blockMesh": sh(case, "blockMesh", "log.blockMesh"), "checkMesh": sh(case, "checkMesh", "log.checkMesh")}
    post.init_isentropic(case, e)
    rc["rhoCentralFoam"] = sh(case, "rhoCentralFoam", "log.rhoCentralFoam")
    wall = time.time() - t0
    hist = post.history(case) if rc["rhoCentralFoam"] == 0 else []
    q = post.qa(case, hist) if hist else {"status": "FAIL", "reason": f"solver rc={rc['rhoCentralFoam']}"}
    ver = subprocess.run(["bash", "-c", f"source {BASHRC} >/dev/null 2>&1; echo $WM_PROJECT_VERSION"],
                         capture_output=True, text=True).stdout.strip()
    res = {"engine": eng_name, "level": level, "case": str(case), "returncodes": rc, "wall_time_s": wall,
           "openfoam_version": ver, "qa": q, "history": hist,
           "mesh_ok": "Mesh OK" in (case / "log.checkMesh").read_text()}
    (case / "cfd_result.json").write_text(json.dumps(res, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="+", default=["conical_A", "bell_B"])
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "runs/cfd"))
    ap.add_argument("--post-only", action="store_true", help="re-run QA/GCI/maps on existing cases")
    a = ap.parse_args()
    if a.post_only:
        results = []
        for e in a.engines:
            for lv in a.levels:
                r = json.loads((Path(a.out) / f"{e}_L{lv}" / "cfd_result.json").read_text())
                r["qa"] = post.qa(Path(r["case"]), r["history"]) if r["history"] else r["qa"]
                Path(r["case"], "cfd_result.json").write_text(json.dumps(r, indent=1))
                results.append(r)
    else:
        jobs = [(e, l, a.out) for l in sorted(a.levels, reverse=True) for e in a.engines]  # longest first
        with ProcessPoolExecutor(a.jobs) as ex:
            results = list(ex.map(run_case, jobs))
    summary = {"refinement_ratio": REFINEMENT_RATIO, "cases": [{k: v for k, v in r.items() if k != "history"} | {
        "final": post.window_stats(r["history"]) if r["history"] else None} for r in results], "gci": {}, "maps": {}}
    for e in a.engines:
        rs = sorted([r for r in results if r["engine"] == e and r["history"]], key=lambda r: -r["level"])
        if len(rs) >= 3:
            ws = [post.window_stats(r["history"]) for r in rs[:3]]
            f = [w["stream_thrust_n"] for w in ws]
            m = [w["mdot_out"] for w in ws]
            summary["gci"][e] = {"levels": [r["level"] for r in rs[:3]],
                                 "stream_thrust": post.gci(*f, REFINEMENT_RATIO, REFINEMENT_RATIO),
                                 "mdot": post.gci(*m, REFINEMENT_RATIO, REFINEMENT_RATIO)}
            finest = rs[0]
            summary["gci"][e]["all_levels_qa_pass"] = all(r["qa"]["status"] == "PASS" for r in rs[:3])
            if finest["qa"]["status"] == "PASS":
                cmap = build_cfd_map(load_engine(REPO / f"configs/engines/{e}.yaml"), finest, summary["gci"][e])
                path = Path(a.out) / "maps" / f"{e}_cfd.json"
                summary["maps"][e] = {"path": str(path), "map_hash": cmap.save(path)}
    Path(a.out, "cfd_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("gci", "maps")}, indent=1))
    for c in summary["cases"]:
        print(c["engine"], c["level"], c["qa"]["status"], round(c["wall_time_s"]), "s",
              c["final"] and round(c["final"]["stream_thrust_n"], 1))


if __name__ == "__main__":
    main()
