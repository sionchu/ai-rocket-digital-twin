"""Collect runs/ppo_win-isaac/eval_v1/<scenario>/<controller>/s{1000,2000}/eval_*.json into summary_v1.json/.md
next to the v0 numbers (ADR-0010, README; label win-isaac). usage: python scripts/summarize_eval_v1.py"""
from __future__ import annotations

import json
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "runs/ppo_win-isaac/eval_v1"
SCEN = ["nominal", "wind", "rcs_fail", "noise", "lateral", "hop"]
CTRL = ["pd", "pd_tuned", "pd_tuned_try1", "rl_nominal", "rl_scenario"]
LAYOUTS = ["tri_diff", "quad_diff", "coldgas"]
# v0 (win-isaac, physics v0): PD / RL nominal (r3) / RL scenario, from ADR-0010 and README
V0 = {"nominal": (60, 57, None), "wind": (43, 20, 54), "rcs_fail": (60, 27, 57), "noise": (60, 60, 17),
      "lateral": (5, 28, 52), "hop": ("-", "-", None)}  # hop is new in v1 (ADR-0012)
V0_LAYOUT = {"tri_diff": 60, "quad_diff": 60, "coldgas": 59}


def load(scen, ctrl):
    blocks = []
    for s0 in (1000, 2000):
        f = sorted((E / scen / ctrl / f"s{s0}").glob("eval_*.json"))
        if not f:
            return None
        blocks.append(json.loads(f[0].read_text()))
    eps = [e for b in blocks for e in b["episodes"]]
    reasons = {}
    for e in eps:
        reasons[e["termination_reason"]] = reasons.get(e["termination_reason"], 0) + 1
    return {"n": len(eps), "success": sum(e["success"] for e in eps),
            "physics_valid": sum(e["physics_valid"] for e in eps),
            "fuel_mean_kg": sum(e["fuel_used_kg"] for e in eps) / len(eps), "reasons": reasons,
            "checkpoint": blocks[0]["checkpoint"], "pd_gains": blocks[0]["pd_gains"], "map_hash": blocks[0]["map_hash"]}


def main():
    res = {s: {c: load(s, c) for c in CTRL if (E / s / c).exists()} for s in SCEN}
    res["layouts"] = {lay: load("nominal", f"rl_{lay}") for lay in LAYOUTS}
    (E / "summary_v1.json").write_text(json.dumps(res, indent=1))

    def cell(r):
        return "NOT_RUN" if r is None else f"{r['success']}/{r['n']}"

    lines = ["| 시나리오 | PD 기본 v0 → v1 | PD 재튜닝 v1 | RL 기본 v0(r3) → v1(v1_nominal) | RL 시나리오 v0 → v1 | v1 실패 사유 (PD 기본 / RL 시나리오) |",
             "|---|---|---|---|---|---|"]
    for s in SCEN:
        r, (p0, n0, s0) = res[s], V0[s]
        rs = r.get("rl_scenario")
        reason = lambda x: ", ".join(f"{k} {v}" for k, v in sorted(x["reasons"].items())) if x else "-"  # noqa: E731
        lines.append(f"| {s} | {p0} → {cell(r.get('pd'))} | {cell(r.get('pd_tuned'))} | {n0} → {cell(r.get('rl_nominal'))} | "
                     f"{'-' if s0 is None else s0} → {'-' if s == 'nominal' else cell(rs)} | "
                     f"{reason(r.get('pd'))} / {reason(rs)} |")
    lines += ["", "| 배치 (RL, nominal) | v0 | v1 |", "|---|---|---|"]
    lines += [f"| {lay} | {V0_LAYOUT[lay]}/60 | {cell(res['layouts'][lay])} |" for lay in LAYOUTS]
    if res["wind"].get("pd_tuned_try1"):
        lines += ["", f"wind 재튜닝 PD try1(`wind_try1_8seeds.json`, 실패 기록): {cell(res['wind']['pd_tuned_try1'])}"]
    (E / "summary_v1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
