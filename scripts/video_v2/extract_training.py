"""Extract per-iteration training scalars from the win-isaac TensorBoard logs into one JSON."""
import glob
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]
RUNS = ["r0_base", "r1_pos1", "r2_pos2_ent", "r3_guided", "m1_mix_ft", "m2_mix_scratch"]
TAGS = {
    "success": "Metrics/success_rate",
    "touchdown": "Metrics/touchdown_rate",
    "tip_over": "Episode_Termination/tip_over",
    "oob": "Episode_Termination/oob",
    "timeout": "Episode_Termination/timeout",
    "td_vz": "Metrics/td_vz_abs_mean",
    "fuel": "Metrics/fuel_used_mean",
    "reward": "Train/mean_reward",
    "physics_valid": "Metrics/physics_valid_rate",
}


def main():
    out = {}
    for run in RUNS:
        f = glob.glob(str(REPO / f"runs/ppo_win-isaac/train/{run}/events*"))[0]
        acc = EventAccumulator(f, size_guidance={"scalars": 0})
        acc.Reload()
        have = set(acc.Tags()["scalars"])
        series = {}
        for key, tag in TAGS.items():
            if tag in have:
                s = acc.Scalars(tag)
                series[key] = [[int(x.step), float(x.value)] for x in s]
        out[run] = {"source": Path(f).name, "series": series}
    dst = REPO / "runs/video/v2/training_scalars.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out))
    for run, d in out.items():
        st = d["series"]["success"]
        print(run, len(st), st[0][0], st[-1][0], round(st[-1][1], 3))


if __name__ == "__main__":
    main()
