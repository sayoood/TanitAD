"""Re-derive the 416x1024 TRAIN cache size from the real per-episode sizes (E20 input).

The only 416x1024 episodes that exist are the 139 eval clips, and they are not a random draw
from the train split (their source mp4s run larger). So the plain "mean x N" is set beside
models that let each episode's size follow its source clip's MEASURED covariates (mp4 bytes,
camera frames). Every coefficient is fit on the 139 ONLY and applied to the 4,572 train clips'
covariates; models are ranked by leave-one-out error on the 139, and every total carries a
bootstrap interval over the 139.

⚠️ What the interval does NOT cover: a train/eval difference the covariates do not see. Only a
build of a random train sample measures that (deferred: see RESULT.md).

    python size_model.py <inventory.json> <out.json>
"""
from __future__ import annotations

import json
import random
import statistics as st
import sys
from pathlib import Path

B = 2000
SEED = 20260919


def fit(rows):
    """Four per-episode size models, each returning (predict(mp4, frames), train_total)."""
    mp4 = [r[0] for r in rows]; fr = [r[1] for r in rows]; by = [r[2] for r in rows]
    mean_b = st.mean(by)
    per_frame = sum(by) / sum(fr)
    ratio = sum(by) / sum(mp4)
    mx, my = st.mean(mp4), mean_b
    sxx = sum((x - mx) ** 2 for x in mp4)
    b = sum((x - mx) * (y - my) for x, y in zip(mp4, by)) / sxx if sxx else 0.0
    a = my - b * mx
    return {
        "M0 mean x N": lambda m, f: mean_b,
        "M1 per stored frame": lambda m, f: per_frame * f,
        "M2 linear in mp4 bytes": lambda m, f: a + b * m,
        "M3 ratio to mp4 bytes": lambda m, f: ratio * m,
    }, {"a": a, "b": b, "per_frame": per_frame, "ratio": ratio, "mean": mean_b}


def main(inv_path: str, out_path: str) -> int:
    inv = json.loads(Path(inv_path).read_text(encoding="utf-8"))
    rows = [(r["mp4_mb"] * 1e6, r["n_frames"], r["mb"] * 1e6) for r in inv["eval139_416"]["rows"]]
    # the cache keeps every 3rd camera frame: n_frames / timestamp rows = 0.331-0.333 on all 139
    train = [(m * 1e6, t / 3.0) for _, m, t in inv["train_covariates"]["per_clip"]]
    n = len(train)
    models, coef = fit(rows)

    def total(ms, name):
        return sum(ms[name](m, f) for m, f in train)

    out = {"n_eval": len(rows), "n_train": n, "coefficients": coef, "models": {}}
    # leave-one-out on the 139: which model predicts an UNSEEN episode best?
    for name in models:
        err = []
        for i in range(len(rows)):
            ms, _ = fit(rows[:i] + rows[i + 1:])
            err.append(ms[name](rows[i][0], rows[i][1]) - rows[i][2])
        out["models"][name] = {"train_total_GB": total(models, name) / 1e9,
                               "loo_rmse_MB": (sum(e * e for e in err) / len(err)) ** 0.5 / 1e6,
                               "loo_bias_MB": st.mean(err) / 1e6}
    rng = random.Random(SEED)
    boots = {name: [] for name in models}
    for _ in range(B):
        s = [rows[rng.randrange(len(rows))] for _ in rows]
        ms, _ = fit(s)
        for name in models:
            boots[name].append(total(ms, name) / 1e9)
    for name, v in boots.items():
        v.sort()
        out["models"][name]["ci95_GB"] = [v[int(0.025 * B)], v[int(0.975 * B) - 1]]
    # R^2 of the linear model, and the covariate shift it corrects for
    by = [r[2] for r in rows]; my = st.mean(by)
    ss_tot = sum((y - my) ** 2 for y in by)
    ss_res = sum((y - models["M2 linear in mp4 bytes"](m, f)) ** 2 for m, f, y in rows)
    out["M2_r2"] = 1 - ss_res / ss_tot
    out["covariate_shift"] = {
        "eval_mp4_mean_MB": st.mean(r[0] for r in rows) / 1e6,
        "train_mp4_mean_MB": st.mean(m for m, _ in train) / 1e6,
        "eval_frames_mean": st.mean(r[1] for r in rows),
        "train_frames_mean": st.mean(f for _, f in train),
        "eval_bytes_mean_MB": my / 1e6, "eval_bytes_sd_MB": st.stdev(by) / 1e6,
    }
    best = min(out["models"], key=lambda k: out["models"][k]["loo_rmse_MB"])
    out["best_by_loo"] = best
    out["legacy_estimate_GB"] = {"value": 386.5, "was": "4,713 x 82.00 MB (the 408x1024 sample)"}
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
