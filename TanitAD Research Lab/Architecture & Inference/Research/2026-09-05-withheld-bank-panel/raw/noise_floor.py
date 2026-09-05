"""⭐ THE RUN-TO-RUN NOISE FLOOR, measured from the panel's OWN logs — free.

⛔⛔ WHY THIS EXISTS. The panel's paired episode-cluster bootstrap resamples EPISODES at
eval time with the two TRAINED MODELS HELD FIXED. It therefore answers *"would this delta
survive on other windows?"* — and NOT *"would it survive on another training run?"*. With
one seed per arm those are different questions, and only the second one licenses "the
lever caused it". A CI that excludes zero is not evidence that a lever did anything if two
runs of the SAME configuration differ by more than the delta.

A1 and A2 carry `--withheld-bank-warmup 450`, so for steps 1..450 they roll the FIXED bank
— i.e. over that stretch they are the SAME CONFIGURATION as A0, same seed, same data
order. Their divergence there is pure run-to-run nondeterminism (non-deterministic cuDNN
kernels / atomic reductions), amplified by training dynamics. That is a noise floor
measured on this exact rig at zero extra GPU cost.

MEASURED 2026-09-05 (A0 vs A1, 9 identical-config log rows): 54/54 differences non-zero;
|Δ withheld_speed_mae| mean 0.194, max 0.701 m/s; |Δ loss| mean 0.495, max 1.728. The
divergence AMPLIFIES: −0.00007 at step 50, −0.70 by step 300.

⇒ Read every headline delta against this floor, and prefer the eval-level replicate
(A0b_replicate: A0's flags and A0's seed, run again) which measures the same thing where
the panel actually scores.

usage: python noise_floor.py [arms_dir]
"""
import json
import statistics
import sys
from pathlib import Path

ARMS = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Admin\run_wbank\arms")
KEYS = ["loss", "traj", "withheld_speed_mae", "kept_speed_mae", "goal2s_err_m", "sel_v3",
        "anchor_acc", "goal_tac"]


def rows(arm):
    p = ARMS / arm / "metrics.jsonl"
    if not p.exists():
        return {}
    out = {}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            if "step" in r and "eval_loss" not in r:
                out[r["step"]] = r
    return out


def warmup_of(arm):
    p = ARMS / arm / "config.json"
    if not p.exists():
        return None
    return json.load(open(p, encoding="utf-8")).get("withheld_bank", {}).get("warmup_steps", 0)


def main():
    A0 = rows("A0_fixed")
    if not A0:
        raise SystemExit("no A0_fixed")
    print("# run-to-run noise floor, from the identical-config warm-up stretch\n")
    print("An arm with `--withheld-bank-warmup N` rolls the FIXED bank for steps <= N, so")
    print("over that stretch it IS A0's configuration. Divergence there is nondeterminism.\n")
    any_rows = False
    for arm in ("A1_pred", "A2_random"):
        R = rows(arm)
        N = warmup_of(arm)
        if not R or not N:
            print(f"{arm}: absent or no warm-up — skipped")
            continue
        pre = [s for s in sorted(A0) if s <= N and s in R]
        if not pre:
            continue
        any_rows = True
        print(f"## {arm} vs A0_fixed — {len(pre)} identical-config rows (steps <= {N})")
        nz = tot = 0
        per_key = {}
        for k in KEYS:
            d = [R[s][k] - A0[s][k] for s in pre if k in R[s] and k in A0[s]]
            if not d:
                continue
            per_key[k] = d
            nz += sum(1 for x in d if x != 0)
            tot += len(d)
        print(f"non-zero differences: {nz}/{tot}"
              f"{'  ⇒ training is NOT deterministic' if nz else '  ⇒ bit-identical'}")
        print("\n| metric | mean |Δ| | max |Δ| | first-row |Δ| | last-row |Δ| |")
        print("|---|---|---|---|---|")
        for k, d in per_key.items():
            a = [abs(x) for x in d]
            print(f"| {k} | {statistics.mean(a):.5f} | {max(a):.5f} | "
                  f"{a[0]:.5f} | {a[-1]:.5f} |")
        print()
    # the eval-level replicate, if it was run
    if (ARMS / "A0b_replicate").exists():
        print("## A0b_replicate exists — the EVAL-level floor is in the panel report")
        print("   (A0b vs A0 paired delta = same config, same seed, run twice)")
    else:
        print("## A0b_replicate NOT run — only the TRAIN-LOG floor is available")
    if not any_rows:
        print("no identical-config stretch found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
