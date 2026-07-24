"""Post-mortem experiment **A** — did the ego-dropout zero-fill damage the WEIGHTS,
or was it only a measurement artifact?

Stage 2 of 2. Stage 1 is the training run itself:

    flagship4b-v3enc-expA-nodrop-2k   (tanitad-pod / RTX A6000)
      = v3enc's verbatim invocation, from step 0, for 2,000 steps, with EXACTLY
        one lever changed:  v2_ego_dropout 0.25 -> 0.0   (``--ego-dropout 0.0``)

Experiment B (``postmortem_b_egodropout_v3enc10k.json``) probed FIXED weights and
therefore could not separate *"corruption baked into the weights during training"*
from *"corruption applied at measurement time"*. A separates them, because exp-A's
own training log is mask-FREE by construction, exactly like v1's.

The three-way read, all on the SAME step grid (0, 50, ..., 1950; n = 40 rows/arm):

    exp-A   dropout 0.0, mask-free log      <- new
    v3enc   dropout 0.25, mask-CONTAMINATED log
    v1      no ego-dropout at all, mask-free log

plus the no-speed ablation control for the level-free "fraction of the speed-channel
benefit recovered" statistic.

ESTIMATORS (CLAUDE.md: never quote an interval without its estimator).
  * ``bucket_mean``  — arithmetic mean of the per-batch (B=16x4) logged values in a
    step range. A DESCRIPTIVE SUMMARY OF A TRAINING LOG. It is not val ADE and it is
    not a held-out interval.
  * ``step_matched_paired_circular_moving_block_bootstrap`` — all arms log at the
    same steps, so differences are taken PER STEP (which cancels the shared descent
    trend) and the per-step differences are resampled in circular moving blocks of
    ``L`` consecutive rows to respect their autocorrelation. B = 10000. Reported for
    the full 0-2k window only; sub-buckets (n = 10 rows) are descriptive means.
  * ``artifact_only_null_band`` — what exp-A's log WOULD read if the mask changed
    the logged number but not the weights: ``v3enc_logged x r``, where
    ``r = off/on``. r is bounded: r <= 1.0 always (at init the rollout is
    action-insensitive, so off == on), and B MEASURED r = 0.6081 at the step-10,000
    weights. Landing BELOW the band is weight damage no measurement artifact can
    explain; landing INSIDE it is consistent with a pure artifact.

  python taniteval/postmortem_a_analyze.py [--log <expA train_log.jsonl>]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOGS = REPO / "taniteval/results/trainlogs"

_ap = argparse.ArgumentParser()
_ap.add_argument("--log", default=str(LOGS / "expA-nodrop_train_log.jsonl"),
                 help="exp-A train_log.jsonl (pulled from the pod)")
_ap.add_argument("--out", default=str(
    REPO / "taniteval/results/postmortem_a_egodropout_off_expA2k.json"))
_ap.add_argument("--n-boot", type=int, default=10000)
_ap.add_argument("--block", type=int, default=4,
                 help="primary circular moving-block length, in LOG ROWS "
                      "(1 row = 50 steps); 4 rows = 200 steps")
_ap.add_argument("--seed", type=int, default=1234)
_args = _ap.parse_args()

# Bucket convention: ``lo < step <= hi``. This is REPLICATED from the post-mortem's
# published tables, not assumed — it is the only one of four candidates that
# reproduces SS1.1's 0-2k row exactly (v1 0.6458, v3enc 1.0364, no-speed 1.3152) and
# Appendix B's 8-10k row (v1 0.1062, v3enc 0.4699, no-speed 0.5740).
WINDOW = (0, 2000)
# The four 500-step buckets + the published 0-2k window + two ROBUSTNESS windows
# that drop the initial transient. The transient is real and it REVERSES the
# ordering: at steps 50/100 v3enc BEATS v1 by 0.97 / 0.79 on g_op_fwd_ade_m (v1
# starts at 3.3203 and only falls below v3enc at step 150), so the (0,500] bucket
# carries an arm-ordering flip that has nothing to do with the mask.
BUCKETS = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (0, 2000),
           (200, 2000), (500, 2000)]
# Every arm is restricted to exp-A's OWN step set, so all four are exactly matched
# row-for-row. exp-A ran `--steps 2000` = steps 0..1999, so it has no step-2000 row;
# the last bucket therefore holds 9 rows (1550..1950) for ALL arms, not 10. The
# published reference numbers (which do include step 2000) are reported alongside.
KEYS = ["g_op_fwd_ade_m", "g_tac_fwd_ade_m", "g_str_fwd_ade_m",
        "g_op_mid_de_m", "g_tac_mid_de_m", "g_str_mid_de_m", "inv"]
# The rollout/encoder split the post-mortem's SS2 is built on.
ROLLOUT = ["g_op_fwd_ade_m", "g_tac_fwd_ade_m", "g_str_fwd_ade_m"]
ENCODER = ["g_op_mid_de_m", "g_tac_mid_de_m", "g_str_mid_de_m"]
REFS = {"v1_speedjerk": "v1-speedjerk_train_log.jsonl",
        "v3enc": "v3enc_train_log.jsonl",
        "nospeed_phase0": "nospeed-phase0_train_log.jsonl"}


def load(path: Path) -> dict[int, dict]:
    """{step: row}, deduped on step keeping the LAST occurrence — v1 and the
    no-speed arm replay steps after a resume (same rule as experiment B)."""
    bystep: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            bystep[int(r["step"])] = r
    return bystep


GRID: set[int] = set()          # exp-A's own step set, filled in below


def bmean(rows: dict[int, dict], key: str, lo: int, hi: int, matched=True):
    v = [r[key] for s, r in rows.items()
         if lo < s <= hi and key in r and (not matched or s in GRID)]
    return round(st.fmean(v), 4) if v else None


def block_boot(diffs: list[float], stat, n_boot: int, L: int, seed: int):
    """Circular moving-block bootstrap over an ORDERED list of per-step values."""
    n = len(diffs)
    if n == 0:
        return None
    rng = random.Random(seed)
    nb = max(1, -(-n // L))                       # ceil(n / L)
    out = []
    for _ in range(n_boot):
        samp: list[float] = []
        for _b in range(nb):
            s0 = rng.randrange(n)
            samp.extend(diffs[(s0 + j) % n] for j in range(L))
        out.append(stat(samp[:n]))
    out.sort()
    return {"lo": round(out[int(0.025 * n_boot)], 4),
            "hi": round(out[int(0.975 * n_boot) - 1], 4)}


def paired_diff(a: dict[int, dict], b: dict[int, dict], key: str,
                lo: int, hi: int) -> tuple[list[int], list[float]]:
    steps = sorted(s for s in a if lo < s <= hi and s in b and s in GRID
                   and key in a[s] and key in b[s])
    return steps, [a[s][key] - b[s][key] for s in steps]


def ci(diffs, n_boot, L, seed, stat=None):
    stat = stat or st.fmean
    r = block_boot(diffs, stat, n_boot, L, seed)
    return None if r is None else {
        "delta": round(stat(diffs), 4), "lo": r["lo"], "hi": r["hi"],
        "n_paired_steps": len(diffs), "block_rows": L,
        "estimator": "step_matched_paired_circular_moving_block_bootstrap"}


# --------------------------------------------------------------------------- #
A = load(Path(_args.log))
R = {k: load(LOGS / f) for k, f in REFS.items()}
V1, V3, NS = R["v1_speedjerk"], R["v3enc"], R["nospeed_phase0"]

GRID.update(s for s in A if WINDOW[0] < s <= WINDOW[1])
grid = {k: sorted(s for s in d if WINDOW[0] < s <= WINDOW[1])
        for k, d in {"expA": A, **R}.items()}
# every reference must COVER exp-A's grid, else the matching is not exact
covers = {k: GRID.issubset(set(v)) for k, v in grid.items()}
same_grid = all(covers.values())

# --- Provenance / integrity: is exp-A really on v3enc's batch sequence? -------
# nav_valid_frac, route_acc, man_acc, wp, man, route are functions of the BATCH
# (labels/targets), not of the weights, at step 0; nav_valid_frac stays a pure
# batch statistic at EVERY step. Row-for-row equality with v3enc proves the two
# runs consumed the same windows in the same order -> the pairing is real.
shared = sorted(GRID & set(grid["v3enc"]))
nav_match = sum(1 for s in shared
                if abs(A[s]["nav_valid_frac"] - V3[s]["nav_valid_frac"]) < 1e-12)
step0 = {k: {"expA": A[0].get(k), "v3enc": V3[0].get(k),
             "bit_identical": A[0].get(k) == V3[0].get(k)}
         for k in ("g_op_fwd_ade_m", "g_op_mid_de_m", "pred", "roll", "wp",
                   "man", "route", "cls", "decorr", "erank", "dim_std",
                   "ego_r2", "nav_valid_frac", "inv", "sigreg", "loss")}

# --- B's measured artifact ratio (for the null band) ------------------------
BJ = json.loads((REPO / "taniteval/results/postmortem_b_egodropout_v3enc10k.json")
                .read_text(encoding="utf-8"))
r10k = {}
for k in ROLLOUT:
    d = BJ["derived"].get(k, {})
    on_a, off = d.get("measured_mask_ON_analytic_p0.25"), d.get("measured_mask_OFF")
    if on_a and off:
        r10k[k] = round(off / on_a, 4)

# --------------------------------------------------------------------------- #
buckets: dict[str, dict] = {}
for k in KEYS:
    per_bucket = {}
    for lo, hi in BUCKETS:
        tag = f"{lo}-{hi}"
        eA, v1, v3, ns = (bmean(A, k, lo, hi), bmean(V1, k, lo, hi),
                          bmean(V3, k, lo, hi), bmean(NS, k, lo, hi))
        row = {"expA_nodrop": eA, "v3enc_logged": v3, "v1": v1,
               "nospeed_control": ns}
        if eA is not None and v1:
            row["ratio_expA_over_v1"] = round(eA / v1, 3)
        if v3 is not None and v1:
            row["ratio_v3enc_over_v1"] = round(v3 / v1, 3)
        # How much of the v3enc-vs-v1 LOGGED gap does turning the mask off in
        # TRAINING remove?  1.0 = exp-A landed on v1; 0.0 = it landed on v3enc.
        if None not in (eA, v3, v1) and abs(v3 - v1) > 1e-9:
            row["recovery_fraction_of_logged_gap"] = round((v3 - eA) / (v3 - v1), 4)
        # Level-free: fraction of the speed-channel benefit recovered.
        if ns:
            for nm, val in (("v1", v1), ("v3enc_logged", v3),
                            ("expA_nodrop", eA)):
                if val is not None:
                    row[f"speed_benefit_recovered_{nm}"] = round((ns - val) / ns, 4)
        # Artifact-only null band (rollout terms only; the encoder terms are
        # mask-invariant by construction, B: max|delta| = 0.0).
        if k in ROLLOUT and v3 is not None and k in r10k:
            band = sorted((round(v3 * r10k[k], 4), v3))
            row["artifact_only_null_band"] = band
            row["expA_below_null_band"] = (eA is not None and eA < band[0])
            row["expA_inside_null_band"] = (eA is not None
                                            and band[0] <= eA <= band[1])
        per_bucket[tag] = row
    buckets[k] = per_bucket

# --------------------------------------------------------------------------- #
PAIRED_WINDOWS = [(0, 2000), (500, 2000)]
paired: dict[str, dict] = {}
for k in KEYS:
  for lo, hi in PAIRED_WINDOWS:
    ent: dict[str, object] = {}
    for nm, other in (("expA_minus_v3enc", V3), ("expA_minus_v1", V1),
                      ("v3enc_minus_v1", None)):
        if nm == "v3enc_minus_v1":
            steps, d = paired_diff(V3, V1, k, lo, hi)
        else:
            steps, d = paired_diff(A, other, k, lo, hi)
        if d:
            ent[nm] = ci(d, _args.n_boot, _args.block, _args.seed)
            ent[nm]["separated_from_zero"] = not (
                ent[nm]["lo"] <= 0.0 <= ent[nm]["hi"])
    # block-length sensitivity on the headline contrast
    if k == "g_op_fwd_ade_m":
        sens = {}
        for L in (1, 2, 4, 8):
            _, d = paired_diff(A, V3, k, lo, hi)
            sens[f"L{L}"] = ci(d, _args.n_boot, L, _args.seed)
        ent["block_length_sensitivity_expA_minus_v3enc"] = sens
    # recovery fraction with a block bootstrap over the aligned TRIPLES
    steps = sorted(s for s in GRID if lo < s <= hi and s in V3 and s in V1
                   and k in A.get(s, {}) and k in V3[s] and k in V1[s])
    if steps:
        trip = [(A[s][k], V3[s][k], V1[s][k]) for s in steps]
        idx = list(range(len(trip)))

        def _rho(sub_idx, _t=trip):
            num = st.fmean(_t[i][1] - _t[i][0] for i in sub_idx)
            den = st.fmean(_t[i][1] - _t[i][2] for i in sub_idx)
            return num / den if abs(den) > 1e-12 else float("nan")

        boot = block_boot([float(i) for i in idx],
                          lambda s: _rho([int(x) for x in s]),
                          _args.n_boot, _args.block, _args.seed)
        ent["recovery_fraction"] = {
            "rho": round(_rho(idx), 4), "lo": boot["lo"], "hi": boot["hi"],
            "n_paired_steps": len(steps), "block_rows": _args.block,
            "estimator": "step_matched_circular_moving_block_bootstrap_of_ratio",
            "denominator_mean_v3enc_minus_v1": round(
                st.fmean(t[1] - t[2] for t in trip), 4),
            "reading": "1.0 = exp-A landed on v1 (mask fully causal for the "
                       "logged metric); 0.0 = exp-A landed on v3enc (mask "
                       "irrelevant)"}
    paired.setdefault(k, {})[f"{lo}-{hi}"] = ent

# --------------------------------------------------------------------------- #
op = buckets["g_op_fwd_ade_m"]["0-2000"]
op5 = buckets["g_op_fwd_ade_m"]["500-2000"]
rho = paired["g_op_fwd_ade_m"]["0-2000"].get("recovery_fraction", {})
rho5 = paired["g_op_fwd_ade_m"]["500-2000"].get("recovery_fraction", {})
res = {
    "experiment": "post-mortem A — v2_ego_dropout 0.0 re-run, 2,000 steps",
    "date": "2026-07-21",
    "arm": "flagship4b-v3enc-expA-nodrop-2k",
    "NOT_A_MODEL": ("2,000-step attribution diagnostic. Never evaluate as an arm, "
                    "never leaderboard, never register as a flagship version."),
    "design": {
        "base": "flagship4b-v3enc-30k invocation, verbatim (MODEL_REGISTRY SS1.4)",
        "single_lever": "v2_ego_dropout 0.25 -> 0.0 (--ego-dropout 0.0)",
        "config_diff_vs_v3enc": "1 entry out of 118 compared fields",
        "lr_schedule": ("bit-identical to v3enc over the whole window: --warmup "
                        "2000 means cosine_lr never reads --steps for step < 2000 "
                        "(train_flagship4b.py:438 -> train_worldmodel.py:90-94)"),
        "levers_INACTIVE_in_this_window": [
            "rollout_k > 4 — the staged schedule is K=4 for step < 5000, and v1 "
            "also ran K=4, so DO-NOT-CARRY #2 is untestable here",
            "v2_encoder_ego_decorr — decorr_w = 0.0 for step < 10000",
        ],
        "levers_ACTIVE_and_still_confounded_vs_v1": [
            "v2_fa_dropout 0.15", "v2_invdyn_gradscale 0.5",
            "v2_ego_to_planners", "v2_goal_decode", "v2_nav_dropout 0.5",
            "v2_traj_jerk 0.02", "v2_gated_intent", "v2_anchor_tactical",
            "v2_route_from_vision", "v2_labels", "needed_fut 16 vs v1's 10",
            "v1's aux_accel head + jerk_weight absent",
        ],
    },
    "integrity": {
        "all_references_cover_expA_grid": same_grid,
        "reference_covers_expA_grid": covers,
        "matched_grid_n_rows": len(GRID),
        "matched_grid_first_last": [min(GRID), max(GRID)] if GRID else None,
        "n_log_rows_per_arm_0-2k_unrestricted": {k: len(v)
                                                 for k, v in grid.items()},
        # Replicates the post-mortem's published SS1.1 row from the RAW logs, on
        # the references' own full grid (which includes step 2000, exp-A's does
        # not). Proves the bucket convention used here is the published one.
        "published_convention_crosscheck_0-2k_g_op_fwd_ade_m": {
            "v1": bmean(V1, "g_op_fwd_ade_m", 0, 2000, matched=False),
            "v3enc": bmean(V3, "g_op_fwd_ade_m", 0, 2000, matched=False),
            "nospeed_control": bmean(NS, "g_op_fwd_ade_m", 0, 2000,
                                     matched=False),
            "post_mortem_SS1.1_prose": {"v1": 0.6458, "v3enc": 1.0364,
                                        "nospeed": 1.3152}},
        "batch_sequence_shared_with_v3enc": {
            "nav_valid_frac_rows_matching": nav_match,
            "of_rows": len(shared),
            "why": ("nav_valid_frac is a pure function of the sampled batch; "
                    "row-for-row equality proves both runs consumed the same "
                    "windows in the same order, so exp-A vs v3enc is a genuinely "
                    "PAIRED comparison"),
        },
        "step_0_parity": step0,
        "b_measured_artifact_ratio_off_over_on_at_10k_weights": r10k,
    },
    "estimators": {
        "bucket_mean": ("arithmetic mean of per-batch (16x4) training-log values "
                        "in a step range; DESCRIPTIVE, not a held-out interval"),
        "intervals": ("step_matched_paired_circular_moving_block_bootstrap, "
                      f"B={_args.n_boot}, block={_args.block} log rows "
                      "(= 200 steps), n=40 rows per arm over 0-2k"),
        "null_band": ("artifact_only_null_band = [v3enc_logged x r_10k, "
                      "v3enc_logged x 1.0]; r <= 1 holds at any weights, "
                      "r = 0.6081 was MEASURED by experiment B at step-10,000 "
                      "weights. Applying r_10k at 0-2k weights is a BOUNDED "
                      "EXTRAPOLATION, flagged UNVERIFIED"),
    },
    "bucket_means": buckets,
    "paired_contrasts_0-2k": paired,
    "headline": {
        "metric": "g_op_fwd_ade_m",
        "primary_window": "0-2000 (the post-mortem's published bucket)",
        "0-2000": {
            "v1": op["v1"], "v3enc_logged": op["v3enc_logged"],
            "expA_nodrop": op["expA_nodrop"],
            "recovery_fraction_of_logged_gap": op.get(
                "recovery_fraction_of_logged_gap"),
            "recovery_fraction_ci": {k: rho.get(k) for k in ("rho", "lo", "hi")},
            "artifact_only_null_band": op.get("artifact_only_null_band"),
            "expA_below_null_band": op.get("expA_below_null_band")},
        "500-2000_transient_free": {
            "v1": op5["v1"], "v3enc_logged": op5["v3enc_logged"],
            "expA_nodrop": op5["expA_nodrop"],
            "recovery_fraction_of_logged_gap": op5.get(
                "recovery_fraction_of_logged_gap"),
            "recovery_fraction_ci": {k: rho5.get(k) for k in ("rho", "lo", "hi")},
            "artifact_only_null_band": op5.get("artifact_only_null_band"),
            "expA_below_null_band": op5.get("expA_below_null_band")},
        # The ARTIFACT-FREE channel: B proved g_*_mid_de_m reads only
        # (z_t, fut_states) and NO actions, so the mask cannot touch the
        # MEASUREMENT (max|delta| = 0.0 at fixed weights). Any exp-A-vs-v3enc
        # difference here is therefore PURE WEIGHT EFFECT, with no transport and
        # no extrapolation. This is the one channel B could say nothing about.
        "encoder_terms_artifact_free": {
            k: {"expA_nodrop": buckets[k]["0-2000"]["expA_nodrop"],
                "v3enc": buckets[k]["0-2000"]["v3enc_logged"],
                "v1": buckets[k]["0-2000"]["v1"],
                "paired_expA_minus_v3enc":
                    paired[k]["0-2000"].get("expA_minus_v3enc")}
            for k in ENCODER},
    },
    "what_2k_steps_CANNOT_settle": [
        "The held-out EVAL gap (4.60x / 3.19x). Nothing here is a val number, and "
        "TanitEval loads .eval() so the mask was never active in any eval.",
        "v3enc's PLATEAU. It flatlined from ~step 4,500 — after this window ends. "
        "0-2k measures the descent, not the plateau.",
        "rollout_k. It is 4 in v3enc AND v1 for step < 5000, so DO-NOT-CARRY #2 "
        "is not under test here.",
        "decorr. decorr_w = 0.0 for step < 10000.",
        "Steady-state learning rate. The whole 0-2k window is inside the 2,000-"
        "step LR warmup, so it measures the warmup transient.",
        "The other 11 active v3enc-vs-v1 lever differences: exp-A isolates the "
        "ego-dropout against v3enc, NOT against v1.",
    ],
}
Path(_args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
print(json.dumps(res["integrity"]["batch_sequence_shared_with_v3enc"], indent=2))
print(json.dumps(res["headline"], indent=2))
print(f"[out] {_args.out}")
