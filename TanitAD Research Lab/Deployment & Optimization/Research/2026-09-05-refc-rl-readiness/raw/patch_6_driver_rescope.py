#!/usr/bin/env python3
"""Patch 6 — wire the PI's re-scoped PRIMARY endpoint into `rl_refcv3_min.py`.

Exact-string edits with assertions, so a silent no-op is impossible (the class
where `git add` exits 0 and stages nothing: an edit tool that reports success is
not evidence the edit happened). Every replacement is asserted present exactly
once BEFORE it is applied and asserted applied afterwards.

Five edits:
  1. import `fan_safety.py` at START-UP (the analysis chain is paid before the
     GPU work — the 2026-08-11 class where an analysis-time import destroys a
     completed rollout);
  2. ARMS gains the two V2-faithful ingredients per arm (`use_gt_bar`,
     `noise_mode`) plus the `ctrl_const` constant-only control arm;
  3. `make_cfg` passes them through to `PostTrainConfig`;
  4. `make_sample_fn` computes the >=GT BAR — the human's logged 2 s future
     scored under the SAME reward spec and the SAME scene ctx as the candidates —
     and hands it to the objective as `extras["gt_bar"]`;
  5. `readout` gains the FAN-SAFETY block: per-candidate flags on the emitted fan
     via `fan_safety.score_paths`, summarised over the whole fan / top-8 / top-32 /
     the selected path, with the probability mass `softmax(sel_score)` and
     `softmax(anchor_logits)` put on each flag.
"""
import io
import os
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\Admin\refcv4b_repo\stack\scripts\rl_refcv3_min.py"

with io.open(SRC, encoding="utf-8") as fh:
    t = fh.read()
orig = t

EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# --------------------------------------------------------------------------- #
# 1. the fan-safety module, imported at start-up                                #
# --------------------------------------------------------------------------- #
edit("import fan_safety",
     '''SUITE = _load_by_path("openloop_suite_for_rl",
                      os.path.join(_TE_TOOLS, "openloop_suite.py"))
''',
     '''SUITE = _load_by_path("openloop_suite_for_rl",
                      os.path.join(_TE_TOOLS, "openloop_suite.py"))
#: ⭐ THE RE-SCOPED PRIMARY ENDPOINT (PI, 2026-09-05). Imported HERE, at start-up,
#: with everything else the analysis needs — a fan-safety import that fails after
#: the arms have trained destroys paid compute and reports it as a total failure.
FS = _load_by_path("fan_safety_for_rl", os.path.join(_TE_TOOLS, "fan_safety.py"))
''')

# --------------------------------------------------------------------------- #
# 2. the arm table                                                              #
# --------------------------------------------------------------------------- #
edit("ARMS table",
     '''ARMS = {
    # name: (reward weights, w_anchor, lr, steps)  — ONE variable across rl/base:
    # the RL stage. reg_echo is the deliberate regression; ctrl0 the reproduction control.
    "rl":       dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=1e-5, steps=2000),
    "reg_echo": dict(weights={"gt_similarity": 1.0}, w_anchor=0.0, lr=1e-5, steps=2000),
    "ctrl0":    dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=0.0, steps=200),
}
''',
     '''#: ⭐ RE-SCOPED 2026-09-05 (the PI's correction). The `rl` arm is now the
#: V2-FAITHFUL stage: DiffusionDriveV2's released RL adds two things our earlier
#: arm did not have, and both are what make the update push mass AWAY from
#: collision-prone and infeasible candidates rather than toward the human:
#:   `use_gt_bar`  — positive inter-anchor advantage ONLY above the human's own
#:                   score on that window (`_model_rl.py:891-893`); collisions
#:                   stay pinned at -1 by the veto, applied AFTER the bar.
#:   `noise_mode`  — "two_scalar": one longitudinal and one lateral scalar per
#:                   trajectory, broadcast over the waypoints (`:646-654`).
#: ⚠️ The one variable against `base` is still THE RL STAGE (on/off). These two
#: are INSIDE the stage's definition, named here so the arm cannot be read as the
#: predecessor's arm with a different result.
ARMS = {
    # name: reward weights, w_anchor, lr, steps, use_gt_bar, noise_mode
    "rl":         dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=1e-5, steps=2000,
                       use_gt_bar=True, noise_mode="two_scalar"),
    # ⛔ the deliberate regression: the ego GT future INSIDE the advantage. No bar
    # (the echo IS the objective) — G-FAN must fire on it or the panel is VOID.
    "reg_echo":   dict(weights={"gt_similarity": 1.0}, w_anchor=0.0, lr=1e-5, steps=2000,
                       use_gt_bar=False, noise_mode="two_scalar"),
    # the reproduction control: lr 0, weights hash-identical, every readout delta
    # EXACTLY 0. Same ingredients as `rl` so it controls the arm that ran.
    "ctrl0":      dict(weights=dict(DEFAULT_WEIGHTS), w_anchor=1.0, lr=0.0, steps=200,
                       use_gt_bar=True, noise_mode="two_scalar"),
    # ⭐ the CONSTANT-ONLY control (the probe-panel rule, CLAUDE.md 2026-08-22):
    # EVERY component weight is 0.0, so the reward is exactly 0.0 for every
    # candidate — MEASURED, not asserted: RewardSpec over these weights returns a
    # tensor whose unique value is {0.0} and whose std is 0.0. A constant reward
    # has an identically-zero group-relative advantage, so the ONLY gradient left
    # is the anchor trust region, which is itself zero while the live fan equals
    # the frozen reference. ⇒ this arm must read the no-information value EXACTLY.
    # It differs from ctrl0 in what it proves: ctrl0 disables the OPTIMIZER
    # (lr = 0), ctrl_const leaves the optimizer live (lr = 1e-5) and removes only
    # the INFORMATION. If ctrl_const moves, something other than the reward is
    # driving the update, and no `rl` result above it means anything.
    "ctrl_const": dict(weights={k: 0.0 for k in DEFAULT_WEIGHTS}, w_anchor=1.0,
                       lr=1e-5, steps=200, use_gt_bar=False, noise_mode="two_scalar"),
}
''')

# --------------------------------------------------------------------------- #
# 3. make_cfg passes them through                                               #
# --------------------------------------------------------------------------- #
edit("make_cfg",
     '''        method="grpo", group_size=int(a.group), normalize="none",
        noise_mode="multiplicative", noise_scale=float(a.noise),''',
     '''        method="grpo", group_size=int(a.group), normalize="none",
        # ⭐ from the ARM SPEC, not from a flag default: the arm's definition is
        # what the record must show, and `to_dict()` writes both into config.json.
        noise_mode=str(spec.get("noise_mode", "multiplicative")),
        use_gt_bar=bool(spec.get("use_gt_bar", False)),
        noise_scale=float(a.noise),''')

# --------------------------------------------------------------------------- #
# 4. the >=GT bar in the sample_fn                                              #
# --------------------------------------------------------------------------- #
edit("gt_bar in make_sample_fn",
     '''def make_sample_fn(model, reference, *, echo_reward: bool):
    """(batch, cfg) -> (traj2 [B,N,G,5,2], logp [B,N,G], ctx, extras) — the 2 s PREFIX
    of the sampled fan goes to the reward; the FULL mean fan pair goes to the anchor."""
    def sample_fn(batch, cfg):''',
     '''def make_sample_fn(model, reference, *, echo_reward: bool, gt_bar: bool = False):
    """(batch, cfg) -> (traj2 [B,N,G,5,2], logp [B,N,G], ctx, extras) — the 2 s PREFIX
    of the sampled fan goes to the reward; the FULL mean fan pair goes to the anchor.

    ``gt_bar`` adds V2's >=GT POSITIVE MASK: the human's logged 2 s future is scored
    as ONE MORE CANDIDATE under the SAME RewardSpec and the SAME scene ctx, and the
    resulting per-window scalar is handed to the objective as ``extras["gt_bar"]``.

    ⛔ WHY THIS IS NOT THE ECHO THE PANEL FORBIDS. The bar is the human's SCORE,
    not the human's SHAPE. It enters the objective as a THRESHOLD on the reward
    axis — a candidate that scores above it earns positive advantage no matter how
    far it is from the logged path, and a candidate that merely reproduces the
    logged path earns nothing beyond it. The reward CONTEXT is untouched, so
    `FORBIDDEN_FUTURE_CTX` still holds on the honest arm and the preflight's
    permutation test (garble every future_* field; ctx must be bit-identical)
    still passes. The distinction is exactly the one the DDv2 analysis drew:
    a bar, not a target. `reg_echo` is what a TARGET looks like, and it is the
    arm the echo gate must fail.
    """
    def sample_fn(batch, cfg):''')

edit("gt_bar computation",
     '''        extras = {}
        if reference is not None:''',
     '''        extras = {}
        if gt_bar:
            # the human's own 2 s future, scored as a candidate under the SAME
            # spec + ctx. Shape [B,1,1,5,2] -> reward [B,1,1] -> the [B] bar.
            spec_bar = RewardSpec(weights=dict(cfg.reward_weights), dt=cfg.dt)
            gt5 = with_origin(batch["gt_traj"]).reshape(
                traj2.shape[0], 1, 1, traj2.shape[-2], 2)
            with torch.no_grad():
                extras["gt_bar"] = spec_bar(gt5, ctx).reshape(traj2.shape[0])
        if reference is not None:''')

# --------------------------------------------------------------------------- #
# 5. the fan-safety readout                                                     #
# --------------------------------------------------------------------------- #
edit("readout fan safety",
     '''        oracle = (fan[..., :N_REWARD_SLOTS, :] - gt[:, None]).norm(dim=-1).mean(dim=-1).min(dim=1).values  # [B] oracle-in-fan 2 s
        for j, wi in enumerate(b["wis"]):''',
     '''        oracle = (fan[..., :N_REWARD_SLOTS, :] - gt[:, None]).norm(dim=-1).mean(dim=-1).min(dim=1).values  # [B] oracle-in-fan 2 s

        # ⭐ THE RE-SCOPED PRIMARY ENDPOINT — fan safety on the EMITTED fan.
        # Per-candidate flags [B, N] from the shared scorer (ONE definition, the
        # same module the --dump reference and the pairing use), then summarised
        # over the whole fan, the top-8 and top-32 by the score the argmax
        # actually uses, and the selected path — plus the probability mass the
        # ranking softmax and the raw conf_head put on each flag.
        # ⚠️ lead5 is [B,1,5,2] and broadcasts over N: contact and TTC are
        # TIME-ALIGNED against the lead's own track, never a static snapshot.
        lead5 = b["lead_track"].reshape(-1, 1, len(GRID_S), 2)
        has_lead = b["has_lead"].reshape(-1)
        sc = FS.score_paths(fan2, b["v0"], lead5, lead_len_m=LEAD_LEN_DEFAULT_M)
        rank = out["sel_score"].detach().float()                        # [B, N]
        if "reach_keep" in out and out["reach_keep"] is not None:
            # rows the selector excludes carry ZERO mass, not small mass
            rank = rank.masked_fill(~out["reach_keep"].bool(), float("-inf"))
        fs = FS.summarise_fan(sc, rank, out["anchor_logits"].detach().float(),
                              sel.reshape(-1))
        for j, wi in enumerate(b["wis"]):''')

edit("readout row fields",
     '''            rows.append({"wi": int(wi), "eid": int(e_i), "clip": corp.clip_ids[e_i],
                         "has_lead": bool(b["has_lead"][j]),
                         "R1": float(r1[j].mean()), "R2": float(coll[j].float().mean()),
                         "R3": float(r3[j]), "R_FAN": float(spread[j]),
                         "R_REACH": float(reach[j]), "R_ORACLE": float(oracle[j]),
                         "sel_idx": int(sel[j])})''',
     '''            row = {"wi": int(wi), "eid": int(e_i), "clip": corp.clip_ids[e_i],
                   "has_lead": bool(has_lead[j]),
                   "R1": float(r1[j].mean()), "R2": float(coll[j].float().mean()),
                   "R3": float(r3[j]), "R_FAN": float(spread[j]),
                   "R_REACH": float(reach[j]), "R_ORACLE": float(oracle[j]),
                   "sel_idx": int(sel[j])}
            # every fan-safety summary, per window. The lead-only ones are kept
            # for EVERY window and filtered by `has_lead` at aggregation time —
            # dropping them here would make the population invisible in the file.
            for k, v in fs.items():
                row[k] = float(v[j])
            rows.append(row)''')

edit("readout aggregate",
     '''    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}
    return {"n_windows": len(rows), "n_episodes": len({r["eid"] for r in rows}),
            "n_with_lead": int(sum(r["has_lead"] for r in rows)),
            "aggregate": agg, "per_window": rows, "eval_mode": True,''',
     '''    agg = {k: float(np.mean([r[k] for r in rows])) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}
    # ⭐ the fan-safety aggregate, EACH ON ITS OWN POPULATION and each carrying
    # its n. A contact rate averaged over no-lead windows is a rate over a
    # population where the event cannot occur — the denominator, not the model,
    # would then carry the improvement.
    lead_rows = [r for r in rows if r["has_lead"]]
    fan_agg, fan_n = {}, {}
    for k in (kk for kk in rows[0] if kk.split("_", 1)[-1] in FS.FLAGS
              or kk in ("fan_peak_g_mean", "sel_peak_g", "fan_v_mean_2s_spread")):
        pop = lead_rows if any(f in k for f in FS.LEAD_ONLY) else rows
        fan_agg[k] = float(np.mean([r[k] for r in pop])) if pop else float("nan")
        fan_n[k] = len(pop)
    return {"n_windows": len(rows), "n_episodes": len({r["eid"] for r in rows}),
            "n_with_lead": int(sum(r["has_lead"] for r in rows)),
            "aggregate": agg, "fan_safety": fan_agg, "fan_safety_n": fan_n,
            "fan_safety_tool": FS.TOOL,
            "per_window": rows, "eval_mode": True,''')

# --------------------------------------------------------------------------- #
# 6. mode_arm — hand the bar to the sample_fn, and pair the fan-safety keys      #
# --------------------------------------------------------------------------- #
edit("mode_arm sample_fn + gt batch",
     '''    reference = ReferencePolicy(model).to(device) if pcfg.w_anchor > 0 else None
    sample_fn = make_sample_fn(model, reference, echo_reward=(a.arm == "reg_echo"))
    pool = scoreable_windows(corp, lead)
    rng = random.Random(int(a.seed))

    def batches():
        for _ in range(pcfg.steps):
            yield build_batch(corp, lead, rng.sample(pool, pcfg.batch), device,
                              with_gt=(a.arm == "reg_echo"))''',
     '''    reference = ReferencePolicy(model).to(device) if pcfg.w_anchor > 0 else None
    sample_fn = make_sample_fn(model, reference, echo_reward=(a.arm == "reg_echo"),
                               gt_bar=bool(pcfg.use_gt_bar))
    pool = scoreable_windows(corp, lead)
    rng = random.Random(int(a.seed))
    # ⚠️ the >=GT bar SCORES the logged future, so the batch must CARRY it. Getting
    # this wrong is a KeyError 2,000 steps in, after the compute is paid.
    need_gt = (a.arm == "reg_echo") or bool(pcfg.use_gt_bar)

    def batches():
        for _ in range(pcfg.steps):
            yield build_batch(corp, lead, rng.sample(pool, pcfg.batch), device,
                              with_gt=need_gt)''')

edit("mode_arm fan-safety deltas",
     '''    deltas = {k: paired_delta(before, after, k) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}''',
     '''    deltas = {k: paired_delta(before, after, k) for k in ("R1", "R2", "R3", "R_FAN", "R_REACH", "R_ORACLE")}
    # ⭐ THE PRIMARY ENDPOINT'S PAIRED DELTAS — same estimator, same windows, each
    # on its own population. `paired_delta` clusters by EPISODE, so a lead-only
    # key is restricted to lead windows first or its zeros dilute the estimate.
    fan_keys = [k for k in before["per_window"][0]
                if k.split("_", 1)[-1] in FS.FLAGS
                or k in ("fan_peak_g_mean", "sel_peak_g", "fan_v_mean_2s_spread")]
    fan_deltas = {}
    for k in fan_keys:
        lead_only = any(f in k for f in FS.LEAD_ONLY)
        if lead_only:
            bsub = {**before, "per_window": [r for r in before["per_window"] if r["has_lead"]]}
            asub = {**after, "per_window": [r for r in after["per_window"] if r["has_lead"]]}
        else:
            bsub, asub = before, after
        d = paired_delta(bsub, asub, k)
        d["population"] = "lead windows" if lead_only else "all windows"
        fan_deltas[k] = d''')

edit("mode_arm summary carries fan safety",
     '''    summary.update({"arm": a.arm, "gate3": gate3, "readout_deltas_paired": deltas,
                    "sel_idx_agreement_with_base": sel_same, "ckpt_after": ck_path,
                    "_tier": "T0 training-side; the capability read is the taniteval harness"})''',
     '''    summary.update({"arm": a.arm, "gate3": gate3, "readout_deltas_paired": deltas,
                    "fan_safety_before": before.get("fan_safety"),
                    "fan_safety_after": after.get("fan_safety"),
                    "fan_safety_n": after.get("fan_safety_n"),
                    "fan_safety_deltas_paired": fan_deltas,
                    "sel_idx_agreement_with_base": sel_same, "ckpt_after": ck_path,
                    "_primary_endpoint": "fan safety on the EMITTED fan (PI re-scope "
                                         "2026-09-05); the four families are SECONDARY",
                    "_tier": "T0 training-side; the capability read is the taniteval harness"})''')

# --------------------------------------------------------------------------- #
# apply                                                                         #
# --------------------------------------------------------------------------- #
fail = 0
for name, old, new in EDITS:
    n = t.count(old)
    if n != 1:
        print(f"  ⛔ {name}: anchor found {n} times, expected exactly 1")
        fail += 1
        continue
    t = t.replace(old, new, 1)
    if new not in t:
        print(f"  ⛔ {name}: replacement not present after apply")
        fail += 1
        continue
    print(f"  ok {name}")

if fail:
    raise SystemExit(f"{fail} edit(s) failed — NOTHING WRITTEN")

if t == orig:
    raise SystemExit("no change — refusing to report success")

with io.open(SRC, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(t)
print(f"patched {SRC}: {len(orig)} -> {len(t)} bytes, {len(EDITS)} edits")
