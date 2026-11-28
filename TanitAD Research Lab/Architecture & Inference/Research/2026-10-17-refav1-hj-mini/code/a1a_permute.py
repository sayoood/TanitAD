"""PREREG_RA1_ES Addendum D: A1a (A1 with its action sequences permuted ACROSS windows) against the banked A1, at two settings.

usage: a1a_permute.py <raw/ra1_es dir> <out.json>
  S1 = st250 (250 steps, wd 1e-3), S2 = st1000_wd3.0 (1,000 steps, wd 3.0). In each root: A1a_b0.08g50_s0 (new), A1_b0.08g50_s0/_s1,
  A1x_b0.08g50_s0 (banked); A0_b0.08_s0 and R0_s0 at S1; R0_s0 at S2 is the unregularised st1000 run.
  Paired episode-cluster bootstrap (B = 2000, seed 0) on the SAME 2,399 eval-141 windows (asserted) -- EPISODE variance only.
  The manipulation check reads each run's train_log.jsonl (unweighted idm/L1, last row); the same-clip leak is recomputed EXACTLY
  from (seed, step): the trainer's sample_batch draw and the A1a derangement are both pure functions of them."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

CELL = "b0.08g50"
SETTINGS = {"S1": dict(root="st250", steps=250, wd="1e-3", r0="st250"),
            "S2": dict(root="st1000_wd3.0", steps=1000, wd="3.0", r0="st1000")}
BATCH, N_TRAIN, SEED = 64, 540, 0


def pair(a, b, eid):
    from taniteval import ci
    r = ci.paired_episode_cluster_bootstrap(np.asarray(a, np.float64), np.asarray(b, np.float64), eid, n_boot=2000, seed=0)
    return {k: (bool(r[k]) if k == "separated" else float(r[k])) for k in ("delta", "lo", "hi", "separated") if k in r}


def last_idm(run_dir):
    rows = [json.loads(l) for l in open(os.path.join(run_dir, "train_log.jsonl"), encoding="utf-8") if l.strip()]
    return float(rows[-1]["idm/L1"]), int(rows[-1]["step"])


def same_clip_leak(steps, seed=SEED, batch=BATCH, n_train=N_TRAIN):
    """Fraction of (step, window) pairs whose donor window comes from the SAME clip (an UPPER bound of the true-action leak: a same
    clip need not mean overlapping frames). Mirrors ra1_pooled.sample_batch and the trainer's shuffle_generator exactly."""
    import torch
    from tanitad.models import ra1_hjepa as R
    hits = 0
    for step in range(steps):
        rng = np.random.default_rng((seed, 104729, step))
        idx = rng.integers(0, n_train, size=batch)
        perm = R.window_derangement(batch, torch.Generator().manual_seed(seed * 1_299_709 + step)).numpy()
        hits += int((idx == idx[perm]).sum())
    return hits / (steps * batch)


def reading(rows: dict) -> dict:
    """The committed readings of Addendum D, applied mechanically. ``rows[S]`` holds ``d`` (A1a - A1: delta/lo/hi/separated),
    ``idm_a1a`` / ``idm_a1`` (manipulation check) and ``floor`` (A1's banked training-seed floor at that setting)."""
    manip = {s: bool(r["idm_a1a"] > r["idm_a1"]) for s, r in rows.items()}
    if not all(manip.values()):
        bad = [s for s, ok in manip.items() if not ok]
        return dict(outcome="INSTRUMENT_FAIL", manipulation_ok=manip,
                    text=f"MANIPULATION CHECK FAILS at {', '.join(bad)}: A1a's idm/L1 is not above A1's, so the permutation did not remove "
                         "the information the IDM uses. Neither reading is made.",
                    next="find what the permuted target still carries (same-clip donors, the action marginal) before any reading")
    pos = {s: bool(r["d"]["separated"] and r["d"]["delta"] > 0) for s, r in rows.items()}
    qual = []
    if all(pos.values()):
        within = [s for s, r in rows.items() if abs(r["d"]["delta"]) <= 2 * r["floor"]]
        for s in within:
            qual.append(f"within 2x A1's seed floor at {s} (|delta| {abs(rows[s]['d']['delta']):.4f} <= 2 x {rows[s]['floor']:.4f}): "
                        "needs A1a seed 1 before it is quoted as a lever effect")
        return dict(outcome="TRUE_ACTION_INFO", manipulation_ok=manip, separated_positive=pos, qualifiers=qual,
                    text="READING 1: A1a - A1 > 0, separated, at both settings => the gain needs the TRUE action information "
                         "(ego-motion supervision of the visual latent, H-RA1-ES-3a supported).",
                    next="make that supervision explicit and cheap (an IDM-only arm, or the same IDM term on R0), each with its own pre-registration")
    hit = [s for s, ok in pos.items() if ok]
    if len(hit) == 1:
        other = [s for s in rows if s not in hit][0]
        qual.append(f"mixed: separated at {hit[0]} only; the cheapest next experiment is A1a seed 1 at {other}")
    return dict(outcome="GRADIENT_REGULARISATION_ARTIFACT", manipulation_ok=manip, separated_positive=pos, qualifiers=qual,
                text="READING 2: A1a - A1 is not > 0 separated at both settings => the gain is a gradient-regularisation artifact "
                     "(H-RA1-ES-3b): the IDM gradient helps without the true pairing.",
                next="isolate the regulariser (weight noise, or a random-target auxiliary loss of the same magnitude on R0), with its own pre-registration")


def main(d, out):
    def pw(root, run):
        return np.load(os.path.join(d, root, run, "eval", "perwindow.npz"), allow_pickle=True)

    def ev(root, run):
        return json.load(open(os.path.join(d, root, run, "eval", "eval.json"), encoding="utf-8"))

    res = dict(schema="ra1-a1a-permute-v1", prereg="PREREG_RA1_ES.md Addendum D (landed bc570e07 before any A1a data)", cell=CELL, seed=SEED,
               estimator="paired episode-cluster bootstrap B=2000 seed 0 (EPISODE variance only); ONE training seed; the readout is "
                         "deterministic (no inference variance)",
               tier="T1-class open-loop trajectory (eval-141 ADE@6s)", settings={}, h_hj_k_1="NOT reopened: the A1x gate binds; reading K "
               "under a replacement control is the PI's call")
    ref = None
    rows = {}
    for s, cfg in SETTINGS.items():
        root = cfg["root"]
        runs = {"A1a": f"A1a_{CELL}_s0", "A1_s0": f"A1_{CELL}_s0", "A1_s1": f"A1_{CELL}_s1", "A1x": f"A1x_{CELL}_s0"}
        z = {k: pw(root, r) for k, r in runs.items()}
        z["R0"] = pw(cfg["r0"], "R0_s0")
        if s == "S1":
            z["A0"] = pw(root, "A0_b0.08_s0")
        for k, x in z.items():
            ref = x if ref is None else ref
            assert np.array_equal(x["eid"], ref["eid"]) and np.array_equal(x["t"], ref["t"]), f"{s} {k}: eval windows differ"
            assert np.array_equal(x["hold_eid"], ref["hold_eid"]) and np.array_equal(x["hold_t"], ref["hold_t"]), f"{s} {k}: holdout windows differ"
        e = ev(root, runs["A1a"])
        assert e["arm"] == "A1a" and e["kin_mode"] == "yaw" and int(e["ckpt_step"]) == cfg["steps"], (s, e["arm"], e["kin_mode"], e["ckpt_step"])
        assert e["instrument_ok"] is True, f"{s}: A1a's eval controls failed"
        lc = json.load(open(os.path.join(d, root, runs["A1a"], "launch_check.json"), encoding="utf-8"))
        wd_row = [r for r in lc["G_DVB"] if r["lever"] == "weight_decay"][0]
        assert wd_row["ok"] and float(wd_row["declared"]) == float(cfg["wd"]), (s, wd_row)
        sa_row = [r for r in lc["G_DVB"] if r["lever"] == "shuffle_act"][0]
        assert sa_row["ok"] and sa_row["declared"] is True and sa_row["built"] is True, (s, sa_row)
        eid, heid = ref["eid"], ref["hold_eid"]
        a1 = [z["A1_s0"]["ade6"], z["A1_s1"]["ade6"]]
        floor = abs(float(a1[0].mean()) - float(a1[1].mean()))
        idm_a1a, st_a = last_idm(os.path.join(d, root, runs["A1a"]))
        idm_a1, st_1 = last_idm(os.path.join(d, root, runs["A1_s0"]))
        assert st_a == st_1 == cfg["steps"], (s, st_a, st_1)
        dlt = pair(z["A1a"]["ade6"], a1[0], eid)
        rows[s] = dict(d=dlt, idm_a1a=idm_a1a, idm_a1=idm_a1, floor=floor)
        blk = dict(flags=f"--steps {cfg['steps']} --wd {cfg['wd']}", root=root,
                   A1a_eval_ade6=float(z["A1a"]["ade6"].mean()), A1_eval_ade6=[float(x.mean()) for x in a1], P10_eval_ade6=float(ref["p10_ade6"].mean()),
                   A1a_minus_A1=dlt, A1_seed_floor=floor, two_x_floor=2 * floor,
                   manipulation=dict(idm_L1_A1a=idm_a1a, idm_L1_A1=idm_a1, ok=bool(idm_a1a > idm_a1),
                                     idm_L1_A1x=last_idm(os.path.join(d, root, runs["A1x"]))[0]),
                   gate_rows=dict(G_LIVE_act_shuffle_nonidentity_frac=lc["G_LIVE"].get("act_shuffle_nonidentity_frac"),
                                  G_LIVE_act_shuffle_changed_frac=lc["G_LIVE"].get("act_shuffle_changed_frac"),
                                  G_DVB_ok=f"{sum(r['ok'] for r in lc['G_DVB'])}/{len(lc['G_DVB'])}", G_LIVE_ok=lc["G_LIVE"]["ok"]),
                   same_clip_donor_frac_UPPER_BOUND=same_clip_leak(cfg["steps"]),
                   REPORTED_NOT_READ=dict(
                       A1a_minus_P10=pair(z["A1a"]["ade6"], ref["p10_ade6"], eid),
                       A1_s0_minus_P10=pair(a1[0], ref["p10_ade6"], eid),
                       A1a_minus_A1x=pair(z["A1a"]["ade6"], z["A1x"]["ade6"], eid),
                       A1a_minus_R0=pair(z["A1a"]["ade6"], z["R0"]["ade6"], eid),
                       A1a_minus_A0=(pair(z["A1a"]["ade6"], z["A0"]["ade6"], eid) if "A0" in z else "A0 not run at this setting"),
                       holdout_ade6=dict(A1a=float(z["A1a"]["hold_ade6"].mean()), A1_s0=float(z["A1_s0"]["hold_ade6"].mean()),
                                         P10=float(ref["hold_p10_ade6"].mean()),
                                         A1a_minus_A1=pair(z["A1a"]["hold_ade6"], z["A1_s0"]["hold_ade6"], heid))),
                   families_A1a=e.get("families"))
        res["settings"][s] = blk
    res["reading"] = reading(rows)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for s, b in res["settings"].items():
        x = b["A1a_minus_A1"]
        print(f"{s} {b['flags']}: A1a - A1 {x['delta']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]{' SEP' if x['separated'] else ''} | floor "
              f"{b['A1_seed_floor']:.4f} | idm/L1 A1a {b['manipulation']['idm_L1_A1a']:.4f} vs A1 {b['manipulation']['idm_L1_A1']:.4f} | "
              f"same-clip donors {b['same_clip_donor_frac_UPPER_BOUND']:.4%}")
    print(res["reading"]["outcome"], "|", res["reading"]["text"], "|", res["reading"].get("qualifiers"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
