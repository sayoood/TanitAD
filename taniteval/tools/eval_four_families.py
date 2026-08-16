#!/usr/bin/env python3
"""The BINDING four-family eval for a flagship checkpoint, on any episode corpus.

⛔ WHY THIS SCRIPT EXISTS. ``stack/scripts/eval_flagship_v4.py`` gates its full
metric path on ``is_v4 = isinstance(ck, dict) and ("head" in ck)``. A v1-shaped
flagship checkpoint (keys ``grounding``/``model``/``opt``/``step``) has no
``head``, so that script can only ever run ``MODE_A_canary_only_validation`` on
it — it never emits per-window ``pred``/``gt`` and therefore cannot produce a
single one of the four families Sayed made binding on 2026-08-02. This script is
the path that can: it drives ``taniteval.rollout.collect`` (LONGITUDINAL +
LATERAL, off the dense 10 Hz path) and ``taniteval.hierarchy.run`` (TACTICAL +
STRATEGIC, which actually traverses the brains), and assembles them through
``four_families.all_families``.

⚠️ **ADE IS ONE ROW OF FOUR FAMILIES, NOT "THE RESULT".** It is reported here
because cross-arm comparability needs it, never as the headline.

⚠️ **VISION-ONLY IS THE DEPLOYABLE READ.** ``hierarchy.run`` scores the route
head under three conditioning regimes and this script surfaces all three, but the
admissible one is ``route_acc_follow`` — ``route_acc_nav`` FEEDS THE MODEL THE
ANSWER (the nav command is derived from the ego's own future) and v1 already
scored 1.0000 on it, which measured an echo of its own input. Labels may use ego;
inference may not.

⚠️ **The corpus is NOT asserted to be the canonical val split.** This runs on
whatever ``--corpus`` names, precisely so an OOD/official-split corpus can be
scored — so every output carries the corpus path and key, and numbers from
different corpora may not be compared.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch


def _p(*a):
    print(*a, flush=True)


#: fields the lead block may declare to TIME-JOIN a coarse lead track onto a dense
#: path. `four_families._distance_keeping` reads them; dropping them silently
#: reverts the join to a shape match, which only works while the two grids happen
#: to agree. See that function's ⛔ comment for why truncation is not the fallback.
_LEAD_JOIN_KEYS = ("path_steps", "dt_s")
_LEAD_REQUIRED = ("leads", "lead_lens", "speeds", "state", "eid")


def load_lead_block(path: str) -> dict:
    """Load a lead block from EITHER container the programme has actually written.

    ⛔ WHY THIS IS NOT JUST ``torch.load``. Two builders emit lead blocks and they
    do not agree on the container: ``taniteval/tools/build_lead_block.py`` ends in
    ``torch.save`` (a ``.pt`` zip), while the only lead block banked IN THIS REPO —
    ``…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`` (881 rows,
    LEAD 270 / NO_LEAD 551 / NO_LABEL 60) — was written with ``np.savez``.
    ``torch.load`` on the ``.npz`` does not degrade, it raises
    ``RuntimeError: file in archive is not in a subdirectory: leads.npy``, so the
    banked artifact could not be fed to this tool at all and the distance-keeping
    half stayed UNAVAILABLE **despite the data being present and correct**.
    MEASURED 2026-08-16.

    Returns a plain dict so the caller cannot depend on the container. ``NpzFile``
    is lazy and its ``.get`` semantics differ from ``dict``'s, which is exactly how
    an optional join key would go missing without an error.
    """
    if str(path).endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            block = {k: z[k] for k in z.files}
    else:
        block = torch.load(path, map_location="cpu", weights_only=False)
        block = dict(block)
    missing = [k for k in _LEAD_REQUIRED if k not in block]
    if missing:
        sys.exit(f"lead block {path} is missing {missing} — it cannot score "
                 f"distance-keeping. Rebuild it with tools/build_lead_block.py.")
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="dir of ep_*.pt episodes")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", default=None,
                    help="the run's config.json. Supplied -> the model is "
                         "rebuilt from the RUN'S OWN cfg and loaded STRICT, "
                         "which is the only way a drifted stack default cannot "
                         "silently produce a different architecture.")
    ap.add_argument("--arm", required=True, help="label for the output record")
    ap.add_argument("--out", required=True, help="JSON output FILE (not a dir)")
    ap.add_argument("--windows-out", default=None,
                    help="also dump the per-window tensors here")
    ap.add_argument("--episodes", type=int, default=0,
                    help="0 = ALL. ⛔ a non-zero value silently changes the "
                         "denominator; it is recorded in the output.")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--speed-input", action="store_true")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--skip-hierarchy", action="store_true",
                    help="⛔ leaves TACTICAL and STRATEGIC UNAVAILABLE. For a "
                         "smoke test only — the result is NOT admissible.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--windows-in", default=None,
                    help="RESCORE an already-banked windows dump instead of "
                         "running inference. The whole point of the lead block "
                         "is that it attaches row-for-row to a scored dump, so "
                         "closing the distance-keeping half must not cost "
                         "another scoring pass.")
    ap.add_argument("--lead", default=None,
                    help="lead block from tools/build_lead_block.py. Without it "
                         "the distance-keeping half of LONGITUDINAL stays "
                         "UNAVAILABLE — a WORK ITEM, not a pass.")
    ap.add_argument("--carry-hierarchy-from", default=None,
                    help="a prior output JSON whose TACTICAL/STRATEGIC blocks "
                         "were computed by hierarchy.run on THESE windows. "
                         "Spliced verbatim rather than recomputed; the corpus "
                         "and n_windows are checked to match.")
    a = ap.parse_args()

    if os.path.isdir(a.out):
        sys.exit(f"--out must be a FILE, got a directory: {a.out}")
    if a.windows_in and not a.skip_hierarchy and not a.carry_hierarchy_from:
        sys.exit("--windows-in cannot run hierarchy.run (it needs the model and "
                 "the episodes). Pass --carry-hierarchy-from <prior json>, or "
                 "--skip-hierarchy and accept an inadmissible result.")

    from taniteval import ci, four_families as ff, hierarchy, loaders, rollout
    from taniteval.data import load_frames

    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))
    if not files and not a.windows_in:
        sys.exit(f"no ep_*.pt under {a.corpus}")
    # --windows-in rescore never loads episodes (inference and hierarchy are both
    # skipped/carried), so a corpus in another on-disk format (e.g. the w120
    # v2-compressed cache) is fine as PROVENANCE — recorded, not globbed.
    n_avail = len(files)
    if a.episodes:
        files = files[:a.episodes]
    _p(f"[corpus] {a.corpus}\n[corpus] episodes {len(files)} of {n_avail} available")

    hier, carried = None, None
    if a.windows_in:
        win = rollout.load_windows(a.windows_in)
        _p(f"[windows] {win['pred'].shape[0]} rows loaded from {a.windows_in} "
           f"(NO inference)")
    else:
        entry = {"arch": "flagship-worldmodel-v2" if a.run_config
                 else "flagship-worldmodel",
                 "ckpt": a.ckpt, "run_config": a.run_config,
                 "speed_input": bool(a.speed_input)}
        t0 = time.time()
        h = loaders.load(entry, device=a.device)
        model, step_readout = h["model"].eval(), h["step_readout"]
        if step_readout is None:
            sys.exit("this checkpoint exposes no grounded step readout — the "
                     "operative rollout cannot be decoded from it")
        _p(f"[model] loaded STRICT in {time.time()-t0:.1f}s  arch={entry['arch']}"
           f"  step={h.get('step')}  state_dim={h.get('state_dim')}  "
           f"speed_input={a.speed_input}")

        eps = load_frames(files)
        _p(f"[episodes] {len(eps)} loaded, T[0]={eps[0].feats.shape}")

        # ---- LONGITUDINAL + LATERAL: the world-model fidelity pass ---------- #
        t0 = time.time()
        win = rollout.collect(model, step_readout, eps, a.device,
                              stride=a.stride, batch=a.batch,
                              speed_input=a.speed_input,
                              # inert without a decision_fn, but the arm's label
                              # family must never be inherited from a default
                              labels_v2=bool(h["labels_v2"]))
        _p(f"[collect] {win['pred'].shape[0]} windows in {time.time()-t0:.0f}s")
        if a.windows_out:
            rollout.save_windows(win, a.windows_out)
            _p(f"[collect] windows -> {a.windows_out}")

        # ---- TACTICAL + STRATEGIC: the pass that traverses the brains ------- #
        if not a.skip_hierarchy:
            t0 = time.time()
            hier = hierarchy.run(model, step_readout, eps, a.device,
                                 speed_input=a.speed_input, max_eps=len(eps),
                                 stride=a.stride, batch=max(a.batch, 16),
                                 n_boot=a.n_boot,
                                 # ⛔ the arm's OWN cfg.v2_labels, read from the
                                 # run config the loader already rebuilt from.
                                 # Without --run-config a `--v2` arm resolves to
                                 # v1 and its maneuver_acc is scored against
                                 # labels the head never saw (fixed 2026-08-17).
                                 labels_v2=bool(h["labels_v2"]))
            _p(f"[hierarchy] {hier.get('n_windows')} windows in "
               f"{time.time()-t0:.0f}s  skipped={hier.get('skipped')}  "
               f"maneuver_labels={'v2' if h['labels_v2'] else 'v1'}")

    # ---- the lead block: the distance-keeping half of LONGITUDINAL ---------- #
    if a.lead:
        lead = load_lead_block(a.lead)
        n_lead = int(np.asarray(lead["leads"]).shape[0])
        n_win = int(win["pred"].shape[0])
        # ⛔ POSITIONAL JOIN. A length mismatch means the block was built on a
        # different window grid, and every row after the first divergence would
        # score this arm's trajectory against another episode's traffic — a
        # plausible number, not an error. Refuse rather than truncate.
        if n_lead != n_win:
            sys.exit(f"lead block has {n_lead} rows for {n_win} scored windows — "
                     f"different window grids. Rebuild the lead block against "
                     f"this corpus/stride; do NOT truncate.")
        win["lead"] = {"leads": np.asarray(lead["leads"]),
                       "lead_lens": np.asarray(lead["lead_lens"]),
                       "speeds": np.asarray(lead["speeds"]),
                       "state": np.asarray(lead["state"]),
                       "eid": list(lead["eid"]), "n_boot": a.n_boot}
        # ⛔ CARRY THE TIME-JOIN THROUGH. A block built on the lead's own coarse
        # samples must tell the scorer WHICH path steps it is defined on; without
        # these keys `_distance_keeping` falls back to a bare shape match, which
        # silently works only while the grids coincide and refuses (or worse,
        # mis-joins) the moment a dense path is scored.
        for k in _LEAD_JOIN_KEYS:
            if k in lead and lead[k] is not None:
                v = lead[k]
                win["lead"][k] = (v.tolist() if isinstance(v, np.ndarray)
                                  else v)
        states, counts = np.unique(np.asarray(lead["state"]), return_counts=True)
        _p(f"[lead] attached {n_lead} rows  "
           f"states={dict(zip(states.tolist(), counts.tolist()))}")

    fam = ff.all_families(win, hier=hier)

    if a.carry_hierarchy_from:
        # The decision families were computed by hierarchy.run on THESE windows;
        # recomputing them would cost a second ~15-minute pass and could only
        # reproduce the same numbers. Splice, and check the provenance matches.
        prior = json.load(open(a.carry_hierarchy_from))
        if prior.get("n_windows") != int(win["pred"].shape[0]):
            sys.exit(f"--carry-hierarchy-from has n_windows "
                     f"{prior.get('n_windows')} vs {int(win['pred'].shape[0])} "
                     f"here — different windows, refusing to splice")
        for k in ("tactical", "strategic"):
            fam[k] = prior["four_families"][k]
            fam[k]["_carried_from"] = a.carry_hierarchy_from
        fam["_families_unavailable"] = [
            k for k, v in fam.items()
            if isinstance(v, dict) and v.get("status") == "UNAVAILABLE"]
        fam["_complete"] = (not fam["_families_unavailable"]
                            and fam["longitudinal"]["distance_keeping"]["status"] == "OK"
                            # ⛔ the PI's 2026-08-16 anti-echo condition is part of
                            # completeness, not an optional diagnostic — a splice
                            # that dropped it would re-open the hole it closes.
                            and fam["longitudinal"]["anti_echo"]["status"] == "OK")
        carried = a.carry_hierarchy_from

    # ---- ADE + per-family CIs, ONE episode-cluster resampling --------------- #
    # ⛔ The components are computed with four_families' OWN `_seq_geometry`, not
    # a re-derivation. A second implementation here would let the interval and
    # the point estimate drift apart silently, which is the exact failure the
    # estimator rule exists to prevent.
    pred = win["pred_dense"].float()
    gt = win["gt_dense"].float()
    dt = float(win.get("dt_s", 0.1) or 0.1)
    P, G = ff._seq_geometry(pred, dt), ff._seq_geometry(gt, dt)
    eid = list(win["eid"])

    sp_err = (P["speed"] - G["speed"]).abs().mean(1)
    al_err = (P["along"] - G["along"]).abs().mean(1)
    ct_err = (P["cross"] - G["cross"]).abs().mean(1)
    both = P["valid"] & G["valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    # per-window masked means; a window with NO valid step contributes NaN and is
    # dropped from that component only (its n is reported beside it).
    def _wmean(x, m):
        n = m.sum(1)
        out = torch.where(n > 0, (x * m).sum(1) / n.clamp_min(1), torch.nan)
        return out
    head_w = _wmean(dh.abs(), both) * 180.0 / np.pi
    pv = P["pair_valid"] & G["pair_valid"]
    curv_w = _wmean((P["curvature"] - G["curvature"]).abs(), pv)

    sparse_pred, sparse_gt = win["pred"].float(), win["gt"].float()
    ade_w = torch.linalg.norm(sparse_pred - sparse_gt, dim=-1).mean(1)
    ade2s_w = torch.linalg.norm(sparse_pred[:, -1] - sparse_gt[:, -1], dim=-1)

    comps = {
        "ade_mean_4wp_m": (ade_w.numpy(), "mean"),
        "fde_2s_m": (ade2s_w.numpy(), "mean"),
        "LON_speed_mae_mps": (sp_err.numpy(), "mean"),
        "LON_along_mae_m": (al_err.numpy(), "mean"),
        "LAT_cross_mae_m": (ct_err.numpy(), "mean"),
    }
    boots = ci.bootstrap_metrics(comps, eid, n_boot=a.n_boot)
    # heading/curvature carry NaNs (masked windows) -> their own resampling on
    # the surviving windows, with the denominator stated.
    for name, v in (("LAT_heading_mae_deg", head_w), ("LAT_curvature_mae_1pm", curv_w)):
        v = v.numpy()
        keep = ~np.isnan(v)
        boots[name] = ci.episode_cluster_bootstrap(
            v[keep], [e for e, k in zip(eid, keep) if k], n_boot=a.n_boot)
        boots[name]["n_windows_dropped_no_valid_step"] = int((~keep).sum())

    rec = {
        "arm": a.arm,
        "ckpt": a.ckpt,
        "run_config": a.run_config,
        "corpus": a.corpus,
        "corpus_key": os.path.basename(a.corpus.rstrip("/")),
        # --windows-in never materialises `eps`; the corpus listing is the
        # denominator either way, and it is the one already checked above.
        "episodes_scored": len(files),
        "episodes_available": n_avail,
        "episodes_flag": a.episodes,
        "stride": a.stride,
        "speed_input": bool(a.speed_input),
        "n_windows": int(win["pred"].shape[0]),
        "four_families": fam,
        "intervals": boots,
        "pc2": win.get("pc2"),
        "windows_in": a.windows_in,
        "lead_block": a.lead,
        "hierarchy_carried_from": carried,
        "_estimator": ("episode-cluster bootstrap over the corpus's episodes "
                       "(taniteval.ci). ⛔ overlapping_holdout_se is NOT used: it "
                       "biases the POINT ESTIMATE, not only the interval."),
        "_binding": ("Sayed 2026-08-02 — LONGITUDINAL + LATERAL + TACTICAL + "
                     "STRATEGIC in ADDITION to ADE, per-family, never pooled. A "
                     "family reported UNAVAILABLE is a WORK ITEM, not a pass."),
        "_vision_only": ("Sayed 2026-08-03 — inference is VISION-ONLY. The "
                         "deployable route read is route_acc_follow; "
                         "route_acc_nav feeds the model a label-derived nav "
                         "command and is reference-only."),
    }
    with open(a.out, "w") as f:
        json.dump(rec, f, indent=2, default=str)
    _p(f"[out] {a.out}")
    for k in ("longitudinal", "lateral", "tactical", "strategic"):
        v = fam[k]
        _p(f"  {k:14s} status={v.get('status', 'OK')}")
    _p(f"  families_unavailable={fam['_families_unavailable']}  "
       f"complete={fam['_complete']}")
    # ⛔ PRINTED, not merely serialised (PI 2026-08-16). A longitudinal number is
    # not a longitudinal RESULT until the arm is shown to beat a hold-v0 copy of
    # its own input, separated — so the operator sees that bit on the console.
    _p(f"  anti_echo      {fam.get('_anti_echo_summary')}")
    _p(f"  ⛔ longitudinal_claim_admissible="
       f"{fam.get('_longitudinal_claim_admissible')}")


if __name__ == "__main__":
    main()
