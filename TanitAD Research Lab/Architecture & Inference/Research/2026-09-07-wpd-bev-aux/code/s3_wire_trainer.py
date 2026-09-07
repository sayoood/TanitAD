"""WP-D s3 — wire the BEV auxiliary term into ``refc_v3_train.py``.

⛔ Applied as SIX ANCHORED EDITS by a script rather than by hand, so the change
is reproducible, reviewable as a diff, and each anchor is asserted to occur
**exactly once**. ``refc_v3_train.py`` is 4,954 lines and is the live refcv5
trainer; a silent double-apply or a near-miss anchor in it is expensive.

⚠️ The trainer file this patches is the REPO copy. The A40 running refcv5-v2 has
its own shipped copy and is untouched by this edit (pods receive code by
file-ship, never by git — ``CLAUDE.md``), so nothing here can reach a live run.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. CLI FLAGS
# ---------------------------------------------------------------------------
A1_OLD = '''    g5.add_argument("--agent-presence-hard", action="store_true",
                    help="hard-mask sub-threshold slots instead of soft "
                         "scaling. Soft is the default BECAUSE a hard mask has "
                         "zero gradient to the presence head through the "
                         "planner loss.")
'''
A1_NEW = A1_OLD + '''    # ---- WP-D: the BEV auxiliary loss (TRAINING-ONLY) -------------------
    g5.add_argument("--bev-aux", default="off", choices=["off", "col", "xcol"],
                    help="WP-D. Supervise the trunk's feature map with a POLAR "
                         "agent-occupancy target built from the same "
                         "obstacle.offline join --agent-join loads. TRAINING-"
                         "ONLY: the head is constructed LAST and removing it "
                         "is BIT-IDENTICAL at the planner's output (pinned by "
                         "stack/tests/test_bev_aux.py). 'col' = one ray MLP "
                         "per azimuth column, no cross-column mixing (the "
                         "CHEAP FLOOR, and the arm the cylindrical projection "
                         "justifies); 'xcol' = one self-attention block over "
                         "the column tokens, so cross-column reasoning is an "
                         "ABLATION rather than an assumption.")
    g5.add_argument("--w-bev-aux", type=float, default=0.0,
                    help="weight on the BEV auxiliary loss. ⛔ Checked against "
                         "the trajectory loss at step 0 by "
                         "refc_bev_aux.assert_loss_parity: E-DEC-18b MEASURED "
                         "the sibling PSG term sitting 10-30x above the "
                         "objective it was meant to support, at EVERY weight "
                         "tested, and destroying the encoder.")
    g5.add_argument("--bev-aux-occlusion", default="mask",
                    choices=["mask", "none"],
                    help="the THIRD STATE. 'mask' (default) marks cells behind "
                         "an occluder UNOBSERVABLE and does not supervise "
                         "them. ⛔ 'none' is the DELIBERATE REGRESSION: it "
                         "supervises occluded space as FREE, which is the "
                         "two-state merge WP-A flagged. MEASURED on B1 EVAL "
                         "(26,394 frames): 17.66 % of cells are occluded and "
                         "27.96 % of OCCUPIED cells are, so 'none' mislabels "
                         "more than a quarter of the agents as empty road.")
    g5.add_argument("--bev-aux-detach", action="store_true",
                    help="⛔ DELIBERATE REGRESSION: the head reads a DETACHED "
                         "feature map, so it learns the target and teaches the "
                         "trunk NOTHING. The arm that proves the aux gradient "
                         "actually reaches the trunk.")
    g5.add_argument("--bev-aux-rng", type=int, default=24,
                    help="range bins over --bev-aux-rmax (24 x 2.5 m = 60 m, "
                         "matching bev_raster's x_fwd_m).")
    g5.add_argument("--bev-aux-rmax", type=float, default=60.0,
                    help="target range, metres.")
    g5.add_argument("--bev-aux-pos-weight", type=float, default=30.61,
                    help="BCE positive-class weight. ⭐ A pre-registered "
                         "CONSTANT from the MEASURED corpus base rate "
                         "(3.1634 %% of supervised cells over the whole B1 "
                         "EVAL join => (1-p)/p = 30.61), NEVER computed from "
                         "the batch: a batch-derived weight makes two arms "
                         "with identical flags optimise different objectives.")
    g5.add_argument("--bev-aux-hidden", type=int, default=256)
    g5.add_argument("--bev-aux-dtok", type=int, default=64)
'''

# ---------------------------------------------------------------------------
# 2. DECLARED-FLAG GATE (the "declared but inert" registry)
# ---------------------------------------------------------------------------
A2_OLD = '''    "agent_w_project": {
'''
A2_NEW = '''    "w_bev_aux": {
        "flag": "--w-bev-aux", "term": "WP-D BEV auxiliary loss",
        "gate": lambda a: (str(getattr(a, "bev_aux", "off")) != "off"
                           and bool(getattr(a, "agent_join", None)),
                           "--w-bev-aux needs `--bev-aux col|xcol` AND "
                           "`--agent-join` (no head => no `bev_logits` in "
                           "`out`; no join => no target)"),
        "mask": None,
        "already": "_pin_refcv5_seams: the `--bev-aux off` branch and the "
                   "`--bev-aux` w/join refusals",
    },
    "agent_w_project": {
'''

# ---------------------------------------------------------------------------
# 3. CONFIG ASSEMBLY + REFUSALS
# ---------------------------------------------------------------------------
A3_OLD = '''    # --- WP-6: the agent seam -------------------------------------------- #
    if getattr(args, "agents", "off") != "off":
'''
A3_NEW = '''    # --- WP-D: the BEV auxiliary head (TRAINING-ONLY) --------------------- #
    # ⛔ Its refusals mirror the `--w-agent` family EXACTLY, because the failure
    # they prevent is the same one and it has already happened here: a seam
    # declared in config.json whose loss term is silently skipped, so the run
    # record claims a lever that never entered the gradient.
    _bev_mode = str(getattr(args, "bev_aux", "off"))
    _w_bev = float(getattr(args, "w_bev_aux", 0.0))
    if _bev_mode != "off":
        if _w_bev <= 0.0:
            raise SystemExit(
                "[v3] ⛔ --bev-aux %s with --w-bev-aux 0 builds a head, stamps "
                "it into config.json and puts ZERO gradient into the trunk. "
                "Set --w-bev-aux > 0, or --bev-aux off." % _bev_mode)
        if not getattr(args, "agent_join", None):
            raise SystemExit(
                "[v3] ⛔ --bev-aux %s without --agent-join has NO LABELS. The "
                "BEV target is built from the SAME obstacle.offline join the "
                "detection head uses; without it every window is NO_LABEL and "
                "the term would be an exact 0.0 for the whole run. Pass "
                "--agent-join <joins/train2400_agents.jsonl.xz>." % _bev_mode)
        _gh, _gw = core.encoder.grid_shape
        core.bev_aux = _refc_bev_aux.BEVAuxConfig(
            enable=True, kind=_bev_mode,
            n_rng=int(getattr(args, "bev_aux_rng", 24)),
            n_az=int(_gw),
            d_tok=int(getattr(args, "bev_aux_dtok", 64)),
            hidden=int(getattr(args, "bev_aux_hidden", 256)),
            w=_w_bev,
            pos_weight=float(getattr(args, "bev_aux_pos_weight", 30.61)),
            occlusion=str(getattr(args, "bev_aux_occlusion", "mask")),
            detach_trunk=bool(getattr(args, "bev_aux_detach", False)))
        print("[v3] WP-D BEV aux ON: kind=%s grid=%dx%d target=%dx%d w=%.4g "
              "occlusion=%s detach=%s pos_weight=%.4g"
              % (_bev_mode, _gh, _gw, core.bev_aux.n_rng, core.bev_aux.n_az,
                 _w_bev, core.bev_aux.occlusion, core.bev_aux.detach_trunk,
                 core.bev_aux.pos_weight), flush=True)
    elif _w_bev > 0.0:
        # ⛔⛔ THE SILENT ONE, and the reason this branch exists: with
        # `--bev-aux off` no head is built, so `out` carries no `bev_logits`
        # and the loss-time guard skips the term -- while `w_bev_aux` is
        # stamped into config.json. That is the `w_agent` defect verbatim.
        raise SystemExit(
            "[v3] ⛔ --bev-aux off, but --w-bev-aux %.4g > 0. No head is "
            "built, so `bev_logits` never appears in `out` and the term is "
            "SILENTLY SKIPPED while the weight is stamped into config.json. "
            "Pass --bev-aux col|xcol, or --w-bev-aux 0." % _w_bev)
    # --- WP-6: the agent seam -------------------------------------------- #
    if getattr(args, "agents", "off") != "off":
'''

# ---------------------------------------------------------------------------
# 4. DATASET — the per-window target, off the join already in hand
# ---------------------------------------------------------------------------
A4_OLD = '''    agent_join_stats: dict | None = None
'''
A4_NEW = '''    agent_join_stats: dict | None = None
    #: WP-D. ``PolarBEVSpec | None`` -- while it is None the batch carries NO
    #: ``bev_occ``/``bev_mask`` and ``--w-bev-aux > 0`` REFUSES at loss time,
    #: for exactly the reason the ``agent_join`` guard above exists.
    bev_spec = None
    bev_occlusion: str = "mask"
'''

A5_OLD = '''        return {"agent_ep": torch.tensor(eid, dtype=torch.long),
                "agent_box": t["box"][0], "agent_yaw": t["yaw"][0],
                "agent_cls": t["cls"][0], "agent_valid": t["valid"][0],
                "agent_occ": t["occ"][0], "agent_rates": t["rates"][0],
                "agent_rates_mask": t["rates_mask"][0],
                "agent_label": torch.tensor(bool(has)),
                "agent_n_raw": torch.tensor(int(n_raw), dtype=torch.long),
                "agent_n_truncated": torch.tensor(max(n_raw - pad, 0),
                                                  dtype=torch.long)}
'''
A5_NEW = '''        item = {"agent_ep": torch.tensor(eid, dtype=torch.long),
                "agent_box": t["box"][0], "agent_yaw": t["yaw"][0],
                "agent_cls": t["cls"][0], "agent_valid": t["valid"][0],
                "agent_occ": t["occ"][0], "agent_rates": t["rates"][0],
                "agent_rates_mask": t["rates_mask"][0],
                "agent_label": torch.tensor(bool(has)),
                "agent_n_raw": torch.tensor(int(n_raw), dtype=torch.long),
                "agent_n_truncated": torch.tensor(max(n_raw - pad, 0),
                                                  dtype=torch.long)}
        # ⭐ WP-D. Built from the RAW join rows (``ag``, pre-truncation and
        # pre-visibility-filter), not from the padded slot block: the slot
        # targets are a DETECTION parameterisation with a fixed query budget,
        # and rasterising them would silently drop whatever `--agent-pad`
        # truncated. `labelled=has` keeps NO_LABEL and labelled-clear apart --
        # both give an all-zero occupancy and they differ ONLY in the mask.
        if self.bev_spec is not None:
            _occ, _msk = _bev_aux.build_target(
                ag, labelled=bool(has), spec=self.bev_spec,
                occlusion=self.bev_occlusion)
            item["bev_occ"] = torch.from_numpy(_occ)
            item["bev_mask"] = torch.from_numpy(_msk)
        return item
'''

# ---------------------------------------------------------------------------
# 5. THE LOSS TERM
# ---------------------------------------------------------------------------
A6_OLD = '''    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
'''
A6_NEW = '''    # ---- WP-D: the BEV auxiliary loss -------------------------------------
    # ⛔ `obstacle.offline` enters HERE, in the loss, and nowhere in the
    # forward -- the vision-only rule enforced by WHERE the tensor is read.
    w_bev = float(getattr(model, "_w_bev_aux", 0.0))
    if w_bev > 0.0 and "bev_occ" not in batch:
        # ⛔⛔ REFUSE, DO NOT SKIP -- the `--w-agent` guard one level down, and
        # for the same measured reason: a run that trains, converges, writes a
        # checkpoint and stamps `w_bev_aux` while the head was never supervised
        # would read as "the BEV auxiliary does not help".
        raise SystemExit(
            "[v3] ⛔ --w-bev-aux > 0 but the batch carries no `bev_occ`: this "
            "dataset has no BEV target wired (`ds.bev_spec` is None), so the "
            "auxiliary loss would be SILENTLY SKIPPED while config.json "
            "stamps the weight. Pass --agent-join, or --w-bev-aux 0.")
    if w_bev > 0.0 and "bev_logits" in out and "bev_occ" in batch:
        _bv = _refc_bev_aux.bev_aux_loss(
            out["bev_logits"], batch["bev_occ"].to(device),
            batch["bev_mask"].to(device),
            pos_weight=float(core.bev_aux.pos_weight))
        loss = loss + w_bev * _bv["loss"]
        extra["bev"] = _bv["loss"]
        # ⭐ n PER TERM, in the log row, always. `bev_n_supervised` is how a
        # batch of NO_LABEL windows becomes VISIBLE as an exact 0.0 with its
        # reason, instead of a term that quietly contributes nothing; and
        # `bev_n_pos` is the base rate the controls must be read against.
        extra["bev_n_supervised"] = float(_bv["n_supervised"])
        extra["bev_n_pos"] = float(_bv["n_pos"])

    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
'''

# ---------------------------------------------------------------------------
# 6. THE WEIGHT CARRIER + THE IMPORTS + THE JOIN CALL SITE
# ---------------------------------------------------------------------------
A7_OLD = '''    model._w_agent = float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT))
'''
A7_NEW = '''    model._w_agent = float(getattr(args, "w_agent", AGENT_WEIGHT_DEFAULT))
    model._w_bev_aux = float(getattr(args, "w_bev_aux", 0.0))   # WP-D
'''

A8_OLD = '''from tanitad.refs import refc_agents as _refc_agents  # noqa: E402
'''
A8_NEW = '''from tanitad.refs import refc_agents as _refc_agents  # noqa: E402
from tanitad.refs import refc_bev_aux as _refc_bev_aux  # noqa: E402  (WP-D)
from tanitad.data import bev_aux as _bev_aux  # noqa: E402  (WP-D target)
'''

A9_OLD = '''        agent_stats = ds.enable_agent_join(
'''
A9_NEW = '''        # ⭐ WP-D: the SAME reader, the SAME join, one target more. The BEV
        # spec is attached BEFORE `enable_agent_join` runs its census so a
        # single pass over the window index covers both.
        if str(getattr(args, "bev_aux", "off")) != "off":
            ds.bev_spec = _bev_aux.PolarBEVSpec(
                n_az=int(cfg.core.encoder.grid_shape[1]),
                n_rng=int(getattr(args, "bev_aux_rng", 24)),
                r_max_m=float(getattr(args, "bev_aux_rmax", 60.0)))
            ds.bev_occlusion = str(getattr(args, "bev_aux_occlusion", "mask"))
        agent_stats = ds.enable_agent_join(
'''

EDITS = [("1 CLI flags", A1_OLD, A1_NEW),
         ("2 declared-flag gate", A2_OLD, A2_NEW),
         ("3 config assembly + refusals", A3_OLD, A3_NEW),
         ("4 dataset attrs", A4_OLD, A4_NEW),
         ("5 dataset target", A5_OLD, A5_NEW),
         ("6 loss term", A6_OLD, A6_NEW),
         ("7 weight carrier", A7_OLD, A7_NEW),
         ("8 imports", A8_OLD, A8_NEW),
         ("9 join call site", A9_OLD, A9_NEW)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    t = io.open(a.src, encoding="utf-8", newline="").read()
    if "WP-D BEV aux ON" in t:
        raise SystemExit("[s3] the file is ALREADY patched — refusing to "
                         "double-apply (a second copy of the loss term would "
                         "double the weight silently)")
    for name, old, new in EDITS:
        n = t.count(old)
        # ⛔ EXACTLY ONCE. A near-miss anchor in a 4,954-line file produces a
        # patch that looks applied and is not; a duplicate anchor produces two
        # loss terms.
        if n != 1:
            raise SystemExit(f"[s3] anchor for edit {name!r} matched {n} times "
                             f"— refusing")
        t = t.replace(old, new)
        print(f"[s3] applied edit {name}", flush=True)
    Path(a.dst).parent.mkdir(parents=True, exist_ok=True)
    io.open(a.dst, "w", encoding="utf-8", newline="").write(t)
    print(f"[s3] wrote {a.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
