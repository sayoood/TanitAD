"""Apply the S1b / S1c source edits to a refc.py + refc_train.py pair. IDEMPOTENT.

Exists because the repo lives on a Google Drive mount that went into a whole-mount
read-failure state mid-edit; this reproduces the identical edit on any checkout
(Thor's, and the repo's again once the mount returns) from ONE source of truth, so
the two cannot drift.

Verify after running:  python -c "import tanitad.refs.refc as r; ..." + param_breakdown
"""
from __future__ import annotations
import sys
from pathlib import Path

SEL_FIELD_OLD = """    refined: bool = False             # S1 rank on the refined confidence
    reach_clamp: bool = False         # S2 bounded-acceleration candidate band"""
SEL_FIELD_NEW = """    refined: bool = False             # S1 rank on the refined confidence
    score_emitted: bool = False       # S1b read that confidence FROM THE EMITTED
    #                                   fan (one extra conf-only pass), so the
    #                                   scored object IS the ranked object
    score_emitted_t: int = -1         # ...with WHICH timestep token. -1 continues
    #                                   the loop's own schedule; >=0 pins one.
    #                                   t=0 is the ONLY token `loss_cls` ever
    #                                   supervises, and "this estimate is clean"
    #                                   is arguably what a denoised fan IS.
    #                                   0 parameters either way (the embedding
    #                                   table already carries every index).
    reach_clamp: bool = False         # S2 bounded-acceleration candidate band"""

ANYON_OLD = """        return bool(self.refined or self.reach_clamp or self.graft_cons
                    or self.graft_route or self.graft_goal
                    or self.seam_clamp > 0.0)"""
ANYON_NEW = """        return bool(self.refined or self.score_emitted or self.reach_clamp
                    or self.graft_cons or self.graft_route or self.graft_goal
                    or self.seam_clamp > 0.0)"""

CFG_OLD = """    sel_reach_clamp: bool = False     # S2 bounded-acceleration band on the"""
CFG_NEW = '''    # --- S1's CLIMB-OUT: two ZERO-PARAMETER distribution matches --------------
    # E-SEL-0 MEASURED that the refined readout, UNSUPERVISED, ranks 0.8372 m
    # (base) / 0.9187 m (XL) WORSE than the shipped t=0 score - separated, both
    # arms - while still scoring 8.7x / 16.6x chance. So it is off-distribution,
    # not uninformative, and S1 must CLIMB OUT (supervise it) rather than HARVEST
    # it. These two flags remove the two places where the object that is SCORED,
    # the object that is SUPERVISED and the object that is EMITTED are still
    # three different things. Both cost 0 parameters.
    sel_score_emitted: bool = False   # S1b the ranked confidence is read from the
    #                                   EMITTED fan. Today the last denoise pass
    #                                   scores its own INPUT `x_in` and the fan
    #                                   that leaves the decoder is `x_in + off` -
    #                                   the emitted trajectories are never scored
    #                                   by any head. Costs one extra conf-only
    #                                   decoder pass; 0 parameters. Inert at
    #                                   steps=0 BY CONSTRUCTION (as S1 is).
    sel_ce_reach: bool = False        # S1c the ranked-score CE normalises over
    #                                   EXACTLY the survivor set the argmax ranks
    #                                   over, and its target is the best candidate
    #                                   IN that set. Today the CE is a full-fan
    #                                   softmax while the selector solves a
    #                                   ~26-28 % sized problem: MEASURED 73.76 %
    #                                   (base) / 72.08 % (XL) of the fan is
    #                                   unreachable and never selected, and
    #                                   S3_DEPLOYABLE 3.2 measured that a
    #                                   statistic over the whole candidate axis is
    #                                   DOMINATED by candidates no selector ever
    #                                   picks. REQUIRES sel_reach_clamp (the mask
    #                                   is its survivor set); the trainer refuses
    #                                   the combination without it. 0 parameters.
    sel_score_emitted_t: int = -1     # ...and with WHICH timestep token (-1 =
    #                                   continue the loop's schedule). MEASURED
    #                                   POST-HOC: `loss_cls` supervises the conf
    #                                   head ONLY at t=0, so the token matters.
    sel_reach_clamp: bool = False     # S2 bounded-acceleration band on the'''

SELFN_OLD = """        return SelectionConfig(
            refined=self.sel_refined, reach_clamp=self.sel_reach_clamp,"""
SELFN_NEW = """        return SelectionConfig(
            refined=self.sel_refined, score_emitted=self.sel_score_emitted,
            score_emitted_t=self.sel_score_emitted_t,
            reach_clamp=self.sel_reach_clamp,"""

FWD_OLD = """        # ---- the RANKED score ------------------------------------------------
        base = refined if sel.refined else conf"""
FWD_NEW = '''        # S1b: THE READOUT ABOVE SCORES THE WRONG OBJECT, AND IT IS ONE LINE
        # OF SOURCE. `_decode(kv, cond, x_in, t)` returns the confidence OF
        # `x_in` alongside the offset that improves it, and the loop then emits
        # `x = x_in + off`. So `refined` is the confidence of the estimate the
        # LAST pass CONSUMED, never of the fan that leaves this method - the
        # emitted trajectories are scored by no head at all. That is D1 again,
        # one denoise step less severe: the shipped ranker is 2 passes stale and
        # S1's refined ranker is 1 pass stale.
        #
        # THE EMITTED FAN IS NOT TOUCHED. The extra pass keeps its confidence and
        # DISCARDS its offset, so `anchor_traj` - and therefore the published
        # oracle-in-fan (0.1914 base / 0.1640 XL) that every D-SEL contrast is
        # paired against - is bit-unchanged. The cost is one extra decoder pass
        # and ZERO parameters.
        #
        # `t_idx` continues the loop's own schedule (pass i used `i + 1`),
        # clamped to the embedding table exactly as the loop clamps it.
        prefinal = None
        if sel.score_emitted and steps > 0:
            t_e = (min(steps + 1, self.cfg.diffusion_steps)
                   if sel.score_emitted_t < 0
                   else min(sel.score_emitted_t, self.cfg.diffusion_steps))
            e_conf, _ = self._decode(kv, cond, x, t_e)
            prefinal = refined
            refined, _ = self._apply_grafts(e_conf, terms, self._seam_refined,
                                            "refined", 0)

        # ---- the RANKED score ------------------------------------------------
        base = refined if sel.refined else conf'''

REACH_OLD = """        rank = score
        if sel.reach_clamp and v_ms is not None:"""
REACH_NEW = """        rank = score
        reach_keep = None
        if sel.reach_clamp and v_ms is not None:"""

KEEP_OLD = """            keep = keep | dead[:, None]
            rank = score.masked_fill(~keep, float("-inf"))"""
KEEP_NEW = """            keep = keep | dead[:, None]
            reach_keep = keep
            rank = score.masked_fill(~keep, float("-inf"))"""

OUT_OLD = """        if cons_s is not None:
            out["cons_score"] = cons_s
        return out"""
OUT_NEW = '''        if cons_s is not None:
            out["cons_score"] = cons_s
        if prefinal is not None:
            # S1b's own control, carried in the SAME forward: the readout S1
            # ships today, next to the one that scores the emitted fan. A
            # cross-forward comparison would confound the change with float
            # non-determinism; this one cannot.
            out["prefinal_logits"] = prefinal
        if reach_keep is not None:
            # S1c consumes this in the trainer. The argmax above already ranks
            # over exactly this set, so exporting it is what lets the CROSS-
            # ENTROPY normalise over the same support instead of over a fan that
            # is 72-74 % unpickable.
            out["reach_keep"] = reach_keep
        return out'''

MODEL_OLD = """        if "cons_score" in dec:
            out["cons_score"] = dec["cons_score"]"""
MODEL_NEW = '''        if "cons_score" in dec:
            out["cons_score"] = dec["cons_score"]
        for _k in ("prefinal_logits", "reach_keep"):
            # S1b's in-forward control and S1c's CE support, passed through
            # VERBATIM: `compute_losses` reads `reach_keep` and the probes read
            # both. Re-deriving either outside the decoder is how two
            # definitions of the same mask drift apart - the reason
            # `refc_select.reachability_mask` is a re-export and not a copy.
            if _k in dec:
                out[_k] = dec[_k]'''

REFC_EDITS = [("SelectionConfig.score_emitted", SEL_FIELD_OLD, SEL_FIELD_NEW),
              ("RefCModel.forward pass-through", MODEL_OLD, MODEL_NEW),
              ("SelectionConfig.any_on", ANYON_OLD, ANYON_NEW),
              ("RefCConfig.sel_score_emitted+sel_ce_reach", CFG_OLD, CFG_NEW),
              ("RefCConfig.selection()", SELFN_OLD, SELFN_NEW),
              ("decoder.forward S1b pass", FWD_OLD, FWD_NEW),
              ("decoder.forward reach_keep init", REACH_OLD, REACH_NEW),
              ("decoder.forward reach_keep set", KEEP_OLD, KEEP_NEW),
              ("decoder.forward out dict", OUT_OLD, OUT_NEW)]

# ---------------------------------------------------------------- refc_train.py
TR_CE_OLD = """        fan_err = (out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1).mean(-1)
        r_star = fan_err.argmin(dim=1)                      # [B] oracle index
        loss_rcls = F.cross_entropy(out["sel_score"], r_star.detach())"""
TR_CE_NEW = '''        fan_err = (out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1).mean(-1)
        # S1c - THE CE MUST NORMALISE OVER THE SET THE ARGMAX RANKS OVER.
        # Without this the objective is a full-fan softmax while the selector
        # solves a ~26-28 % sized problem: MEASURED 73.76 % (base) / 72.08 % (XL)
        # of the emitted fan is outside the bounded-acceleration band, is never
        # selected, and deleting it moves ADE by EXACTLY 0.0. A softmax that
        # normalises over those candidates spends its mass on classes that can
        # never win - the same mechanism S3_DEPLOYABLE 3.2 measured when a rank
        # correlation over the full candidate axis turned out to be disconnected
        # from selection quality FOR EVERY SCORE, including the future-seeing one.
        #
        # The TARGET moves with the support: `r_star` becomes the best candidate
        # IN the survivor set, so the CE is never asked to put mass on a class its
        # own softmax has masked out (which is how a masked CE produces NaN).
        # `reach_keep` always has >= 1 survivor per row - the decoder's empty-set
        # fallback ORs a dead row back to its whole fan - so both the argmin and
        # the log-softmax are finite by construction, not by luck.
        ce_score, ce_err = out["sel_score"], fan_err
        if cfg.sel_ce_reach:
            keep = out["reach_keep"]                          # [B, N] bool
            ce_err = fan_err.masked_fill(~keep, float("inf"))
            ce_score = out["sel_score"].masked_fill(
                ~keep, torch.finfo(out["sel_score"].dtype).min / 4)
        r_star = ce_err.argmin(dim=1)                       # [B] oracle index
        loss_rcls = F.cross_entropy(ce_score, r_star.detach())'''

TR_TELE_OLD = """                "frac_sel_2x_worse": (sel_err > 2.0 * oracle).float().mean(),
            }"""
TR_TELE_NEW = """                "frac_sel_2x_worse": (sel_err > 2.0 * oracle).float().mean(),
            }
            if cfg.sel_ce_reach:
                # telemetry, not a metric: how much of the fan the CE is actually
                # normalising over, so a silent collapse to "the whole fan" (or
                # to one candidate) is visible in train_log.jsonl, not inferred.
                sel_extra["ce_support_frac"] = keep.float().mean()"""

TR_SET_OLD = """    cfg.sel_refined = bool(args.sel_refined)
    cfg.sel_reach_clamp = bool(args.sel_reach_clamp)"""
TR_SET_NEW = '''    cfg.sel_refined = bool(args.sel_refined)
    cfg.sel_score_emitted = bool(args.sel_score_emitted)
    cfg.sel_score_emitted_t = int(args.sel_score_emitted_t)
    cfg.sel_ce_reach = bool(args.sel_ce_reach)
    cfg.sel_reach_clamp = bool(args.sel_reach_clamp)'''

TR_GUARD_OLD = """    # --- D-SEL: the selection surface (gated BEFORE build, module presence) ---
    cfg.sel_refined = bool(args.sel_refined)"""
TR_GUARD_NEW = '''    # S1c NEEDS S2's SURVIVOR SET, AND A SILENTLY-INERT FLAG IS THE WORST
    # OUTCOME. Without --sel-reach-clamp the decoder computes no mask, so
    # --sel-ce-reach would train the FULL-FAN CE while config.json claimed the
    # restricted one - an arm that reads as a treatment and behaves as a control.
    # That is the D-TAC1 `tactical_speed_input` failure exactly ("a conservative
    # guard that makes an effect unattributable is not conservative"). Refuse at
    # parse time, not after a GPU-day.
    # ...and the SAME silent-inert class for S1b: with `sel_refined` off the
    # ranked score is `conf`, so moving the REFINED readout cannot reach the
    # argmax at all. ⚠️ MEASURED (E-S1-0, frozen 30 k weights): scoring the
    # emitted fan is 0.99 m WORSE than scoring its predecessor, because it moves
    # the readout FURTHER from the only distribution `loss_cls` supervises. S1b
    # is therefore admissible ONLY inside an arm that also SUPERVISES the readout
    # it moves - which is exactly what --sel-refined turns on.
    if bool(args.sel_score_emitted) and not bool(args.sel_refined):
        raise SystemExit(
            "--sel-score-emitted requires --sel-refined. Without S1 the ranked "
            "score is the t=0 classifier `conf`, so moving the REFINED readout "
            "cannot reach the argmax - the flag would be SILENTLY INERT while "
            "config.json recorded it as ON. And at frozen weights the emitted "
            "readout is MEASURED 0.99 m WORSE (E-S1-0), so it must never be "
            "shipped without the supervision that is supposed to repair it.")
    # ⚠️ MEASURED, and loud because the default is the WORSE half of a 5.5x
    # difference. E-S1-0 dose-response, refc-base-30k, 881 windows, frozen 30 k
    # weights, selection ADE@2s over the S2-reachable survivors:
    #     conf(anchors, t=0)  SUPERVISED  0.4728   <- the shipped ranker
    #     conf(X_1,     t=2)              1.3100   <- what S1 ranks on TODAY
    #     conf(X_2,     t=2)              2.3024   <- S1b at the default token
    #     conf(X_2,     t=0)              0.6253   <- S1b at the SUPERVISED token
    # Pinning the token is -1.6771 m [-1.9430, -1.4091], separated, for ZERO
    # parameters: `loss_cls` supervises `conf_head` at t=0 ONLY, so t=1/t=2 are
    # tokens no confidence objective ever shaped.
    if bool(args.sel_score_emitted) and int(args.sel_score_emitted_t) < 0:
        print("[d-sel] ⚠️ --sel-score-emitted with --sel-score-emitted-t -1 "
              "(continue the denoise schedule). MEASURED at frozen 30 k weights "
              "this is the WORSE choice: 2.3024 vs 0.6253 selection ADE@2s, a "
              "paired -1.6771 m [-1.9430, -1.4091] in favour of `-t 0`. Both "
              "cost 0 parameters. Pass `--sel-score-emitted-t 0` unless the "
              "arm's purpose is to test the token itself.", flush=True)
    if bool(args.sel_ce_reach) and not bool(args.sel_reach_clamp):
        raise SystemExit(
            "--sel-ce-reach requires --sel-reach-clamp. S1c normalises the "
            "ranked-score CE over the SAME survivor set the argmax ranks over, "
            "and that set is S2's reachability mask - with S2 off there is no "
            "mask, and the flag would be SILENTLY INERT while config.json "
            "recorded it as ON.")
    # --- D-SEL: the selection surface (gated BEFORE build, module presence) ---
    cfg.sel_refined = bool(args.sel_refined)'''

TR_BANNER_OLD = """        "sel_on": _sel.any_on, "sel_refined": cfg.sel_refined,
        "sel_reach_clamp": cfg.sel_reach_clamp,"""
TR_BANNER_NEW = """        "sel_on": _sel.any_on, "sel_refined": cfg.sel_refined,
        "sel_score_emitted": cfg.sel_score_emitted,
        "sel_score_emitted_t": cfg.sel_score_emitted_t,
        "sel_ce_reach": cfg.sel_ce_reach,
        "sel_reach_clamp": cfg.sel_reach_clamp,"""

TR_BANNER2_OLD = """        "s1_inert_because_classifier_mode": (cfg.sel_refined
                                             and args.mode == "classifier"),"""
TR_BANNER2_NEW = """        "s1_inert_because_classifier_mode": (cfg.sel_refined
                                             and args.mode == "classifier"),
        # S1b is inert at 0 denoise steps for the same structural reason.
        "s1b_inert_because_classifier_mode": (cfg.sel_score_emitted
                                              and args.mode == "classifier"),
        "ce_support": ("reachable_only" if cfg.sel_ce_reach else "full_fan"),"""

TR_LOG_OLD = """                   ("cls_refined", "oracle_ade", "sel_ade", "sel_gap",
                    "rank_acc", "frac_sel_2x_worse", "goal_dir", "goal_dist",
                    "goal_valid_frac", "goal_gate", "goal_dist_gate")"""
TR_LOG_NEW = """                   ("cls_refined", "ce_support_frac", "oracle_ade",
                    "sel_ade", "sel_gap",
                    "rank_acc", "frac_sel_2x_worse", "goal_dir", "goal_dist",
                    "goal_valid_frac", "goal_gate", "goal_dist_gate")"""

TR_VLOG_OLD = """                               ("cls_refined", "oracle_ade", "sel_ade",
                                "sel_gap", "rank_acc", "frac_sel_2x_worse")"""
TR_VLOG_NEW = """                               ("cls_refined", "ce_support_frac",
                                "oracle_ade", "sel_ade",
                                "sel_gap", "rank_acc", "frac_sel_2x_worse")"""

TR_ARG_OLD = '''    ap.add_argument("--sel-refined", action="store_true",'''
TR_ARG_NEW = '''    ap.add_argument("--sel-score-emitted", action="store_true",
                    help="S1b: read the ranked confidence FROM THE EMITTED fan "
                         "(one extra conf-only decoder pass; the offset is "
                         "discarded so anchor_traj is bit-unchanged). Today the "
                         "last denoise pass scores its own INPUT and the fan "
                         "that leaves the decoder is scored by no head at all. "
                         "0 parameters. Inert at --mode classifier.")
    ap.add_argument("--sel-score-emitted-t", type=int, default=-1,
                    help="S1b timestep token for the emitted-fan readout. -1 "
                         "continues the denoise schedule; 0 pins the ONLY token "
                         "`loss_cls` ever supervises. 0 parameters either way.")
    ap.add_argument("--sel-ce-reach", action="store_true",
                    help="S1c: normalise the ranked-score CE over EXACTLY the "
                         "survivor set the argmax ranks over, with its target "
                         "the best candidate IN that set. REQUIRES "
                         "--sel-reach-clamp. 0 parameters.")
    ap.add_argument("--sel-refined", action="store_true",'''

TRAIN_EDITS = [("compute_losses S1c CE", TR_CE_OLD, TR_CE_NEW),
               ("compute_losses ce_support_frac", TR_TELE_OLD, TR_TELE_NEW),
               ("train() guard", TR_GUARD_OLD, TR_GUARD_NEW),
               ("train() cfg set", TR_SET_OLD, TR_SET_NEW),
               ("banner flags", TR_BANNER_OLD, TR_BANNER_NEW),
               ("banner inert rows", TR_BANNER2_OLD, TR_BANNER2_NEW),
               ("step-log ce_support_frac", TR_LOG_OLD, TR_LOG_NEW),
               ("val-log ce_support_frac", TR_VLOG_OLD, TR_VLOG_NEW),
               ("argparse flags", TR_ARG_OLD, TR_ARG_NEW)]


def apply(path: Path, edits) -> list[str]:
    src = path.read_text(encoding="utf-8")
    done = []
    for name, old, new in edits:
        if new in src:
            done.append(f"SKIP (already applied): {name}")
            continue
        if src.count(old) != 1:
            raise SystemExit(f"REFUSING: anchor for {name!r} occurs "
                             f"{src.count(old)} times in {path} (need exactly 1)")
        src = src.replace(old, new)
        done.append(f"applied: {name}")
    path.write_text(src, encoding="utf-8", newline="\n")
    return done


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    if len(a) != 2:
        raise SystemExit("usage: s1_patch_refc.py <refc.py> <refc_train.py>")
    for line in apply(Path(a[0]), REFC_EDITS):
        print(f"[refc.py]       {line}")
    for line in apply(Path(a[1]), TRAIN_EDITS):
        print(f"[refc_train.py] {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
