#!/usr/bin/env python3
"""The committed weight rule for L_rank (DESIGN_SCORER_FT.md 2.5), as a tool against the PATCHED refe/train.py.  Mirrors refe/weight_rule.py (the L_field rule).

    On 8 real micro-batches of the arm's own bank at the host's init (seeded draw, seed 20261012), with REAL online labels from the arm's own labeller, measure
        r(w)   = || grad_S (w * L_rank) || / || grad_S l_score ||       S = {score_q_mlp, score_dec, score_head}
        rho(w) = || grad_T (w * L_rank) || / || grad_T l_traj  ||       T = the trunk group (backbone LoRA + scene_proj)
    Candidates w in {1/64, 1/32, ..., 2, 4}.  Take the SMALLEST w with r(w) >= 1.0 ("the new term is at least as strong on the scorer as the BCE that already reached AUC 0.93-0.96
    without ranking").
    K2  : r(4)    <  1.0 -> the arm is NOT launched (the pairs are too few or too easy: report the pair counts).
    K2b : rho(w*) >  1.0 -> the arm is NOT launched (the ranking would out-pull the trajectory loss on the shared trunk): back to the Master Mind.

    python weight_rule_rank.py --args "<the arm's train.py argv, as the chain's $ARGS>" [--refe <dir of the patched refe>] [--out weight_rule_rank.json]
    exit 0  a weight was selected (written to --out; the value goes into the prereg's blank, then into the arm's argv as --rank-w)
    exit 5  K2 or K2b: the arm is not launched          exit 4  the tool could not measure (a refusal, never a guess)

HOW IT MEASURES.  It runs the REAL `train.run(a)` on the arm's argv under instrumentation until the optimiser is built (the same technique as the launch gate's BUILD run), capturing the
MODEL, the TargetBank and the OnlineLabeller the trainer itself built (warm start from --init-from, nav mode, cameras, amp, label backend and metric cache included), so the tool cannot
drift from the trainer.  Nothing is trained and nothing is written.  It then draws 8 micro-batches of --batch rows with `torch.Generator().manual_seed(20261012)` + `randperm`, forwards each
exactly as the trainer does (autocast per --amp, `traj.float()`), submits the DETACHED proposals to the arm's own labeller and collects the lag-0 labels, and takes the gradients of
l_score (the trainer's own expression: pointwise BCE over the six heads, mean(-1), constant denominator n_proposals x B, x --score-w), of L_rank (RankLoss.term, weight 1, denominator --batch)
and of l_traj (`wta_loss`, the argv's --yaw-wrap).  The gradients are POOLED over the 8 micro-batches (their mean, what an accumulation window hands the optimiser) before the norms are
taken; per-micro-batch ratios are diagnostics (min / median / max) and do not decide.  L_rank enters the trainer as `rank_w * L_rank` inside `(l_traj + l_score + ...) / accum`, so r and rho
are EXACTLY LINEAR in w: r(w) = w * r(1), and every candidate is a power of two (w * g is then bit-identical to the gradient of (w * L_rank)).

The argv must name --rank-heads (and --rank-ep-gap / --rank-gate-topk if the arm does): the term is built from them.  Its --rank-w is ignored (the rule is defined at w = 1; a 0.0 or absent
value is replaced by 1.0 for the BUILD only, because the trainer refuses heads without a weight); the argv_sha256 reported is of the argv AS GIVEN.  The pair counts per head are reported:
a named head with zero pairs over the 8 sets is listed in `dead_heads` (G-LIVE L2 consumes it).

WHAT IT CANNOT SAY: it needs the arm's real bank, images, D9 metric cache and the host's init checkpoint (C1's final: pod-side).  On the dev box it is exercised on a self-made tiny bank with the
SYNTHETIC label backend only (test_rank_loss.py), and the result is then stamped NOT A DECISION.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 20261012
N_MICRO = 8
CANDIDATES = tuple(2.0 ** k for k in range(-6, 3))            # 1/64 ... 4
R_TARGET, RHO_MAX = 1.0, 1.0
REPLAY_SEED = 20261122                                        # A3: the RNG state the three forwards of one micro-batch replay (REPLAY_SEED + k); fixes dropout masks, not the draw
GRAD_PASSES = 3                                               # A3: one forward + one backward per gradient (l_score, L_rank, l_traj); never a retained graph


class Stop(Exception):
    pass


def decide(r_by_w: dict, rho_by_w: dict) -> dict:
    """The committed rule on {w: r(w)} and {w: rho(w)}.  Pure: no I/O, no model."""
    ws = sorted(r_by_w)
    if tuple(ws) != tuple(sorted(CANDIDATES)) or sorted(rho_by_w) != ws:
        raise ValueError(f"the candidates must be exactly {CANDIDATES}, got {ws}")
    if any(not math.isfinite(r_by_w[w]) or r_by_w[w] < 0.0 or not math.isfinite(rho_by_w[w]) or rho_by_w[w] < 0.0 for w in ws):
        return {"status": "ERROR", "rank_w": None, "why": f"a non-finite or negative r(w) / rho(w): {r_by_w} {rho_by_w}"}
    if r_by_w[ws[-1]] < R_TARGET:
        return {"status": "K2", "rank_w": None, "why": f"r({ws[-1]:g}) = {r_by_w[ws[-1]]:.6g} < {R_TARGET}: the pairs are too few or too easy to pull on the scorer as hard as the BCE; report the pair counts"}
    w = next(w for w in ws if r_by_w[w] >= R_TARGET)
    if rho_by_w[w] > RHO_MAX:
        return {"status": "K2b", "rank_w": None, "w_star": w, "why": f"rho({w:g}) = {rho_by_w[w]:.6g} > {RHO_MAX}: at the smallest sufficient weight the ranking would out-pull the trajectory loss on the shared trunk; back to the Master Mind"}
    return {"status": "SELECT", "rank_w": w, "why": f"the smallest candidate with r(w) >= {R_TARGET}: r({w:g}) = {r_by_w[w]:.6g}, rho({w:g}) = {rho_by_w[w]:.6g} <= {RHO_MAX}"}


def build_instrumented(refe: Path, argv: list):
    """Run the real train.run(a) until the optimiser is built; return (T, a, model, ds, lab, dev, rank)."""
    sys.path.insert(0, str(refe))
    import torch
    import torch._functorch.config as FC
    FC.donated_buffer = False          # MM 2026-11-21: measure() takes THREE torch.autograd.grad(..., retain_graph=True) on ONE graph. Under the arm's --compile the backbone's
    # compiled backward donates saved buffers and then refuses retain_graph (pod 2026-11-21 16:43:52Z, torch 2.7.1: "This backward function was compiled with non-empty donated
    # buffers which requires create_graph=False and retain_graph=False"; the check is gated on this flag, runtime_wrappers.py:2200-2203, and so is the donation itself,
    # jit_compile_runtime_wrappers.py:878). Donation only reuses saved-activation memory: no gradient value changes. Set BEFORE train.run builds and compiles the model.
    import train as T
    import online_labeller as OLm
    ap = T.build_parser()
    argv = list(argv)
    a = ap.parse_args(argv)
    if a.labeller != "online":
        raise SystemExit("weight_rule_rank: the argv is not --labeller online: there is no label to measure L_rank against")
    if not a.rank_heads:
        raise SystemExit("weight_rule_rank: the argv has no --rank-heads: there is no term to measure")
    if not (a.rank_w > 0.0):
        a = ap.parse_args(argv + ["--rank-w", "1.0"])      # the build only: the trainer refuses heads without a weight; the rule is defined at w = 1
    if not a.out:
        a.out = str(Path(tempfile.gettempdir()) / "weight_rule_rank_unused_run_dir")          # run() refuses a long run without --out; it is never created (we stop before it)
    cap = {"model": None, "ds": None, "lab": None, "rank": None}
    Orig, Bank, AdamW0, Lab0, Rank0 = T.REFe, T.TargetBank, torch.optim.AdamW, OLm.OnlineLabeller, T.RK.RankLoss

    class Rec(Orig):
        def __init__(self, cfg):
            super().__init__(cfg)
            cap["model"] = self

    class RecBank(Bank):
        def __init__(self, *a_, **k_):
            super().__init__(*a_, **k_)
            cap["ds"] = self

    class RecLab(Lab0):
        def __init__(self, *a_, **k_):
            super().__init__(*a_, **k_)
            cap["lab"] = self

    class RecRank(Rank0):
        def __init__(self, *a_, **k_):
            super().__init__(*a_, **k_)
            cap["rank"] = self

    class StopAdam(AdamW0):
        def zero_grad(self, *a_, **k_):
            raise Stop()

    patches = [(T, "REFe", Orig), (T, "TargetBank", Bank), (torch.optim, "AdamW", AdamW0), (OLm, "OnlineLabeller", Lab0), (T.RK, "RankLoss", Rank0)]
    T.REFe, T.TargetBank, torch.optim.AdamW, OLm.OnlineLabeller, T.RK.RankLoss = Rec, RecBank, StopAdam, RecLab, RecRank
    rc = None
    try:
        rc = T.run(a)
    except Stop:
        rc = "STOPPED_AT_BUILD"
    finally:
        for obj, name, orig in patches:
            setattr(obj, name, orig)
    if rc != "STOPPED_AT_BUILD" or any(cap[k] is None for k in cap):
        raise SystemExit(f"weight_rule_rank: the trainer did not reach the optimiser build with a model, a bank, a labeller and a rank term (rc={rc!r}, built {[k for k in cap if cap[k] is not None]}); "
                         f"nothing was measured")
    dev = next(cap["model"].parameters()).device
    return T, a, cap["model"], cap["ds"], cap["lab"], dev, cap["rank"]


def _groups(model):
    """S = the scorer's three modules; T = the trunk group (backbone LoRA + scene_proj).  Trainable parameters only."""
    S, Tr = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        top = n.split(".")[0]
        if top in ("score_q_mlp", "score_dec", "score_head"):
            S.append(p)
        elif top == "scene_proj" or (top == "backbone" and n.rsplit(".", 1)[-1] in ("A", "B")):
            Tr.append(p)
    return S, Tr


def _vec(gs, ps):
    import torch
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().double().flatten() for g, p in zip(gs, ps)])


def measure(T, a, model, ds, lab, dev, rank) -> dict:
    """Per micro-batch the gradients of l_score, L_rank (weight 1) and l_traj on S and T; pooled norms; r and rho at every candidate."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import default_collate
    S, Tr = _groups(model)
    if not S or not Tr:
        raise SystemExit(f"weight_rule_rank: a parameter group is empty (scorer {len(S)}, trunk {len(Tr)} tensors): the rule needs both (--scorer-sees-trunk, LoRA)")
    n_rows = len(ds)
    need = N_MICRO * a.batch
    if n_rows < need:
        raise SystemExit(f"weight_rule_rank: the bank has {n_rows} rows, fewer than the {need} the rule draws ({N_MICRO} micro-batches x --batch {a.batch})")
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(n_rows, generator=g).tolist()[:need]
    batches = [perm[k * a.batch:(k + 1) * a.batch] for k in range(N_MICRO)]
    ids = [[(ds.rows[i].get("log_name"), ds.rows[i].get("token"), int(ds.rows[i].get("rank", 0))) for i in b] for b in batches]
    cfg = model.cfg
    acc = {k: None for k in ("sS", "sR", "sTt", "sTr")}
    per, pairs, sets, labelled_sets, total_sets = [], {h: 0 for h in rank.heads}, {h: 0 for h in rank.heads}, 0, 0
    # MM 2026-11-22 (A3): ONE FORWARD PER GRADIENT, ONE BACKWARD PER FORWARD, NO RETAINED GRAPH. A2 turned donated buffers off and the pod still crashed at the SECOND
    # torch.autograd.grad on one --compile graph (2026-11-22 21:39:51Z, torch 2.7.1: "a variable needed for gradient computation has been modified by an inplace operation:
    # [16, 1925, 1024] is at version 18; expected version 0"): the compiled backward reuses a saved activation in place, so a retained compiled graph cannot be backpropagated
    # twice. The trainer takes exactly one backward per forward, so each gradient here does the same. The three forwards replay ONE RNG state (fork_rng + manual_seed), so they
    # are the same graph; `replay_max_abs_diff_*` measures that (0.0 on a deterministic device) and is reported, not used. The labels come from the FIRST forward, as before.
    rng_devs = [dev] if dev.type == "cuda" else []

    def forward(img, ego, goal, cal, seed):
        with torch.random.fork_rng(devices=rng_devs):
            torch.manual_seed(seed)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.amp == "bf16" and dev.type == "cuda")):
                traj, score = model(img, ego, goal, calib=cal if cal.numel() else None)[:2]
        traj, score = traj.float(), score.float()
        return traj, score

    for k, b in enumerate(batches):
        batch = default_collate([ds[i] for i in b])
        img, ego, goal, tgt, cand_xy, cand_tg, cand_m, cand_id, cal = batch[:9]
        img, ego, goal, tgt = (x.to(dev) for x in (img, ego, goal, tgt))
        seed_k = REPLAY_SEED + k
        # pass 1: l_score (and the labels, from this forward's DETACHED proposals)
        traj1, score1 = forward(img, ego, goal, cal, seed_k)
        metas = [{kk: ds.rows[int(j)].get(kk) for kk in ("log_name", "token", "rank", "aug", "step")} for j in b]
        h = lab.submit(k, 0, metas, traj1.detach().double().cpu().numpy())
        labels_np, mask_np, _ = lab.collect(h, 0)
        lab_t, msk = torch.from_numpy(labels_np).to(dev), torch.from_numpy(mask_np).to(dev)
        bce = F.binary_cross_entropy_with_logits(score1, lab_t, reduction="none")
        denom = max(float(cfg.n_proposals) * float(img.shape[0]), 1.0)          # the online trainer's constant denominator (train.py: cov_norm = n_proposals)
        l_score = (bce.mean(-1) * msk[:, None]).sum() / denom * a.score_w
        gs = torch.autograd.grad(l_score, S + Tr, retain_graph=False, allow_unused=True)
        traj1, score1, l_score_v = traj1.detach(), score1.detach(), float(l_score)
        del bce, l_score
        # pass 2: L_rank (weight 1) on the same graph, replayed
        traj2, score2 = forward(img, ego, goal, cal, seed_k)
        l_rank = rank.term(score=score2, labels=lab_t, msk=msk, batch=a.batch)
        for hh in rank.heads:
            pairs[hh] += rank.last["pairs"][hh]
            sets[hh] += rank.last["sets"][hh]
        labelled_sets += int((msk > 0).sum())
        total_sets += int(msk.numel())
        d_score = float((score2.detach() - score1).abs().max())
        gr = torch.autograd.grad(l_rank, S + Tr, retain_graph=False, allow_unused=True)
        l_rank_v = float(l_rank)
        del traj2, score2, l_rank
        # pass 3: l_traj on the trunk, replayed
        traj3, score3 = forward(img, ego, goal, cal, seed_k)
        l_traj = T.wta_loss(traj3, tgt, yaw_wrap=(a.yaw_wrap == "on"))[0]
        d_traj = float((traj3.detach() - traj1).abs().max())
        gt = torch.autograd.grad(l_traj, Tr, retain_graph=False, allow_unused=True)
        l_traj_v = float(l_traj)
        del traj3, score3, l_traj
        ns = len(S)
        v = {"sS": _vec(gs[:ns], S), "sR": _vec(gr[:ns], S), "sTt": _vec(gt, Tr), "sTr": _vec(gr[ns:], Tr)}
        for kk in acc:
            acc[kk] = v[kk] if acc[kk] is None else acc[kk] + v[kk]
        nS, nR = float(v["sS"].norm()), float(v["sR"].norm())
        per.append({"l_score": l_score_v, "l_rank": l_rank_v, "l_traj": l_traj_v, "norm_score_S": nS, "norm_rank_S": nR, "norm_traj_T": float(v["sTt"].norm()),
                    "norm_rank_T": float(v["sTr"].norm()), "r1": (nR / nS) if nS > 0 else None,
                    "replay_max_abs_diff_score": d_score, "replay_max_abs_diff_traj": d_traj})
        model.zero_grad(set_to_none=True)
    pooled = {kk: float((acc[kk] / N_MICRO).norm()) for kk in acc}
    if not pooled["sS"] > 0.0:
        raise SystemExit("weight_rule_rank: ||grad_S l_score|| is zero on the pooled batches: r is undefined (no labelled set among the 8?)")
    if not pooled["sTt"] > 0.0:
        raise SystemExit("weight_rule_rank: ||grad_T l_traj|| is zero on the pooled batches: rho is undefined")
    r1, rho1 = pooled["sR"] / pooled["sS"], pooled["sTr"] / pooled["sTt"]
    rs = sorted(p["r1"] for p in per if p["r1"] is not None)
    return {"seed": SEED, "n_micro": N_MICRO, "batch": a.batch, "micro_batch_ids": ids, "scorer_tensors": len(S), "trunk_tensors": len(Tr),
            "grad_passes": GRAD_PASSES, "retained_graph": False, "replay_seed": REPLAY_SEED,
            "replay_max_abs_diff": {"score": max(p["replay_max_abs_diff_score"] for p in per), "traj": max(p["replay_max_abs_diff_traj"] for p in per)},
            "pooled": {"norm_grad_S_l_score": pooled["sS"], "norm_grad_S_rank_w1": pooled["sR"], "norm_grad_T_l_traj": pooled["sTt"], "norm_grad_T_rank_w1": pooled["sTr"], "r1": r1, "rho1": rho1},
            "per_micro_batch": per, "r1_per_micro_min_median_max": [rs[0], rs[len(rs) // 2], rs[-1]] if rs else None,
            "r_by_w": {repr(w): w * r1 for w in CANDIDATES}, "rho_by_w": {repr(w): w * rho1 for w in CANDIDATES},
            "pairs_by_head": pairs, "sets_with_pairs_by_head": sets, "dead_heads": [h for h in rank.heads if pairs[h] == 0],
            "labelled_sets": labelled_sets, "total_sets": total_sets, "label_coverage_of_the_8_batches": labelled_sets / max(total_sets, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--args", required=True, help="the arm's train.py argv as one string (the chain's $ARGS); must carry --labeller online and --rank-heads")
    ap.add_argument("--refe", default=str(HERE), help="the directory holding the PATCHED train.py (default: this file's directory)")
    ap.add_argument("--out", default=None, help="write the result JSON here")
    a = ap.parse_args(argv)
    targv = a.args.split()                              # the shell's word splitting of an unquoted $ARGS (default IFS) -- exactly gate_token.argv_from_string, which the chain and the gate use
    T, ns, model, ds, lab, dev, rank = build_instrumented(Path(a.refe).resolve(), targv)
    try:
        res = measure(T, ns, model, ds, lab, dev, rank)
    finally:
        lab.close()
    rule = decide({w: res["r_by_w"][repr(w)] for w in CANDIDATES}, {w: res["rho_by_w"][repr(w)] for w in CANDIDATES})
    synthetic = ns.label_backend == "synthetic"
    import torch._functorch.config as FC
    res.update({"donated_buffer": bool(FC.donated_buffer), "rule": rule, "argv_sha256": hashlib.sha256("\x00".join(targv).encode()).hexdigest(), "init_from": ns.init_from, "heads": list(rank.heads), "ep_gap": rank.ep_gap,
                "gate_topk": rank.gate_topk, "label_backend": ns.label_backend, "device": str(dev), "NOT_A_DECISION": synthetic, "ignored_resume": bool(ns.resume)})
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    p = res["pooled"]
    print(f"weight_rule_rank: pooled ||grad_S l_score|| = {p['norm_grad_S_l_score']:.6g}   ||grad_S L_rank|| (w=1) = {p['norm_grad_S_rank_w1']:.6g}   r(1) = {p['r1']:.6g}")
    print(f"  pooled ||grad_T l_traj|| = {p['norm_grad_T_l_traj']:.6g}   ||grad_T L_rank|| (w=1) = {p['norm_grad_T_rank_w1']:.6g}   rho(1) = {p['rho1']:.6g}")
    print("  r(w):   " + "  ".join(f"{w:g}:{res['r_by_w'][repr(w)]:.4g}" for w in CANDIDATES))
    print("  rho(w): " + "  ".join(f"{w:g}:{res['rho_by_w'][repr(w)]:.4g}" for w in CANDIDATES))
    print(f"  pairs by head {res['pairs_by_head']}   sets with pairs {res['sets_with_pairs_by_head']}   label coverage {res['label_coverage_of_the_8_batches']:.3f}"
          + (f"   DEAD HEADS: {res['dead_heads']}" if res["dead_heads"] else ""))
    print(f"  grad passes {res['grad_passes']} (one forward + one backward each, no retained graph)   replay max |diff| score {res['replay_max_abs_diff']['score']:.3g}  traj {res['replay_max_abs_diff']['traj']:.3g}")
    print(f"  RULE: {rule['status']}  rank_w = {rule['rank_w']}  -- {rule['why']}" + ("   [SYNTHETIC LABELS: NOT A DECISION]" if synthetic else ""))
    return {"SELECT": 0, "K2": 5, "K2b": 5}.get(rule["status"], 4)


if __name__ == "__main__":
    sys.exit(main())
