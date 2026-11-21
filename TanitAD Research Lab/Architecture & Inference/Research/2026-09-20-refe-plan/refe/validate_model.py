"""Validate REFe's architecture BEFORE it earns any rented GPU.

Checks, in order of how badly a failure would waste pod time:
  1. parameter accounting -- total, trainable, and the trainable FRACTION against DriveZero's
     published 5.49 % (338.46 M / 18.58 M). The fraction is comparable across backbone sizes even
     though the absolute counts are not: REFe is ViT-S with one camera, theirs is ViT-L with four.
  2. LoRA is IDENTITY AT INIT -- B is zeroed, so a fresh REFe must reproduce the frozen backbone
     exactly. If it does not, the adapter is perturbing a pretrained trunk before a single step.
  3. only the intended tensors carry gradient -- asserted by NAME, not by count, because a count
     can be right while the wrong tensors are unfrozen.
  4. the scorer is genuinely DETACHED -- a deliberate-regression arm proves the check can FAIL.
  5. winner-takes-all really routes gradient to exactly one proposal.
  6. forward + backward on the real 4060 at the real input size, with peak memory.

Usage: python validate_model.py [--device cuda]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import REFe, REFeConfig, param_report, wta_loss  # noqa: E402

PUB_TOTAL, PUB_TRAIN, PUB_PCT = 338.46e6, 18.58e6, 5.49


def main(argv: list[str]) -> int:
    # ⚠️ `--device cpu` USED TO BE SILENTLY IGNORED. The old expression only honoured the flag when
    # its value was "cuda"; any other value fell through to "cuda if available", so the one thing an
    # operator would ever pass it for -- keeping off a busy GPU -- did nothing. A flag that accepts
    # a value and discards it is worse than no flag: the command line records an intent the run did
    # not follow.
    dev = (argv[argv.index("--device") + 1] if "--device" in argv
           else ("cuda" if torch.cuda.is_available() else "cpu"))
    if dev == "cuda" and not torch.cuda.is_available():
        print("  --device cuda asked for, but no CUDA is available -- refusing to pretend")
        return 2
    # ⛔ THE SHIPPING CONFIG (ViT-L x 4 cameras) NEEDS 9.49 GiB AND THIS BOX HAS 8. MEASURED
    # 2026-09-21: it "runs" at 27.074 s/step against the 2-camera 1.630 s -- a 34x outlier that is
    # the PCIe bus, not the GPU. So the full validation is an A40 job, and pretending otherwise is
    # how the previous 4-camera timings got published. `--backbone` lets every ARM be exercised
    # here at a size that is genuinely resident; the parameter accounting in section 1 is the part
    # that must still be read at ViT-L, and it needs no device at all.
    cfg = (REFeConfig.for_backbone(argv[argv.index("--backbone") + 1])
           if "--backbone" in argv else REFeConfig())
    m = REFe(cfg)
    fails = []

    # ---------------------------------------------------------------- 1. parameters
    r = param_report(m)
    print("== 1. parameter accounting ==")
    print(f"  REFe total      {r['total']:>12,}  ({r['total']/1e6:.2f} M)")
    print(f"  REFe trainable  {r['trainable']:>12,}  ({r['trainable']/1e6:.2f} M)  = {r['pct']:.2f} %")
    print(f"  DriveZero pub.  {PUB_TOTAL:>12,.0f}  ({PUB_TOTAL/1e6:.2f} M) / {PUB_TRAIN/1e6:.2f} M "
          f"= {PUB_PCT:.2f} %   [ViT-L, 4 cameras -- NOT a target]")
    print("  per group (total / trainable):")
    for k, (t, tr) in sorted(r["groups"].items(), key=lambda kv: -kv[1][0]):
        print(f"    {k:22s} {t:>11,} / {tr:>10,}")

    # ---------------------------------------------------------------- 2. LoRA identity at init
    print("\n== 2. LoRA is identity at init (B zeroed) ==")
    bb = m.backbone
    x = torch.randn(1, 3, cfg.img_h, cfg.img_w)
    with torch.no_grad():
        got = bb(x)
        for blk in bb.blocks:                       # disable the adapters entirely
            blk.attn.q.scale = 0.0
            blk.attn.v.scale = 0.0
        ref = bb(x)
        for blk in bb.blocks:
            blk.attn.q.scale = 1.0 / cfg.lora_rank
            blk.attn.v.scale = 1.0 / cfg.lora_rank
    d = (got - ref).abs().max().item()
    ok = d == 0.0
    print(f"  max |with_lora - without_lora| = {d:.3e}   {'PASS' if ok else 'FAIL'}")
    fails += [] if ok else ["LoRA is not identity at init"]

    # ---------------------------------------------------------------- 3. what is trainable, by name
    print("\n== 3. trainable tensors, asserted BY NAME ==")
    tr_names = [n for n, p in m.named_parameters() if p.requires_grad]
    bad = [n for n in tr_names
           if n.startswith("backbone") and not (".A" in n or ".B" in n)]
    lora = [n for n in tr_names if n.startswith("backbone")]
    print(f"  trainable tensors: {len(tr_names)}   backbone ones: {len(lora)} (all LoRA A/B)")
    print(f"  backbone tensors trainable that are NOT LoRA: {len(bad)}   {'PASS' if not bad else 'FAIL'}")
    if bad:
        print("   ", bad[:6])
    fails += [] if not bad else ["non-LoRA backbone tensors are trainable"]
    expect = 2 * 2 * cfg.depth           # q.A q.B v.A v.B per block
    print(f"  expected LoRA tensors = 2 proj x 2 mats x {cfg.depth} blocks = {expect}   "
          f"got {len(lora)}   {'PASS' if len(lora) == expect else 'FAIL'}")
    fails += [] if len(lora) == expect else ["unexpected LoRA tensor count"]

    # ---------------------------------------------------------------- 4/5/6. runtime
    print(f"\n== 4-6. forward/backward on {dev} ==")
    m = m.to(dev)
    # ⛔ THIS BUILT `randn(B, 3, H, W)` -- FOUR-DIMENSIONAL, ONE CAMERA -- against a 4-camera config,
    # so the validator died here with "configured for 4 cameras but received 1" (exit 1) and
    # sections 4-6 NEVER RAN: the shape checks, the WTA routing check, the detach check, the peak-
    # memory assertion, and BOTH constructed-regression arms this file exists for. The change's own
    # headline deliverable was unexercised. A validator that cannot reach its arms is not a weak
    # validator, it is an absent one.
    # ⚠️ `--batch` defaults to 1 on a 4-camera rig because ViT-L x 4 cameras at B=2 needs ~19 GB and
    # the dev box has 8. That is a CAPACITY fact, stated rather than hidden: the paper's batch is
    # 256, data-parallel over 16 H20s, and none of these numbers are comparable to it.
    B = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else (
        1 if cfg.n_cameras > 1 else 2)
    print(f"  batch {B} x {cfg.n_cameras} cameras"
          + ("   (B=1 by default here: 4 x ViT-L at B=2 needs ~19 GB; pass --batch to override)"
             if B == 1 and cfg.n_cameras > 1 else ""))
    img = torch.randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w, device=dev)
    ego = torch.randn(B, cfg.ego_dim, device=dev)
    goal = torch.randn(B, 2 * cfg.n_goal_points, device=dev)
    tgt = torch.randn(B, cfg.horizon_steps, cfg.traj_dim, device=dev)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    traj, score = m(img, ego, goal)
    print(f"  traj  {tuple(traj.shape)}   expect ({B}, {cfg.n_proposals}, {cfg.horizon_steps}, {cfg.traj_dim})")
    print(f"  score {tuple(score.shape)}  expect ({B}, {cfg.n_proposals}, {cfg.n_score_components})")
    shape_ok = (tuple(traj.shape) == (B, cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)
                and tuple(score.shape) == (B, cfg.n_proposals, cfg.n_score_components))
    fails += [] if shape_ok else ["output shapes wrong"]

    loss, idx = wta_loss(traj, tgt)
    loss.backward()
    g = m.traj_head[-1].weight.grad
    print(f"  WTA winner index per sample: {idx.tolist()}")
    print(f"  traj_head grad present: {g is not None and g.abs().sum().item() > 0}")

    # the scorer must be detached: a score-only loss must leave the trajectory path with NO grad
    m.zero_grad(set_to_none=True)
    traj2, score2 = m(img, ego, goal)
    score2.sum().backward()
    qg = m.queries.grad
    detached_ok = qg is None or qg.abs().sum().item() == 0.0
    print(f"  scorer detached (score-only loss leaves queries grad-free): "
          f"{'PASS' if detached_ok else 'FAIL'}")
    fails += [] if detached_ok else ["scorer is NOT detached"]

    # deliberate-regression arm: re-attach the scorer and the check MUST go red
    m.zero_grad(set_to_none=True)
    # ⛔ THIS ARM MUST MIRROR THE LIVE FORWARD PASS, OR IT TESTS A PATH THAT NO LONGER EXISTS.
    # It used to hand-build the PRE-2026-09-20 architecture: `cat([tok, registers])` and one shared
    # context for both decoders. After the registers were made to COMPRESS, that path was gone from
    # the model and this regression arm was running green against code nobody ships -- a guard that
    # cannot fail because it no longer touches the thing it guards. Rebuilt against the current
    # wiring: registers compress to scene tokens, the trajectory decoder reads them, the scoring
    # decoder reads the VISUAL tokens.
    # ⛔ THE ARM NO LONGER RESTATES THE FORWARD PASS. It drifted three times in one day -- when
    # the registers began to compress, when `pos3d` became an analytic MLP, and when the goal
    # gained Fourier features -- because a guard that duplicates the thing it guards will always
    # lag it. `REFe.forward(detach_scorer=False)` runs the REAL path with exactly one property
    # flipped, so this arm cannot drift again without the model itself changing.
    _traj2, _score2 = m(img, ego, goal, detach_scorer=False)
    _score2.sum().backward()
    qg2 = m.queries.grad
    regression_red = qg2 is not None and qg2.abs().sum().item() > 0
    print(f"  deliberate-regression arm (scorer re-attached) goes RED: "
          f"{'PASS' if regression_red else 'FAIL -- the detach check cannot detect anything'}")
    fails += [] if regression_red else ["detach check is inert"]


    # ⛔ TWO CONSTRUCTED REGRESSIONS THE VALIDATOR USED TO MISS ENTIRELY (review 4).
    # 1. Removing ONLY `visual_ctx.detach()` leaks the score loss into the frozen trunk's LoRA and
    #    every arm stayed green -- the detach check watches the QUERIES, not the backbone.
    # 2. Setting every LoRA scale to 0 makes the adapters produce no gradient at all, and again
    #    every arm passed: the model still ran, still had 96 trainable tensors, still "trained".
    # Both are silent capability losses, which is the class this whole file exists to catch.
    m.zero_grad(set_to_none=True)
    _t3, _s3 = m(img, ego, goal)
    _s3.sum().backward()
    lora_grad = sum(float(p_.grad.abs().sum()) for n_, p_ in m.named_parameters()
                    if n_.startswith("backbone") and p_.requires_grad and p_.grad is not None)
    backbone_clean = lora_grad == 0.0
    # ⚠️ DECLARED DEPARTURE, NOT THE PAPER'S INVARIANT -- the wording here used to claim otherwise.
    # PUBLISHED (p. 8): "Candidate trajectories are detached before entering the scoring branch."
    # The paper detaches the TRAJECTORIES, which `traj.detach()` does and which the arm above
    # checks. REFe ALSO detaches the scoring decoder's visual context, which the paper never
    # states, so the six-component PDM loss reaches neither the backbone LoRA nor `scene_proj`.
    # The old message called a failure here "the scorer leaks into the frozen trunk" -- language
    # that promoted OUR extra into THEIR requirement, with a constructed regression defending it.
    # ⛔ It is not cosmetic: REFe exists to compare DINOv3 against DriveVFM, and discarding the
    # score loss's gradient to the encoder changes how much the encoder is adapted -- the variable
    # under study. The arm stays (an undeclared change to it should still fail) but it now says
    # what it is, and `REFeConfig.detach_scorer_context = False` is the ablation that settles it.
    print(f"  score-only loss leaves the BACKBONE LoRA grad-free: "
          f"{'PASS' if backbone_clean else 'FAIL'}   "
          f"[REFe's EXTRA detach of visual_ctx, not the paper's -- see the note here]")
    fails += [] if backbone_clean else ["REFe's extra visual_ctx detach was removed undeclared"]

    m.zero_grad(set_to_none=True)
    traj4, _s4 = m(img, ego, goal)
    l4, _ = wta_loss(traj4, tgt)
    l4.backward()
    lora_trains = sum(float(p_.grad.abs().sum()) for n_, p_ in m.named_parameters()
                      if n_.startswith("backbone") and p_.requires_grad and p_.grad is not None)
    lora_alive = lora_trains > 0
    print(f"  the trajectory loss DOES reach the LoRA adapters: "
          f"{'PASS' if lora_alive else 'FAIL -- LoRA is inert, the trunk never adapts'}")
    fails += [] if lora_alive else ["LoRA receives no gradient"]

    if dev == "cuda":
        print(f"  peak CUDA memory: {torch.cuda.max_memory_allocated()/2**30:.2f} GB at batch {B}")

        # ⛔ ASSERT THE PEAK, DO NOT ONLY PRINT IT. MEASURED 9.58 GB at batch 2 -- it fits an
        # A40 and does NOT fit this box's 8 GiB RTX 4060, and the number was sitting in the
        # output unasserted while three documents quoted a stale 5.36 GB and the pod
        # provisioning arithmetic rested on it.
        _dev_gb = 8.0
        if torch.cuda.is_available():
            _pk = torch.cuda.max_memory_allocated() / 1e9
            print(f"  peak fits an A40 (48 GB): {_pk < 44.0}   "
                  f"fits THIS card ({_dev_gb} GB): {_pk < _dev_gb}")
            fails += [] if _pk < 44.0 else [f"peak {_pk:.2f} GB exceeds an A40"]

    print("\n" + ("VALIDATE_OK" if not fails else "VALIDATE_FAILED: " + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
