#!/usr/bin/env python3
"""REF-A v1 trainer — feature prediction on cached frozen DINOv3 fields.

⭐ WHAT IS DIFFERENT FROM `refa_train_plus.py`, IN ONE LINE: the loss is the
FUTURE PATCH FIELD, not a trajectory label. Everything else that made REF-A
stable is kept verbatim (feature-cache training, fit-once standardizer, no
BatchNorm/dropout, adapter-vs-predictor LR groups, the adapter-collapse
monitor).

CACHE CONTRACT (stage 1, separate job — this trainer never touches an image):
    <cache>/<episode_id>.pt  ->  fp16 tensor [T, 640, 1024]
      * DINOv3 ViT-L/16 patch tokens, CLS DISCARDED
      * 256x640 crop at 120 deg HFOV (grid 16x40)
    <cache>/index.json       ->  {"episodes": [...], "parity_key": "...",
                                  "skip_hash": "...", "geometry": {...}}
⛔ The trainer REFUSES a cache whose geometry disagrees with
``refa_v1.DINOV3_GEOMETRY`` — a silently narrower interface is exactly the
defect v1 exists to remove, and it would still train.

Run (smoke, CPU, no cache needed):
    python stack/scripts/refa_v1_train.py --smoke
Run (real):
    python stack/scripts/refa_v1_train.py --cache /path/dinov3_w120 \
        --steps 30000 --bs 8 --out ~/experiments/refa-v1
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from tanitad.config import StrategicPolicyConfig, TacticalPolicyConfig
from tanitad.refs.refa_v1 import DINOV3_GEOMETRY, RefAV1, RefAV1Config


def build_model(args) -> RefAV1:
    cfg = RefAV1Config(
        strategic_cfg=None if args.no_hierarchy else StrategicPolicyConfig(),
        tactical_cfg=None if args.no_hierarchy else TacticalPolicyConfig(),
        w_cf=args.w_cf, cf_negs=args.cf_negs, cf_at_step=args.cf_at_step,
        motion_inject=args.motion_inject,
        target_space=args.target_space,
        w_aux_head=args.w_aux_head, proposal_k=args.proposal_k,
    )
    if args.smoke:
        cfg.d_enc, cfg.n_tokens, cfg.d_state = 32, 8, 32
        cfg.op_layers, cfg.op_heads, cfg.tac_layers = 1, 2, 1
        cfg.tac_queries, cfg.str_dim, cfg.str_layers = 4, 16, 1
        if not args.no_hierarchy:
            cfg.strategic_cfg = StrategicPolicyConfig(d_model=32, depth=1,
                                                      n_heads=2, d_ctx=16,
                                                      d_cmd=8)
            cfg.tactical_cfg = TacticalPolicyConfig(d_model=32, depth=1,
                                                    n_heads=2, d_intent=16)
    return RefAV1(cfg)


def verify_cache(cache: Path) -> dict:
    """⛔ Geometry is a CONTRACT, not a hint (see module docstring)."""
    idx = json.loads((cache / "index.json").read_text(encoding="utf-8"))
    geo = idx.get("geometry", {})
    for k in ("n_tokens", "d_enc", "hfov_deg"):
        want, got = DINOV3_GEOMETRY[k], geo.get(k)
        if got != want:
            raise SystemExit(
                f"REFUSING this cache: geometry.{k} = {got!r}, v1 requires "
                f"{want!r}. A narrowed visual interface trains happily and is "
                "the defect v1 was built to remove.")
    return idx


class SmokeData:
    """Random fields with the right shapes — proves the loop, never a number."""

    def __init__(self, cfg: RefAV1Config, bs: int):
        self.cfg, self.bs = cfg, bs

    def batch(self):
        c = self.cfg
        return (torch.randn(self.bs, c.op_window, c.n_tokens, c.d_enc),
                torch.randn(self.bs, c.op_steps, c.a_dim) * 0.1,
                torch.randn(self.bs, c.op_steps, c.n_tokens, c.d_enc))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path,
                    help="stage-1 DINOv3 feature cache (0.2 s grid)")
    ap.add_argument("--episodes", type=Path,
                    help="the v2ep episode dir (actions/poses at 10 Hz)")
    ap.add_argument("--lru", type=int, default=32,
                    help="episodes held in RAM (each ~130 MB fp16 fields)")
    # ⭐ THE v7.2 JOIN. The loader has accepted these since 2026-09-01 and the
    # model has masked -100 correctly for longer; only the TRAINER could not
    # pass them, so a real run trained the trajectory path with the tactical and
    # strategic heads unsupervised while every component reported itself ready.
    ap.add_argument("--labels", type=Path, default=None,
                    help="v7.2 s2 labels .jsonl.gz — supervises the tactical "
                         "and strategic heads. REQUIRED with a real --cache; "
                         "out-of-band windows emit -100 and the model skips "
                         "that family rather than averaging a NaN")
    ap.add_argument("--nav", type=Path, default=None,
                    help="nav/route source joined on clip_id, fed to all three "
                         "layers. ⛔ must NOT carry the situation classifier's "
                         "output in any form (PI 2026-08-03)")
    ap.add_argument("--allow-unlabelled", action="store_true",
                    help="⛔ deliberate: run a real --cache WITHOUT --labels. "
                         "Only for a diagnostic that does not touch the "
                         "tactical/strategic heads — never for a registered arm")
    ap.add_argument("--out", type=Path, default=Path("./refa_v1_run"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--adapter-lr-mult", type=float, default=0.1,
                    help="REF-A stability item 4: the adapter warms up SLOWER "
                         "than the predictor (10x longer warmup == 0.1x LR).")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="ablation arm: drop both brains (they are a matched "
                         "set) — the change-#7 control")
    # --- change #10 was UNREACHABLE FROM THE LAUNCH LINE until these existed - #
    # ⛔ The counterfactual term shipped in the model with no flag to turn it
    # on. That is the advertised-but-inert defect one level out: `sanity()`
    # refuses w_cf with zero negatives, and meanwhile NO launch could set w_cf
    # at all. A capability with no switch is not a capability.
    ap.add_argument("--w-cf", type=float, default=0.0,
                    help="weight on the counterfactual-action InfoNCE (change "
                         "#10). 0 = off. Its no-information floor is "
                         "ln(1+cf_negs), so cf_excess > 0 PROVES the predictor "
                         "used the action.")
    ap.add_argument("--cf-negs", type=int, default=3)
    ap.add_argument("--cf-at-step", type=int, default=4)
    # --- PI 2026-08-31: "implement and try 1" ------------------------------- #
    ap.add_argument("--motion-inject", action="store_true",
                    help="add a CROSS-CHANNEL projection of z_t - z_(t-1) to "
                         "the initial rollout state. Default history path is "
                         "the adapter's DEPTHWISE temporal conv only -- each "
                         "channel mixes its own past, so cross-channel motion "
                         "(parallax, an edge crossing patches) has no route "
                         "into the state.")
    ap.add_argument("--target-space", choices=("adapter", "frozen"),
                    default="adapter",
                    help="'adapter' = original form, whose primary loss has a "
                         "COLLAPSE MINIMUM (the target passes the trained "
                         "adapter); 'frozen' = predict std(DINOv3) itself -- "
                         "fixed target variance, minimum removed (DINO-WM).")
    # --- MM-E19 carried over: full-chain BPTT needs a tighter clip ---------- #
    ap.add_argument("--clip", type=float, default=1.0,
                    help="grad-norm clip. ⚠️ MM-E19 MEASURED a 60-step "
                         "full-chain rollout diverging (gnorm 2.1e9) at the "
                         "loose default and surviving at 0.5. v1 rolls 30 "
                         "steps with full-chain gradient through an 80 M "
                         "predictor — consider 0.5 for the first real arm.")
    # --- Drive-JEPA-adapted multimodal proposals (2026-09-01) --------------- #
    ap.add_argument("--w-aux-head", type=float, default=0.0,
                    help="imitation weight on the proposal head (WTA over "
                         "proposal-k modes vs the demonstrated (a, kappa)). "
                         "SEEDS the planner only -- behaviour stays planned.")
    ap.add_argument("--proposal-k", type=int, default=1,
                    help="number of proposal modes (Drive-JEPA proposal-set "
                         "idea); >1 adds a score head, and at plan() time all "
                         "modes join the iCEM seed pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/ckpt.pt if present")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    a = ap.parse_args(argv)

    if not a.smoke and (a.cache is None or a.episodes is None):
        raise SystemExit("--cache AND --episodes are required unless --smoke")
    # ⛔ REFUSE AN UNSUPERVISED REAL RUN — and refuse it HERE, before
    # `verify_cache` touches the disk, so the message is about the mistake and
    # not about whichever file the cache reader happened to open first.
    # The PI made v7.2 tactical/strategic labels and nav-to-all-layers
    # mandatory. Without --labels those heads take NO gradient and the run
    # still LOOKS healthy — losses fall, checkpoints land, the trajectory path
    # learns — so nothing in the log would say the hierarchy was never trained.
    # That is what this refuses: not a crash, a silently narrower experiment.
    if a.cache is not None and not a.labels and not a.allow_unlabelled:
        raise SystemExit(
            "⛔ --cache given without --labels: the tactical and strategic "
            "heads would take NO gradient and the run would still look "
            "healthy. Pass --labels <s2_labels_v7.2_*.jsonl.gz> (and --nav), "
            "or --allow-unlabelled if this is a diagnostic that deliberately "
            "does not touch those heads.")
    if a.cache is not None and not a.nav:
        print("⚠️ [refav1] no --nav: the nav command reaches no layer. "
              "Admissible only for an arm that does not claim route "
              "conditioning.", flush=True)
    if a.cache is not None:
        verify_cache(a.cache)

    torch.manual_seed(a.seed)
    model = build_model(a).to(a.device)
    cfg = model.cfg
    # ⛔ SANITY BEFORE THE SPEND, NOT AFTER. `sanity()` is what refuses an
    # inexpressible ladder (a rate that is not an integer multiple of op_dt) and
    # an advertised-but-inert counterfactual term. It was never called here, so
    # a 30k-step arm could have run to completion on a config the model itself
    # would have rejected.
    cfg.sanity()
    a.out.mkdir(parents=True, exist_ok=True)

    # Stability item 4: adapter and predictor are SEPARATE param groups.
    adapter_p = list(model.adapter.parameters())
    ids = {id(p) for p in adapter_p}
    rest_p = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW(
        [{"params": adapter_p, "lr": a.lr * a.adapter_lr_mult},
         {"params": rest_p, "lr": a.lr}], weight_decay=0.01)

    if a.smoke:
        data = SmokeData(cfg, a.bs)
        with torch.no_grad():
            feats, _, _ = data.batch()
            model.std.fit(feats.to(a.device))
    else:
        # ⭐ THE LOADER GAP IS CLOSED (2026-09-01): real windows over the
        # stage-1 cache + v2ep kinematics. The loader emits (a, kappa) with the
        # MEASURED channel repair (v2ep stores kappa first, r=0.995) and the
        # str-extension pairs.
        # ⭐⭐ AND THE v7.2 LABEL / NAV JOIN IS NOW WIRED (2026-09-02). The line
        # above used to end "labels/nav join is the next increment" — the LOADER
        # had supported `labels_path`/`nav_path` for a day, and the model side
        # had supported `-100` masking for longer, but the TRAINER had no flag
        # to supply either. So a real run would have trained the trajectory path
        # with the tactical/strategic heads unsupervised and nav absent, while
        # every component reported itself ready. A capability that exists at
        # both ends and is not connected in the middle is not a capability.
        from tanitad.data.refav1_loader import RefAV1Windows
        data = RefAV1Windows(a.cache, a.episodes, op_window=cfg.op_window,
                             op_steps=cfg.op_steps, str_dt=cfg.str_dt,
                             str_ext_steps=cfg.str_ext_steps,
                             lru=a.lru, seed=a.seed,
                             labels_path=a.labels, nav_path=a.nav)
        print(f"loader: {len(data)} windows over {len(data.names)} episodes")
        print(f"loader: labels={a.labels or 'NONE'} nav={a.nav or 'NONE'}",
              flush=True)
        with torch.no_grad():
            fit = data.batch(max(a.bs, 8))["feats"]
            model.std.fit(fit.reshape(-1, cfg.d_enc).to(a.device))

    # ⛔ THE AUXILIARY HEAD IS ADVERTISED AND INERT — REFUSED RATHER THAN FAKED.
    # The loop read `out["loss"] + cfg.w_aux_head * torch.zeros(())`: the
    # config carries w_aux_head 0.1, the launch record would show it, and the
    # term contributes EXACTLY nothing because no imitation target is wired.
    # That is the defect `sanity()` refuses for w_cf, sitting in the trainer.
    # ⚠️ Fixing it means supplying a proposal target, which the cache contract
    # does not yet carry — so this REFUSES instead of pretending.
    if cfg.w_aux_head and a.smoke:
        raise SystemExit(
            f"w_aux_head is {cfg.w_aux_head} under --smoke: SmokeData has no "
            "demonstrated actions, so the term would be advertised and fed "
            "noise. With a real --cache the loader's (a, kappa) IS the demo "
            "and the term is live.")

    start_step = 0
    if a.resume and (a.out / "ckpt.pt").exists():
        ck = torch.load(a.out / "ckpt.pt", map_location=a.device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = int(ck["step"])
        print(f"resumed from step {start_step}")

    # the set  will actually take — derived from the model, never a
    # hand-maintained list that would drift from it
    import inspect as _inspect
    _FWD_PARAMS = {n for n in _inspect.signature(model.forward).parameters
                   if n != "self"}

    log = (a.out / "train_log.jsonl").open("a", encoding="utf-8")
    t0 = time.time()
    for step in range(start_step + 1, a.steps + 1):
        if a.smoke:
            feats, actions, future = (x.to(a.device) for x in data.batch())
            kw = {}
        else:
            b = data.batch(a.bs)
            feats = b["feats"].to(a.device)
            actions = b["actions"].to(a.device)
            future = b["future_feats"].to(a.device)
            # ⛔ FILTER BY THE MODEL'S ACTUAL SIGNATURE, AND SAY WHAT WAS
            # DROPPED. This used to forward EVERY batch key, which worked only
            # while the loader emitted exactly what `forward` accepted. Turning
            # on --labels/--nav made the loader emit its validity masks
            # (`nav_valid`, …) and the run died on
            # `TypeError: RefAV1.forward() got an unexpected keyword argument`.
            # A blind pass-through is a contract between two files that nobody
            # checks; it breaks the moment either side grows a field.
            # ⚠️ The report matters as much as the filter: dropping a VALIDITY
            # MASK is correct (the model masks -100 itself), but silently
            # dropping a LABEL would narrow the experiment invisibly — the
            # exact failure the --labels guard exists to prevent. So name them
            # once, and let the reader judge which kind they are.
            kw = {k: (v.to(a.device) if torch.is_tensor(v) else v)
                  for k, v in b.items()
                  if k in _FWD_PARAMS
                  and k not in ("feats", "actions", "future_feats")
                  and v is not None}
            if step == start_step + 1:
                dropped = sorted(set(b) - _FWD_PARAMS
                                 - {"feats", "actions", "future_feats"})
                print(f"[refav1] forward() consumes {sorted(kw)}", flush=True)
                if dropped:
                    print(f"⚠️ [refav1] loader emits but forward() does NOT "
                          f"accept: {dropped} — verify each is a validity mask "
                          f"(safe: the model masks -100 itself) and not a "
                          f"label (which would narrow the run silently)",
                          flush=True)
        out = model(feats, actions, future_feats=future, **kw)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()

        if step % a.log_every == 0 or step == 1:
            with torch.no_grad():
                # THE COLLAPSE MONITOR, carried over from REF-A: adapter output
                # per-dim std. Part 2 measured the trained REF-A adapter at
                # 0.8011 vs 0.220 random-init — i.e. NOT collapsed. If v1 ever
                # drives this toward 0 the run is dead regardless of the loss.
                adapter_std = float(model.encode(feats).std(dim=(0, 1, 2)).mean())
            row = {"step": step, "loss": float(loss.detach()),
                   "loss_feat_op": float(out["loss_feat_op"].detach()),
                   "loss_feat_tac": float(out["loss_feat_tac"].detach()),
                   "loss_feat_str": float(out["loss_feat_str"].detach()),
                   "grad_norm": float(gnorm), "adapter_std": adapter_std,
                   "clip": a.clip,
                   # ⭐ THE REALISED LADDER, IN EVERY ROW. The horizon a level
                   # actually trains on is now readable from the log instead of
                   # inferred from the config — which is how a strategic rung
                   # ran at 1.6 s reaching 5.0 s while its config said 1.5/6.0
                   # and every test passed.
                   "tac_target_s": [round((k + 1) * cfg.op_dt, 3)
                                    for k in out["tac_target_idx"]],
                   "str_target_s": [round((k + 1) * cfg.op_dt, 3)
                                    for k in out["str_target_idx"]],
                   "elapsed_s": round(time.time() - t0, 1)}
            if cfg.w_cf:
                # ⭐ cf_excess is measured against a KNOWN floor, ln(1+cf_negs):
                # > 0 proves the predictor used its action, with no baseline to
                # argue about. This is the number the arm exists to move.
                row.update(cf_loss=float(out["cf_loss"].detach()),
                           cf_no_info_floor=out["cf_no_info_floor"],
                           cf_excess=out["cf_excess"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row))
            if not math.isfinite(row["loss"]):
                raise SystemExit("non-finite loss — refusing to continue")

        if step % a.save_every == 0 or step == a.steps:
            torch.save({"step": step, "model": model.state_dict(),
                        "opt": opt.state_dict(), "cfg": vars(cfg)},
                       a.out / "ckpt.pt")
    log.close()
    print(f"done: {a.steps} steps, {model.trainable_parameters():,} trainable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
