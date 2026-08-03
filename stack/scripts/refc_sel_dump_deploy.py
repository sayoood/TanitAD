"""E-SEL-1D dump — the score S3 can ACTUALLY produce at inference.

``refc_sel_dump_refined.py`` banked ``cons_score`` =
``-||law_head([pooled, fan_i]) - encode_pooled(frame_{t+5})||^2`` — a statistic
built on the **FUTURE FRAME**. E-SEL-1 read rho = 0.6657 / 0.6212 off it and
correctly refused to quote that as S3's effect size, because the thing S3
deploys never sees ``z_{t+5}``.

⭐ WHAT THIS ADDS. The deployed call is ``refc.py:1246-1251``::

    cons_s = sl.consequence_scores(x, cons_ctx, cons_head,
                                   self.feat_proj, self.conf_head,
                                   detach=sel.cons_detach)
    r_terms.append(self.cons_gate * cons_s)

with ``cons_head = self.law_head`` and ``cons_ctx = pooled`` (``refc.py:1835-1836``).
So the deployable score is

    s_deploy = conf_head(layer_norm(feat_proj(law_head([pooled, fan]))))

and it is computed here by **calling that exact function**, not by
re-implementing it — re-implementing is how two definitions of the headline
quantity drift apart.

Banked per window, so every downstream statistic is reproducible OFF-Thor
forever (the 256x256 val cache is a SINGLE-DISK dependency; see ESEL_VERDICT
escalation 3, and this dump is the artifact that stops it being one for the
consequence question):

  * ``pooled``        [B, F]     the decoder context - the ONLY thing S3 reads
  * ``law_tgt``       [B, F]     z_{t+5}, banked so the ORACLE leg stays checkable
  * ``cons_deploy``   [B, N]     s_deploy - **the headline**
  * ``cons_oracle``   [B, N]     E-SEL's statistic, recomputed on the SAME fan
  * ``cons_ctxswap``  [B, N]     s_deploy with ``pooled`` globally DERANGED
  * everything ``refc_sel_dump_refined.py`` banked, byte-compatible

⛔ DETERMINISM IS THE CONTROL, NOT AN ASSUMPTION. ``refc.py:1217-1218`` puts the
denoise noise behind ``if self.training``, so an eval decode is deterministic and
this pass must reproduce E-SEL's bank BIT-FOR-BIT. ``--verify-against`` asserts
it. If it deviates, rho_deploy and rho_oracle are not on the same fan and the
paired contrast is void — that is a control that CAN fire, unlike a shuffled
control whose expectation is 0 by construction.

⛔ RASTER GATE (R-2026-08-02-a) is inherited verbatim: base returns plausible
wrong numbers SILENTLY at the wrong token count.

⚠️ NAV MODE is ``follow_constant``, deliberately — the C6 confound, inherited so
the baseline this is paired against does not move.

Run (Thor, tanitad-edge = the INFERENCE venv; never tanitad-train):
    PYTHONPATH=~/TanitAD/stack:~/TanitAD/taniteval OMP_NUM_THREADS=6 \\
    ~/venvs/tanitad-edge/bin/python ~/refc_sel_dump_deploy.py \\
        --ckpt ~/models/refc-base/ckpt.pt --preset base \\
        --val  ~/valdata/physicalai-val-0c5f7dac3b11 \\
        --verify-against ~/fan_refined_refc-base-30k.pt \\
        --out  ~/fan_deploy_refc-base-30k.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("OMP_NUM_THREADS", "6")

LAW_AHEAD = 5                    # refc_train.LAW_AHEAD — the LAW target horizon
WINDOW, STRIDE = 8, 8            # canonical val protocol
NAV_MODE = "follow_constant"     # matches the banked fans and the published rows
CTXSWAP_SEED = 20260803          # fixed in PREREG_S3_DEPLOYABLE.md §2


def build_model(ckpt: str, preset: str):
    from taniteval.loaders import _apply_overrides
    from tanitad.refs.refc import (RefCModel, refc_config, refc_small_config,
                                   refc_smoke_config, refc_xl_config)
    presets = {"small": refc_small_config, "base": refc_config,
               "xl": refc_xl_config, "smoke": refc_smoke_config}
    cfg = presets[preset]()
    cj = Path(ckpt).parent / "config.json"
    if cj.exists():
        _apply_overrides(cfg, json.loads(cj.read_text()).get("cfg", {}))
    model = RefCModel(cfg)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    return model, cfg, ck.get("step"), missing, unexpected


def assert_raster(cfg, feats) -> tuple:
    """⛔ The gate whose ABSENCE produced the retracted 2026-08-02 numbers."""
    gh, gw = cfg.encoder.grid_shape
    exp = (gh * 32, gw * 32)                      # refc.py:214-218 patch stride
    got = tuple(feats.shape[-2:])
    if got != exp:
        raise RuntimeError(
            f"C-raster REFUSES: val raster {got} but the arm declares "
            f"grid_shape {(gh, gw)} => trained at {exp}. Scoring here is a "
            f"SILENT instrument failure — base accepts any token count and "
            f"returns a plausible wrong number (R-2026-08-02-a).")
    return got, (gh, gw), gh * gw


def sattolo(n: int, seed: int) -> np.ndarray:
    """A guaranteed DERANGEMENT (no fixed point), not merely a shuffle.

    A plain ``rng.permutation`` leaves ~1 window in e (37 %... no: exactly 1 on
    average) mapped to ITSELF, and a self-map is a leg of C-ctxswap that
    silently is NOT swapped. Sattolo's algorithm yields a single cycle, so every
    window is scored against a DIFFERENT window's context by construction — the
    property the control's reading depends on.
    """
    rng = np.random.default_rng(seed)
    p = np.arange(n)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))               # strictly < i => no fixed point
        p[i], p[j] = p[j], p[i]
    return p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-against", default=None,
                    help="a refc_sel_dump_refined bank; C-reproduce-esel")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import driving_diagnostic as dd
    from taniteval import data
    from taniteval.refc_eval import resolve_nav
    from tanitad.refs import refc_select as sl

    t0 = time.time()
    model, cfg, step, missing, unexpected = build_model(a.ckpt, a.preset)
    model = model.to(a.device).eval()
    steps = cfg.decoder.diffusion_steps
    K_MAX = max(dd.WP_STEPS)

    files = data.list_val_episodes(a.val, a.episodes)
    # ⛔⛔ EPISODE ID FROM POSITION IN THE **FULL** LIST. `data.load_frames`
    # numbers episodes by list position, so a dropped clip RENUMBERS every later
    # one — that produced "agreement 0.7183, fans 22 m apart" before a GT-based
    # assert caught it. Gaps are left as HOLES, never closed.
    dropped, eps = [], []
    for i, f in enumerate(files):
        try:
            eps.append(data.RawEp(data.load_episode(str(f), mmap=True), i))
        except Exception as exc:
            dropped.append({"file": Path(f).name, "episode_id": i,
                            "bytes": Path(f).stat().st_size,
                            "error": type(exc).__name__})
    if dropped:
        print(f"[deploy] ⚠️ {len(dropped)} UNREADABLE clip(s) quarantined: "
              f"{[(x['file'], x['episode_id']) for x in dropped]}", flush=True)
    raster, grid, n_tok = assert_raster(cfg, eps[0].feats)
    print(f"[deploy] raster {raster} grid {grid} = {n_tok} tokens :: C-raster PASS",
          flush=True)

    FAN, LOG, REF, SEL, GT, CV, EID, V0, SPD, HDG, POOL, TGT = (
        [], [], [], [], [], [], [], [], [], [], [], [])
    for ep in eps:
        fr, poses = ep.feats, ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - WINDOW - K_MAX, STRIDE))
        for i in range(0, len(starts), a.batch):
            ch = starts[i:i + a.batch]
            last = torch.tensor([t + WINDOW - 1 for t in ch])
            fw = torch.stack([torch.as_tensor(fr[t:t + WINDOW]) for t in ch]) \
                .to(a.device).float().div_(255.0)
            v0 = poses[last, 3].to(a.device)
            nav_cmd, _ = resolve_nav(model, fw, v0, steps, NAV_MODE,
                                     poses=poses, last=last)
            with torch.no_grad():
                o = model(fw, nav_cmd=nav_cmd, v0=v0, steps=steps)
                fut = torch.stack([torch.as_tensor(fr[t + WINDOW - 1 + LAW_AHEAD])
                                   for t in ch]).to(a.device).float().div_(255.0)
                law_tgt = model.encode_pooled(fut)              # [b,F] = z_{t+5}
            FAN.append(o["anchor_traj"].float().cpu())
            LOG.append(o["anchor_logits"].float().cpu())
            REF.append(o["refined_logits"].float().cpu())
            SEL.append(o["sel_idx"].cpu())
            POOL.append(o["pooled"].float().cpu())
            TGT.append(law_tgt.float().cpu())
            GT.append(dd.gt_ego_waypoints(ep.poses, last))
            CV.append(dd.baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            V0.append(poses[last, 3])
            SPD.append(ep.poses[last, 3])
            HDG.append(dd.net_heading_change_deg(ep.poses, last))
        print(f"[deploy] ep{ep.episode_id} {len(starts)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)

    fan = torch.cat(FAN)
    pooled = torch.cat(POOL)
    law_tgt = torch.cat(TGT)
    B, N = fan.shape[:2]
    eid = np.asarray(EID)

    # ---------------------------------------------------------------- scores
    # ⭐ THE DEPLOYABLE SCORE IS COMPUTED BY CALLING THE DEPLOYED FUNCTION.
    # `guard=False` so a DEGENERATE score is MEASURED (C-degenerate) instead of
    # raising — a control has to be able to report its own failure. The guard is
    # then run separately and its verdict recorded.
    swap = sattolo(B, CTXSWAP_SEED)
    dep, orc, cxs, cons_norm = [], [], [], []
    CH = 32
    with torch.no_grad():
        for s in range(0, B, CH):
            e = min(s + CH, B)
            f_ = fan[s:e].to(a.device)
            p_ = pooled[s:e].to(a.device)
            ps = pooled[swap[s:e]].to(a.device)
            dep.append(sl.consequence_scores(f_, p_, model.law_head,
                                             model.decoder.feat_proj,
                                             model.decoder.conf_head,
                                             detach=True, guard=False).float().cpu())
            cxs.append(sl.consequence_scores(f_, ps, model.law_head,
                                             model.decoder.feat_proj,
                                             model.decoder.conf_head,
                                             detach=True, guard=False).float().cpu())
            # E-SEL's ORACLE statistic, on the SAME fan, same chunking
            inp = torch.cat([p_[:, None].expand(e - s, N, p_.shape[-1]),
                             f_.reshape(e - s, N, -1)], dim=-1)
            cons = model.law_head(inp)                          # [b,N,F]
            orc.append(-(cons - law_tgt[s:e].to(a.device)[:, None])
                       .pow(2).mean(-1).float().cpu())
            cons_norm.append(cons.norm(dim=-1).float().cpu())
    cons_deploy = torch.cat(dep)
    cons_oracle = torch.cat(orc)
    cons_ctxswap = torch.cat(cxs)

    # C-degenerate: would `assert_candidate_axis` raise on the deployable score?
    try:
        sl.assert_candidate_axis(cons_deploy.to(a.device), N,
                                 name="refc consequence score")
        guard_verdict = "PASS — score varies along the candidate axis"
    except Exception as exc:                                    # pragma: no cover
        guard_verdict = f"RAISED — {type(exc).__name__}: {exc}"

    d = dict(fan=fan, logits=torch.cat(LOG), refined_logits=torch.cat(REF),
             cons_score=cons_oracle,          # same key/meaning as E-SEL's bank
             cons_oracle=cons_oracle, cons_deploy=cons_deploy,
             cons_ctxswap=cons_ctxswap,
             pooled=pooled, law_tgt=law_tgt,
             sel=torch.cat(SEL), gt=torch.cat(GT).float(),
             cv=torch.cat(CV).float(), eid=EID, v0=torch.cat(V0).float(),
             speed=torch.cat(SPD).float(), head_deg=torch.cat(HDG).float(),
             wp_steps=list(dd.WP_STEPS), ckpt=a.ckpt, ckpt_step=step,
             steps=steps, nav_mode=NAV_MODE, n_anchors=int(N),
             raster=raster, grid_shape=grid, n_tokens=n_tok,
             law_ahead=LAW_AHEAD, host=os.uname().nodename,
             episodes_requested=a.episodes, episodes_scored=len(eps),
             episodes_dropped=dropped,
             ctxswap_seed=CTXSWAP_SEED, ctxswap_perm=swap.tolist(),
             ctxswap_is_derangement=bool((swap != np.arange(B)).all()),
             ctxswap_cross_episode_frac=float((eid[swap] != eid).mean()),
             candidate_axis_guard=guard_verdict,
             cons_latent_norm_mean=float(torch.cat(cons_norm).mean()),
             sd_missing=len(missing), sd_unexpected=len(unexpected),
             torch_version=torch.__version__,
             deploy_provenance=(
                 "conf_head(layer_norm(feat_proj(law_head([pooled, fan_i])))) — "
                 "computed by CALLING refc_select.consequence_scores, the exact "
                 "function refc.py:1248-1250 calls, with cons_head=law_head and "
                 "ctx=pooled per refc.py:1835-1836. NO FUTURE FRAME IN THE PATH. "
                 "detach=True mirrors sel.cons_detach's default."),
             oracle_provenance=(
                 "-mean_sq(law_head([pooled, fan_i]) - encode_pooled(frame_{t+5})) "
                 "— E-SEL-1's statistic, recomputed HERE on the SAME fan so the "
                 "upper bound and the deployable score are PAIRED. Uses the "
                 "FUTURE FRAME and is an UPPER BOUND ONLY."),
             ctxswap_provenance=(
                 "s_deploy with `pooled` replaced by a Sattolo DERANGEMENT of the "
                 "windows (seed 20260803): every window scored against a "
                 "DIFFERENT window's context, no fixed points. Asks whether the "
                 "score reads the scene at all or only trajectory shape."),
             wall_s=round(time.time() - t0, 1))

    # -------------------------------------------------- C-reproduce-esel
    if a.verify_against:
        old = torch.load(a.verify_against, map_location="cpu", weights_only=False)
        rep = {"bank": a.verify_against}
        for k in ("fan", "logits", "refined_logits", "cons_score", "gt", "cv"):
            if k not in old:
                rep[k] = "ABSENT in bank"
                continue
            same = bool(torch.equal(old[k], d[k]))
            mx = float((old[k].float() - d[k].float()).abs().max())
            rep[k] = {"bit_identical": same, "max_abs_diff": mx}
        rep["eid_identical"] = bool(list(old.get("eid", [])) == list(d["eid"]))
        rep["sel_agreement"] = float((old["sel"] == d["sel"]).float().mean()) \
            if "sel" in old else None
        rep["reading"] = (
            "eval decode is deterministic (refc.py:1217-1218: noise only when "
            "self.training), so BIT-IDENTICAL is the expected result on the same "
            "host. Any deviation means rho_deploy and rho_oracle are not on the "
            "same fan and the paired contrast is VOID.")
        d["c_reproduce_esel"] = rep
        print(f"[deploy] C-reproduce-esel: {json.dumps(rep, indent=2)[:900]}",
              flush=True)

    torch.save(d, a.out)
    print(f"[deploy] {B} windows x {N} anchors -> {a.out} ({d['wall_s']}s)",
          flush=True)
    print(f"[deploy] guard: {guard_verdict}", flush=True)
    print(f"[deploy] ctxswap derangement={d['ctxswap_is_derangement']} "
          f"cross_episode={d['ctxswap_cross_episode_frac']:.4f}", flush=True)
    sd = cons_deploy.std(dim=1)
    print(f"[deploy] s_deploy per-window std: median {float(sd.median()):.6g} "
          f"min {float(sd.min()):.6g} · s_oracle median "
          f"{float(cons_oracle.std(dim=1).median()):.6g}", flush=True)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
