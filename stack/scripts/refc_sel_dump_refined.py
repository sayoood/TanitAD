"""AUGMENTED fan dump — recovers the two things ``refc_rerank.dump`` threw away.

This is the GPU-side half of ``scripts/refc_sel_probe.py`` (prereg §6.4). It runs
one forward over the canonical val episodes and banks, per window:

  * ``refined_logits``  the LAST denoise pass's confidence — the score REF-C's
                        decoder computed and then DISCARDED. That discarding IS
                        defect D1, which is why no existing bank contains it, and
                        it is E-SEL-0's treatment leg.
  * ``cons_score``      per candidate, ``-||law_head([pooled, fan_i]) - z_{t+5}||^2``
                        — the consequence's agreement with what actually happened
                        next. E-SEL-1's statistic.

Everything else it writes is byte-compatible with ``refc_rerank.dump``'s schema,
so ``refc_sel_probe.py`` consumes the augmented bank with no special case.

⛔ WHY IT MUST DECODE RATHER THAN READ A BANK. ``AnchoredDiffusionDecoder``
computes the refined confidence inside the denoise loop and returns only the
offset (pre-D-SEL: ``_, off = self._decode(...)``). No banked artifact in the
programme contains it. E-SEL-0 is therefore not a re-analysis; it needs weights.

⛔ RASTER GATE (R-2026-08-02-a). REF-C was once scored at 176x624 = 120 tokens
against its trained 8x8 = 64. **XL crashed loudly; base RETURNED NUMBERS
SILENTLY**, because base has ``graft_imagination=False`` and nothing inside it
validates the token count — ``feat_proj`` accepts any n. So the fed raster is
asserted against the arm's OWN declared ``grid_shape`` before a single window is
scored, and a mismatch raises. The w120 256x640 cylindrical caches built for the
flagship are NOT admissible input here.

⚠️ NAV MODE IS ``follow_constant``, DELIBERATELY. The banked fans this run must
be comparable with were collected that way, and it is the condition the
published 0.4728 / 0.4714 were measured in. That is the 07-21 C6 confound and it
is a REAL limitation of those numbers — but changing it here would move the
baseline E-SEL-0 is paired against, which is a worse error. The mode is stamped
into the output.

⭐ E-SEL-0'S PAIRED COMPARISON IS INTERNAL TO ONE FORWARD. ``argmax(refined)``
and ``argmax(anchor)`` index the SAME ``anchor_traj`` from the SAME forward, so
the contrast is exact regardless of any host-to-host float difference. The
agreement with the banked fan is reported SEPARATELY, as a reproduction check —
never as the source of the delta.

Run (Thor, tanitad-edge venv — the INFERENCE venv; never tanitad-train):
    PYTHONPATH=~/TanitAD/stack:~/TanitAD/taniteval OMP_NUM_THREADS=6 \\
    ~/venvs/tanitad-edge/bin/python ~/refc_sel_dump_refined.py \\
        --ckpt ~/models/refc-base/ckpt.pt --preset base \\
        --val  ~/valdata/physicalai-val-0c5f7dac3b11 \\
        --out  ~/fan_refined_refc-base-30k.pt
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--preset", default="base")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    import driving_diagnostic as dd
    from taniteval import data
    from taniteval.refc_eval import resolve_nav

    t0 = time.time()
    model, cfg, step, missing, unexpected = build_model(a.ckpt, a.preset)
    model = model.to(a.device).eval()
    steps = cfg.decoder.diffusion_steps
    K_MAX = max(dd.WP_STEPS)

    files = data.list_val_episodes(a.val, a.episodes)
    # ⛔ UNREADABLE CLIPS ARE QUARANTINED AND STAMPED, NEVER SILENTLY DROPPED.
    # A relay over ssh truncates with exit 0, so a short clip is indistinguishable
    # from a good one by `ls` (RETRACTION_LOG 07-25: a 48.9 MB-short checkpoint
    # that md5 caught). Parity is SACRED, so the honest handling is to name the
    # dropped episodes in the artifact and let every downstream comparison
    # EPISODE-MATCH against them — not to quietly report a smaller number as if
    # it were the canonical one.
    #
    # ⛔⛔ THE EPISODE ID MUST COME FROM THE POSITION IN THE **FULL** FILE LIST.
    # `data.load_frames` assigns `episode_id = enumerate(files)`, so handing it a
    # SHORTENED list silently RENUMBERS every clip after the gap: dropping
    # ep_00028 made the decode's "episode 28" the file ep_00029 and produced a
    # set with no episode 39 at all. MEASURED here on the first attempt — the
    # episode-matched cross-check against the bank then compared DIFFERENT CLIPS
    # and its GT did not line up. The bootstrap clusters stay valid either way
    # (they are still real clips), but every cross-artifact join breaks. So the
    # episode is constructed with its ORIGINAL index and the gap is left as a
    # HOLE, not closed.
    dropped, eps = [], []
    for i, f in enumerate(files):
        try:
            eps.append(data.RawEp(data.load_episode(str(f), mmap=True), i))
        except Exception as exc:
            dropped.append({"file": Path(f).name, "episode_id": i,
                            "bytes": Path(f).stat().st_size,
                            "error": type(exc).__name__})
    if dropped:
        print(f"[dump] ⚠️ {len(dropped)} UNREADABLE clip(s) quarantined: "
              f"{[(x['file'], x['episode_id']) for x in dropped]} — n is REDUCED "
              f"and stamped, and the episode ids of the SURVIVORS ARE UNCHANGED "
              f"so an episode-match against a banked fan is exact", flush=True)
    raster, grid, n_tok = assert_raster(cfg, eps[0].feats)
    print(f"[dump] raster {raster} grid {grid} = {n_tok} tokens :: C-raster PASS",
          flush=True)

    FAN, LOG, REF, SEL, GT, CV, EID, V0, SPD, HDG, CONS = (
        [], [], [], [], [], [], [], [], [], [], [])
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
                fan = o["anchor_traj"]                          # [b,N,4,2]
                pooled = o["pooled"]                            # [b,F]
                # E-SEL-1: the CONSEQUENCE of flying each candidate, scored
                # against what actually happened. `law_tgt` is built exactly as
                # refc_train.compute_losses builds it — encode_pooled of the
                # frame LAW_AHEAD steps past the window, through the SAME
                # encoder — so the quantity being correlated is the world model's
                # own training target, not a re-invented proxy.
                fut = torch.stack([torch.as_tensor(fr[t + WINDOW - 1 + LAW_AHEAD])
                                   for t in ch]).to(a.device).float().div_(255.0)
                law_tgt = model.encode_pooled(fut)              # [b,F]
                b, N = fan.shape[:2]
                inp = torch.cat([pooled[:, None].expand(b, N, pooled.shape[-1]),
                                 fan.reshape(b, N, -1)], dim=-1)
                cons = model.law_head(inp)                      # [b,N,F]
                cons_s = -(cons - law_tgt[:, None]).pow(2).mean(-1)   # [b,N]
            FAN.append(fan.float().cpu())
            LOG.append(o["anchor_logits"].float().cpu())
            REF.append(o["refined_logits"].float().cpu())
            SEL.append(o["sel_idx"].cpu())
            CONS.append(cons_s.float().cpu())
            GT.append(dd.gt_ego_waypoints(ep.poses, last))
            CV.append(dd.baseline_waypoints(ep.poses, last)["constant_velocity"])
            EID.extend([ep.episode_id] * len(ch))
            V0.append(poses[last, 3])
            SPD.append(ep.poses[last, 3])
            HDG.append(dd.net_heading_change_deg(ep.poses, last))
        print(f"[dump] ep{ep.episode_id} {len(starts)} windows "
              f"({time.time() - t0:.0f}s)", flush=True)

    d = dict(fan=torch.cat(FAN), logits=torch.cat(LOG),
             refined_logits=torch.cat(REF), cons_score=torch.cat(CONS),
             sel=torch.cat(SEL), gt=torch.cat(GT).float(),
             cv=torch.cat(CV).float(), eid=EID, v0=torch.cat(V0).float(),
             speed=torch.cat(SPD).float(), head_deg=torch.cat(HDG).float(),
             wp_steps=list(dd.WP_STEPS), ckpt=a.ckpt, ckpt_step=step,
             steps=steps, nav_mode=NAV_MODE,
             n_anchors=int(torch.cat(LOG).shape[1]),
             raster=raster, grid_shape=grid, n_tokens=n_tok,
             law_ahead=LAW_AHEAD, host=os.uname().nodename,
             episodes_requested=a.episodes, episodes_scored=len(eps),
             episodes_dropped=dropped,
             parity_note=(
                 "canonical val is 40 episodes -> 881 stride-8 windows. Any "
                 "`episodes_dropped` entry REDUCES n; the reduced set is a "
                 "SUBSET of the canonical one (no re-selection, no re-cache), "
                 "so it stays comparable window-for-window after an EPISODE "
                 "MATCH against a banked fan — but the headline 881-window "
                 "number is NOT reproduced and must not be quoted as if it were."),
             sd_missing=len(missing), sd_unexpected=len(unexpected),
             torch_version=torch.__version__,
             refined_provenance=(
                 "AnchoredDiffusionDecoder's LAST denoise pass confidence, kept "
                 "by S1 (`sel_refined`). Pre-D-SEL this was discarded — that is "
                 "defect D1. UNSUPERVISED here: these weights never trained it "
                 "as a ranker, so E-SEL-0 is a LOWER BOUND on S1, not S1."),
             cons_provenance=(
                 "-mean_sq(law_head([pooled, fan_i]) - encode_pooled(frame_{t+5})). "
                 "law_head is REF-C's trajectory-conditioned world model and "
                 "encode_pooled is its own LAW target path (refc_train.compute_"
                 "losses:410), so this is the trained quantity, not a proxy."),
             wall_s=round(time.time() - t0, 1))
    torch.save(d, a.out)
    print(f"[dump] {d['fan'].shape[0]} windows x {d['n_anchors']} anchors -> "
          f"{a.out} ({d['wall_s']}s)", flush=True)
    # cheap self-checks, printed so a failure is visible in the log
    same = float((d["logits"].argmax(1) == d["sel"]).float().mean())
    ident = bool(torch.equal(d["refined_logits"], d["logits"]))
    corr = float(np.corrcoef(d["logits"].flatten().numpy(),
                             d["refined_logits"].flatten().numpy())[0, 1])
    print(f"[dump] selftest argmax(logits)==sel_idx {same:.4f} · "
          f"refined IS anchor: {ident} (must be False at steps={steps}) · "
          f"corr(anchor, refined) {corr:.4f}", flush=True)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
