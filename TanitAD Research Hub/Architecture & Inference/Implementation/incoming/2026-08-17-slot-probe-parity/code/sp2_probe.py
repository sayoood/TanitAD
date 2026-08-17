"""SP2 — train the agent-slot readout on the BANKED frozen latents and score
AGENT_SLOT_DECODER.md §4 against its pre-registered outcomes.

⛔ TIER: T0-DIAGNOSTIC. A frozen-latent readout is a world-model diagnostic and
is NEVER a driving-performance number (EVAL_DOCTRINE; prereg §4.6).

⛔ ESTIMATOR: ``taniteval.ci.paired_episode_cluster_bootstrap`` (n_boot 2000),
clustered on the EVAL episodes, paired on the SAME windows for every
arm-vs-control comparison. ``overlapping_holdout_se`` is FORBIDDEN and is not
imported here.

THE ARMS (§1.4) — a control pair, not a choice:
  ``cells``  failure => the LATENT does not carry agents
  ``tokens`` failure => the ENCODER does not carry agents
  ``cells`` failing while ``tokens`` succeeds is a POSITIVE finding: it
  localises the loss to the readout/aggregation, not to the world model (D2).

THE CONTROLS (§4.3):
  C-CONST  predict the probe-TRAIN median lead gap for every window. A head
           that cannot beat a constant has measured nothing.
  C-SHUF   the trained head, same weights, latents PERMUTED ACROSS WINDOWS
           WITHIN THE EPISODE. ⛔ Without this a result is inadmissible: this
           programme has repeatedly scored an echo as skill (nav-echo 1.0000;
           T1 action echo 97.9 % open-loop / 0.0 % hold-action).
  C-EPMEAN ⭐ ADDED 2026-08-17 — the leave-one-out mean GT gap of the window's OWN
           eval episode. THE CONFOUND C-SHUF CANNOT SEE: the lead gap varies
           ~3.9 m within an episode against ~6.2 m between, so a head that only
           RECOGNISES THE EPISODE beats the global constant while perceiving no
           agent — and it scores IDENTICALLY under C-SHUF, which permutes within
           the episode. An ORACLE (it reads eval labels), so it is a CEILING on
           that strategy, never a legitimate baseline.
  C-TOK    the ``tokens`` arm (run as its own arm).

⚠️ WINDOW SET AND PAIRING. The primary metric is scored on windows that (a)
carry a GT in-corridor lead within 30 m and (b) on which EVERY scored arm emits
an in-corridor slot. The intersection is taken ONCE and applied to all arms, so
the paired bootstrap compares arms on identical windows. Abstentions are counted
and reported, never silently dropped.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

STACK = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\stack")
TANITEVAL = Path(r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\taniteval")
for p in (str(STACK), str(STACK / "scripts"), str(TANITEVAL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tanitad.models.agent_slots import (  # noqa: E402
    AgentSlotDecoder, PARAM_BAND, SlotDecodeRanges, match_slots, slot_set_loss,
    targets_from_join)
from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

CORRIDOR_M = 1.75          # §4.1 in-corridor rule, fixed for BOTH sides
LEAD_MAX_M = 30.0          # §4.1 recall stratum
K4_THRESHOLD_M = 0.9769    # §4.4 K4: the D-LEAD-1 GT-vs-CV Δ headway
N_BOOT = 2000              # §4.2
#: ⚠️ A PHYSICAL FLOOR ON THE CLOSING RATE, not a numerical epsilon. With
#: eps=1e-3 a lead closing at 1 mm/s yields a 30 000 s "time to collision" and
#: the mean of that column is meaningless (MEASURED: 199.96 s mean error, CI
#: [26.9, 589.4] — an interval that says nothing). TTC is reported only where
#: BOTH sides are closing faster than this, with its own n.
CLOSING_FLOOR = 0.5        # m/s


# ---------------------------------------------------------------------------
# targets / batching
# ---------------------------------------------------------------------------
def build_targets(rows, idx, device):
    """Padded target dict for the frames ``idx`` (a batch of the cache rows)."""
    n_pad = max(1, max(int(rows[i]["agents"].shape[0]) for i in idx))
    parts = []
    for i in idx:
        r = rows[i]
        parts.append(targets_from_join(
            r["agents"].numpy(), classes=r["classes"],
            rates=r["rates"].numpy(), rates_mask=r["rates_mask"].numpy(),
            n_pad=n_pad, device=device))
    return {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}


def memory_of(rows, idx, key, device):
    return torch.stack([rows[i][key] for i in idx]).to(device).float()


# ---------------------------------------------------------------------------
# the lead readout — §4.1, applied IDENTICALLY to prediction and to GT
# ---------------------------------------------------------------------------
def gt_lead_gap(agents: torch.Tensor) -> float | None:
    """Nearest in-corridor GT agent within LEAD_MAX_M, or None.

    ⭐ The occlusion flag is deliberately NOT applied: an agent with
    ``|cy| <= 1.75`` and ``cx > 1.01 m`` is inside the 120 deg field BY
    GEOMETRY (out-of-field needs ``|cy| > cx*tan 60 deg``), so an `occ` filter
    here would be a no-op dressed as a criterion.
    """
    if agents.numel() == 0:
        return None
    cx, cy = agents[:, 0], agents[:, 1]
    m = (cx > 0) & (cy.abs() <= CORRIDOR_M) & (cx <= LEAD_MAX_M)
    if not bool(m.any()):
        return None
    return float(cx[m].min())


def gt_lead_row(agents: torch.Tensor) -> int | None:
    """Index of the GT lead agent (the one :func:`gt_lead_gap` selects)."""
    if agents.numel() == 0:
        return None
    cx, cy = agents[:, 0], agents[:, 1]
    m = (cx > 0) & (cy.abs() <= CORRIDOR_M) & (cx <= LEAD_MAX_M)
    if not bool(m.any()):
        return None
    idx = torch.nonzero(m).flatten()
    return int(idx[cx[idx].argmin()])


def ttc_of(gap_m, v_rel_x, eps: float = 1e-3):
    """``TTC = cx / max(-v_rel_x, eps)`` — §4.1's secondary, in seconds.

    Only a CLOSING lead has a finite time-to-collision; a receding one is
    reported as NaN rather than as a huge number, so the mean is over the
    windows where the quantity is defined and its ``n`` travels with it.
    """
    closing = np.maximum(-np.asarray(v_rel_x, dtype=np.float64), 0.0)
    out = np.where(closing > eps, np.asarray(gap_m, dtype=np.float64)
                   / np.maximum(closing, eps), np.nan)
    return out


def pred_lead(pred: dict):
    """Per-window lead readout. Returns
    ``(gap, emitted_geom, presence, v_rel_x, gap_oracle, oracle_ok)``.

    * ``gap`` / ``presence`` / ``v_rel_x`` — the §4.1 rule: the HIGHEST-presence
      slot with ``cx > 0`` and ``|cy| <= 1.75``.
    * ``emitted_geom`` — some slot is in-corridor at all. This is the PAIRING
      predicate (it must not depend on a threshold, or the paired window set
      would move with τ and the arms would stop being compared on the same
      windows). The τ-gated emission used for RECALL is computed by the caller
      from ``presence``.
    * ⭐ ``gap_oracle`` — the in-corridor slot whose ``cx`` is CLOSEST to the GT
      lead, filled in by the caller. It is a **DIAGNOSTIC, never a criterion**:
      it cannot enter K1-K4. Its only job is to split a null into its two very
      different causes — *the latent carries no agent geometry* (oracle also
      bad) versus *the latent carries it and the presence-ranked SELECTION
      cannot find it* (oracle good, argmax bad). Reporting a null without that
      split would leave the actionable half unsaid.
    """
    cx = pred["box"][..., 0]
    cy = pred["box"][..., 1]
    p = torch.sigmoid(pred["presence_logit"])
    ok = (cx > 0) & (cy.abs() <= CORRIDOR_M)
    score = torch.where(ok, p, torch.full_like(p, -1.0))
    best = score.argmax(dim=1)
    b = torch.arange(cx.shape[0], device=cx.device)
    return (cx[b, best], ok.any(dim=1), p[b, best],
            pred["rates"][b, best, 0], cx, ok)


# ---------------------------------------------------------------------------
def evaluate(head, rows, idx, mem_key, device, batch=64, shuffle_within_ep=None):
    """Per-window (gap_hat, emitted, presence) over ``idx``.

    ``shuffle_within_ep``: a permutation of ``idx`` supplying the MEMORY while
    the TARGETS stay with the original window — the C-SHUF anti-echo control.
    """
    head.eval()
    gaps, emit, pres, vrx, orc = [], [], [], [], []
    src = idx if shuffle_within_ep is None else shuffle_within_ep
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            js = src[s:s + batch]
            mem = memory_of(rows, js, mem_key, device)
            out = head(mem)
            g, e, p, v, allcx, allok = pred_lead(out)
            gaps.append(g.cpu()); emit.append(e.cpu())
            pres.append(p.cpu()); vrx.append(v.cpu())
            # oracle DIAGNOSTIC: the in-corridor slot closest to this window's
            # GT lead. Needs the GT, so it is resolved against the ORIGINAL
            # window (idx), never the shuffled one.
            for k in range(allcx.shape[0]):
                gt = gt_lead_gap(rows[idx[s + k]]["agents"])
                if gt is None or not bool(allok[k].any()):
                    orc.append(float("nan")); continue
                cand = allcx[k][allok[k]]
                orc.append(float((cand - gt).abs().min()))
    return (torch.cat(gaps).numpy(), torch.cat(emit).numpy(),
            torch.cat(pres).numpy(), torch.cat(vrx).numpy(),
            np.asarray(orc, dtype=np.float64))


def train_head(head, rows, tr_idx, mem_key, device, *, steps, lr, batch,
               seed, log_every=200):
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    rng = np.random.default_rng(seed)
    head.train()
    hist = []
    t0 = time.time()
    for st in range(steps):
        js = [int(tr_idx[k]) for k in rng.integers(0, len(tr_idx), batch)]
        mem = memory_of(rows, js, mem_key, device)
        tgt = build_targets(rows, js, device)
        out = head(mem)
        loss = slot_set_loss(out, tgt)
        opt.zero_grad(set_to_none=True)
        loss["total"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        if st % log_every == 0 or st == steps - 1:
            row = {"step": st,
                   **{k: round(float(v.detach()), 4) for k, v in loss.items()
                      if k.startswith("loss_") or k == "total"},
                   "n_matched": loss["n"]["matched"],
                   "n_dropped": loss["n"]["dropped"]}
            hist.append(row)
            print(f"[sp2:{mem_key}] {row}", flush=True)
    return {"history": hist, "wall_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-episodes", type=int, default=40,
                    help="episodes (provider order) reserved for EVAL; the "
                         "remainder train the probe. Episode-DISJOINT. "
                         "⚠️ CONVENIENCE ORDER — superseded by --split-json.")
    ap.add_argument("--split-json", default=None,
                    help="p3_selection.json: the DECLARED eval/train clip lists. "
                         "⛔ Preferred over --eval-episodes, which slices clips in "
                         "PROVIDER ORDER — the convenience selection that left the "
                         "2026-08-16 run with 13 lead-carrying eval episodes. When "
                         "given, the split is a pure function of the declared file "
                         "and any clip in the cache but not in either list is "
                         "REFUSED rather than silently assigned.")
    ap.add_argument("--arms", nargs="+", default=["cells"],
                    choices=["cells", "tokens"])
    ap.add_argument("--n-queries", type=int, default=32,
                    help="MEASURED from the join's in-grid p99 (§2 requires "
                         "this be fitted, not inherited)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)

    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    blob = torch.load(a.cache, map_location="cpu", weights_only=False)
    rows, meta = blob["rows"], blob["meta"]
    stamp = meta["run_stamp"]
    print(f"[sp2] cache {len(rows)} frames · {stamp} · tier T0-DIAGNOSTIC",
          flush=True)

    # ---- episode-DISJOINT split ---------------------------------------------
    clips = sorted({r["clip_id"] for r in rows})
    if a.split_json:
        decl = json.loads(Path(a.split_json).read_text("utf-8"))
        eval_clips = set(decl["eval_clips"]); train_clips = set(decl["train_clips"])
        if eval_clips & train_clips:
            raise SystemExit("[sp2] declared split is not disjoint")
        stray = [c for c in clips if c not in eval_clips and c not in train_clips]
        if stray:
            raise SystemExit(
                f"[sp2] {len(stray)} cached clips are in NEITHER declared list "
                f"(e.g. {stray[:3]}). Refusing to assign them — a clip that "
                f"arrived outside the declared selection must not be scored.")
        split_src = (f"DECLARED {Path(a.split_json).name} "
                     f"(stratum {decl['stratum']['rule']}, "
                     f"seed {decl['sample']['seed']})")
    else:
        eval_clips = set(clips[:int(a.eval_episodes)])
        train_clips = set(clips) - eval_clips
        split_src = "provider order (convenience) — --eval-episodes"
    ev_idx = [i for i, r in enumerate(rows) if r["clip_id"] in eval_clips]
    tr_idx = [i for i, r in enumerate(rows) if r["clip_id"] in train_clips]
    print(f"[sp2] split source: {split_src}", flush=True)
    if not tr_idx or not ev_idx:
        raise SystemExit(f"[sp2] empty split: {len(tr_idx)} train / "
                         f"{len(ev_idx)} eval windows over {len(clips)} clips")
    ev_eid = np.array([rows[i]["episode_uid"] for i in ev_idx])
    print(f"[sp2] clips {len(clips)} -> EVAL {len(eval_clips)} clips / "
          f"{len(ev_idx)} windows · TRAIN {len(train_clips)} clips / "
          f"{len(tr_idx)} windows (episode-disjoint)", flush=True)

    # ---- GT leads -----------------------------------------------------------
    gt_ev = np.array([gt_lead_gap(rows[i]["agents"]) if
                      gt_lead_gap(rows[i]["agents"]) is not None else np.nan
                      for i in ev_idx])
    gt_tr = [gt_lead_gap(rows[i]["agents"]) for i in tr_idx]
    gt_tr = np.array([g for g in gt_tr if g is not None])
    has_gt = ~np.isnan(gt_ev)
    # ⛔ C-CONST is calibrated on the PROBE-TRAIN split only — a constant fitted
    # on the eval split would be peeking at the answer it is the control for.
    # GT lead's own closing rate + ego speed, for the §4.1 TTC secondary and
    # the LONGITUDINAL time-gap. Both are LABEL-side quantities.
    gt_vrx = np.full(len(ev_idx), np.nan)
    v0_ev = np.array([float(rows[i].get("v0", np.nan)) for i in ev_idx])
    for k, i in enumerate(ev_idx):
        r = gt_lead_row(rows[i]["agents"])
        if r is not None and bool(rows[i]["rates_mask"][r]):
            gt_vrx[k] = float(rows[i]["rates"][r, 0])
    tr_vrx = []
    for i in tr_idx:
        r = gt_lead_row(rows[i]["agents"])
        if r is not None and bool(rows[i]["rates_mask"][r]):
            tr_vrx.append(float(rows[i]["rates"][r, 0]))
    const_vrx = float(np.median(tr_vrx)) if tr_vrx else 0.0
    const_m = float(np.median(gt_tr))
    print(f"[sp2] GT lead present in {int(has_gt.sum())}/{len(ev_idx)} eval "
          f"windows ({has_gt.mean():.3f}); C-CONST = train-median "
          f"{const_m:.4f} m (n_train_leads={len(gt_tr)})", flush=True)

    # ---- C-SHUF permutation: WITHIN episode, memory only --------------------
    rng = np.random.default_rng(a.seed + 7)
    by_ep: dict[int, list[int]] = {}
    for pos, i in enumerate(ev_idx):
        by_ep.setdefault(rows[i]["episode_uid"], []).append(pos)
    shuf_pos = np.arange(len(ev_idx))
    for _ep, poss in by_ep.items():
        perm = list(poss)
        if len(perm) > 1:                       # derangement-ish: rotate then shuffle
            p = rng.permutation(len(perm))
            # avoid identity fixed points where possible
            for k in range(len(perm)):
                if p[k] == k and len(perm) > 1:
                    p[k], p[(k + 1) % len(perm)] = p[(k + 1) % len(perm)], p[k]
            for k, src in enumerate(p):
                shuf_pos[poss[k]] = poss[src]
    n_fixed = int((shuf_pos == np.arange(len(ev_idx))).sum())
    shuf_idx = [ev_idx[j] for j in shuf_pos]

    # ---- C-SHUF-XEP: the BETWEEN-episode permutation ------------------------
    # ⭐ PREREG AMENDMENT 2026-08-17 (see PREREG_AMENDMENT_EPISODE_IDENTITY.md).
    # C-SHUF above permutes memory WITHIN an episode. C-SHUF-XEP takes each
    # window's memory from a DIFFERENT episode entirely (episode blocks cycled by
    # one, deterministic). The two are NOT interchangeable and rule out different
    # things:
    #   C-SHUF     destroys WINDOW identity, PRESERVES episode identity.
    #              Δ≈0 => the head is not using anything that varies within the
    #              episode. It is BLIND to a head that reads only episode identity.
    #   C-SHUF-XEP destroys BOTH.
    #              Δ≈0 => the head is not using the input AT ALL (pure prior).
    #              Δ<0 (arm better) WITH C-SHUF Δ≈0 => the head IS reading its
    #              input, but only at EPISODE granularity — i.e. scene
    #              recognition, not agent perception.
    #   C-EPMEAN   the label-side CEILING on exactly that episode-granularity
    #              strategy, so it says HOW MUCH of the arm's score it explains.
    ep_keys = sorted(by_ep)
    xep_pos = np.arange(len(ev_idx))
    if len(ep_keys) > 1:
        for n_e, ep in enumerate(ep_keys):
            src_poss = by_ep[ep_keys[(n_e + 1) % len(ep_keys)]]
            for k, pos in enumerate(by_ep[ep]):
                xep_pos[pos] = src_poss[k % len(src_poss)]
    xep_idx = [ev_idx[j] for j in xep_pos]
    n_same_ep = int(sum(1 for k in range(len(ev_idx))
                        if rows[ev_idx[k]]["clip_id"]
                        == rows[xep_idx[k]]["clip_id"]))

    # ⭐ HOW DISCRIMINATING CAN C-SHUF POSSIBLY BE? — a control ON THE CONTROL.
    # C-SHUF swaps the MEMORY between windows of the SAME episode. If the GT lead
    # gap barely moves within an episode (steady-state car-following), then the
    # shuffled memory answers almost the right question and K2 CANNOT separate
    # however well the head reads — a guard structurally unable to fail, the C13
    # family, one level up. The 2026-08-16 run reported K2 unseparated at both
    # checkpoints without ever measuring this ceiling. It is measured here, from
    # the LABELS ONLY, before any head exists:
    #   within_sd  the mean over eval episodes of the within-episode SD of the GT
    #              gap — the signal C-SHUF destroys, i.e. the most K2 could see.
    #   swap_mae   the mean |gap(i) - gap(shuf(i))| actually induced by THIS
    #              permutation — the realised perturbation, not an idealisation.
    # A swap_mae that is small next to the arm's own error means K2 is weak BY
    # CONSTRUCTION and must not be read as evidence either way.
    _w_sd = []
    for _ep, poss in by_ep.items():
        g = gt_ev[np.asarray(poss)]
        g = g[~np.isnan(g)]
        if g.size >= 2:
            _w_sd.append(float(np.std(g)))
    _sw = np.abs(gt_ev - gt_ev[shuf_pos])
    c_shuf_power = {
        "within_episode_gt_gap_sd_m": (round(float(np.mean(_w_sd)), 4)
                                       if _w_sd else None),
        "between_episode_gt_gap_sd_m": round(float(np.nanstd(gt_ev)), 4),
        "realised_swap_mae_m": round(float(np.nanmean(_sw)), 4),
        "n_episodes_with_ge2_leads": len(_w_sd),
        "fixed_points": n_fixed,
        "_read": ("realised_swap_mae_m is the perturbation C-SHUF actually "
                  "applies to the QUESTION. If an arm's own error is much larger "
                  "than this, K2 has little to detect and its 'not separated' is "
                  "uninformative — say so rather than reading it as a null."),
    }
    print(f"[sp2] C-SHUF discriminability: {json.dumps(c_shuf_power)}", flush=True)

    results: dict = {"run_stamp": stamp, "eval_tier": "T0-DIAGNOSTIC",
                     "_evidence_class": "MEASURED (ours)",
                     "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
                     "n_boot": N_BOOT, "forbidden": "overlapping_holdout_se",
                     "split_source": split_src,
                     "n_eval_episodes": len(eval_clips),
                     # ⭐ THE POWER STATISTIC. The bootstrap CLUSTERS ON EPISODES,
                     # so this — not the window count — is the n behind every
                     # interval below. The 2026-08-16 run's was 13.
                     "n_eval_episodes_with_gt_lead": int(len(
                         {rows[i]["clip_id"] for k, i in enumerate(ev_idx)
                          if not np.isnan(gt_ev[k])})),
                     "n_eval_windows": len(ev_idx),
                     "n_train_windows": len(tr_idx),
                     "c_shuf_fixed_points": n_fixed,
                     "c_shuf_xep_same_episode_windows": n_same_ep,
                     "c_shuf_discriminability": c_shuf_power,
                     "prereg_amendment": (
                         "2026-08-17 — C-EPMEAN + C-SHUF-XEP added to §1.4's "
                         "control list (C-CONST/C-SHUF/C-TOK/C-V5F), which does "
                         "not cover EPISODE IDENTITY. Made BEFORE any arm on "
                         "this corpus was fitted. See "
                         "PREREG_AMENDMENT_EPISODE_IDENTITY.md"),
                     "c_const_m": const_m,
                     "gt_lead_rate": float(has_gt.mean()),
                     "n_queries": int(a.n_queries),
                     "cache_meta": {k: meta[k] for k in
                                    ("step", "step_source", "n_frames",
                                     "stride", "n_cells", "d_readout",
                                     "token_grid", "cuda_max_mem_gb")},
                     "arms": {}}

    arm_pred: dict[str, np.ndarray] = {}
    arm_emit: dict[str, np.ndarray] = {}
    arm_vrx: dict[str, np.ndarray] = {}
    arm_pres: dict[str, np.ndarray] = {}
    arm_oracle: dict[str, np.ndarray] = {}

    for arm in a.arms:
        key = "cells" if arm == "cells" else "tokens"
        if rows[0][key] is None:
            print(f"[sp2] arm {arm}: NOT BANKED in this cache — skipped",
                  flush=True)
            results["arms"][arm] = {"status": "NOT_BANKED"}
            continue
        d_mem = int(rows[0][key].shape[-1])
        n_mem = int(rows[0][key].shape[0])
        head = AgentSlotDecoder(d_mem, n_mem, n_queries=int(a.n_queries),
                                d_model=256, depth=3, n_heads=8,
                                ranges=SlotDecodeRanges(),
                                enforce_band=False).to(device)
        n_par = head.n_params
        in_band = PARAM_BAND[0] <= n_par <= PARAM_BAND[1]
        print(f"[sp2] arm {arm}: memory [{n_mem}, {d_mem}] · head {n_par:,} "
              f"params · §6 band {PARAM_BAND} -> {'IN' if in_band else 'OUT'}",
              flush=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        tr = train_head(head, rows, tr_idx, key, device, steps=a.steps,
                        lr=a.lr, batch=a.batch, seed=a.seed)
        peak = (float(torch.cuda.max_memory_allocated()) / 1e9
                if device.type == "cuda" else None)
        g, e, p, vx, orc = evaluate(head, rows, ev_idx, key, device)
        gs, es, ps, vxs, orcs = evaluate(head, rows, ev_idx, key, device,
                                         shuffle_within_ep=shuf_idx)
        gx, ex_, px, vxx, orcx = evaluate(head, rows, ev_idx, key, device,
                                          shuffle_within_ep=xep_idx)
        arm_pred[arm] = g; arm_emit[arm] = e; arm_vrx[arm] = vx
        arm_pres[arm] = p; arm_oracle[arm] = orc
        arm_pred[f"{arm}__C-SHUF"] = gs; arm_emit[f"{arm}__C-SHUF"] = es
        arm_vrx[f"{arm}__C-SHUF"] = vxs; arm_pres[f"{arm}__C-SHUF"] = ps
        arm_oracle[f"{arm}__C-SHUF"] = orcs
        arm_pred[f"{arm}__C-SHUF-XEP"] = gx; arm_emit[f"{arm}__C-SHUF-XEP"] = ex_
        arm_vrx[f"{arm}__C-SHUF-XEP"] = vxx; arm_pres[f"{arm}__C-SHUF-XEP"] = px
        arm_oracle[f"{arm}__C-SHUF-XEP"] = orcx
        results["arms"][arm] = {
            "status": "TRAINED", "n_params": n_par,
            "param_band_ok": bool(in_band),
            "memory_shape": [n_mem, d_mem],
            "train": tr, "cuda_max_mem_gb": peak,
            "mean_presence_at_lead": float(np.mean(p)),
        }
        torch.save({"head": head.state_dict(), "arm": arm, "stamp": stamp},
                   out / f"head_{arm}.pt")

    # C-CONST + the paired comparisons ---------------------------------------
    arm_pred["C-CONST"] = np.full(len(ev_idx), const_m)
    arm_emit["C-CONST"] = np.ones(len(ev_idx), dtype=bool)
    arm_vrx["C-CONST"] = np.full(len(ev_idx), const_vrx)
    arm_pres["C-CONST"] = np.ones(len(ev_idx))
    arm_oracle["C-CONST"] = np.abs(np.full(len(ev_idx), const_m) - gt_ev)

    # ⭐ C-EPMEAN — THE CONFOUND C-SHUF STRUCTURALLY CANNOT CATCH.
    # The GT lead gap barely moves inside an episode (within-episode SD ~3.9 m vs
    # ~6.2 m between). So a head that merely RECOGNISES WHICH EPISODE it is
    # looking at — trivial from appearance, and nothing to do with perceiving an
    # agent — can emit that episode's typical gap and beat the global constant.
    # ⛔ C-SHUF is blind to this by construction: it permutes memory WITHIN an
    # episode, so an episode-identity reader scores IDENTICALLY under it.
    # C-EPMEAN is the CEILING on that strategy: the leave-one-out mean GT gap of
    # the window's own eval episode. It is an ORACLE (it reads eval labels) and is
    # therefore NOT a legitimate baseline — it is an upper bound on how much of an
    # arm's apparent skill could be episode recognition. An arm that beats
    # C-CONST but not C-EPMEAN has shown nothing about agents.
    ep_of = np.array([rows[i]["clip_id"] for i in ev_idx])
    epmean = np.full(len(ev_idx), const_m, dtype=np.float64)
    for _c in np.unique(ep_of):
        m = (ep_of == _c) & (~np.isnan(gt_ev))
        pos = np.nonzero(m)[0]
        if pos.size == 0:
            continue
        tot = float(np.sum(gt_ev[pos]))
        for k in pos:                       # leave-one-out: never sees its own label
            epmean[k] = ((tot - gt_ev[k]) / (pos.size - 1)) if pos.size > 1 \
                else const_m
    arm_pred["C-EPMEAN"] = epmean
    arm_emit["C-EPMEAN"] = np.ones(len(ev_idx), dtype=bool)
    arm_vrx["C-EPMEAN"] = np.full(len(ev_idx), const_vrx)
    arm_pres["C-EPMEAN"] = np.ones(len(ev_idx))
    arm_oracle["C-EPMEAN"] = np.abs(epmean - gt_ev)

    scored = [k for k in arm_pred]
    common = has_gt.copy()
    for k in scored:
        common &= arm_emit[k].astype(bool)
    n_common = int(common.sum())
    results["n_scored_windows"] = n_common
    results["abstentions"] = {k: int((~arm_emit[k].astype(bool) & has_gt).sum())
                              for k in scored}
    print(f"[sp2] paired window set: {n_common} of {int(has_gt.sum())} "
          f"GT-lead windows (all arms emit); abstentions "
          f"{results['abstentions']}", flush=True)
    if n_common < 30:
        print("[sp2] ⚠️ fewer than 30 paired windows — the interval will be "
              "wide and the verdict weak. Reported anyway, with its n.",
              flush=True)

    eid_c = ev_eid[common]
    gt_c = gt_ev[common]
    # ⛔ THE n THAT ACTUALLY SETS EVERY INTERVAL'S WIDTH. `episode_cluster_bootstrap`
    # resamples CLUSTERS, so the effective sample size is the number of distinct
    # eval episodes surviving into the paired window set — never the window count,
    # which is ~100x larger and which a reader will otherwise take for the n.
    results["n_bootstrap_clusters"] = int(len(np.unique(eid_c)))
    print(f"[sp2] bootstrap clusters (distinct eval episodes in the paired set): "
          f"{results['n_bootstrap_clusters']}  ⇐ this, not "
          f"{n_common} windows, is the n behind every interval", flush=True)
    err = {k: np.abs(arm_pred[k][common] - gt_c) for k in scored}
    results["per_arm"] = {}
    for k in scored:
        orc = arm_oracle[k][common]
        results["per_arm"][k] = {
            "lead_gap_abs_err_m": episode_cluster_bootstrap(
                err[k], eid_c, n_boot=N_BOOT, seed=a.seed),
            "median_abs_err_m": float(np.median(err[k])),
            "mean_pred_gap_m": float(np.mean(arm_pred[k][common])),
            "mean_gt_gap_m": float(np.mean(gt_c)),
            # ⭐ DIAGNOSTIC ONLY — never a criterion (see pred_lead's docstring)
            "_diag_oracle_slot_abs_err_m": {
                "mean": float(np.nanmean(orc)),
                "median": float(np.nanmedian(orc)),
                "n": int(np.isfinite(orc).sum()),
                "_read": ("the BEST in-corridor slot vs the GT lead. Small "
                          "here with a large headline = the geometry is in the "
                          "latent and the presence-ranked SELECTION fails; "
                          "large here too = the latent does not carry it.")},
        }
    results["paired"] = {}
    for arm in a.arms:
        if arm not in err:
            continue
        for ctrl in ("C-CONST", "C-EPMEAN", f"{arm}__C-SHUF",
                     f"{arm}__C-SHUF-XEP"):
            if ctrl not in err:
                continue
            d = paired_episode_cluster_bootstrap(
                err[arm], err[ctrl], eid_c, n_boot=N_BOOT, seed=a.seed)
            results["paired"][f"{arm} vs {ctrl}"] = d
            print(f"[sp2] Δ {arm} − {ctrl}: {d.get('delta')} "
                  f"[{d.get('lo')}, {d.get('hi')}] separated="
                  f"{d.get('separated')}", flush=True)

    # ---- §4.1 SECONDARIES + the LONGITUDINAL family -------------------------
    # ⛔ THE FOUR FAMILIES (prereg §3). This head serves LONGITUDINAL directly
    # and TACTICAL only as an enabling condition; LATERAL and STRATEGIC it does
    # NOT serve, and saying so per family with the reason IS the rule.
    gt_ttc = ttc_of(gt_c, gt_vrx[common], eps=CLOSING_FLOOR)
    ttc_ok = np.isfinite(gt_ttc)
    results["four_families"] = {
        "LONGITUDINAL": {
            "served": True,
            "headway_error": "lead_gap_abs_err_m — the PRIMARY above; the "
                             "lead slot's cx IS the headway",
            "time_gap_s_GT_mean": float(np.nanmean(
                gt_c / np.maximum(v0_ev[common], 0.1))),
            "n_windows": int(n_common),
            "ttc": {},
        },
        "LATERAL": {"served": False,
                    "reason": "the head emits AGENT geometry, not ego path; "
                              "heading / curvature / yaw-rate / cross-track "
                              "are ego quantities this readout never produces"},
        "TACTICAL": {"served": "ENABLING CONDITION ONLY",
                     "reason": "slot-referent agreement needs a TRAINED goal "
                               "head with the categorical `agent_slot` arg on; "
                               "S-T has not run, so the four agent-referencing "
                               "g_tac tokens still index an empty set. NOT "
                               "COMPUTED — a follow-on, per prereg §3."},
        "STRATEGIC": {"served": False,
                      "reason": "`obstacle.offline` has no map, lane graph, "
                                "junction, traffic-light or route feature — 10 "
                                "classes, all dynamic agents. Nothing strategic "
                                "is derivable from this label at all."},
        "_note": ("`taniteval.lead_metrics.distance_keeping` is NOT attached: "
                  "it consumes a predicted ego PATH (W,K,2) over K steps, and "
                  "this is a single-frame perception readout that produces no "
                  "path. Reporting it would require a planner rollout, which "
                  "is the T1 integration prereg §4.6 defers. Stated rather "
                  "than silently skipped."),
    }
    for k in scored:
        pt = ttc_of(arm_pred[k][common], arm_vrx[k][common], eps=CLOSING_FLOOR)
        both = ttc_ok & np.isfinite(pt)
        if int(both.sum()) >= 10:
            e_ttc = np.abs(pt[both] - gt_ttc[both])
            results["four_families"]["LONGITUDINAL"]["ttc"][k] = {
                "lead_ttc_abs_err_s": episode_cluster_bootstrap(
                    e_ttc, eid_c[both], n_boot=N_BOOT, seed=a.seed),
                "n": int(both.sum())}
        else:
            results["four_families"]["LONGITUDINAL"]["ttc"][k] = {
                "status": "UNAVAILABLE", "n": int(both.sum()),
                "reason": "fewer than 10 windows where BOTH the GT lead and "
                          "the predicted lead are CLOSING (TTC is undefined "
                          "for a receding lead)"}

    # ---- recall (§4.1) at the ENCODED-arm operating point -------------------
    # τ* is chosen on the `cells` (ENCODED) arm — the conservative side — and
    # then FROZEN for every other arm (the P8 τ* discipline).
    # ⛔ RECALL IS tau-GATED, or K3 CANNOT FAIL. The first cut of this script
    # called a window "emitted" whenever ANY slot was in-corridor, which is
    # true ~99.8 % of the time by geometry alone — a K3 that passes without
    # measuring anything (the C13 family: a guard structurally unable to fail).
    # tau* is chosen on the ENCODED (cells) arm and FROZEN for every other arm
    # (the P8 tau* discipline), and the emission additionally requires the slot
    # to be inside the LEAD_MAX_M stratum recall is defined over.
    results["recall"] = {}
    tau = 0.5
    if "cells" in arm_pres:
        tau = float(np.median(arm_pres["cells"][has_gt])) if has_gt.any() else 0.5
    results["recall"]["tau_star"] = round(tau, 6)
    results["recall"]["tau_source"] = (
        "median lead-slot presence of the ENCODED (cells) arm over GT-lead "
        "windows; FROZEN and reused by every other arm")
    for k in scored:
        if k == "C-CONST":
            continue
        emit_tau = (arm_emit[k].astype(bool) & (arm_pres[k] >= tau)
                    & (arm_pred[k] <= LEAD_MAX_M))
        results["recall"][k] = {
            "lead_presence_recall": float(
                (emit_tau & has_gt).sum() / max(int(has_gt.sum()), 1)),
            "n_gt_lead_windows": int(has_gt.sum()),
            "geometric_emission_rate": float(
                (arm_emit[k].astype(bool) & has_gt).sum()
                / max(int(has_gt.sum()), 1))}

    # ---- the pre-registered verdict (§4.4 / §4.5) ---------------------------
    verdict = {}
    for arm in a.arms:
        if arm not in err:
            continue
        k1 = results["paired"].get(f"{arm} vs C-CONST", {})
        k2 = results["paired"].get(f"{arm} vs {arm}__C-SHUF", {})
        k5 = results["paired"].get(f"{arm} vs C-EPMEAN", {})
        k1p = bool(k1.get("separated") and k1.get("delta", 0) < 0)
        k2p = bool(k2.get("separated") and k2.get("delta", 0) < 0)
        # K5 is NOT pre-registered and NEVER gates KEEP. It is the attribution
        # test that decides whether a PASSED K1 is about agents or about episode
        # recognition, and it only ever has to be read when K1 passes.
        k5p = bool(k5.get("separated") and k5.get("delta", 0) < 0)
        rec = results["recall"].get(arm, {}).get("lead_presence_recall", 0.0)
        k3p = bool(rec >= 0.50)
        med = results["per_arm"][arm]["median_abs_err_m"]
        k4p = bool(k1p and k2p and k3p and med < K4_THRESHOLD_M)
        verdict[arm] = {
            "K1_better_than_C-CONST": k1p, "K2_better_than_C-SHUF": k2p,
            "K3_recall_ge_0.50": k3p, "K3_recall": rec,
            "K4_median_err_lt_0.9769m": k4p, "median_abs_err_m": med,
            "K5_better_than_C-EPMEAN": k5p,
            "K5_note": ("NOT pre-registered; never gates KEEP. Read it ONLY when "
                        "K1 passes: K1 true with K5 false means the gain could be "
                        "episode recognition, which C-SHUF cannot detect."),
            "KEEP": bool(k1p and k2p and k3p),
            "admissible_as": ("inference-time lead source" if k4p else
                              ("T0 DIAGNOSTIC ONLY" if (k1p and k2p and k3p)
                               else "DROP/RE-SCOPE")),
        }
    results["verdict"] = verdict
    # D-rules
    cells_k1 = verdict.get("cells", {}).get("K1_better_than_C-CONST")
    toks_k1 = verdict.get("tokens", {}).get("K1_better_than_C-CONST")
    d_rule = None
    if cells_k1 is False and toks_k1 is False:
        d_rule = "D1 — K1 fails on BOTH arms: the ENCODER does not carry agent geometry"
    elif cells_k1 is False and toks_k1 is True:
        d_rule = "D2 — K1 fails on cells, PASSES on tokens: the READOUT GRID is the bottleneck"
    elif cells_k1 is True and verdict.get("cells", {}).get("K2_better_than_C-SHUF") is False:
        d_rule = "D3 — K1 passes, K2 fails: the head reads a corpus prior, not the window. DROP the number."
    results["d_rule"] = d_rule

    (out / "slot_probe_results.json").write_text(
        json.dumps(results, indent=1, default=str), "utf-8")
    print("[sp2] VERDICT " + json.dumps(verdict, default=str), flush=True)
    print("[sp2] D-RULE " + str(d_rule), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
