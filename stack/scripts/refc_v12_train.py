"""REF-C v1.2 — head-only trainer + the temperature/margin SWEEP.

The frozen refc-xl-30k decoder never runs here: ``refc_v12_cache.py`` already
recorded its fan, its per-candidate embeddings and its selection score, so this
trains ONLY :class:`~tanitad.models.refc_rescorer.RefCRescorer` (~1.7 M params)
over a cache that fits in RAM. An arm takes minutes, which is what makes a real
sweep affordable — and the sweep IS the experiment:

**flagship v1.5 already proved the hard-argmin target degenerates** once the fan
sharpens (``frac_sel_2x_worse`` 0.099 -> 0.40). The soft target with temperature
``tau`` reduces to that exact objective as ``tau -> 0``, so sweeping tau walks
from the known failure to whatever the safe regime is, and the DEV curve over
training steps shows whether an arm degenerates rather than only where it ends.

WHAT IS MEASURED, ON EVERY ARM, ON THE SAME WINDOWS
---------------------------------------------------
  ``base_ade``   the FROZEN refc-xl-30k selection — the before
  ``sel_ade``    the re-scorer's selection — the after
  ``oracle_ade`` best plan present in the fan (GT-informed, unreachable)
  ``*_gap``      selected - oracle, in metres: the ranking deficit itself
  ``*_2x``       fraction of windows picking >2x worse than the oracle

Model selection uses the DEV split only (episode-disjoint from train, never the
881-window TanitEval val set). The final number is produced by the harness.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python3 scripts/refc_v12_train.py \
      --cache /root/refc_v12_cache --out /workspace/experiments/refc-v12 --sweep
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import torch

from tanitad.models.refc_rescorer import (RefCRescorer, RescorerConfig,
                                          fan_ade_from, param_breakdown,
                                          q_width, rank_metrics,
                                          rescorer_loss, select_q)

# The sweep. ``soft`` tau in metres of ADE; ``pair`` margin_scale in score
# units per metre. The soft arm at tau -> 0 IS the v1.5 hard target, so the
# temperature axis walks from the known failure mode to whatever is safe; the
# explicit ``hard`` arm pins that endpoint. ``topk`` is the second axis, and
# after the v1.0 study it is a first-class one: the top-8 oracle holds 87 % of
# the ranking gap, the full-fan oracle is a lottery.
DEFAULT_SWEEP = [
    ("soft", 0.05, None), ("soft", 0.10, None), ("soft", 0.20, None),
    ("soft", 0.40, None), ("soft", 0.80, None), ("soft", 1.60, None),
    ("soft", 3.20, None), ("soft", 6.40, None),
    ("hard", None, None),
    ("pair", None, 1.0), ("pair", None, 4.0),
    ("regress", None, None),
]
DEFAULT_TOPK_SWEEP = (4, 8, 16, 32, 0)      # 0 == the full 256-wide fan


def parse_arms(spec: str) -> list[tuple]:
    """``"soft:0.4,soft:1.6,regress,pair@4"`` -> [(target, tau, margin), ...].

    Lets a focused sub-sweep be requested without editing DEFAULT_SWEEP, which
    matters once the temperature curve tells you which region to resolve.
    """
    arms = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "@" in tok:
            t, m = tok.split("@")
            arms.append((t, None, float(m)))
        elif ":" in tok:
            t, tau = tok.split(":")
            arms.append((t, float(tau), None))
        else:
            arms.append((tok, None, None))
    assert arms, f"no arms parsed from {spec!r}"
    return arms


# ---- cache -------------------------------------------------------------------

def load_split(root: Path, split: str, limit: int = 0,
               device: str = "cpu") -> dict:
    """Concatenate every shard of a split into tensors (fp16 for the wide ones).

    Shards are per-EPISODE, so a split boundary is episode-disjoint by
    construction. ``device='cuda'`` parks the whole split in VRAM (~7 GB for
    26 k windows at N=256/d=512) which turns every training step into a pure
    index op — that is what makes a 20-arm two-axis sweep affordable.
    """
    files = sorted((root / split).glob("sh_*.pt"))
    assert files, f"no shards under {root / split}"
    if limit:
        files = files[:limit]
    keys = ("q", "q0", "base_logit", "refined_conf", "fan", "pooled",
            "cond", "tgt", "v0")
    acc: dict[str, list] = {k: [] for k in keys}
    eid: list[str] = []
    for f in files:
        r = torch.load(f, map_location="cpu", weights_only=True)
        for k in keys:
            acc[k].append(r[k])
        eid.extend([r["eid"]] * int(r["v0"].shape[0]))
    out = {k: torch.cat(v).to(device) for k, v in acc.items()}
    out["eid"] = eid
    out["fan_ade"] = fan_ade_from(out["fan"].float(), out["tgt"].float())
    nb = sum(v.numel() * v.element_size() for v in out.values()
             if torch.is_tensor(v))
    print(f"[v12] {split}: {len(files)} episodes / {out['v0'].shape[0]} "
          f"windows / {nb / 2**30:.2f} GiB on {device}", flush=True)
    return out


def _batch(d: dict, idx: torch.Tensor, device: str) -> dict:
    idx = idx.to(d["v0"].device)
    return {k: d[k][idx].to(device, non_blocking=True)
            for k in ("q", "q0", "base_logit", "fan", "pooled", "cond",
                      "v0", "fan_ade", "refined_conf")}


@torch.no_grad()
def evaluate(head: RefCRescorer, d: dict, device: str, target: str,
             batch: int = 512) -> dict:
    """Full pass over a split -> the G3 mechanism read, plus the free baseline
    of selecting on the DISCARDED refined-pass confidence (the "what if REF-C
    had simply used the refined logits, untrained" control)."""
    head.eval()
    n = d["v0"].shape[0]
    acc: dict[str, float] = {}
    ref_ade = 0.0
    for i in range(0, n, batch):
        idx = torch.arange(i, min(i + batch, n))
        b = _batch(d, idx, device)
        o = head(select_q(b, head.cfg.q_source), b["base_logit"],
                 b["fan"], b["pooled"], b["cond"], b["v0"], target=target)
        m = rank_metrics(o, b["base_logit"].float(), b["fan_ade"])
        w = idx.shape[0]
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v * w
        ar = torch.arange(w, device=device)
        ref_ade += float(b["fan_ade"][ar, b["refined_conf"].float().argmax(1)]
                         .sum())
    head.train()
    out = {k: v / n for k, v in acc.items()}
    out["refined_conf_ade"] = ref_ade / n
    out["n"] = n
    # gap_recovered is a ratio of means, not a mean of ratios — recompute it
    # from the aggregated means rather than averaging per-batch ratios.
    denom = out["base_ade"] - out["oracle_ade"]
    out["gap_recovered"] = ((out["base_ade"] - out["sel_ade"]) / denom
                            if abs(denom) > 1e-9 else 0.0)
    return out


# ---- one arm -----------------------------------------------------------------

def train_arm(tr: dict, dv: dict, cfg: RescorerConfig, args, target: str,
              tau: float | None, margin: float | None, device: str) -> dict:
    name = (f"k{cfg.topk or 'all'}-{target}"
            + (f"-tau{tau:g}" if tau is not None else "")
            + (f"-m{margin:g}" if margin is not None else ""))
    torch.manual_seed(args.seed)
    head = RefCRescorer(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr,
                            weight_decay=args.wd)
    n = tr["v0"].shape[0]
    g = torch.Generator().manual_seed(args.seed)
    # A FIXED slice of the training split, evaluated with the same code as dev.
    # This is the diagnostic that separates the two ways a re-scorer can fail:
    #   fits train, not dev  -> the signal is memorisable but not predictive:
    #                           the residual ranking error is FUTURE uncertainty
    #   fits neither         -> the head/objective is simply too weak
    tr_probe = {k: (v[:args.train_probe] if torch.is_tensor(v) else v)
                for k, v in tr.items()}
    log: list[dict] = []
    best = {"dev_sel_ade": float("inf")}
    best_state = None
    t0 = time.time()

    for step in range(args.steps):
        lr = (args.lr * (step + 1) / max(args.warmup, 1) if step < args.warmup
              else 0.5 * args.lr * (1 + math.cos(math.pi * (step - args.warmup)
                                                 / max(args.steps - args.warmup,
                                                       1))))
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(n, (args.batch,), generator=g)
        b = _batch(tr, idx, device)
        o = head(select_q(b, cfg.q_source), b["base_logit"], b["fan"],
                 b["pooled"], b["cond"], b["v0"], target=target)
        loss = rescorer_loss(o, b["fan_ade"], target=target,
                             tau=tau if tau is not None else 0.2,
                             margin_scale=margin if margin is not None else 2.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0))
        opt.step()

        if step % args.eval_every == 0 or step == args.steps - 1:
            dev = evaluate(head, dv, device, target)
            trm = evaluate(head, tr_probe, device, target)
            row = {"arm": name, "step": step,
                   "tr_sel_ade": round(trm["sel_ade"], 5),
                   "tr_gap_recovered": round(trm["gap_recovered"], 5),
                   "loss": round(float(loss.detach()), 5),
                   "gnorm": round(gn, 3), "lr": round(lr, 7),
                   "base_gain": round(float(head.base_gain), 4),
                   **{f"dev_{k}": (round(v, 5) if isinstance(v, float) else v)
                      for k, v in dev.items()}}
            log.append(row)
            print(json.dumps(row), flush=True)
            if dev["sel_ade"] < best["dev_sel_ade"]:
                best = {"dev_sel_ade": dev["sel_ade"], "step": step,
                        "tr_gap_recovered": trm["gap_recovered"],
                        "tr_sel_ade": trm["sel_ade"],
                        **{f"dev_{k}": v for k, v in dev.items()}}
                best_state = {k: v.detach().clone().cpu()
                              for k, v in head.state_dict().items()}

    final = log[-1]
    out = {"arm": name, "target": target, "tau": tau, "margin_scale": margin,
           "topk": cfg.topk,
           "best": best, "final": {k: v for k, v in final.items()
                                   if k.startswith("dev_") or k == "step"},
           "wall_s": round(time.time() - t0, 1), "log": log,
           "n_params": param_breakdown(head)["total"]}
    # DEGENERATION READ — the v1.5 signature: dev worsening after its best.
    out["degeneration_m"] = round(final["dev_sel_ade"] - best["dev_sel_ade"], 5)
    out["degeneration_2x"] = round(final["dev_sel_2x"] - best["dev_sel_2x"], 5)

    d_out = Path(args.out) / name
    d_out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": best_state, "cfg": dataclasses.asdict(cfg),
                "target": target, "tau": tau, "margin_scale": margin,
                "step": best.get("step"), "dev": best}, d_out / "head.pt")
    (d_out / "arm.json").write_text(json.dumps(out, indent=2))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--train-probe", type=int, default=4000,
                    help="windows of the TRAIN split evaluated alongside dev "
                         "(the fit-vs-generalise diagnostic)")
    ap.add_argument("--q-source", default="final",
                    choices=("final", "t0", "both"),
                    help="which frozen embedding the head consumes")
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="a 1.6 M head over ~17 k windows can memorise; this "
                         "is the regularisation axis for the follow-up sweep")
    ap.add_argument("--no-q", action="store_true",
                    help="ablate the frozen refined query embedding")
    ap.add_argument("--no-geom", action="store_true",
                    help="ablate the explicit candidate kinematics")
    ap.add_argument("--no-context", action="store_true")
    ap.add_argument("--target", default=None,
                    choices=("soft", "hard", "pair", "regress"))
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--margin-scale", type=float, default=None)
    ap.add_argument("--topk", type=int, default=8,
                    help="score only the top-K by frozen confidence (0 = all)")
    ap.add_argument("--topk-sweep", default=None,
                    help="comma list of K to sweep, e.g. 4,8,16,32,0")
    ap.add_argument("--sweep", action="store_true",
                    help="run DEFAULT_SWEEP (all targets/temperatures)")
    ap.add_argument("--arms", default=None,
                    help="focused sub-sweep, e.g. 'soft:0.4,soft:1.6,pair@4,"
                         "regress,hard' (overrides --sweep)")
    ap.add_argument("--train-episodes", type=int, default=0)
    ap.add_argument("--dev-episodes", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-device", default=None,
                    help="park the cache here (default: the compute device)")
    args = ap.parse_args(argv)

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    cdev = args.cache_device or device
    root = Path(args.cache)
    man = json.loads((root / "manifest.json").read_text())
    tr = load_split(root, "train", args.train_episodes, cdev)
    dv = load_split(root, "dev", args.dev_episodes, cdev)

    def _cfg(topk: int) -> RescorerConfig:
        return RescorerConfig(n_steps=len(man["horizons"]),
                              d_q=q_width(man["d_q"], args.q_source),
                              q_source=args.q_source,
                              d_pooled=man["d_pooled"], d_cond=man["d_q"],
                              d=args.d, layers=args.layers, topk=topk,
                              dropout=args.dropout,
                              use_q=not args.no_q, use_geom=not args.no_geom,
                              use_context=not args.no_context)

    # Identity-at-init sanity: an untrained head must reproduce the frozen pick.
    probe = RefCRescorer(_cfg(args.topk)).to(device).eval()
    pb = _batch(dv, torch.arange(min(64, dv["v0"].shape[0])), device)
    with torch.no_grad():
        po = probe(select_q(pb, args.q_source), pb["base_logit"],
                   pb["fan"], pb["pooled"], pb["cond"], pb["v0"])
    assert bool((po["sel_idx"] == pb["base_logit"].float().argmax(1)).all()), \
        "zero-init re-scorer does NOT reproduce the frozen selection"
    print(f"[v12] identity-at-init OK · head params "
          f"{param_breakdown(probe)['total']:,}", flush=True)

    ks = ([int(x) for x in args.topk_sweep.split(",")] if args.topk_sweep
          else [args.topk])
    arms = (parse_arms(args.arms) if args.arms else
            DEFAULT_SWEEP if args.sweep else
            [(args.target or "soft", args.tau, args.margin_scale)])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    results = []
    for k in ks:
        cfg_k = _cfg(k)
        for target, tau, margin in arms:
            results.append(train_arm(tr, dv, cfg_k, args, target, tau, margin,
                                     device))
    results.sort(key=lambda r: r["best"]["dev_sel_ade"])

    # Frozen baselines on the dev split (identical windows, one read per K:
    # base/oracle are K-independent, the top-K oracle is not).
    b0 = evaluate(RefCRescorer(_cfg(ks[0])).to(device).eval(), dv, device,
                  "soft")
    topk_oracles, topk_incumbent = {}, {}
    for k in sorted(set(ks + list(DEFAULT_TOPK_SWEEP))):
        bk = evaluate(RefCRescorer(_cfg(k)).to(device).eval(), dv, device,
                      "soft")
        topk_oracles[str(k or "all")] = round(bk["oracle_k_ade"], 5)
        # the incumbent's skill INSIDE the set the head is allowed to re-order:
        # this is what a learned ranker actually has to beat.
        topk_incumbent[str(k or "all")] = round(bk["rank_acc_k"], 5)
    summary = {"cache": str(root), "cache_manifest": {
        k: man[k] for k in ("ckpt", "ckpt_step", "n_anchors", "horizons",
                            "windows", "stride", "decode")},
        "dev_windows": dv["v0"].shape[0], "train_windows": tr["v0"].shape[0],
        "frozen_dev_baseline": {
            "base_ade": round(b0["base_ade"], 5),
            "oracle_ade": round(b0["oracle_ade"], 5),
            "base_gap": round(b0["base_gap"], 5),
            "base_2x": round(b0["base_2x"], 5),
            "refined_conf_ade": round(b0["refined_conf_ade"], 5),
            "rank_acc": round(b0["rank_acc"], 5),
            "topk_oracle": topk_oracles,
            "topk_incumbent_rank_acc": topk_incumbent},
        "args": vars(args),
        "ranking": [{"arm": r["arm"], "topk": r["topk"],
                     "dev_sel_ade": r["best"]["dev_sel_ade"],
                     "dev_gap_recovered": r["best"].get("dev_gap_recovered"),
                     "train_gap_recovered": r["best"].get("tr_gap_recovered"),
                     "final_train_gap_recovered":
                         r["log"][-1].get("tr_gap_recovered"),
                     "dev_sel_gap": r["best"]["dev_sel_gap"],
                     "dev_sel_2x": r["best"]["dev_sel_2x"],
                     "dev_rank_acc_k": r["best"].get("dev_rank_acc_k"),
                     "best_step": r["best"].get("step"),
                     "degeneration_m": r["degeneration_m"],
                     "degeneration_2x": r["degeneration_2x"]}
                    for r in results]}
    (Path(args.out) / "sweep.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"done": True, **summary["frozen_dev_baseline"],
                      "best_arm": summary["ranking"][0]}, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
