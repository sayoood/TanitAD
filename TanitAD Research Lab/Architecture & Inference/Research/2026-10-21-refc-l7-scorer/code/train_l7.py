"""train_l7.py -- train an L7 scorer (modes E / F; S once a scene-token bank exists) on a BANKED fan + its targets.

    python train_l7.py --fan-npz <bank_s0>/fan_extras.npz [<bank_s1>/fan_extras.npz ...] \
        --targets-npz <targets_s0.npz> [<targets_s1.npz> ...] --mode F --seed 0 --out-dir <run> [--epochs 30] [--val-frac 0.1]
    python train_l7.py --synthetic 256 --mode F --seed 0 --out-dir <run>       # machinery smoke, no real data

Several banks (e.g. the PILOT's two inference seeds of the SAME windows) are concatenated; the episode-disjoint validation split is drawn
on the episode id, so both seeds of a validation episode land on the validation side.

⛔ The bank must be a TRAIN-split capture (PREREG sec. 2).  This script REFUSES a bank whose provenance names eval139 (the roll json's
mode / --windows-json, or the dump manifest's labels / episodes path) unless --i-know-this-is-eval is passed, and it ALSO refuses a bank
whose provenance it cannot establish at all (fail CLOSED) unless --allow-unknown-provenance is passed.  Both flags exist only for a
machinery smoke whose output is never read as a metric.  Validation is an EPISODE-disjoint slice of the bank (seeded); early stopping
reads only it.

Changed after the 2026-10-22 real-capture preflight (each is a deviation of the first version from the sealed PREREG / the capture's
actual layout; see PREFLIGHT_L7.md):
  1. several banks per run (the PILOT is two inference seeds);
  2. the loss / validation / selection masks are the e9_rank masks (reach AND set-speed ceiling AND emit, with the shipped empty-row
     fallbacks) -- the first version used reach_keep ALONE, which on the real capture is ~all-True (149.0 of 149) while the ceiling
     removes 15-17 candidates from 66 % of the rows, so it trained and early-stopped on a fan the shipped selection can never emit.
     The mode-F FEATURES still use reach_keep only: that is what integration.diff passes at inference, and training must match it;
  3. the eval-bank refusal read a ``windows_json`` key the capture wrapper never writes and a roll json the archive copy does not
     carry, i.e. it could never fire on the real layout; it now reads the roll json (mode, --windows-json) AND the dump manifest, and
     fails closed;
  4. early stopping is on the validation LOSS (PREREG sec. 1: "patience 3 on its validation loss"); the first version stopped on the
     validation pick cost.  ``--early-stop-on pick`` restores it; both are logged every epoch;
  5. per-head held-out AUC is logged (PREREG sec. 7 K7: the ``comf`` head must reach >= 0.95).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refc_scorer as RS  # noqa: E402

TARGET_KEYS = ("cost", "along", "cross", "comf", "ep", "nc", "ttc", "dac")
BANK_KEYS = ("anchor_traj", "sel_score", "refined_logits", "reach_keep", "sel_idx", "clip_index", "ws")
MASK_KEYS = ("ceil_keep", "r8_emit_keep")
AUC_HEADS = ("comf", "nc", "ttc", "dac")           # binary oracle targets; ``ep`` is continuous


def synthetic_bank(n: int, seed: int, N: int = 149) -> tuple[dict, dict]:
    """A fan with a KNOWN best candidate per window: GT = a constant-speed arc; candidates = GT + structured offsets; the
    model's sel_score is a noisy function of the true cost, so a scorer that learns anything beats it."""
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(RS.SLOT_T_REFC)
    v = 5.0 + 10.0 * torch.rand(n, generator=g)
    k = (torch.rand(n, generator=g) - 0.5) * 0.04
    s = v[:, None] * t[None]
    gt = torch.stack([s, 0.5 * k[:, None] * s ** 2], -1)                 # constant speed, constant curvature (small angle)
    dv = (torch.rand(n, N, generator=g) - 0.5) * 0.6                     # speed scale error (longitudinal)
    dy = (torch.rand(n, N, generator=g) - 0.5) * 4.0                     # lateral offset growing with t
    fan = gt[:, None] * (1 + dv[..., None, None])
    fan[..., 1] = fan[..., 1] + dy[..., None] * (t / 6.0)[None, None]
    cost = (fan - gt[:, None]).norm(dim=-1).mean(-1)
    sel_score = -cost + torch.randn(n, N, generator=g) * cost.std() * 1.5
    refined = sel_score + torch.randn(n, N, generator=g) * 0.1
    eid = np.asarray([str(i // 8) for i in range(n)])
    fanz = {"anchor_traj": fan.numpy(), "sel_score": sel_score.numpy(), "refined_logits": refined.numpy(),
            "reach_keep": np.ones((n, N), np.float32), "v0": v.numpy(), "sel_idx": sel_score.argmax(1).numpy(),
            "clip_index": np.arange(n) // 8, "ws": np.arange(n) % 8}
    fanz["keep"] = effective_keep(fanz["reach_keep"], None, None)
    from tanitad.refs import refcv7_oracle as O
    o = O.oracle_subscores(fan, t, v, gt)
    al, cr = (fan - gt[:, None]).abs().unbind(-1)
    tg = {"cost": cost.numpy(), "along": al.mean(-1).numpy(), "cross": cr.mean(-1).numpy(), "eid": eid,
          "clip_index": fanz["clip_index"], "ws": fanz["ws"],
          **{k: o[k].numpy() for k in ("comf", "ep", "nc", "ttc", "dac")},
          **{k + "_mask": o[k + "_mask"].numpy() for k in ("comf", "ep", "nc", "ttc", "dac")}}
    return fanz, tg


# ------------------------------------------------------------------------------------------------ masks
def effective_keep(reach, ceil=None, emit=None) -> np.ndarray:
    """The set of candidates the SHIPPED selection can emit: ``RS.e9_masked`` (= ``refc_v3.e9_rank``, refc_v3.py:1025-1053) on a zero
    score, so reach, then the ceiling with the reach-only fallback, then the emit mask with its fallback.  Finite = kept."""
    r = torch.from_numpy(np.asarray(reach) > 0.5)
    c = None if ceil is None else torch.from_numpy(np.asarray(ceil) > 0.5)
    e = None if emit is None else torch.from_numpy(np.asarray(emit) > 0.5)
    return torch.isfinite(RS.e9_masked(torch.zeros(r.shape, dtype=torch.float32), r, c, e)).numpy()


def _bank_provenance(fan_npz: str) -> dict:
    """What a bank says it is.  Reads (a) ``<step>/fan_roll_s*.json`` (the capture wrapper's roll json: ``mode``, ``child_argv``'s
    --windows-json, and the legacy ``windows_json`` key) and (b) ``<dump>/manifest.json`` (``corpus.labels_path`` / ``corpus.episodes``).
    Returns {"kind": "eval" | "train" | "unknown", "evidence": [...]}.  Any eval evidence wins over train evidence."""
    dump = os.path.dirname(os.path.abspath(fan_npz))
    step = os.path.dirname(dump)
    ev_eval, ev_train = [], []

    def _eval_if(s: str, why: str):
        s = str(s).replace("\\", "/")
        base = os.path.basename(s)
        if "eval139" in s or base.startswith("windows_s") or "_eval" in base or "/eval" in s.lower():
            ev_eval.append(f"{why}: {s}")
        elif "train" in s.lower() or "a6cache" in s.lower():
            ev_train.append(f"{why}: {s}")

    for rj in sorted(glob.glob(os.path.join(step, "fan_roll_s*.json"))):
        try:
            j = json.load(open(rj, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            ev_eval.append(f"unreadable roll json {os.path.basename(rj)}: {type(e).__name__}")
            continue
        mode = str(j.get("mode", ""))
        if mode == "eval139":
            ev_eval.append(f"{os.path.basename(rj)} mode=eval139")
        elif mode == "train139":
            ev_train.append(f"{os.path.basename(rj)} mode=train139")
        if j.get("windows_json"):
            _eval_if(j["windows_json"], f"{os.path.basename(rj)} windows_json")
        argv = [str(a) for a in j.get("child_argv", [])]
        for a, b in zip(argv, argv[1:]):
            if a == "--windows-json":
                ev_eval.append(f"{os.path.basename(rj)} child_argv --windows-json {b}")
    mf = os.path.join(dump, "manifest.json")
    if os.path.exists(mf):
        try:
            c = json.load(open(mf, encoding="utf-8")).get("corpus", {})
            for key in ("labels_path", "episodes"):
                if c.get(key):
                    _eval_if(c[key], f"manifest corpus.{key}")
        except Exception as e:  # noqa: BLE001
            ev_eval.append(f"unreadable manifest: {type(e).__name__}")
    kind = "eval" if ev_eval else ("train" if ev_train else "unknown")
    return {"fan_npz": fan_npz.replace("\\", "/"), "kind": kind, "evidence": ev_eval + ev_train}


def _load_one(fan_npz: str, targets_npz: str) -> tuple[dict, dict]:
    z = np.load(fan_npz, allow_pickle=True)
    t = np.load(targets_npz, allow_pickle=True)
    key = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(z["clip_index"], z["ws"]))}
    try:
        rows = np.asarray([key[(int(a), int(b))] for a, b in zip(t["clip_index"], t["ws"])])
    except KeyError as e:
        raise SystemExit(f"[train_l7] {targets_npz} has a window (clip_index, ws) = {e} that {fan_npz} does not: wrong pairing")
    fanz = {k: np.asarray(z[k])[rows] for k in BANK_KEYS}
    for k in MASK_KEYS:
        if k in z.files:
            fanz[k] = np.asarray(z[k])[rows]
    fanz["inference_seed"] = int(np.asarray(z["inference_seed"]).ravel()[0]) if "inference_seed" in z.files else -1
    fanz["v0"] = np.asarray(t["v0"])
    tg = {k: np.asarray(t[k]) for k in t.files}
    return fanz, tg


def load_bank(fan_npz, targets_npz) -> tuple[dict, dict]:
    """One bank (two paths) or several (two equal-length lists), concatenated along the window axis.  ``fanz['keep']`` is the e9
    effective keep (the loss / validation mask); ``fanz['reach_keep']`` stays the raw reach mask (the mode-F FEATURES use it)."""
    fl = [fan_npz] if isinstance(fan_npz, str) else list(fan_npz)
    tl = [targets_npz] if isinstance(targets_npz, str) else list(targets_npz)
    if len(fl) != len(tl) or not fl:
        raise SystemExit(f"[train_l7] --fan-npz ({len(fl)}) and --targets-npz ({len(tl)}) must pair up, one targets file per bank")
    parts = [_load_one(f, t) for f, t in zip(fl, tl)]
    have_masks = [all(k in p[0] for k in MASK_KEYS) for p in parts]
    if any(have_masks) and not all(have_masks):
        raise SystemExit("[train_l7] some banks carry ceil_keep / r8_emit_keep and some do not: refusing to mix mask semantics")
    seeds = [p[0].pop("inference_seed") for p in parts]
    fanz = {k: np.concatenate([p[0][k] for p in parts], 0) for k in parts[0][0]}
    fanz["bank_of_row"] = np.concatenate([np.full(len(p[0]["ws"]), i, np.int64) for i, p in enumerate(parts)])
    fanz["inference_seeds"] = seeds
    keys = set.intersection(*[set(p[1]) for p in parts])
    tg = {k: np.concatenate([p[1][k] for p in parts], 0) for k in keys}
    fanz["keep"] = effective_keep(fanz["reach_keep"], fanz.get("ceil_keep"), fanz.get("r8_emit_keep"))
    return fanz, tg


def to_batch(fanz: dict, tg: dict, ii: np.ndarray, mode: str) -> tuple[dict, dict, torch.Tensor]:
    """-> (scorer inputs, targets, LOSS mask).  The inputs are assembled with ``reach_keep`` (what integration.diff passes at
    inference: ``_rec["reach_keep"] = out["reach_keep"]``); the loss mask is the e9 effective keep."""
    rec = {"anchor_traj": torch.from_numpy(np.asarray(fanz["anchor_traj"][ii], np.float32)),
           "v0": torch.from_numpy(np.asarray(fanz["v0"][ii], np.float32)),
           "sel_score": torch.from_numpy(np.asarray(fanz["sel_score"][ii], np.float32)),
           "refined_logits": torch.from_numpy(np.asarray(fanz["refined_logits"][ii], np.float32)),
           "reach_keep": torch.from_numpy(np.asarray(fanz["reach_keep"][ii]) > 0.5)}
    keep = torch.from_numpy(np.asarray(fanz["keep"][ii])) if "keep" in fanz else rec["reach_keep"]
    inp = RS.assemble_inputs(rec, mode)
    t = {}
    for k in TARGET_KEYS:
        if k in tg:
            t[k] = torch.from_numpy(np.asarray(tg[k][ii], np.float32))
        if k + "_mask" in tg:
            t[k + "_mask"] = torch.from_numpy(np.asarray(tg[k + "_mask"][ii]).astype(bool))
    return inp, t, keep


def select_masks(fanz: dict, ii: np.ndarray) -> tuple:
    """(reach, ceil, emit) for ``RS.select`` -- the shipped pick's own three masks (None where the bank does not carry them)."""
    def g(k):
        return torch.from_numpy(np.asarray(fanz[k][ii]) > 0.5) if k in fanz else None
    return g("reach_keep"), g("ceil_keep"), g("r8_emit_keep")


def episode_split(eid: np.ndarray, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.unique(eid)
    rng = np.random.default_rng(seed)
    val = set(rng.choice(u, size=max(1, int(round(frac * len(u)))), replace=False).tolist())
    m = np.asarray([e in val for e in eid])
    return np.where(~m)[0], np.where(m)[0]


def auc_binary(score: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Mann-Whitney AUC (NaN if a class is empty)."""
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, np.float64)
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    uniq, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    rk = np.empty(len(s))
    rk[order] = avg[inv]
    return float((rk[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def evaluate(model, fanz, tg, ii, mode, bs=64, temperature=0.5) -> dict:
    model.eval()
    picks, costs, owns, orac, losses, wts = [], [], [], [], [], []
    hs = {h: ([], []) for h in AUC_HEADS if h in model.cfg.heads}
    with torch.no_grad():
        for s in range(0, len(ii), bs):
            jj = ii[s:s + bs]
            inp, t, keep = to_batch(fanz, tg, jj, mode)
            out = model(**inp)
            reach, ceil, emit = select_masks(fanz, jj)
            p = RS.select(RS.rule_imit(out), reach, ceil, emit)
            loss, _ = RS.pai_loss(out, t, keep, temperature)
            losses.append(float(loss))
            wts.append(len(jj))
            c = t["cost"]
            ar = torch.arange(len(jj))
            picks.append(p)
            costs.append(c[ar, p])
            owns.append(c[ar, torch.from_numpy(np.asarray(fanz["sel_idx"][jj]).astype(np.int64))])
            orac.append(c.masked_fill(~keep, float("inf")).min(1).values)
            for h, (sc, yy) in hs.items():
                if h in t and h + "_mask" in t:
                    m = t[h + "_mask"] & keep
                    sc.append(out[h][m].double().numpy())
                    yy.append((t[h][m] > 0.5).numpy())
    model.train()
    cat = lambda x: torch.cat(x).double()
    res = {"n": int(len(ii)), "loss": round(float(np.average(losses, weights=wts)), 5),
           "pick_cost_m": round(float(cat(costs).mean()), 5), "own_pick_cost_m": round(float(cat(owns).mean()), 5),
           "oracle_cost_m": round(float(cat(orac).mean()), 5)}
    aucs = {h: (float("nan") if not sc else auc_binary(np.concatenate(sc), np.concatenate(yy))) for h, (sc, yy) in hs.items()}
    res["head_auc"] = {h: (None if not np.isfinite(v) else round(v, 4)) for h, v in aucs.items()}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fan-npz", nargs="+")
    ap.add_argument("--targets-npz", nargs="+")
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--mode", default="F", choices=("E", "F"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30, help="0 = DRY RUN (load, split, build model + optimizer, write dry_run_log.json, take NO step)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--early-stop-on", default="loss", choices=("loss", "pick"),
                    help="loss = PREREG sec. 1 (validation loss); pick = the first version's validation pick cost")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many optimiser steps (smoke)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--i-know-this-is-eval", action="store_true")
    ap.add_argument("--allow-unknown-provenance", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    torch.manual_seed(a.seed)
    if a.synthetic:
        fanz, tg = synthetic_bank(a.synthetic, a.seed)
        src = {"synthetic": a.synthetic}
    else:
        if not a.fan_npz or not a.targets_npz:
            raise SystemExit("[train_l7] --fan-npz and --targets-npz are required without --synthetic")
        prov = [_bank_provenance(f) for f in a.fan_npz]
        for p in prov:
            if p["kind"] == "eval" and not a.i_know_this_is_eval:
                raise SystemExit(f"[train_l7] REFUSING: {p['fan_npz']} is an EVAL capture ({p['evidence']}); PREREG sec. 2/8")
            if p["kind"] == "unknown" and not a.allow_unknown_provenance:
                raise SystemExit(f"[train_l7] REFUSING: cannot establish that {p['fan_npz']} is a TRAIN capture (no roll json, "
                                 f"no manifest evidence); pass --allow-unknown-provenance for a machinery smoke")
        fanz, tg = load_bank(a.fan_npz, a.targets_npz)
        src = {"fan_npz": [f.replace("\\", "/") for f in a.fan_npz], "targets_npz": [t.replace("\\", "/") for t in a.targets_npz],
               "provenance": prov, "inference_seeds": fanz["inference_seeds"]}
    eid = np.asarray(tg["eid"]).astype(str)
    tr, va = episode_split(eid, a.val_frac, 20261021)
    assert not (set(eid[tr].tolist()) & set(eid[va].tolist())), "episode-disjointness of the validation slice violated"
    cfg = RS.L7Config(mode=a.mode)
    model = RS.L7Scorer(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    log = {"args": vars(a), "source": src, "params": RS.n_params(model), "n_train": int(len(tr)), "n_val": int(len(va)),
           "n_train_episodes": int(len(set(eid[tr].tolist()))), "n_val_episodes": int(len(set(eid[va].tolist()))),
           "keep_mean_candidates": round(float(np.asarray(fanz["keep"]).sum(1).mean()), 3), "epochs": []}
    if a.epochs == 0:                      # DRY RUN: everything above ran (provenance, load, split, model, optimizer object); no step, no checkpoint
        log["dry_run"] = True
        with open(os.path.join(a.out_dir, "dry_run_log.json"), "w", encoding="utf-8") as fh:
            json.dump(log, fh, indent=1)
        print("ZZL7DRYRUN " + json.dumps({"mode": a.mode, "seed": a.seed, "n_train": log["n_train"], "n_val": log["n_val"],
                                          "n_train_episodes": log["n_train_episodes"], "n_val_episodes": log["n_val_episodes"],
                                          "params": log["params"], "keep_mean_candidates": log["keep_mean_candidates"],
                                          "provenance": [p["kind"] for p in src.get("provenance", [])]}))
        return 0
    best, bad, step = float("inf"), 0, 0
    rng = np.random.default_rng(a.seed)
    t_train = 0.0
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        t0 = time.time()
        losses = []
        for s in range(0, len(perm), a.batch):
            jj = perm[s:s + a.batch]
            inp, t, keep = to_batch(fanz, tg, jj, a.mode)
            out = model(**inp)
            loss, lg = RS.pai_loss(out, t, keep, a.temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(lg)
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
        dt = time.time() - t0
        t_train += dt
        vr = evaluate(model, fanz, tg, va, a.mode, temperature=a.temperature)
        trr = evaluate(model, fanz, tg, tr[:512], a.mode, temperature=a.temperature)
        mean = {k: round(float(np.nanmean([d[k] for d in losses if k in d])), 5) for k in losses[0]}
        log["epochs"].append({"epoch": ep, "steps": step, "train_loss": mean, "val": vr, "train_subset": trr,
                              "epoch_s": round(dt, 2), "s_per_window": round(dt / max(len(perm), 1), 6)})
        print("ZZEP " + json.dumps(log["epochs"][-1]), flush=True)
        score = vr["loss"] if a.early_stop_on == "loss" else vr["pick_cost_m"]
        if score < best - 1e-6:
            best, bad = score, 0
            torch.save({"cfg": vars(cfg), "state_dict": model.state_dict(), "epoch": ep}, os.path.join(a.out_dir, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                break
        if a.max_steps and step >= a.max_steps:
            break
    log["train_wall_s"] = round(t_train, 2)
    with open(os.path.join(a.out_dir, "train_log.json"), "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=1)
    print("ZZL7TRAIN " + json.dumps({"early_stop_on": a.early_stop_on, "best_val": best, "epochs": len(log["epochs"]),
                                     "params": log["params"], "train_wall_s": log["train_wall_s"],
                                     "K7_comf_auc_last_epoch": log["epochs"][-1]["val"]["head_auc"].get("comf")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
