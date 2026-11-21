#!/usr/bin/env python3
"""WP YAW / YAX: AXOFF vs AXON decoder-side fine-tune of u400 on the cached encoder context (frozen design: PREREG_AXOFF.md).
Copy of the YF2 trainer's skeleton (yaw_ft_ref.py = unchanged copy of YF2/yaw_ft.py supplies build_decoder / Cache / load_rows / original_decode / cosine) with an --ax-policy zero|real switch.
policy zero (AXOFF): ego[:,2] = 0.0 in EVERY row; column 2 of ego_enc.0.weight set to exactly 0.0 and its gradient masked to exactly 0.0 (so it stays 0.0 and ANY fed ax gives the same plan).
policy real  (AXON) : the bank row's ax untouched; column 2 trains. Same rows, optimiser, batch order per seed in both.
Trainable (both): ego_enc.* (AXOFF minus column 2), queries, dec.*, traj_head.*  (YC.DEC). The training forward IS the deployed forward (original_decode on the model's own modules).
GATES (gates.json BEFORE the first step; red -> rc 3 ZZAXFT_GATE_RED): GA_FEED, GA_COL, GB_STEP0 (real), G_REACH (+ zero and positive controls), G_DECL.
DELIBERATE-REGRESSION ARMS (--regress): ax_fed_in_zero (AXOFF fed real ax), ax_not_fed (AXON fed 0), keep_ax_col (AXOFF keeps u400's column 2).
    python ax_ft.py --cache <dir> --manifest <manifest_yaw_4k.json> --ax-policy zero|real --steps 200 --lr 5e-5 --seed 0 --out <dir>
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_ft_ref as FT  # noqa: E402  (unchanged copy of YF2/yaw_ft.py)
import yaw_common as YC  # noqa: E402
from model import wta_loss  # noqa: E402  (REFE_DIR is on sys.path via yaw_common/yaw_ft_ref)

AX = 2
GROUPS = {"ego_enc": "ego_enc.", "queries": "queries", "dec": "dec.", "traj_head": "traj_head."}


def make_batch(rows, ids, cache, policy, regress="none"):
    ego = torch.tensor([rows[i]["ego"] for i in ids], dtype=torch.float32)
    if (policy == "zero" and regress != "ax_fed_in_zero") or (policy == "real" and regress == "ax_not_fed"):
        ego[:, AX] = 0.0
    return (torch.stack([cache.scene(rows[i]["key"]) for i in ids]), ego, torch.tensor([rows[i]["goal"] for i in ids], dtype=torch.float32),
            torch.tensor([rows[i]["traj"] for i in ids], dtype=torch.float32))


def real_ego(rows, ids):
    return torch.tensor([rows[i]["ego"] for i in ids], dtype=torch.float32)


def grads(m, cfg, sc, ego, go, tg):
    """abs-sum of the gradient per declared group and for column 2 of ego_enc.0.weight; leaves grads cleared."""
    for p in m.parameters():
        p.grad = None
    loss, _ = wta_loss(FT.original_decode(m, cfg, sc, ego, go).float(), tg)
    loss.backward()
    g = {}
    for k, pre in GROUPS.items():
        s = 0.0
        for n, p in m.named_parameters():
            if n.startswith(pre) and p.grad is not None:
                gg = p.grad
                if n == "ego_enc.0.weight":
                    gg = torch.cat([gg[:, :AX], gg[:, AX + 1:]], 1)         # column 2 reported separately
                s += float(gg.abs().sum())
        g[k] = s
    w = m.ego_enc[0].weight.grad
    col2 = float(w[:, AX].abs().sum()) if w is not None else 0.0
    for p in m.parameters():
        p.grad = None
    return g, col2


def gates(model, cfg, base_model, rows, cache, policy, regress, n) -> dict:
    ids = list(range(min(n, len(rows))))
    sc, eg, go, tg = make_batch(rows, ids, cache, policy, regress)
    er = real_ego(rows, ids)
    out = {"n_frames": len(ids), "policy": policy, "regress": regress}
    ax, axr = eg[:, AX], er[:, AX]
    out["ax_real_rms"], out["ax_real_std"] = float(axr.pow(2).mean().sqrt()), float(axr.std())
    W = model.ego_enc[0].weight
    out["col2_abs_sum"] = float(W[:, AX].abs().sum())
    out["GA_FEED"] = bool((ax == 0).all()) if policy == "zero" else bool(torch.equal(ax, axr) and out["ax_real_rms"] > 0.3 and out["ax_real_std"] > 0)
    out["GA_COL"] = bool(out["col2_abs_sum"] == 0.0) if policy == "zero" else True
    # AMENDMENT_1: compare WEIGHTS (alignment-proof) and decode with the SAME model object; a decode across two separately allocated model copies is not bit-stable under MKL (MEASURED: False at OMP=10, True at OMP=8)
    sd_m, sd_b = model.state_dict(), base_model.state_dict()
    out["weights_equal_u400_at_step0"] = bool(all(torch.equal(sd_m[k], sd_b[k]) for k in sd_m if k.startswith(YC.DEC)))
    with torch.no_grad():
        out["GB_STEP0_train_forward_equals_u400_real_ax"] = bool(out["weights_equal_u400_at_step0"] and torch.equal(eg, er) and torch.equal(FT.original_decode(model, cfg, sc, eg, go), FT.original_decode(model, cfg, sc, er, go)))
    out["GB_STEP0"] = out["GB_STEP0_train_forward_equals_u400_real_ax"] if policy == "real" else True      # AXOFF at step 0 is NOT u400 by design (ax = 0): informational only
    g_fed, c2_fed = grads(model, cfg, sc, eg, go, tg)
    eg0 = er.clone()
    eg0[:, AX] = 0.0
    _, c2_zero = grads(model, cfg, sc, eg0, go, tg)                                             # ZERO CONTROL: ax = 0 -> column-2 gradient exactly 0.0
    ref = copy.deepcopy(base_model)
    for p in ref.parameters():
        p.requires_grad_(True)
    _, c2_pos = grads(ref, cfg, sc, er, go, tg)                                                 # POSITIVE CONTROL: u400's own column, real ax -> > 0
    del ref
    out.update({"G_REACH_groups": g_fed, "G_REACH_col2_policy_feed": c2_fed, "G_REACH_zero_control_col2": c2_zero, "G_REACH_positive_control_col2": c2_pos})
    col_ok = (c2_fed == 0.0) if policy == "zero" else (c2_fed > 0.0)
    out["G_REACH"] = bool(all(v > 0.0 for v in g_fed.values()) and c2_zero == 0.0 and c2_pos > 0.0 and col_ok)
    out["pass"] = bool(out["GA_FEED"] and out["GA_COL"] and out["GB_STEP0"] and out["G_REACH"])
    return out


@torch.no_grad()
def evaluate(model, cfg, rows, cache, policy, batch=64, regress="none"):
    acc = {"n": 0, "loss": 0.0, "l_pos": 0.0, "l_yaw": 0.0, "h1": 0.0, "h2": 0.0}
    wrap = lambda d: (d + math.pi) % (2 * math.pi) - math.pi
    for i0 in range(0, len(rows), batch):
        ids = list(range(i0, min(i0 + batch, len(rows))))
        sc, eg, go, tg = make_batch(rows, ids, cache, policy, regress)
        traj = FT.original_decode(model, cfg, sc, eg, go).float()
        loss, idx, l_pos, l_yaw = wta_loss(traj, tg, parts=True)
        ar, b = torch.arange(len(ids)), len(ids)
        acc["n"] += b
        acc["loss"] += float(loss) * b
        acc["l_pos"] += float(l_pos) * b
        acc["l_yaw"] += float(l_yaw) * b
        acc["h1"] += float(wrap(traj[ar, idx, 4, 2] - tg[:, 4, 2]).abs().sum())
        acc["h2"] += float(wrap(traj[ar, idx, 9, 2] - tg[:, 9, 2]).abs().sum())
    n = max(acc["n"], 1)
    return {"n": acc["n"], "loss": acc["loss"] / n, "pos_L1": acc["l_pos"] / n, "yaw_L1": acc["l_yaw"] / n, "heading_err_1s": acc["h1"] / n, "heading_err_2s": acc["h2"] / n}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--base", default=str(YC.U400))
    ap.add_argument("--ax-policy", choices=("zero", "real"), required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--max-frames", type=int, default=0, help="TEST ONLY")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--gate-frames", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--regress", choices=("none", "ax_fed_in_zero", "ax_not_fed", "keep_ax_col"), default="none", help="TESTS ONLY: deliberate-regression arms")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "final.pt").exists():
        print(f"REFUSING: {out}/final.pt exists -- an arm is never overwritten")
        return 4
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    dev, policy = a.device, a.ax_policy
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    cache = FT.Cache(a.cache)
    rows = FT.load_rows(cache, manifest, {int(x) for x in a.ranks.split(",")})
    if a.max_frames:
        keep = {tuple(fr["key"]) for fr in cache.frames[:a.max_frames]}
        rows = {k: [r for r in v if r["key"] in keep] for k, v in rows.items()}
    tr, va = rows["TRAIN"], rows["VAL"]
    model, cfg, base_state = FT.build_decoder(a.base, dev)
    base_model, _, _ = FT.build_decoder(a.base, dev)
    if policy == "zero" and a.regress != "keep_ax_col":
        model.ego_enc[0].weight.data[:, AX] = 0.0
    train_names = []
    for n, p in model.named_parameters():
        if n.startswith(YC.DEC):
            p.requires_grad_(True)
            train_names.append(n)
    if policy == "zero" and a.regress != "keep_ax_col":
        mask = torch.ones_like(model.ego_enc[0].weight)
        mask[:, AX] = 0.0
        model.ego_enc[0].weight.register_hook(lambda g, m=mask: g * m)
    decl_ok = bool(all(n.startswith(YC.DEC) for n in train_names) and all((not p.requires_grad) or n.startswith(YC.DEC) for n, p in model.named_parameters()) and len(train_names) > 0)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params) - (model.ego_enc[0].weight.shape[0] if policy == "zero" else 0)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    gt = gates(model, cfg, base_model, tr, cache, policy, a.regress, a.gate_frames)
    gt["G_DECL"], gt["trainable_names"] = decl_ok, train_names
    gt["pass"] = bool(gt["pass"] and decl_ok)
    cfgd = {"argv": sys.argv if argv is None else list(argv), "args": vars(a), "ax_policy": policy, "base_sha256": YC.sha256_file(a.base), "cache_manifest_sha256": YC.sha256_file(Path(a.cache) / "manifest.json"),
            "manifest_sha256": YC.sha256_file(a.manifest), "code_sha256": {**YC.code_hashes(), "ax_ft.py": YC.sha256_file(__file__)}, "trainable": n_train, "n_train_rows": len(tr), "n_val_rows": len(va),
            "torch": torch.__version__, "device": dev, "started_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    YC.jdump(cfgd, out / "config.json")
    YC.jdump(gt, out / "gates.json")
    print("gates", json.dumps({k: v for k, v in gt.items() if k != "trainable_names"}), flush=True)
    if not gt["pass"]:
        print("ZZAXFT_GATE_RED")
        return 3
    if a.gates_only:
        print("ZZAXFT_GATES_OK")
        return 0
    gen = torch.Generator().manual_seed(a.seed)
    perm, pos = torch.randperm(len(tr), generator=gen).tolist(), 0
    mlog = open(out / "metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    v0 = FT_eval = evaluate(model, cfg, va, cache, policy, regress=a.regress)
    mlog.write(json.dumps({"step": 0, "val": v0, "val_ax0": evaluate(model, cfg, va, cache, "zero", regress="none")}) + "\n")
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = FT.cosine(step, a.steps, a.lr)
        acc = 0.0
        for _ in range(a.accum):
            ids = []
            for _ in range(a.batch):
                if pos >= len(perm):
                    perm, pos = torch.randperm(len(tr), generator=gen).tolist(), 0
                ids.append(perm[pos])
                pos += 1
            sc, eg, go, tg = make_batch(tr, ids, cache, policy, a.regress)
            loss, _ = wta_loss(FT.original_decode(model, cfg, sc, eg, go).float(), tg)
            (loss / a.accum).backward()
            acc += float(loss) / a.accum
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.val_every == 0 or step == a.steps - 1:
            row = {"step": step + 1, "lr": opt.param_groups[0]["lr"], "train_loss": acc, "grad_norm": gn, "col2_abs_sum": float(model.ego_enc[0].weight[:, AX].abs().sum()),
                   "val": evaluate(model, cfg, va, cache, policy, regress=a.regress), "elapsed_s": round(time.time() - t0, 1)}
            mlog.write(json.dumps(row) + "\n")
            mlog.flush()
            print(f"step {step + 1:4d} loss {acc:.4f} val {row['val']['loss']:.4f} {row['elapsed_s']:.0f}s", flush=True)
    mlog.close()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(YC.DEC)}
    if policy == "zero" and a.regress != "keep_ax_col":
        assert bool((state["ego_enc.0.weight"][:, AX] == 0).all()), "AXOFF column 2 is not exactly 0.0 after training"
    changed = sorted(k for k in state if not torch.equal(state[k], base_state[k]))
    torch.save({"state": state, "arm": "AXOFF" if policy == "zero" else "AXON", "ax_policy": policy, "complete": True, "config": cfgd, "changed_keys": changed, "regress": a.regress}, out / "final.pt")
    YC.jdump({"complete": True, "steps": a.steps, "lr": a.lr, "changed_keys": changed, "seconds": round(time.time() - t0, 1), "val_final": evaluate(model, cfg, va, cache, policy, regress=a.regress),
              "val_final_ax0": evaluate(model, cfg, va, cache, "zero"), "val_start": v0, "gates": {k: v for k, v in gt.items() if k != "trainable_names"}}, out / "summary.json")
    print("ZZAXFT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
