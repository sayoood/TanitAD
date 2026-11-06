#!/usr/bin/env python3
"""WP YAW / YF2: DECODER-SIDE YAW FINE-TUNE of u400 on the cached encoder context (route R1, FINETUNE_PLAN section 4; frozen design in PREREG_YAWFT.md).

The encoder side is frozen and read from the cache (proxy_cache_yf2.py): scene_ctx [64,256] per frame. What trains reads only that context + per-row ego / goal / targets.
ego[4] = YAW_SCALE * angular_rate_z (arm B) or 0 (arm C). Slot 5 stays 0. The yaw input enters through ONE separate parameter `col4` (256) that REPLACES column 4 of
ego_enc.0.weight in the first Linear: W_eff = cat(W[:, :4], col4, W[:, 5:]). So (a) the trained column is a leaf whose gradient is read directly (G5), (b) the working weight's
own column 4 never receives gradient, (c) the training forward IS the deployed forward on the composed checkpoint (same F.linear on the same values, G-CKPT).
Rungs: R1 = col4 only; R2 = + the rest of ego_enc; R3 = + queries, dec, traj_head. Loss = model.wta_loss (P2's live loss, yaw_wrap on). wd 0.

GATES (written to gates.json BEFORE the first optimiser step; a red gate REFUSES the run):
 G4 identity  : at col4 = 0 the decode with yaw != 0 and with yaw = 0 both equal the ORIGINAL u400 decode (torch.equal) on >= --gate-frames frames
 G5 reach     : |grad col4|.sum() > 0 with the table; EXACTLY 0.0 with yaw = 0 (the control)
 G9 table     : every TRAIN/VAL token in the table (a missing token REFUSES), row ego[4] == ego[5] == 0 (two sources for one slot REFUSED)
DELIBERATE-REGRESSION ARMS (tests only, --regress): default_col (col4 left at u400's random column -> G4 RED), table_not_applied (table loaded, yaw fed 0 -> G5 RED).
    python yaw_ft.py --cache <dir> --manifest <manifest_yaw_4k.json> --yaw-table <t.json.gz> --arm B|C --rung R1 --seed 0 --out <dir> [--lr-col 1e-3]
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yaw_common as YC  # noqa: E402
sys.path.insert(0, str(YC.REFE_DIR))
import ckpt_io  # noqa: E402
from model import REFe, REFeConfig, wta_loss  # noqa: E402

RUNG = {"R1": dict(sets=(), steps=400, lr_base=0.0), "R2": dict(sets=("ego_enc.",), steps=400, lr_base=1e-4),
        "R3": dict(sets=("ego_enc.", "queries", "dec.", "traj_head."), steps=200, lr_base=5e-5)}
LR_COL_DEFAULT = 1e-3            # settled in PREREG_YAWFT.md section 2 (numerics: probe_col_lr.json)


class TableRefused(RuntimeError):
    pass


def build_decoder(base: str, device: str = "cpu"):
    """REFe's decoder side (ViT depth 0) holding the base checkpoint's tensors; every DEC tensor MUST come from the base (refused otherwise). Returns model, cfg, base_state."""
    cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone("vitl16"), base)
    cfg.depth = 0
    model = REFe(cfg)
    sd = torch.load(base, map_location="cpu", weights_only=False)["model"]
    want = {k: v for k, v in sd.items() if k.startswith(YC.DEC)}
    own = {k for k in model.state_dict() if k.startswith(YC.DEC)}
    if set(want) != own:
        raise SystemExit(f"base decoder keys != model decoder keys: missing {sorted(own - set(want))[:3]} extra {sorted(set(want) - own)[:3]}")
    model.load_state_dict(want, strict=False)
    if cfg.nav_mode == "goal" and not torch.equal(model.goal_freqs.detach().cpu(), sd["goal_freqs"].float()):
        raise SystemExit("goal_freqs differ from the base checkpoint's")
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device), cfg, {k: v.clone() for k, v in want.items()}


class YawDecoder(torch.nn.Module):
    """REFe.forward after scene_proj, trajectory side, with column 4 of ego_enc.0.weight supplied by the leaf `col4`."""

    def __init__(self, model, cfg, col4_init: torch.Tensor, slot: int = YC.EGO_YAW_SLOT):
        super().__init__()
        if cfg.nav_mode != "goal":
            raise NotImplementedError("command mode (P6 final): rows carry no one-hot; see HANDOFF 'P6 final'")
        self.m, self.cfg, self.slot = model, cfg, slot
        self.col4 = torch.nn.Parameter(col4_init.detach().clone().float())

    def eff_weight(self):
        W = self.m.ego_enc[0].weight
        return torch.cat([W[:, :self.slot], self.col4[:, None].to(W.dtype), W[:, self.slot + 1:]], 1)

    def forward(self, scene_ctx, ego, goal):
        m, cfg = self.m, self.cfg
        g = goal.unsqueeze(-1) * m.goal_freqs
        x = torch.cat([ego, goal, g.sin().flatten(1), g.cos().flatten(1)], -1)
        h = F.linear(x, self.eff_weight(), m.ego_enc[0].bias)
        tok = m.ego_enc[2](m.ego_enc[1](h)).unsqueeze(1)
        q = m.queries.expand(scene_ctx.shape[0], -1, -1) + tok
        for blk in m.dec:
            q = blk(q, scene_ctx)
        return m.traj_head(q).view(scene_ctx.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)


def original_decode(model, cfg, scene_ctx, ego, goal):
    """The deployed forward (the base model's own modules, no col4 indirection) -- the A0 reference for G4."""
    g = goal.unsqueeze(-1) * model.goal_freqs
    q = model.queries.expand(scene_ctx.shape[0], -1, -1) + model.ego_enc(torch.cat([ego, goal, g.sin().flatten(1), g.cos().flatten(1)], -1)).unsqueeze(1)
    for blk in model.dec:
        q = blk(q, scene_ctx)
    return model.traj_head(q).view(scene_ctx.shape[0], cfg.n_proposals, cfg.horizon_steps, cfg.traj_dim)


class Cache:
    """memory-mapped scene shards of proxy_cache(_yf2) (bf16 bits as int16)."""

    def __init__(self, d):
        d = Path(d)
        self.mf = json.load(open(d / "manifest.json", encoding="utf-8"))
        self.frames = [json.loads(l) for l in open(d / "frames.jsonl", encoding="utf-8")]
        dt = np.int16 if self.mf["dtype"] == "bf16" else np.float32
        sizes = {}
        for fr in self.frames:
            sizes[fr["shard"]] = max(sizes.get(fr["shard"], 0), fr["index"] + 1)
        self.mm = {s: np.memmap(d / f"shard_{s:04d}.scene.bin", dtype=dt, mode="r", shape=(n, 64, 256)) for s, n in sizes.items()}
        self.pos = {tuple(fr["key"]): (fr["shard"], fr["index"]) for fr in self.frames}
        self.bf16 = dt is np.int16
        self.rows = [json.loads(l) for l in open(d / "rows.jsonl", encoding="utf-8")]

    def scene(self, key):
        s, i = self.pos[key]
        t = torch.from_numpy(np.array(self.mm[s][i]))
        return (t.view(torch.bfloat16) if self.bf16 else t).float()


def load_rows(cache: Cache, manifest: dict, ranks=None):
    split = {(f["log_name"], f["token"], int(f["step"])): f["yf2_split"] for f in manifest["frames"]}
    out = {"TRAIN": [], "VAL": []}
    for r in cache.rows:
        key = tuple(r["key"][:2]) + (int(r["key"][2]),)
        if key not in split or key not in cache.pos:
            continue
        if ranks is not None and int(r["rank"]) not in ranks:
            continue
        out[split[key]].append({"key": key, "token": key[1], "rank": int(r["rank"]), "ego": r["ego"], "goal": r["goal"][:4], "traj": r["traj"]})
    return out


def check_rows(rows, table):
    miss = sorted({r["token"] for r in rows if r["token"] not in table})
    if miss:
        raise TableRefused(f"{len(miss)} token(s) absent from the yaw table, e.g. {miss[:3]} -- REFUSED rather than feed a yaw of 0.0")
    bad = [r["token"] for r in rows if r["ego"][4] != 0.0 or r["ego"][5] != 0.0]
    if bad:
        raise TableRefused(f"{len(bad)} row(s) already carry a non-zero ego[4]/ego[5] (e.g. {bad[:2]}): two sources for one slot")


def make_batch(rows, ids, cache: Cache, table, source: str, scale: float, regress: str = "none"):
    """yaw source: 'table' -> scale*table[token] (python float64 multiply, then float32: the SAME arithmetic the eval shim uses), 'zeros' -> 0."""
    ego = torch.tensor([rows[i]["ego"] for i in ids], dtype=torch.float32)
    if source == "table" and regress != "table_not_applied":
        ego[:, YC.EGO_YAW_SLOT] = torch.tensor([scale * table[rows[i]["token"]] for i in ids], dtype=torch.float32)
    return (torch.stack([cache.scene(rows[i]["key"]) for i in ids]), ego, torch.tensor([rows[i]["goal"] for i in ids], dtype=torch.float32),
            torch.tensor([rows[i]["traj"] for i in ids], dtype=torch.float32))


def cosine(step, total, lr):
    return lr * (1.0 + math.cos(math.pi * step / total)) / 2.0


def gates(model, cfg, dec: YawDecoder, rows, cache, table, scale, n: int, regress: str) -> dict:
    """G4 identity + G5 reach, computed on the SAME make_batch path the training uses (so a table that is loaded but not applied is seen by G5)."""
    dev = dec.col4.device
    ids = list(range(min(n, len(rows))))
    sc, eg, go, tg = (x.to(dev) for x in make_batch(rows, ids, cache, table, "table", scale, regress))
    eg0 = eg.clone()
    eg0[:, YC.EGO_YAW_SLOT] = 0.0
    out = {"n_frames": len(ids), "col4_init_abs_sum": float(dec.col4.abs().sum())}
    with torch.no_grad():
        a0 = original_decode(model, cfg, sc, eg0, go)                       # A0: the base decode at yaw = 0 (as deployed)
        out["G4_identity_yaw0"] = bool(torch.equal(a0, dec(sc, eg0, go)))
        out["G4_identity_yaw_nonzero"] = bool(torch.equal(a0, dec(sc, eg, go)))
    out["G4_table_has_nonzero_yaw"] = bool((eg[:, YC.EGO_YAW_SLOT] != 0).any())
    for name, e in (("table", eg), ("zero_control", eg0)):
        dec.col4.grad = None
        loss, _ = wta_loss(dec(sc, e, go).float(), tg)
        loss.backward()
        out[f"G5_gradabs_{name}"] = float(dec.col4.grad.abs().sum())
        dec.col4.grad = None
    out["G4_pass"] = bool(out["G4_identity_yaw0"] and out["G4_identity_yaw_nonzero"] and out["col4_init_abs_sum"] == 0.0)
    out["G5_pass"] = bool(out["G5_gradabs_table"] > 0.0 and out["G5_gradabs_zero_control"] == 0.0 and out["G4_table_has_nonzero_yaw"])
    out["pass"] = bool(out["G4_pass"] and out["G5_pass"])
    return out


@torch.no_grad()
def evaluate(dec, rows, cache, table, source, scale, batch=64, regress="none"):
    """T0 (teacher-forced, logged ego) diagnostics on VAL rows: wta loss, position L1, yaw L1, winner heading error at 1 s / 2 s (steps 4 / 9)."""
    acc = {"n": 0, "loss": 0.0, "l_pos": 0.0, "l_yaw": 0.0, "h1": 0.0, "h2": 0.0}
    for i0 in range(0, len(rows), batch):
        ids = list(range(i0, min(i0 + batch, len(rows))))
        sc, eg, go, tg = (x.to(dec.col4.device) for x in make_batch(rows, ids, cache, table, source, scale, regress))
        traj = dec(sc, eg, go).float()
        loss, idx, l_pos, l_yaw = wta_loss(traj, tg, parts=True)
        ar = torch.arange(len(ids), device=dec.col4.device)
        wrap = lambda d: (d + math.pi) % (2 * math.pi) - math.pi
        b = len(ids)
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
    ap.add_argument("--yaw-table", default=str(YC.TABLE_NAVTRAIN))
    ap.add_argument("--yaw-table-sha256", default=None, help="refuse unless the table file has this sha256 (prefix ok)")
    ap.add_argument("--arm", choices=("B", "C"), required=True)
    ap.add_argument("--rung", choices=tuple(RUNG), default="R1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr-col", type=float, default=LR_COL_DEFAULT)
    ap.add_argument("--lr-base", type=float, default=None)
    ap.add_argument("--ranks", default="0,1")
    ap.add_argument("--max-frames", type=int, default=0, help="TEST ONLY: use only the first N cached frames' rows")
    ap.add_argument("--val-every", type=int, default=50)
    ap.add_argument("--gate-frames", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--regress", choices=("none", "default_col", "table_not_applied"), default="none", help="TESTS ONLY: the deliberate-regression arms")
    ap.add_argument("--gates-only", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if (out / "final.pt").exists():
        print(f"REFUSING: {out}/final.pt exists -- an arm is never overwritten")
        return 4
    out.mkdir(parents=True, exist_ok=True)
    rung = RUNG[a.rung]
    steps = a.steps or rung["steps"]
    lr_base = rung["lr_base"] if a.lr_base is None else a.lr_base
    scale = YC.YAW_SCALE
    YC.import_yt_scale()
    torch.manual_seed(a.seed)
    dev = a.device
    table, tmeta, tsha = YC.load_yaw_table(a.yaw_table)
    if a.yaw_table_sha256 and not tsha.startswith(a.yaw_table_sha256):
        raise TableRefused(f"table sha256 {tsha[:12]} != pinned {a.yaw_table_sha256[:12]}")
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    cache = Cache(a.cache)
    ranks = {int(x) for x in a.ranks.split(",")}
    rows = load_rows(cache, manifest, ranks)
    if a.max_frames:
        keep = {tuple(fr["key"]) for fr in cache.frames[:a.max_frames]}
        rows = {k: [r for r in v if r["key"] in keep] for k, v in rows.items()}
    tr, va = rows["TRAIN"], rows["VAL"]
    check_rows(tr + va, table)
    model, cfg, base_state = build_decoder(a.base, dev)
    orig_col = model.ego_enc[0].weight[:, YC.EGO_YAW_SLOT].detach().clone()
    col_init = orig_col if a.regress == "default_col" else torch.zeros_like(orig_col)
    dec = YawDecoder(model, cfg, col_init).to(dev)
    source = "table" if a.arm == "B" else "zeros"
    # ---- trainable sets
    groups = [{"params": [dec.col4], "lr": a.lr_col, "name": "col4"}]
    trained_keys = []
    for n, p in model.named_parameters():
        if n.startswith(rung["sets"]) and rung["sets"]:
            p.requires_grad_(True)
            trained_keys.append(n)
    if trained_keys:
        groups.append({"params": [p for p in model.parameters() if p.requires_grad], "lr": lr_base, "name": "base"})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    base_lrs = [g["lr"] for g in opt.param_groups]
    n_train = sum(p.numel() for g in groups for p in g["params"])
    gt = gates(model, cfg, dec, tr, cache, table, scale, a.gate_frames, a.regress)
    cfgd = {"argv": sys.argv if argv is None else list(argv), "args": vars(a), "rung": rung | {"sets": list(rung["sets"])}, "steps": steps, "lr_base": lr_base, "yaw_scale": scale,
            "yaw_scale_source": YC.yaw_scale_source(), "yaw_source": source, "yaw_table_sha256": tsha, "base_sha256": YC.sha256_file(a.base), "cache_manifest_sha256": YC.sha256_file(Path(a.cache) / "manifest.json"),
            "manifest_sha256": YC.sha256_file(a.manifest), "code_sha256": {**YC.code_hashes(), "yaw_ft.py": YC.sha256_file(__file__)}, "trainable": n_train, "n_train_rows": len(tr), "n_val_rows": len(va),
            "torch": torch.__version__, "device": dev, "started_local": time.strftime("%Y-%m-%dT%H:%M:%S")}
    YC.jdump(cfgd, out / "config.json")
    YC.jdump(gt, out / "gates.json")
    print("gates", json.dumps(gt), flush=True)
    if not gt["pass"]:
        print("ZZYAWFT_GATE_RED")
        return 3
    if a.gates_only:
        print("ZZYAWFT_GATES_OK")
        return 0
    # ---- train
    gen = torch.Generator().manual_seed(a.seed)
    perm, pos = torch.randperm(len(tr), generator=gen).tolist(), 0
    mlog = open(out / "metrics.jsonl", "a", encoding="utf-8")
    t0 = time.time()
    v0 = evaluate(dec, va, cache, table, source, scale, regress=a.regress)
    v0z = evaluate(dec, va, cache, table, "zeros", scale)
    mlog.write(json.dumps({"step": 0, "val": v0, "val_yaw0": v0z, "col4_norm": float(dec.col4.norm())}) + "\n")
    for step in range(steps):
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = cosine(step, steps, b)
        acc = 0.0
        for _ in range(a.accum):
            ids = []
            for _ in range(a.batch):
                if pos >= len(perm):
                    perm, pos = torch.randperm(len(tr), generator=gen).tolist(), 0
                ids.append(perm[pos])
                pos += 1
            sc, eg, go, tg = (x.to(dev) for x in make_batch(tr, ids, cache, table, source, scale, a.regress))
            loss, _ = wta_loss(dec(sc, eg, go).float(), tg)
            (loss / a.accum).backward()
            acc += float(loss) / a.accum
        gn = float(torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0))
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % a.val_every == 0 or step == steps - 1:
            row = {"step": step + 1, "lr_col": opt.param_groups[0]["lr"], "train_loss": acc, "grad_norm": gn, "col4_norm": float(dec.col4.norm()), "col4_absmax": float(dec.col4.abs().max()),
                   "val": evaluate(dec, va, cache, table, source, scale, regress=a.regress), "val_yaw0": evaluate(dec, va, cache, table, "zeros", scale), "elapsed_s": round(time.time() - t0, 1)}
            mlog.write(json.dumps(row) + "\n")
            mlog.flush()
            print(f"step {step + 1:4d} loss {acc:.4f} |col4| {row['col4_norm']:.3f} val {row['val']['loss']:.4f} (yaw0 {row['val_yaw0']['loss']:.4f}) {row['elapsed_s']:.0f}s", flush=True)
    mlog.close()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(YC.DEC)}
    if a.arm == "B":                                                                  # arm B: the trained column IS the deployed column 4; arm C never feeds a yaw, so it keeps u400's own (dead) column
        state["ego_enc.0.weight"][:, YC.EGO_YAW_SLOT] = dec.col4.detach().cpu()
    changed = sorted(k for k in state if not torch.equal(state[k], base_state[k]))
    torch.save({"state": state, "col4": dec.col4.detach().cpu(), "arm": a.arm, "rung": a.rung, "complete": True, "config": cfgd, "changed_keys": changed}, out / "final.pt")
    YC.jdump({"complete": True, "steps": steps, "changed_keys": changed, "col4_norm": float(dec.col4.norm()), "seconds": round(time.time() - t0, 1), "val_final": evaluate(dec, va, cache, table, source, scale, regress=a.regress),
              "val_final_yaw0": evaluate(dec, va, cache, table, "zeros", scale), "val_start": v0, "gates": gt}, out / "summary.json")
    print("ZZYAWFT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
