"""RENC on box 2 (PREREG_T2_RENC.md). One process, one job at a time. Subcommands:

  data         python -I renc_box2.py data --pkg D:/t2v8f/pkg --gap D:/t2v8f/gap48 --kit D:/t2a/box2_kit --hf D:/tanitad/hf --scenes2 D:/t2v8f/scenes2 --scenes3 C:/t2v8f/scenes3 --out D:/t2v8f/renc/data
               GAP48's job with tick stride 2 and the CYLINDERS kept: per scene <s8>.render.npy / <s8>.real.npy (uint8 [n,3,256,640]) and <s8>.meta.pt (ticks, cam_frames, target = P(E(real)) fp16 [n,40,1024],
               gate_idx = the positions of GAP48's ticks, render_gate = P(E(render)) fp16 at those). Imports GAP48's own preflight / Ctx / one-pass decode (t2v8f_gap48.py, sha256-checked by that preflight).
  train        python -I renc_box2.py train --folds A,B --data D:/t2v8f/renc/data --hf D:/tanitad/hf --out D:/t2v8f/renc/out
               E'_A on fold A only, then E'_B on fold B only: LoRA q,v r32 a32, loss MSE(P(E'(render)),T)+1.0*MSE(P(E'(real)),T), AdamW 1e-4 wd 0, batch 8 pairs, EXACTLY 10 epochs, seed 0.
  gate-encode  python -I renc_box2.py gate-encode --data ... --hf ... --out D:/t2v8f/renc/out
               G-ID0 (B=0 reproduces the frozen tokens bit for bit on 100 frames; a perturbed B must fail it) and the tokens of the GAP48 ticks of ALL scenes through each MERGED E' (the stage-5 path).
Markers (exact names): in --out: RENC_DATA.running|.done|.fail, TRAIN_<fold>.done|.fail, RENC_TRAIN.done|.fail, RENC_GATE_ENCODE.done|.fail.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).replace("\\", "/"))      # python -I does not put the script's folder on sys.path
import renc_lib as L  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

E, LM = L.E, L.LM


def marker(d, name, ok, text):
    for ext in ("done", "fail"):
        p = f"{d}/{name}.{ext}"
        if os.path.exists(p):
            os.replace(p, p + ".prev")
    with open(f"{d}/{name}.{'done' if ok else 'fail'}", "w", newline="\n") as f:
        f.write(text + "\n")


def snapshot_of(a):
    return a.snapshot or E.find_local_snapshot(f"{a.hf}/hub")


# ============================================================================================================================ data
def data_scene(ctx, G, enc, sc, out):
    S1 = G.s1()
    import t2v8f_nocv2 as N
    sid, s8 = sc["scene"], sc["scene"][:8]
    t0 = time.time()
    r, tr, intr, stride, info, _R = S1.build_renderer(ctx, sid, objects=True)
    if info is not None and info.get("verdict") == "REFUSED":
        raise RuntimeError("ACTOR_REFUSED")
    n = r.n_frames() // stride
    ticks = L.data_ticks(n)
    gt = L.gate_ticks(n)
    if not set(gt) <= set(ticks):
        raise RuntimeError(f"GATE_TICKS_NOT_IN_DATA_TICKS n={n}: GAP48's ticks are not all on the stride-2 grid")
    frames = [t * stride for t in ticks]
    refs = G.ref_frames_onepass(ctx.mp4(sid), frames, (r.cam.width, r.cam.height))
    gidx = [ticks.index(t) for t in gt]
    rend = np.lib.format.open_memmap(f"{out}/{s8}.render.npy.tmp", mode="w+", dtype=np.uint8, shape=(len(ticks), 3, 256, 640))
    real = np.lib.format.open_memmap(f"{out}/{s8}.real.npy.tmp", mode="w+", dtype=np.uint8, shape=(len(ticks), 3, 256, 640))
    tgt, rg = [], []
    for j, (t, fr) in enumerate(zip(ticks, frames)):
        img = tr.render(r.gt_rig_to_world(fr) @ r.cam.T_sensor_rig, r.frame_timestamps_us(fr)[1])
        rc, qc = enc.rectify(img, intr), enc.rectify(refs[fr], intr)
        rend[j], real[j] = rc.numpy(), qc.numpy()
        tgt.append(enc.tokens(qc, already_cyl=True).to(torch.float16))
        if t in gt:
            rg.append(enc.tokens(rc, already_cyl=True).to(torch.float16))
    rend.flush(), real.flush()
    del rend, real, r, tr, refs
    S1.free_gpu()
    for k in ("render", "real"):
        os.replace(f"{out}/{s8}.{k}.npy.tmp", f"{out}/{s8}.{k}.npy")
    meta = {"scene": sid, "ticks": torch.tensor(ticks), "cam_frames": torch.tensor(frames), "gate_idx": torch.tensor(gidx), "target": torch.stack(tgt), "render_gate": torch.stack(rg), "n_ticks_total": int(n),
            "fold": L.fold_of(sid)}
    torch.save(meta, f"{out}/{s8}.meta.pt.tmp")
    os.replace(f"{out}/{s8}.meta.pt.tmp", f"{out}/{s8}.meta.pt")
    return {"scene": sid, "n_pairs": len(ticks), "n_gate": len(gt), "fold": meta["fold"], "seconds": round(time.time() - t0, 1),
            "sha256": {k: L.sha256_file(f"{out}/{s8}.{k}") for k in ("render.npy", "real.npy", "meta.pt")}}


def cmd_data(a):
    import t2v8f_gap48 as G
    manifest = json.load(open(a.manifest, encoding="utf-8"))
    out = a.out.replace("\\", "/")
    os.makedirs(out, exist_ok=True)
    a.pull, a.min_free_gb = out, a.min_free_gb
    ctx = G.Ctx(a, manifest)
    sys.path.insert(0, f"{ctx.pkg}/code")
    G.set_env(ctx)
    res_path = f"{out}/RENC_DATA_RESULT.json"
    prev = json.load(open(res_path, encoding="utf-8"))["scenes"] if os.path.isfile(res_path) else {}
    res = {"stage": "renc-data", "started": time.strftime("%F %T"), "stub": bool(a.stub), "scenes": {}}
    open(f"{out}/RENC_DATA.running", "w").write(f"pid={os.getpid()}\n")
    pf = G.preflight(ctx)
    res["preflight"] = pf
    if pf["verdict"] != "PASS":
        marker(out, "RENC_DATA", False, f"PREFLIGHT_REFUSED {json.dumps(pf['failures'])[:600]}")
        os.remove(f"{out}/RENC_DATA.running")
        return 1
    S1 = G.s1()
    enc, _E = S1.make_encoder(ctx)
    for sc in manifest["scenes"][:a.limit_scenes or None]:
        sid, s8 = sc["scene"], sc["scene"][:8]
        free = shutil.disk_usage(out).free / 2**30
        if free < a.stop_free_gb:                                                     # a scene is ~0.1 GiB: stop cleanly rather than fill D:
            res["scenes"].setdefault(sid, {"scene": sid, "status": "FAIL", "reason": f"DISK_LOW_STOP {free:.2f} GiB free < {a.stop_free_gb}"})
            L.jdump(res, res_path)
            break
        old = prev.get(sid)
        if old and old.get("status") == "DONE" and all(os.path.isfile(f"{out}/{s8}.{k}") and L.sha256_file(f"{out}/{s8}.{k}") == h for k, h in old["sha256"].items()):
            res["scenes"][sid] = old
            continue
        try:
            row = data_scene(ctx, G, enc, sc, out)
            row["status"] = "DONE"
        except Exception as e:                                                        # noqa: BLE001
            row = {"scene": sid, "status": "FAIL", "reason": repr(e)[:400], "trace": traceback.format_exc()[-1200:]}
        res["scenes"][sid] = row
        L.jdump(res, res_path)
        try:
            S1.free_gpu()
        except Exception:                                                             # noqa: BLE001
            pass
    nd = sum(1 for v in res["scenes"].values() if v["status"] == "DONE")
    res["n_done"], res["finished"] = nd, time.strftime("%F %T")
    L.jdump(res, res_path)
    ok = nd == len(manifest["scenes"][:a.limit_scenes or None])
    os.remove(f"{out}/RENC_DATA.running")
    marker(out, "RENC_DATA", ok, f"n_done={nd}")
    return 0 if ok else 1


# ============================================================================================================================ train
def train_fold(a, fold, snap, device):
    out = f"{a.out}/E_{fold}"
    os.makedirs(out, exist_ok=True)
    all_scenes = L.list_scenes(a.data)
    train_scenes = [s for s in all_scenes if L.fold_of(s) == fold]
    held = [s for s in all_scenes if L.fold_of(s) != fold]
    assert train_scenes and not set(train_scenes) & set(held)
    ds = L.PairData(a.data, train_scenes)                                              # the trainer is only ever GIVEN its own fold
    enc = L.make_encoder(snap, device, dtype=a.train_dtype)
    mods = L.inject_lora(enc.model)
    params = L.lora_params(mods)
    n_tr = sum(p.numel() for p in params)
    blocks = len({i for i, _ in mods})
    enc.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) if a.ckpt else None
    enc.model.train()                                                                  # needed for checkpointing to act; dropout and drop-path rates are 0 (asserted by a test)
    opt = torch.optim.AdamW(params, lr=L.LR, weight_decay=L.WD)
    gen = torch.Generator().manual_seed(L.SEED)
    ck_path, start_epoch, log = f"{out}/ckpt_epoch.pt", 0, []
    if os.path.isfile(ck_path):                                                        # resume by epoch
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        L.set_lora_state(mods, ck["lora"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        start_epoch, log = ck["epoch"], ck["log"]
    N, B = len(ds), L.BATCH
    steps_per_epoch = -(-N // B)
    t_start, step_s = time.time(), []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for ep in range(start_epoch, L.EPOCHS):
        perm = torch.randperm(N, generator=gen)
        tot = torch.zeros(3, dtype=torch.float64)
        for s in range(steps_per_epoch):
            if a.max_steps and len(step_s) >= a.max_steps:
                break
            t0 = time.time()
            idx = perm[s * B:(s + 1) * B].tolist()
            rend, real, tgt = ds.batch(idx)
            denom = len(idx) * L.TOKEN_SHAPE[0] * L.TOKEN_SHAPE[1]
            opt.zero_grad(set_to_none=True)
            lsum = torch.zeros(3, dtype=torch.float64)
            for c in range(0, len(idx), L.MICRO_PAIRS):
                sl = slice(c, c + L.MICRO_PAIRS)
                both = L.pooled_train(enc, torch.cat([rend[sl], real[sl]]))
                k = both.shape[0] // 2
                loss, lr_, lk_ = L.renc_loss(both[:k], both[k:], tgt[sl], denom=denom)
                loss.backward()
                lsum += torch.tensor([float(loss.detach()), float(lr_), float(lk_)], dtype=torch.float64)
            opt.step()
            tot += lsum * len(idx)
            step_s.append(time.time() - t0)
            if (s + 1) % 50 == 0:
                print(f"fold {fold} epoch {ep + 1} step {s + 1}/{steps_per_epoch} loss {float(lsum[0]):.5f} s/step {np.mean(step_s[-50:]):.2f}", flush=True)
        log.append({"epoch": ep + 1, "steps": steps_per_epoch, "loss": float(tot[0] / N), "loss_render": float(tot[1] / N), "loss_keep": float(tot[2] / N)})
        torch.save({"lora": L.lora_state(mods, blocks=blocks), "opt": opt.state_dict(), "gen": gen.get_state(), "epoch": ep + 1, "log": log}, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)
    done_epochs = len(log)
    lora = L.lora_state(mods, blocks=blocks)
    b_abs = float(max(t.abs().max() for k, t in lora["tensors"].items() if k.endswith("lora_B")))
    blob = {"lora": lora, "step": done_epochs * steps_per_epoch, "seed": L.SEED, "arm": f"RENC_{fold}", "fold": fold, "epochs": done_epochs, "train_scenes": train_scenes, "held_out_scenes": held,
            "card_prefix_sha256": L.CARD_PREFIX_SHA256, "argv_sha": L.hashlib.sha256(" ".join(sys.argv).encode()).hexdigest()}
    bp = f"{out}/E_{fold}_lora_blob.pt"
    torch.save(blob, bp)
    del enc, mods
    base = L.make_encoder(snap, "cpu")                                                  # the merged q/v weights: the stage-5 fold on CPU, saved by canonical name
    parsed = LM.parse_lora(lora, r=L.R, alpha=L.ALPHA, targets=L.TARGETS, blocks=blocks)
    LM.fold_lora(base.model, parsed, r=L.R, alpha=L.ALPHA)
    dig = LM.model_digest(base.model, blocks)
    tp = LM.target_params(base.model, blocks)
    mp = f"{out}/E_{fold}_merged_qv.pt"
    torch.save({LM.canon(i, t): p.data.clone() for (i, t), p in tp.items()}, mp)
    row = {"fold": fold, "epochs": done_epochs, "epochs_required": L.EPOCHS, "steps_per_epoch": steps_per_epoch, "n_pairs": N, "n_train_scenes": len(train_scenes), "n_trainable": n_tr, "blocks": blocks,
           "loss_per_epoch": log, "seconds": round(time.time() - t_start, 1), "s_per_step_mean": float(np.mean(step_s)) if step_s else None, "b_abs_max": b_abs,
           "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device == "cuda" else None, "device": device, "grad_checkpointing": bool(a.ckpt),
           "blob": {"file": bp, "sha256": L.sha256_file(bp)}, "merged_qv": {"file": mp, "sha256": L.sha256_file(mp), "digest": dig["merged_qv_sha256"]}, "lora_tensors_sha256": LM.lora_tensors_sha256(parsed),
           "train_scenes": train_scenes, "touched_scenes": sorted(ds.touched), "decisions": "P = pool4x10 then fp8 (straight-through); batch 8 pairs; micro-batch 4 pairs; constant lr; partial last batch kept"}
    L.jdump(row, f"{a.out}/TRAIN_RESULT_{fold}.json")
    return row


def cmd_train(a):
    os.makedirs(a.out, exist_ok=True)
    snap, device = snapshot_of(a), ("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ok = True
    open(f"{a.out}/RENC_TRAIN.running", "w").write(f"pid={os.getpid()}\n")
    for fold in a.folds.split(","):
        try:
            row = train_fold(a, fold, snap, device)
            good = row["epochs"] == L.EPOCHS or bool(a.max_steps)
            marker(a.out, f"TRAIN_{fold}", good, f"epochs={row['epochs']} final_loss={row['loss_per_epoch'][-1]['loss']:.6f}")
            ok = ok and good
        except Exception as e:                                                         # noqa: BLE001
            marker(a.out, f"TRAIN_{fold}", False, f"{repr(e)[:500]} {traceback.format_exc()[-600:]}")
            ok = False
            break
    os.remove(f"{a.out}/RENC_TRAIN.running")
    marker(a.out, "RENC_TRAIN", ok, "folds=" + a.folds)
    return 0 if ok else 1


# ============================================================================================================================ gate encode
def _frames_100(data, scenes):
    """The 100 frames of G-ID0: real cylinders at positions 0 and n//2 of every scene (sorted), then render position 0 of the first scenes until 100."""
    fr = []
    for sid in scenes:
        rl = np.load(f"{data}/{sid[:8]}.real.npy", mmap_mode="r")
        fr += [np.array(rl[0]), np.array(rl[len(rl) // 2])]
    for sid in scenes:
        if len(fr) >= 100:
            break
        fr.append(np.array(np.load(f"{data}/{sid[:8]}.render.npy", mmap_mode="r")[0]))
    return [torch.from_numpy(x) for x in fr[:100]]


def g_id0(a, snap, device, scenes):
    frames = _frames_100(a.data, scenes)
    n = len(frames)
    enc = L.make_encoder(snap, device)
    frozen = [enc.tokens(f, already_cyl=True) for f in frames]
    mods = L.inject_lora(enc.model)                                                    # B = 0: the adapter is in the graph and adds an exact 0
    z = [enc.tokens(f, already_cyl=True) for f in frames]
    gen = torch.Generator().manual_seed(1)
    for w in mods.values():                                                            # regression arm: a perturbed B must fail the identity
        w.lora_B.data.copy_(torch.randn(w.lora_B.shape, generator=gen) * 1e-2)
    p = [enc.tokens(f, already_cyl=True) for f in frames]
    eq = lambda xs: int(sum(torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(xs, frozen)))   # noqa: E731
    row = {"n_frames": n, "adapter_B0_bit_equal": eq(z), "perturbed_B_bit_equal": eq(p)}
    del enc, mods
    tmp = f"{a.out}/_idtmp_blob.pt"                                                    # the merged path with B = 0 (fold of a zero delta)
    base = L.make_encoder(snap, "cpu")
    mods = L.inject_lora(base.model)
    torch.save({"lora": L.lora_state(mods, blocks=len({i for i, _ in mods}))}, tmp)
    menc, _ = L.merged_encoder(tmp, snap, device)
    row["merged_B0_bit_equal"] = eq([menc.tokens(f, already_cyl=True) for f in frames])
    os.remove(tmp)
    row["pass"] = bool(row["adapter_B0_bit_equal"] == n and row["merged_B0_bit_equal"] == n)
    row["regression_arm_flagged"] = bool(row["perturbed_B_bit_equal"] < n)
    return row


def cmd_gate_encode(a):
    os.makedirs(a.out, exist_ok=True)
    snap, device = snapshot_of(a), ("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    scenes = L.list_scenes(a.data)
    open(f"{a.out}/RENC_GATE_ENCODE.running", "w").write(f"pid={os.getpid()}\n")
    try:
        gid = g_id0(a, snap, device, scenes)
        L.jdump(gid, f"{a.out}/G_ID0.json")
        for fold in L.FOLDS:
            enc, dig = L.merged_encoder(f"{a.out}/E_{fold}/E_{fold}_lora_blob.pt", snap, device)
            od = f"{a.out}/gate_tokens/{fold}"
            os.makedirs(od, exist_ok=True)
            for sid in scenes:
                m = torch.load(f"{a.data}/{sid[:8]}.meta.pt", map_location="cpu", weights_only=True)
                gi = m["gate_idx"].tolist()
                rl, rr = np.load(f"{a.data}/{sid[:8]}.render.npy", mmap_mode="r"), np.load(f"{a.data}/{sid[:8]}.real.npy", mmap_mode="r")
                rt = torch.stack([enc.tokens(torch.from_numpy(np.array(rl[j])), already_cyl=True).to(torch.float16) for j in gi])
                qt = torch.stack([enc.tokens(torch.from_numpy(np.array(rr[j])), already_cyl=True).to(torch.float16) for j in gi])
                torch.save({"scene": sid, "encoder": fold, "ticks": m["ticks"][gi], "render": rt, "real": qt, "merged_qv_sha256": dig["merged_qv_sha256"]}, f"{od}/{sid[:8]}.pt")
            del enc
        os.remove(f"{a.out}/RENC_GATE_ENCODE.running")
        marker(a.out, "RENC_GATE_ENCODE", True, f"G_ID0 pass={gid['pass']}")
        return 0
    except Exception as e:                                                             # noqa: BLE001
        if os.path.exists(f"{a.out}/RENC_GATE_ENCODE.running"):
            os.remove(f"{a.out}/RENC_GATE_ENCODE.running")
        marker(a.out, "RENC_GATE_ENCODE", False, f"{repr(e)[:500]} {traceback.format_exc()[-600:]}")
        return 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("data", "train", "gate-encode"))
    ap.add_argument("--data", default="D:/t2v8f/renc/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf", default="D:/tanitad/hf")
    ap.add_argument("--snapshot", default=None, help="tests: a tiny snapshot dir")
    ap.add_argument("--folds", default="A,B")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--train-dtype", default="bf16", choices=("bf16", "fp32"), help="tests only: the dtype of the encoder being trained (box 2 trains bf16)")
    ap.add_argument("--ckpt", type=int, default=1, help="gradient checkpointing (1 on box 2)")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke only: stop after N steps; the fold is then NOT a result")
    ap.add_argument("--pkg"), ap.add_argument("--gap"), ap.add_argument("--kit"), ap.add_argument("--scenes2"), ap.add_argument("--scenes3")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--stub", action="store_true"), ap.add_argument("--hash-volumes", dest="hash_volumes", action="store_true")
    ap.add_argument("--limit-scenes", type=int, default=0), ap.add_argument("--min-free-gb", type=float, default=6.0), ap.add_argument("--stop-free-gb", type=float, default=1.5)
    a = ap.parse_args(argv)
    a.out, a.data = a.out.replace("\\", "/"), a.data.replace("\\", "/")
    if a.cmd == "data":
        a.manifest = a.manifest or f"{a.gap}/MANIFEST_GAP48.json"
        return cmd_data(a)
    return {"train": cmd_train, "gate-encode": cmd_gate_encode}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
