#!/usr/bin/env python3
"""fit_arm_logits.py: ONE forward pass over the 8,489 FIT sets -> the six scorer logits of SEVERAL scorer arms (+ a mandatory control).  RUNS ON THE POD, GPU.

Feeds PREREG_P6_SELECTORS_ARM.md.  The arms (C0, C0r, S1E) differ from the P6 base ONLY in score_q_mlp / score_dec / score_head, so one forward pass with the
base checkpoint gives the proposals and the visual context, and every arm's scorer is applied to the SAME tensors (same idea and same helper classes as
eval/arm_logits_pass.py, which this file IMPORTS: copy both files into one directory on the pod, or keep the repo layout selectors/arm_fit + eval).

The forward pass reproduces the label job's FIT dump (labels/p6_fan_dump.py, a wrapper around the pod's rl_fan_dump.py), because the control compares against
what that dump stored: same TargetBank built from config_for_checkpoint (nav_mode command), same bf16 autocast, same TF32 flags, same default batch of 4, and
the same BATCH COMPOSITION (heldout sets and fit_extra sets were dumped in two runs, each in sorted (log, token) order, 4 at a time; the pack's is_heldout
flag and key order let this tool re-form those batches).  A different composition can move bf16 outputs, which is why it matters.

Output:  OUT/<NAME>/fit_logits.npz  (log, token, step, rank, logits [N,64,6] f32, pack_sha256, ckpt_sha256, arm), one per arm and P6_control, in the pack's set order;
         OUT/report.json.  Prints ZZFITARM_OK (exit 0) only if every control holds, else ZZFITARM_FAIL <reason> (exit 1).
Controls (checked on EVERY batch, fail fast, and again over the assembled result at the end):
  internal   P6_control through the hook path == the planner's own score output, max |diff| EXACTLY 0.0
  reference  P6_control vs the logits stored in fit_pack.npz (stored rounded to 6 decimals): max |diff| <= --ref-tol (default 1e-3)
  proposals  traj vs pack traj_xy / yaw (both stored rounded to 5 decimals, MEASURED on fit_pack.npz): xy <= --prop-xy-tol (2e-5), yaw <= --prop-yaw-tol (2e-5)
  guard      every state key outside the three scorer modules of every arm checkpoint is bit-equal to the base (arm_logits_pass.verify_arm), before and after
Resumable: every --chunk sets one part file in OUT/_progress (a chunk is whole batches); the same command resumes.  Refuses a different run in the same OUT.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _import_pass():
    for d in (HERE, os.path.normpath(os.path.join(HERE, "..", "..", "eval"))):
        if os.path.exists(os.path.join(d, "arm_logits_pass.py")):
            if d not in sys.path:
                sys.path.insert(0, d)
            import arm_logits_pass as mod
            return mod
    raise SystemExit("ZZFITARM_FAIL arm_logits_pass.py not found next to this file or in ../../eval (copy it beside fit_arm_logits.py)")


A = _import_pass()
import torch  # noqa: E402

CONTROL = A.CONTROL
P6_SHA = "968a469d5b9abdd279aff19b8031c53e9fec0e2683092bff12cfa93be6a4d3d1"


class FitFail(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------- pack
def load_pack(path):
    with np.load(path, allow_pickle=False) as zf:
        z = {k: zf[k] for k in zf.files}
    need = ("log", "token", "step", "rank", "traj_xy", "yaw", "logits", "is_heldout")
    miss = [k for k in need if k not in z]
    if miss:
        raise FitFail(f"pack {path} lacks arrays {miss}")
    p = {k: z[k] for k in need}
    n = len(p["log"])
    if p["logits"].shape != (n, 64, 6) or p["traj_xy"].shape[:2] != (n, 64) or p["yaw"].shape[:2] != (n, 64):
        raise FitFail(f"pack shapes: logits {p['logits'].shape}, traj_xy {p['traj_xy'].shape}, yaw {p['yaw'].shape}")
    keys = list(zip(map(str, p["log"]), map(str, p["token"]), map(int, p["step"]), map(int, p["rank"])))
    if len(set(keys)) != n:
        raise FitFail("duplicate keys in the pack")
    p["keys"] = keys
    return p


def plan_batches(is_heldout, batch_size, limit=None):
    """Pack indices -> batches.  Two groups as in the label job (heldout dump, then fit_extra dump), each in the pack's (sorted) order, `batch_size`
    consecutive sets per batch, the last batch of a group short.  --limit N keeps the first N sets of that order."""
    ho = [i for i in range(len(is_heldout)) if is_heldout[i]]
    fe = [i for i in range(len(is_heldout)) if not is_heldout[i]]
    if limit:
        ho = ho[:limit]
        fe = fe[:max(0, limit - len(ho))]
    out = []
    for g in (ho, fe):
        out += [g[j:j + batch_size] for j in range(0, len(g), batch_size)]
    return out


def chunk_batches(batches, chunk_sets, batch_size):
    per = max(1, chunk_sets // batch_size)
    return [batches[j:j + per] for j in range(0, len(batches), per)]


# ----------------------------------------------------------------------------------------------------------- the forward
def one_batch(fwd, arms, tap, items, amp):
    """One planner forward (hooks record what the scorer is handed), then every arm's scorer on the SAME tensors, in the SAME autocast context.
    -> traj [B,64,20,3] f32, {arm: logits [B,64,6] f32}, internal control diff"""
    tap.clear()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        traj, score = fwd(items)
        x, c = tap.get()
        lg = {n: a(x, c).float().cpu().numpy().astype(np.float32) for n, a in arms.items()}
    own = score.float().cpu().numpy().astype(np.float32)
    if own.shape != lg[CONTROL].shape:
        raise FitFail(f"planner score {own.shape} vs hook path {lg[CONTROL].shape}")
    return traj.float().cpu().numpy(), lg, float(np.abs(lg[CONTROL] - own).max())


def ref_diffs(idx, traj, ctl, pack):
    """per-set max |diff| of (xy, yaw, control logits) against the pack rows idx"""
    xy = np.abs(traj[..., :2].astype(np.float64) - pack["traj_xy"][idx]).reshape(len(idx), -1).max(1)
    yw = np.abs(traj[..., 2].astype(np.float64) - pack["yaw"][idx]).reshape(len(idx), -1).max(1)
    lg = np.abs(ctl.astype(np.float64) - pack["logits"][idx]).reshape(len(idx), -1).max(1)
    return xy, yw, lg


def check_tol(idx, xy, yw, lg, pack, tol):
    for name, v, t in (("proposal xy", xy, tol["xy"]), ("proposal yaw", yw, tol["yaw"]), ("control logits", lg, tol["ref"])):
        j = int(np.argmax(v))
        if not v[j] <= t:                                   # also catches NaN
            k = pack["keys"][idx[j]]
            raise FitFail(f"reference control: {name} max|diff| {v[j]:.3e} > tol {t:g} at set {k} (batch composition, a changed model/code, or the wrong "
                          f"checkpoint/dataset; bf16 outputs move with batch shape)")


# ----------------------------------------------------------------------------------------------------------- resumable store
class Store:
    def __init__(self, out, meta):
        self.out, self.dir, self.meta = out, os.path.join(out, "_progress"), meta
        os.makedirs(self.dir, exist_ok=True)
        mp = os.path.join(self.dir, "meta.json")
        if os.path.exists(mp):
            old = json.load(open(mp, encoding="utf-8"))
            if old != meta:
                raise FitFail(f"{mp} belongs to a different run (args/checkpoints/pack differ); use a new --out")
        else:
            with open(mp + ".tmp", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1, sort_keys=True)
            os.replace(mp + ".tmp", mp)
        for n in list(meta["arms"]) + [CONTROL]:
            fp = os.path.join(out, n, "fit_logits.npz")
            if os.path.exists(fp):
                with np.load(fp) as z:
                    old_id = str(z["run_id"])
                if old_id != meta["run_id"]:
                    raise FitFail(f"{fp} exists from a different run; refusing to overwrite")

    def part(self, c):
        return os.path.join(self.dir, f"part_{c:06d}.npz")

    def done(self, c):
        return os.path.exists(self.part(c))

    def save(self, c, d):
        np.savez(self.part(c) + ".tmp.npz", **d)
        os.replace(self.part(c) + ".tmp.npz", self.part(c))

    def load_all(self, n_chunks):
        parts = []
        for c in range(n_chunks):
            with np.load(self.part(c)) as z:
                parts.append({k: z[k] for k in z.files})
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run_fit(*, fwd, arms, tap, pack, items_for, chunks, store, amp, tol, log=print):
    """Loops over the chunks not yet banked.  -> (assembled dict, timing).  Raises FitFail on the first control miss."""
    todo = [c for c in range(len(chunks)) if not store.done(c)]
    flat = [i for c in todo for b in chunks[c] for i in b]
    log(f"  {len(chunks)} chunks, {len(chunks) - len(todo)} banked, {len(flat):,} sets to run; arms {list(arms)}")
    stream = items_for(flat)
    tm = {"n_sets": 0, "wait_s": 0.0, "forward_s": 0.0, "first_batch_s": None}
    t_start = time.time()
    for c in todo:
        rows = {"idx": [], "internal": [], "xy": [], "yaw": [], "ctl": []}
        lgs = {n: [] for n in arms}
        for b in chunks[c]:
            t0 = time.time()
            items = [next(stream) for _ in b]
            t1 = time.time()
            traj, lg, internal = one_batch(fwd, arms, tap, items, amp)
            t2 = time.time()
            if tm["first_batch_s"] is None:
                tm["first_batch_s"] = t2 - t0
            tm["wait_s"] += t1 - t0
            tm["forward_s"] += t2 - t1
            tm["n_sets"] += len(b)
            if traj.shape[0] != len(b) or traj.shape[1] != 64:
                raise FitFail(f"traj shape {traj.shape} for a batch of {len(b)}")
            if internal != 0.0:
                raise FitFail(f"internal control: P6_control via the hooks differs from the planner's own score by {internal:.3e} (must be exactly 0.0) "
                              f"in the batch starting at set {pack['keys'][b[0]]}")
            xy, yw, lgd = ref_diffs(b, traj, lg[CONTROL], pack)
            check_tol(b, xy, yw, lgd, pack, tol)
            rows["idx"] += list(b)
            rows["internal"] += [internal] * len(b)
            rows["xy"] += list(xy)
            rows["yaw"] += list(yw)
            rows["ctl"] += list(lgd)
            for n in arms:
                lgs[n].append(lg[n])
        d = {k: np.asarray(v) for k, v in rows.items()}
        for n in arms:
            d["L__" + n] = np.concatenate(lgs[n])
        store.save(c, d)
        log(f"    chunk {c + 1}/{len(chunks)} banked; {tm['n_sets']:,} sets, {(time.time() - t_start) / max(tm['n_sets'], 1):.3f} s/set")
    tm["seconds_this_run"] = time.time() - t_start
    return store.load_all(len(chunks)), tm


def write_outputs(out, asm, arms, pack, meta):
    order = np.argsort(asm["idx"], kind="stable")
    idx = asm["idx"][order]
    if len(set(idx.tolist())) != len(idx):
        raise FitFail("a set was run twice")
    for n in arms:
        lg = asm["L__" + n][order].astype(np.float32)
        if not np.isfinite(lg).all():
            raise FitFail(f"arm {n}: non-finite logits")
    for n in arms:
        d = os.path.join(out, n)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "fit_logits.npz")
        np.savez(p + ".tmp.npz", log=pack["log"][idx], token=pack["token"][idx], step=pack["step"][idx], rank=pack["rank"][idx],
                 logits=asm["L__" + n][order].astype(np.float32),
                 pack_sha256=np.array(meta["pack_sha256"]), ckpt_sha256=np.array(meta["base_sha"] if n == CONTROL else meta["arms"][n]),
                 arm=np.array(n), run_id=np.array(meta["run_id"]))
        os.replace(p + ".tmp.npz", p)
    return idx


def final_checks(asm, pack, tol, arm_paths, stat0, base_path):
    """the end-of-run controls, over the ASSEMBLED result (includes chunks banked by an earlier run)"""
    idx = asm["idx"]
    if not (asm["internal"] == 0.0).all():
        raise FitFail(f"end check: internal control max {asm['internal'].max():.3e}")
    lg = np.abs(asm["L__" + CONTROL].astype(np.float64) - pack["logits"][idx]).reshape(len(idx), -1).max(1)
    check_tol(idx, asm["xy"], asm["yaw"], lg, pack, tol)
    # the guard again, on freshly re-read files, and the files themselves unchanged since the start
    for n, p in [("base", base_path)] + list(arm_paths.items()):
        st = os.stat(p)
        if (st.st_size, st.st_mtime_ns) != stat0[n]:
            raise FitFail(f"end check: checkpoint {p} changed on disk during the run")
    base_state, _ = A.load_ckpt(base_path)
    for n, p in arm_paths.items():
        st, meta = A.load_ckpt(p)
        A.verify_arm(n, base_state, st, meta)
    return {"max_internal_diff": float(asm["internal"].max()), "max_control_logit_diff_vs_pack": float(lg.max()),
            "n_control_logit_diff_exactly_zero": int((lg == 0.0).sum()), "max_prop_xy_diff": float(asm["xy"].max()),
            "max_prop_yaw_diff": float(asm["yaw"].max()), "n_sets": int(len(idx))}


# ----------------------------------------------------------------------------------------------------------- real wiring
def stream_items(ds, idxs, workers):
    """ds[i] in order; workers > 0 prefetches with a bounded thread pool (rl_fan_dump.iter_items)"""
    import collections
    if workers <= 0:
        for i in idxs:
            yield ds[i]
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as ex:
        q, it = collections.deque(), iter(idxs)
        for _ in range(workers * 2):
            i = next(it, None)
            if i is None:
                break
            q.append(ex.submit(ds.__getitem__, i))
        while q:
            item = q.popleft().result()
            i = next(it, None)
            if i is not None:
                q.append(ex.submit(ds.__getitem__, i))
            yield item


class RealSeam:
    """model + dataset exactly as the label job built them (labels/p6_fan_dump.py: config_for_checkpoint, command-mode TargetBank)."""

    def __init__(self, a):
        refe = a.refe_dir or os.environ.get("REFE_DIR", "")
        if not refe or not os.path.isdir(refe):
            raise FitFail("set --refe-dir (or REFE_DIR) to the pod's refe/ directory")
        if refe not in sys.path:
            sys.path.insert(0, refe)
        self.a = a

    def load_model(self):
        import ckpt_io
        from model import REFe, REFeConfig
        a = self.a
        if a.device.startswith("cuda") and not torch.cuda.is_available():
            raise FitFail("--device cuda but CUDA is not available")
        cfg = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone(a.backbone), a.base_ckpt)
        if cfg.nav_mode != "command":
            raise FitFail(f"{a.base_ckpt} resolves to nav_mode {cfg.nav_mode!r}; the P6 checkpoint is a command-mode model")
        if a.device.startswith("cuda"):
            torch.backends.cuda.matmul.allow_tf32 = True            # rl_fan_dump.run sets the same two flags
            torch.backends.cudnn.allow_tf32 = True
        model = REFe(cfg)
        meta: dict = {}
        ckpt_io.load_for_inference(model, a.base_ckpt, map_location="cpu", backbone=a.backbone, meta_out=meta)
        self.cfg = cfg
        return model.to(a.device).eval()

    def load_dataset(self, keys):
        import train as T
        a = self.a
        try:
            return T.TargetBank(a.targets, a.images, self.cfg, calib=a.calib, max_missing_frac=1.0)
        except RuntimeError as e:
            raise FitFail(f"TargetBank refused: {e}")

    def forward(self, model):
        dev = self.a.device

        def fwd(items):                                             # = rl_fan_dump.forward_batch, score kept
            img = torch.stack([it[0] for it in items]).to(dev, non_blocking=True)
            ego = torch.stack([it[1] for it in items]).to(dev, non_blocking=True)
            goal = torch.stack([it[2] for it in items]).to(dev, non_blocking=True)
            cal = torch.stack([it[8] for it in items]) if items[0][8].numel() else None
            return model(img, ego, goal, calib=cal)
        return fwd


def parse_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--arm", action="append", help="NAME=path/to/model_sft_B.pt (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack", required=True, help="fit_pack.npz of the label job (the 20-step raw proposals form, NOT fit_pack8)")
    ap.add_argument("--targets", default="/workspace/data/refe_navtrain/train_grow_cmd_3c")
    ap.add_argument("--images", default="/workspace/data/navtrain_pixels")
    ap.add_argument("--calib", default="/workspace/data/refe_navtrain/train_grow/calib_table.json")
    ap.add_argument("--refe-dir", default="")
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--batch-size", type=int, default=4, help="4 = the label dump's batch; another value changes bf16 numerics and the reference control may refuse")
    ap.add_argument("--chunk", type=int, default=512, help="sets per progress file (rounded down to whole batches)")
    ap.add_argument("--limit", type=int, default=0, help="smoke: only the first N sets (output is then PARTIAL and make_arm_pack refuses it)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4, help="image-decode threads (the label job used 4)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--expect-base-sha256", default=P6_SHA)
    ap.add_argument("--ref-tol", type=float, default=1e-3, help="max |control logits - pack logits|")
    ap.add_argument("--prop-xy-tol", type=float, default=2e-5, help="the pack stores xy rounded to 5 decimals (half step 5e-6) as float32 (ulp up to 7.6e-6 at 100 m): 5e-6 + 3.8e-6 worst case")
    ap.add_argument("--prop-yaw-tol", type=float, default=2e-5, help="the pack stores yaw rounded to 5 decimals (half step 5e-6, MEASURED: on the 1e-5 grid; the first pod run refused at 5.007e-6 under the old 1e-6), as float32")
    return ap.parse_args(argv)


def main(argv=None, seam=None) -> int:
    a = parse_args(argv)
    try:
        return _main(a, seam)
    except (FitFail, A.ArmRefused) as e:
        print(f"ZZFITARM_FAIL {e}")
        return 1
    except Exception as e:                                          # a crash must still hand the GPU on with a verdict line
        print(f"ZZFITARM_FAIL exception {type(e).__name__}: {e}")
        return 1


def _main(a, seam):
    arm_paths = A.parse_arms(a.arm)
    if not arm_paths:
        raise FitFail("no --arm given")
    torch.set_num_threads(max(1, a.threads))
    amp = a.device.startswith("cuda")
    tol = {"ref": a.ref_tol, "xy": a.prop_xy_tol, "yaw": a.prop_yaw_tol}
    print(f"  tolerances: reference logits {tol['ref']:g}, proposals xy {tol['xy']:g} yaw {tol['yaw']:g}; batch {a.batch_size}; amp bf16 {amp}", flush=True)
    t_begin = time.time()
    base_sha = A.sha256_file(a.base_ckpt)
    if a.expect_base_sha256 and base_sha != a.expect_base_sha256.lower():
        raise FitFail(f"base checkpoint sha256 {base_sha[:16]}.. != expected {a.expect_base_sha256[:16]}..")
    stat0 = {"base": (os.stat(a.base_ckpt).st_size, os.stat(a.base_ckpt).st_mtime_ns)}
    base_state, _ = A.load_ckpt(a.base_ckpt)
    arm_shas = {}
    for n, p in arm_paths.items():
        st, meta = A.load_ckpt(p)
        A.verify_arm(n, base_state, st, meta)                        # refuses naming the first non-scorer key that moved
        arm_shas[n] = A.sha256_file(p)
        stat0[n] = (os.stat(p).st_size, os.stat(p).st_mtime_ns)
        del st
    del base_state
    pack = load_pack(a.pack)
    pack_sha = A.sha256_file(a.pack)
    print(f"  base {base_sha[:12]}; arms { {n: s[:12] for n, s in arm_shas.items()} }; non-scorer keys bit-equal for every arm; pack {len(pack['keys']):,} sets "
          f"({int(pack['is_heldout'].sum()):,} heldout)", flush=True)
    batches = plan_batches(pack["is_heldout"], a.batch_size, a.limit or None)
    chunks = chunk_batches(batches, a.chunk, a.batch_size)
    n_run = sum(len(b) for b in batches)
    run_id = A.tokens_sha([f"{base_sha}|{sorted(arm_shas.items())}|{pack_sha}|{a.batch_size}|{a.chunk}|{a.limit}"])[:32]
    meta = {"run_id": run_id, "base_sha": base_sha, "arms": dict(sorted(arm_shas.items())), "pack_sha256": pack_sha, "batch_size": a.batch_size,
            "chunk": a.chunk, "limit": a.limit, "n_sets": n_run}
    os.makedirs(a.out, exist_ok=True)
    store = Store(a.out, meta)

    seam = seam or RealSeam(a)
    model = seam.load_model()
    ds = seam.load_dataset(pack["keys"])
    row_of = {(r["log_name"], r["token"], int(r.get("step", 0)), int(r.get("rank", 0))): i for i, r in enumerate(ds.rows)}
    lost = [k for k in pack["keys"] if k not in row_of]
    if lost:
        raise FitFail(f"{len(lost)} pack sets have no dataset row, e.g. {lost[0]} (wrong --targets / --images?)")
    arms = {CONTROL: A.ScorerArm(CONTROL, model, None)}
    for n, p in arm_paths.items():
        st, _ = A.load_ckpt(p)
        arms[n] = A.ScorerArm(n, model, st)
        del st
    tap = A.Tap(model)
    fwd = seam.forward(model)

    def items_for(flat):
        return stream_items(ds, [row_of[pack["keys"][i]] for i in flat], a.workers)

    asm, tm = run_fit(fwd=fwd, arms=arms, tap=tap, pack=pack, items_for=items_for, chunks=chunks, store=store, amp=amp, tol=tol)
    if len(asm["idx"]) != n_run:
        raise FitFail(f"{len(asm['idx'])} sets assembled, expected {n_run}")
    end = final_checks(asm, pack, tol, arm_paths, stat0, a.base_ckpt)
    for m in A.SCORER:                                              # the control modules must still be the planner's own
        sd0, sd1 = getattr(model, m).state_dict(), arms[CONTROL].mods[m].state_dict()
        if any(not torch.equal(sd0[k], sd1[k]) for k in sd0):
            raise FitFail(f"end check: control module {m} no longer equals the planner's")
    write_outputs(a.out, asm, list(arms), pack, meta)
    rep = {"verdict": "ZZFITARM_OK", "partial": bool(a.limit), "n_sets": n_run, "arms": list(arm_paths), "base_sha256": base_sha, "arm_sha256": arm_shas,
           "pack_sha256": pack_sha, "tolerances": tol, "batch_size": a.batch_size, "device": a.device, "controls": end,
           "timing_this_run": {**tm, "per_set_s": tm["seconds_this_run"] / tm["n_sets"] if tm["n_sets"] else None, "total_s": time.time() - t_begin},
           "run_id": run_id}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))
    print("ZZFITARM_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
