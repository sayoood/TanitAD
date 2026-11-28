#!/usr/bin/env python3
"""b3_check.py -- control b3 of card PREREG_V8F_LEVER3_V2 (sha256 707c5225..) section 3 step 4: the REACH of the repack.
Question: does a trainer / bank generator that reads the REPACKED cache (pool8x20 dropped) see exactly what it sees on the ORIGINAL-format cache?
Stages (each a subprocess of this file, at the code tree, CPU, CUDA hidden):
  T  trainer path: the trainer's OWN build_data -> V8FMiniWindows -> ChunkedSampler(seed) -> first batch -> D.collate; plus every clip through ClipStore.get / get_kin. sha256 of every array.
  B  bank path: `v8f_dagger_bank.py gen` (the pod's s2 flags) over the first N clips of the TRAIN list; the shard's content_sha256 (G.content_sha256) and the record arrays.
Verdicts:  orig == repack for T and B  AND  a clip of the repack slice with pool4x10 deleted makes T fail (error or different hash) and a second one makes B fail.
Also: the 10 s001 repacked clips equal the MM's cache_union files (md5), when those exist.
usage: b3_check.py run [--n-bank-clips 2] | b3_check.py stageT ... | b3_check.py stageB ...   (run writes <kit>/B3_CHECK.json)"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time

PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = "C:/Users/Admin/lever3/tree_cdb"
KIT = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
B3 = "E:/TanitAD-scratch/lever3/b3"
V9 = "C:/Users/Admin/lever3/merged/v9_labels_union.npz"
P10 = "E:/Projects/TanitAD-artifacts/sfresh_inputs/inputs/p10/p10_prior.pt"
INIT = "C:/Users/Admin/lever3/init/UE_s0/A/ckpt_last.pt"
TMPL = "F:/TanitAD-scratch/rollhist/lever3/rec/x/rollhist/argv/RH_s0.tmpl.json"
UEA = "F:/TanitAD-scratch/rollhist/build/b3gate/ref/pod_argv/UE_s0.A.json"
GRID = KIT + "/inputs/eval_grid_windows.json"
MM_CACHE = "C:/Users/Admin/lever3/cache_union/train/clips"


def env():
    e = dict(os.environ)
    e.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_DISABLE_XET": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
              "PYTHONPATH": ";".join([TREE + "/stack", TREE + "/stack/scripts", TREE])})
    return e


def sub(args, log):
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.run([PY] + args, env=env(), stdout=fh, stderr=subprocess.STDOUT, cwd=TREE)
    return p.returncode, time.time() - t0


def mapped_argv(path, cache, join, out, fit_join):
    t = json.load(open(path))["argv"]
    ov = {"--cache-dir": cache, "--v9-train": V9, "--lead-train": B3 + "/labels/NO_LEAD_SLICE.npz", "--speed-train": B3 + "/labels/speed_slice.jsonl",
          "--join-train": join, "--p10": P10, "--tracks": B3 + "/labels/native_tracks_slice.pt", "--init-ckpt": INIT, "--out": out, "--device": "cpu",
          "--dagger-bank": "@BANK@", "--dagger-bank-sha256": "0" * 64}
    for k, v in ov.items():
        if k in t:
            t[t.index(k) + 1] = v
    if not fit_join and "--fit-join" in t:
        t.remove("--fit-join")
    return t


def hash_obj(o, h, path=""):
    import numpy as np
    import torch
    if isinstance(o, torch.Tensor):
        o = o.detach().cpu().numpy()
    if isinstance(o, np.ndarray):
        h.update(f"{path}|{o.dtype}|{o.shape}|".encode())
        h.update(np.ascontiguousarray(o).tobytes())
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            hash_obj(o[k], h, f"{path}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, x in enumerate(o):
            hash_obj(x, h, f"{path}[{i}]")
    else:
        h.update(f"{path}|{type(o).__name__}|{o!r}".encode())


def stage_T(a):
    """the trainer's own sequence of main(): build_data, V8FMiniWindows, ChunkedSampler(seed), first batch (collate); then every clip through the store."""
    import numpy as np
    import torch
    import v8f_mini_train as TR
    from tanitad.data import v8f_mini_data as D
    argv = mapped_argv(a.argv_template, a.cache, a.join, a.work + "/out", fit_join=True)
    ns = TR.build_parser().parse_args(argv)
    cfg = TR.make_config(ns)
    torch.manual_seed(ns.seed)
    np.random.seed(ns.seed)
    store, J, wins, clips, info = TR.build_data(ns, cfg)
    ds = D.V8FMiniWindows(store, J, wins, hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start, stok=getattr(store, "stok_store", None))
    sampler = D.ChunkedSampler(wins, ns.batch, seed=ns.seed, chunk=max(4, min(48, info["lru"] - 2)), max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s)
    first = next(sampler.epoch(0))
    batch = D.collate([ds.item(*w) for w in first])
    hb = hashlib.sha256()
    hash_obj(batch, hb)
    clip_names = J["clip"].tolist()
    first_clips = sorted({str(clip_names[w[0]]) for w in first})
    hs = {}
    for c in clips:                       # the whole store path of every clip: get (the training read) and get_kin (the bank / table read)
        h = hashlib.sha256()
        hash_obj(store.get(c), h)
        hash_obj(store.get_kin(c), h)
        hs[c] = h.hexdigest()
    hj = hashlib.sha256(open(a.join, "rb").read()).hexdigest()
    json.dump({"n_clips": len(clips), "n_windows": len(wins), "first_batch_n": len(first), "first_batch_sha256": hb.hexdigest(), "first_batch_clips": first_clips,
               "clip_store_sha256": hs, "clip_store_all_sha256": hashlib.sha256("".join(hs[c] for c in sorted(hs)).encode()).hexdigest(), "join_sha256": hj,
               "train_sha256": info["train_sha256"], "train_window_list": hashlib.sha256(repr(wins).encode()).hexdigest(), "argv_cache_dir": ns.cache_dir},
              open(a.out_json, "w"), indent=1)
    print("stageT ok", hb.hexdigest()[:16], len(first), "windows in the first batch")


def stage_B(a):
    """the bank generator over the first N clips of the TRAIN list, the pod's s2 flags; reads back the shard's content hash."""
    uea = mapped_argv(UEA, a.cache, a.join, a.work + "/out_unused", fit_join=False)   # M2b's argv: the join is the file stage T built (no --fit-join: shards must not rewrite it)
    os.makedirs(a.work, exist_ok=True)
    margv = a.work + "/m2b_argv.json"
    json.dump({"argv": uea}, open(margv, "w"))
    out = a.work + "/bank"
    if os.path.isdir(out):
        shutil.rmtree(out)
    args = [TREE + "/stack/scripts/v8f_dagger_bank.py", "gen", "--ckpt", INIT, "--m2b-argv", margv, "--eval-grid", GRID, "--out-dir", out, "--shard", "0", "--n-shards", "1",
            "--limit-clips", str(a.n_bank_clips), "--device", "cpu", "--readout", "causal_g", "--acc-gain", "1", "--path-exec", "--kappa-max", "0.2", "--threads", "4"]
    t0 = time.time()
    rc, _ = sub(args, a.work + "/bank_gen.log")
    dt = time.time() - t0
    prov = out + "/shard_000of001.provenance.json"
    res = {"rc": rc, "wall_s": dt}
    if os.path.isfile(prov):
        p = json.load(open(prov))
        res.update({"content_sha256": p["content_sha256"], "n_records": p["n_records"], "shard_windows": p["plan"]["shard_windows"], "coverage": p["coverage"],
                    "rollout_s": p["seconds"]["rollout"], "controls_pass": {k: v.get("PASS") for k, v in p["controls"].items()}})
    json.dump(res, open(a.out_json, "w"), indent=1)
    print("stageB", res.get("content_sha256", "NO PROVENANCE")[:16], f"rc={rc} wall {dt:.0f}s")
    sys.exit(0 if (rc == 0 and "content_sha256" in res) else 1)


def mutate(src_slice, dst_slice, clip):
    """a copy of the repacked slice with the pool4x10 tensor of ONE clip deleted"""
    import torch
    if os.path.exists(dst_slice):
        shutil.rmtree(dst_slice)
    shutil.copytree(src_slice, dst_slice)
    p = f"{dst_slice}/train/clips/{clip}.pt"
    o = torch.load(p, map_location="cpu", weights_only=True)
    del o["pool4x10"]
    torch.save(o, p)


def run(a):
    res = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tree": TREE, "n_bank_clips": a.n_bank_clips, "controls": {}}
    W = B3 + "/work"
    os.makedirs(W, exist_ok=True)

    def T(name, cache):
        out = f"{W}/T_{name}.json"
        rc, dt = sub([KIT + "/b3_check.py", "stageT", "--cache", cache, "--join", f"{B3}/join_{name}.npz", "--work", f"{W}/T_{name}", "--out-json", out, "--argv-template", TMPL],
                     f"{W}/T_{name}.log")
        return rc, (json.load(open(out)) if rc == 0 and os.path.isfile(out) else None), dt

    def B(name, cache, n):
        out = f"{W}/B_{name}.json"
        rc, dt = sub([KIT + "/b3_check.py", "stageB", "--cache", cache, "--join", f"{B3}/join_{name}.npz", "--work", f"{W}/B_{name}", "--out-json", out, "--n-bank-clips", str(n)],
                     f"{W}/B_{name}.log")
        return rc, (json.load(open(out)) if os.path.isfile(out) else None), dt

    for f in (f"{B3}/join_orig.npz", f"{B3}/join_repack.npz"):
        if os.path.exists(f):
            os.remove(f)                                          # own temp of an earlier run of this check
    rcT0, t_o, dT0 = T("orig", B3 + "/orig")
    rcT1, t_r, dT1 = T("repack", B3 + "/repack")
    res["trainer"] = {"orig": t_o and {k: v for k, v in t_o.items() if k != "clip_store_sha256"}, "repack": t_r and {k: v for k, v in t_r.items() if k != "clip_store_sha256"}, "rc": [rcT0, rcT1], "wall_s": [dT0, dT1]}
    ok_T = bool(t_o and t_r and t_o["first_batch_sha256"] == t_r["first_batch_sha256"] and t_o["clip_store_all_sha256"] == t_r["clip_store_all_sha256"] and
                t_o["join_sha256"] == t_r["join_sha256"] and t_o["train_window_list"] == t_r["train_window_list"])
    res["controls"]["T_first_batch_equal"] = ok_T
    rcB0, b_o, dB0 = B("orig", B3 + "/orig", a.n_bank_clips)
    rcB1, b_r, dB1 = B("repack", B3 + "/repack", a.n_bank_clips)
    res["bank"] = {"orig": b_o, "repack": b_r, "rc": [rcB0, rcB1], "wall_s": [dB0, dB1]}
    ok_B = bool(b_o and b_r and "content_sha256" in b_o and b_o.get("content_sha256") == b_r.get("content_sha256") and b_o["controls_pass"] == b_r["controls_pass"])
    res["controls"]["B_first_chunk_equal"] = ok_B
    # mutations: pool4x10 deleted from one repacked clip, must FAIL each path
    if t_o:
        clipT = t_o["first_batch_clips"][0]
        mutate(B3 + "/repack", B3 + "/repack_mutT", clipT)
        rc, t_m, _ = T("mutT", B3 + "/repack_mutT")
        res["mutation_trainer"] = {"clip": clipT, "rc": rc, "differs": bool(t_m and t_m["first_batch_sha256"] != t_o["first_batch_sha256"]), "log_tail": open(f"{W}/T_mutT.log", encoding="utf-8", errors="replace").read()[-300:]}
        res["controls"]["T_mutation_fails"] = bool(rc != 0 or res["mutation_trainer"]["differs"])
    if b_o and os.path.isfile(f"{W}/B_orig/m2b_argv.json"):
        import torch
        # the first TRAIN clip by the sha12 order the lister uses: the bank's limit-clips takes train clips in that order
        sys.path[:0] = [TREE + "/stack"]
        from tanitad.data import v8f_mini_data as D
        clips = [c for c, _ in D.read_manifest(B3 + "/repack", "train")]
        clipB = D.select_train_clips(clips, 1.0)[0]
        mutate(B3 + "/repack", B3 + "/repack_mutB", clipB)
        rc, b_m, _ = B("mutB", B3 + "/repack_mutB", a.n_bank_clips)
        res["mutation_bank"] = {"clip": clipB, "rc": rc, "differs": bool(b_m and b_m.get("content_sha256") and b_m["content_sha256"] != b_o["content_sha256"]),
                                "log_tail": open(f"{W}/B_mutB.log", encoding="utf-8", errors="replace").read()[-300:]}
        res["controls"]["B_mutation_fails"] = bool(rc != 0 or res["mutation_bank"]["differs"])
    # the MM's cache: the 10k clips of the slice must equal its repacked files
    sl = json.load(open(B3 + "/SLICES.json"))["clips"]
    cmp = {}
    for r in sl:
        if r["part"] == "10k":
            p = f"{MM_CACHE}/{r['clip']}.pt"
            cmp[r["clip"]] = (hashlib.md5(open(p, "rb").read()).hexdigest() == r["md5"]) if os.path.isfile(p) else None
    res["mm_cache_equal_for_10k_slice_clips"] = {"n": len(cmp), "equal": sum(1 for v in cmp.values() if v), "absent": sum(1 for v in cmp.values() if v is None)}
    res["controls"]["mm_cache_md5_equal"] = all(v for v in cmp.values()) if cmp and all(v is not None for v in cmp.values()) else None
    res["pass"] = all(v is True for k, v in res["controls"].items() if k != "mm_cache_md5_equal")
    json.dump(res, open(KIT + "/B3_CHECK.json", "w"), indent=1)
    print(json.dumps(res["controls"]), "PASS" if res["pass"] else "FAIL")
    return 0 if res["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "stageT", "stageB"])
    ap.add_argument("--cache")
    ap.add_argument("--join")
    ap.add_argument("--work")
    ap.add_argument("--out-json")
    ap.add_argument("--argv-template", default=TMPL)
    ap.add_argument("--n-bank-clips", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "run":
        sys.exit(run(a))
    os.makedirs(a.work, exist_ok=True)
    if a.cmd == "stageT":
        sys.path[:0] = [TREE + "/stack", TREE + "/stack/scripts", TREE]
        stage_T(a)
    else:
        stage_B(a)


main()
