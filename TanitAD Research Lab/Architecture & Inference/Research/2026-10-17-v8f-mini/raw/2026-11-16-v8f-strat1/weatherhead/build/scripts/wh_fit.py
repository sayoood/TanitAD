"""wh_fit.py -- PREREG_V8F_WEATHERHEAD form (a), card 4.2: fit the five-sub-head pooled head on the FROZEN driving model's cached features, TRAIN windows only.

  fit   one arm of the card's table (P1..P9): python wh_fit.py fit --arm H_s0 --init S1_s0 --adapter s1 --fit-seed 0 --epochs auto --out OUT/H_s0 --token TOKEN
  k4p   the K4p control: an H checkpoint with its pooled head re-initialised (CTX_POOL_INIT_SEED, forked RNG), eval only
  kid   the fit script's own ``ctx_pool_logits`` on the eval grid (K-ID: equal to the evaluator's to atol 1e-5)

Zero driving parameters are trained (S, T, O, adapter, ``pos``, ``time``, ``e_*`` and the shipped ``ctx_head`` are frozen; the FEAT arm alone trains a fresh adapter WITH the head, by design).
No fit starts without a ``wh-gate-token-v1`` PASS token bound to this exact argv (``--smoke`` is the only exception and marks every output SMOKE: never a result).
The eval set is never loaded by ``fit`` (disjointness, card 4.5)."""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wh_lib as W  # noqa: E402
import torch  # noqa: E402

M = W.M


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def strip_token(argv):
    """the argv the token binds: everything after the subcommand name, without the ``--token PATH`` pair"""
    out, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x == "--token":
            skip = True
            continue
        out.append(x)
    return out


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--arm", required=True)
    f.add_argument("--init", required=True, choices=sorted(W.CKPT_PREFIX), help="the banked driving checkpoint (S1_s0 | S1_s1 | C0_s0 | C0_s1)")
    f.add_argument("--ctx-pool", choices=("on",), required=True, help="the lever of this card: NAMED in every arm's argv (a default that disables a lever must be named)")
    f.add_argument("--adapter", choices=("s1", "random", "feat"), default="s1", help="s1 = the banked adapter, frozen; random = a random frozen adapter (RA); feat = a fresh adapter TRAINED with the head (FEAT)")
    f.add_argument("--adapter-seed", type=int, default=0)
    f.add_argument("--fit-seed", type=int, default=0)
    f.add_argument("--labels", choices=("real", "permuted"), default="real", help="permuted = R1: train labels permuted between train CLIPS (seed 0)")
    f.add_argument("--epochs", default="auto", help="auto = choose E_v per variable on TRAIN only (H arms) | a path to the wh_meta.json of the arm whose E_v this arm uses")
    f.add_argument("--e-grid", default="4,8,16", help="the epoch grid; anything but 4,8,16 needs --smoke")
    f.add_argument("--out", required=True)
    f.add_argument("--token", default="", help="the wh-gate-token-v1 PASS token (required unless --smoke)")
    f.add_argument("--source", choices=("auto", "ram", "stream"), default="auto", help="P-COMMIT: auto = in RAM iff free commit >= features + 5.5 GB, else stream the memory-mapped clips")
    f.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    f.add_argument("--train-cache", default=W.TRAIN_CACHE)
    f.add_argument("--max-clips", type=int, default=0, help="SMOKE only: the first N train clips")
    f.add_argument("--pod-train-manifest", default="", help="P-CACHE: the pod's train manifest.jsonl (a file the MM pulls)")
    f.add_argument("--smoke", action="store_true", help="a code check, never a result")
    k = sub.add_parser("k4p")
    k.add_argument("--ckpt", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--smoke", action="store_true")
    d = sub.add_parser("kid")
    d.add_argument("--ckpt", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--eval-cache", default=W.EVAL_CACHE)
    d.add_argument("--max-windows", type=int, default=0, help="SMOKE only: every k-th window so that N remain")
    d.add_argument("--smoke", action="store_true")
    return ap


def log_to(path):
    f = open(path, "a", encoding="utf-8")

    def log(*a):
        s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return log


def cmd_fit(a, argv):
    os.makedirs(a.out, exist_ok=True)
    log = log_to(f"{a.out}/wh_fit_log.txt")
    t0 = time.time()
    grid = tuple(int(x) for x in a.e_grid.split(","))
    if grid != W.E_GRID and not a.smoke:
        raise SystemExit("REFUSED: the epoch grid is 4,8,16 (card 4.2); another grid is a SMOKE")
    if a.max_clips and not a.smoke:
        raise SystemExit("REFUSED: --max-clips is a SMOKE-only flag")
    tok = None
    if not a.smoke:
        import wh_check as C
        tok, why = C.verify_token(a.token, strip_token(argv[1:]), a.arm)
        if why:
            raise SystemExit("REFUSED (no gate token): " + why)
    elif a.token:
        raise SystemExit("REFUSED: --smoke and --token are two different things; a smoke never carries a token")
    why = W.ckpt_sha_problem(a.init)
    if why:
        raise SystemExit("REFUSED: " + why)
    tab = W.load_table()
    join = dict(np.load(W.pinned("join_eval141.npz")))
    grid_path = W.pinned("eval_grid_windows.json")
    disj = W.disjointness(tab, grid_path, a.train_cache, [str(c) for c in join["clip"]])
    log("DISJOINTNESS", disj)
    if any(v != 0 for v in disj.values()):
        raise SystemExit(f"REFUSED: train / eval overlap {disj}")
    clips = [str(c) for c in tab["train_clip"]]
    rs = W.train_rowset(tab, range(a.max_clips) if a.max_clips else None)
    rs, n_skipped = W.drop_mismatching_clips(a.train_cache, clips, tab, rs)
    y = W.fit_labels(tab, rs["row"])
    if a.labels == "permuted":
        y = W.permute_labels_between_clips(y, rs["ci"], 0)
    log(f"train windows {len(y)} from {len(np.unique(rs['ci']))} clips, skipped clips (frame count != train_n): {n_skipped}; labelled rows per variable {[int((y[:, k] >= 0).sum()) for k in range(5)]}")
    sc = W.choose_source(len(y), a.source)
    log("P-COMMIT", sc)
    src = W.make_source(sc["mode"], a.train_cache, clips, rs, log=log)
    pc = {"train": W.cache_provenance(a.train_cache), "md5_vs_pod": W.p_cache_match(f"{a.train_cache}/manifest.jsonl", a.pod_train_manifest)}
    log("P-CACHE", pc)
    dev = torch.device(a.device)
    lock_job = None
    if a.device == "cuda":
        lock_job = f"wh-{a.arm}"
        if not W.gpu_lock_acquire(lock_job, timeout_s=0):
            raise SystemExit("LOCK-BUSY: devbox_gpu.lock is held or the GPU reads >= 3,500 MiB used; never forced")
    try:
        ck = torch.load(f"{W.CK}/{a.init}/ckpt_last.pt", map_location="cpu", weights_only=False)
        model = W.load_pool_model(ck, a.adapter, a.adapter_seed).to(dev)
        train_adapter = a.adapter == "feat"
        # reachability on ONE real batch before any fit (G-LIVE, card 4.4 item 2)
        b0 = np.arange(min(W.BS, len(y)))
        live = W.live_check(model, src.gather(b0).to(dev), torch.as_tensor(y[b0]).to(dev), train_adapter=train_adapter)
        log("LIVE", {k: live[k] for k in ("pass", "subhead_grad_abs_sum", "other_grad_abs_sum")})
        if not live["pass"]:
            raise SystemExit("REFUSED: G-LIVE reachability failed " + json.dumps(live)[:400])
        meta = {"arm": a.arm, "SMOKE": bool(a.smoke), "result": False if a.smoke else None, "argv": argv, "argv_sha256": argv_sha(strip_token(argv[1:])), "token": None if tok is None else tok.get("sha256"),
                "init": a.init, "init_ckpt_sha256": json.load(open(f"{W.CK}/{a.init}/BANK_MANIFEST.json"))["files"]["ckpt_last.pt"]["sha256"], "adapter": a.adapter, "adapter_seed": a.adapter_seed,
                "fit_seed": a.fit_seed, "labels": a.labels, "label_table_sha256": W.PINS["strat_labels_table.npz"], "train_manifest_sha256": W.sha256_file(f"{a.train_cache}/manifest.jsonl"),
                "base_commit": W.BASE_COMMIT, "n_windows": int(len(y)), "n_clips": int(len(np.unique(rs["ci"]))), "n_skipped_clips": int(n_skipped), "source": sc, "p_cache": pc, "disjointness": disj,
                "labelled_rows": {v: int((y[:, k] >= 0).sum()) for k, v in enumerate(M.CTX_VARS)}, "live": live, "device": a.device, "torch": torch.__version__, "records": {}}
        if a.epochs == "auto":
            if a.adapter != "s1" or a.labels != "real":
                raise SystemExit("REFUSED: --epochs auto is the H arms' selection (banked adapter, real labels); every other arm names the wh_meta.json of its seed's H arm")
            E_v, scores, recs = W.select_epochs(model, src, y, rs["ci"], dev, log, grid, seed=a.fit_seed)
            meta["epoch_selection"] = {"grid": list(grid), "heldout_margin": scores, "E_v": E_v, "records": recs}
        else:
            ref = json.load(open(a.epochs, encoding="utf-8"))
            if ref.get("SMOKE") != bool(a.smoke):
                raise SystemExit("REFUSED: a SMOKE arm and a real arm cannot lend each other their E_v")
            E_v = {v: int(ref["E_v"][v]) for v in M.CTX_VARS}
            meta["epochs_from"] = {"path": a.epochs, "sha256": W.sha256_file(a.epochs), "arm": ref["arm"]}
        meta["E_v"] = E_v
        log("E_v", E_v)
        E_use = dict(E_v)
        if train_adapter:                                                  # FEAT: the adapter is SHARED by the five sub-heads and trained with them, so per-variable epochs cannot be composed from different fits (the heads would sit on different adapters):
            E_use = {v: E_v["weather"] for v in M.CTX_VARS}                  # ONE fit at the weather E_v (the primary variable); recorded, and listed as a deviation in the build report
        meta["E_v_used"] = E_use
        fin = W.fit_final(model, src, y, E_use, a.fit_seed, dev, train_adapter, log)
        meta["final"] = fin
        bad = [(E, r["census"]) for E, r in fin["recs"].items() if not r["census"]["pass"]]
        meta["census_pass"] = not bad
        if bad:
            log("CENSUS FAILED", bad)
        sha = W.save_head_ckpt(f"{a.out}/ckpt_wh.pt", model, ck, {"arm": a.arm, "SMOKE": bool(a.smoke), "E_v": E_v, "fit_seed": a.fit_seed, "init": a.init, "adapter": a.adapter})
        meta["ckpt_sha256"] = sha
        meta["seconds"] = time.time() - t0
        json.dump(meta, open(f"{a.out}/wh_meta.json", "w", encoding="utf-8"), indent=1, default=float)
        open(f"{a.out}/{'SMOKE_' if a.smoke else ''}DONE", "w").write(sha + "\n")
        log("ZZ WH-FIT DONE ZZ", a.arm, f"{time.time() - t0:.0f}s", "SMOKE (never a result)" if a.smoke else "")
        return 0 if not bad else 3
    finally:
        if lock_job:
            W.gpu_lock_release(lock_job)


def cmd_k4p(a):
    """K4p: the H checkpoint with the pooled head re-initialised (CTX_POOL_INIT_SEED, forked RNG)."""
    os.makedirs(a.out, exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if ck.get("cfg", {}).get("ctx_pool") != "on":
        raise SystemExit("REFUSED: this checkpoint has no pooled head")
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, W.p10_zero())
    model.load_state_dict(ck["model"])
    before = W.head_state(model)
    W.reinit_head(model, M.CTX_POOL_INIT_SEED)
    after = W.head_state(model)
    if all(torch.equal(before[k], after[k]) for k in before):
        raise SystemExit("REFUSED: the re-initialised head equals the trained one")
    sha = W.save_head_ckpt(f"{a.out}/ckpt_wh.pt", model, ck, {"arm": "K4p", "SMOKE": bool(a.smoke), "from": a.ckpt, "from_sha256": W.sha256_file(a.ckpt), "init_seed": M.CTX_POOL_INIT_SEED})
    json.dump({"arm": "K4p", "SMOKE": bool(a.smoke), "from_sha256": W.sha256_file(a.ckpt), "ckpt_sha256": sha, "init_seed": M.CTX_POOL_INIT_SEED}, open(f"{a.out}/wh_meta.json", "w"), indent=1)
    print("K4P", a.out, sha)
    return 0


def cmd_kid(a):
    """The fit script's own logits on the eval grid (the FIT path ``ctx_pool_fit_logits``, 4-frame windows read straight from the eval clip files)."""
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, W.p10_zero())
    model.load_state_dict(ck["model"])
    join = dict(np.load(W.pinned("join_eval141.npz")))
    ew = W.eval_windows(W.pinned("eval_grid_windows.json"), join["clip"])
    sel = np.arange(len(ew["ci"]))
    if a.max_windows:
        sel = np.linspace(0, len(sel) - 1, a.max_windows).round().astype(int)
    rs = {"ci": ew["ci"][sel], "t": ew["t"][sel], "row": np.zeros(len(sel), np.int64)}
    src = W.StreamSource(a.eval_cache, [str(c) for c in join["clip"]], rs)
    lg = W.predict(model, src, np.arange(len(sel)), torch.device("cpu"))
    np.savez(a.out, ctx_pool_logits=lg, win_ci=rs["ci"], win_t=rs["t"], sel=sel, SMOKE=np.asarray(bool(a.smoke)))
    print("KID", a.out, lg.shape)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.cmd == "fit":
        return cmd_fit(a, argv)
    return cmd_k4p(a) if a.cmd == "k4p" else cmd_kid(a)


if __name__ == "__main__":
    sys.exit(main())
