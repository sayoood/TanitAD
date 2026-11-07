"""PREREG_V8F_VOE1B step 2: the real run (VOE1's run_voe1.py, copied; the deltas are named in IMPL_CHOICES_VOE1B.md).
Reads the bank (build_bank_voe1b.py), scores the six manipulations and the real-vs-real control for M2b s0 / s1 and for the 32 untrained world models UT{chassis}_{100..115},
the event alignment (full for M2b, AP only for the untrained), writes RESULT_VOE1B.json, traces_voe1b.npz and VOE1B_RUN.done (rc=<n>) into --out-dir (default: this directory).
Judges nothing: report_voe1b.py checks the controls and writes RESULT_VOE1B.md.
Options (smoke tests of the plumbing only, written to another --out-dir): --max-episodes N (first N episodes), --n-untrained K (first K seeds per chassis).
Command: see HANDOFF_VOE1B.md. Needs WMACT_CODE_ROOT, CUDA_VISIBLE_DEVICES=-1."""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voe1b_core as C  # noqa: E402

CARD = os.path.join(HERE, "..", "PREREG_V8F_VOE1B.md")
CARD_PREFIX_BYTES, CARD_PREFIX_SHA = 5785, "aa317fe3f046e9d0522701d4b35c9d029b1fd55a4a4590823c02ff60449493c3"
IMPL_BANK_SHA = "b217db9d1e66fcbc4bd2376e4120ca5393c044413863f4a034fec16ab177565a"      # as recorded in IMPL_CHOICES_VOE1B section 6
CODE_FILES = ("voe1b_core.py", "voe1b_real.py", "build_bank_voe1b.py", "counts_voe1b.py", "run_voe1b.py", "test_voe1b_synth.py")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def ap_task(a):
    """Worker: one event cell. a = (key, sm, sn, Y, ep_w, n_ep, with_boot)."""
    key, sm, sn, Y, ep_w, n_ep, with_boot = a
    t = time.time()
    r = C.ap_cell(sm, sn, Y, ep_w, n_ep, key, with_boot=with_boot, cal_seeds=C.CAL_SEEDS if with_boot else ())
    r["seconds"] = round(time.time() - t, 1)
    return key, r


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def main(workers: int, out_dir: str, max_episodes: int, n_ut: int):
    import torch
    import voe1b_real as R
    from voe1b_real import W
    t_start = time.time()
    log = lambda *a: print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    OUT_JSON, OUT_NPZ = (os.path.join(out_dir, n) for n in ("RESULT_VOE1B.json", "traces_voe1b.npz"))
    smoke = bool(max_episodes) or n_ut != len(R.UT_SEEDS)

    # ---- inputs and their hashes
    card_bytes = open(CARD, "rb").read()
    assert hashlib.sha256(card_bytes[:CARD_PREFIX_BYTES]).hexdigest() == CARD_PREFIX_SHA, "card prefix changed"
    impl = os.path.join(HERE, "IMPL_CHOICES_VOE1B.md")
    impl_sha = sha_file(impl)
    assert open(os.path.join(HERE, "IMPL_CHOICES_VOE1B.sha256")).read().split()[0] == impl_sha, "IMPL_CHOICES changed after its hash was recorded"
    bank_p = R.BANK_PATH
    Bk = torch.load(bank_p, weights_only=False)
    meta = {"card_prefix_sha256": CARD_PREFIX_SHA, "card_file_sha256": sha_file(CARD), "impl_sha256": impl_sha, "bank_sha256": sha_file(bank_p), "bank_path": bank_p,
            "code_sha256": {f: sha_file(os.path.join(HERE, f)) for f in CODE_FILES},
            "latents_pt_sha256": sha_file(os.path.join(HERE, "..", "wmact", "latents.pt")), "wmact_lib_sha256": sha_file(os.path.join(HERE, "..", "wmact", "wmact_lib.py")),
            "counts_sha256": sha_file(os.path.join(HERE, "counts_voe1b.json")), "tanitad_file": W.tanitad.__file__, "workers": workers, "smoke": smoke,
            "max_episodes": max_episodes, "n_untrained_per_chassis": n_ut, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    meta["bank_sha_equals_impl"] = meta["bank_sha256"] == IMPL_BANK_SHA
    chk = Bk["check"]
    assert chk["M2b_s0"]["max_abs_diff_z1"] < 1e-4 and chk["M2b_s1"]["max_abs_diff_z1"] < 1e-4 and chk["actions_max_abs_diff"] == 0.0, chk

    # ---- models: the two trained, the untrained family (hash-checked against the bank build)
    trained = R.load_trained()
    wms = R.build_wms(trained, n_ut)
    rep = R.reinit_report(trained, wms)
    for tag, r_ in rep["ut"].items():
        assert r_["wm_state_sha256"] == Bk["reinit"]["ut"][tag]["wm_state_sha256"], tag
        assert r_["n_tensors_equal_to_trained"] == 0 and r_["n_params_reinitialised"] == r_["n_params_total"], tag
    TAGS = R.tags_for(n_ut)
    UT = tuple(t for t in TAGS if t not in R.DECIDING)
    log(f"models built ({len(TAGS)} tags); untrained hashes equal the bank build's; every wm.* parameter re-initialised")

    # ---- windows
    w = Bk["win"]
    keep = np.asarray(w["keep"])
    eid_all = np.array(w["eid"])
    if max_episodes:
        keep = keep & np.isin(eid_all, sorted(set(eid_all.tolist()))[:max_episodes])
    eid = eid_all[keep]
    base = np.asarray(w["base"])[keep]
    g0 = np.asarray(w["g0"])[keep]
    T_clip = np.array([Bk["T"][e] for e in eid])
    ep, n_ep = C.episode_codes(eid)
    act, spd = Bk["act"].numpy(), Bk["speed"].numpy()
    wt = C.window_table(act, spd, base, g0, T_clip, ep)
    from tanitad.models.hjepa import donor_indices
    don_t = donor_indices(len(base), episode_ids=torch.as_tensor(ep), seed=0).numpy()
    assert (don_t == wt["donor"]["SPLICE"]).all(), "SPLICE donors differ from tanitad donor_indices"
    uq, Wt = W.boot_weights(eid)
    assert np.array_equal(Wt, C.boot_W(n_ep)), "bootstrap weights differ from wmact_lib.boot_weights"
    log(f"windows {len(base)}, episodes {n_ep}; donors and bootstrap weights equal the repo helpers")
    if not smoke:
        cnt = json.load(open(os.path.join(HERE, "counts_voe1b.json"), encoding="utf-8"))
        for k, q in wt["q"].items():
            assert int(q.sum()) == cnt["qualify"][k]["n_windows"], (k, int(q.sum()), cnt["qualify"][k])
        log("qualifying windows equal the label-only counts frozen in IMPL_CHOICES (incl. ACTION-SWAP-M n)")

    qm = wt["q"]["ACTION-SWAP-M"]
    res = {"meta": meta, "repro": chk, "reinit": rep, "window_table": {k: {"n_windows": int(q.sum()), "n_episodes": int(len(set(eid[q])))} for k, q in wt["q"].items()},
           "swapm_candidates": {"min": int(wt["swapm_ncand"][qm].min()) if qm.any() else 0, "median": float(np.median(wt["swapm_ncand"][qm])) if qm.any() else 0.0},
           "ut_tags": list(UT), "tags": {}, "events": {}}
    models, bank_z, bank_a = {}, {}, {}
    act_t = Bk["act"].float()
    traces_out = {}

    # ---- event cells of the two deciding tags go to the pool first (they run while this process scores the tags)
    clip_lo = np.array([Bk["off"][e] for e in eid])
    clip_hi = clip_lo + T_clip
    fr16 = base[:, None] + np.arange(16)[None]
    ac16 = base[:, None] + np.arange(15)[None]
    Yev = {}
    for ev, key in zip(C.EVENTS, ("ev_brake", "ev_turn", "ev_lead")):
        Yev[ev] = {l: C.event_step_labels(Bk[key], base, clip_lo, clip_hi, l, {1: 15, 2: 7, 3: 3}[l]) for l in (1, 2, 3)}
    for tag in TAGS:
        models[tag] = C.HJEPATraces(wms[tag].stack)
        bank_z[tag] = Bk["z1"][tag].float()
        with torch.no_grad():
            bank_a[tag] = wms[tag].scale_actions(act_t)

    def tr16_of(tag):
        return models[tag].traces(bank_z[tag][torch.from_numpy(fr16)], bank_a[tag][torch.from_numpy(ac16)])

    pool, futs = None, []
    tasks = []
    for tag in R.DECIDING:
        t16 = tr16_of(tag)
        for ev in C.EVENTS:
            for l in (1, 2, 3):
                tasks.append((f"{tag}/{ev}/L{l}", t16[l]["sur"], t16[l]["null"], Yev[ev][l], ep, n_ep, True))
    tasks.sort(key=lambda a: -a[1].shape[1])                                           # level 1 first
    if workers > 1:
        import multiprocessing as mp
        os.environ["OMP_NUM_THREADS"] = "1"
        pool = mp.get_context("spawn").Pool(workers - 1)                               # the main process is the sixth process
        futs = [pool.apply_async(ap_task, (a,)) for a in tasks]
    log(f"event cells of the deciding tags: {len(tasks)} ({'pool of %d' % (workers - 1) if pool else 'inline'})")

    # ---- the manipulation rows, per tag
    for tag in TAGS:
        t = time.time()
        deciding = tag in R.DECIDING
        r = C.eval_tag(models[tag], bank_z[tag], bank_a[tag], base, eid, wt, tag, cal=True, keep_traces=deciding, lean=not deciding, rvr_ref_flags=deciding, median_traces=True)
        tr, mt = r.pop("_traces"), r.pop("_mtraces")
        if tr is not None:
            for name, d in tr.items():
                for k, v in d.items():
                    traces_out[f"{tag}/{name}/{k}"] = np.asarray(v, np.float32) if k != "q" else np.asarray(v, bool)
        for name, d in mt.items():
            for k, v in d.items():
                traces_out[f"mt/{tag}/{name}/{k}"] = np.asarray(v, np.float32)
        res["tags"][tag] = r
        log(f"{tag}: rows done ({time.time() - t:.0f}s)")

    # ---- event alignment of the untrained tags: AP only (no interval)
    for tag in UT:
        t16 = tr16_of(tag)
        for ev in C.EVENTS:
            for l in (1, 2, 3):
                k_, r_ = ap_task((f"{tag}/{ev}/L{l}", t16[l]["sur"], t16[l]["null"], Yev[ev][l], ep, n_ep, False))
                res["events"].setdefault(tag, {}).setdefault(ev, {})[f"L{l}"] = r_
    log("untrained event AP done")
    if pool is not None:
        out = dict(f.get() for f in futs)
        pool.close()
        pool.join()
    else:
        out = dict(ap_task(a) for a in tasks)
    for key, r_ in out.items():
        tag, ev, lv = key.split("/")
        res["events"].setdefault(tag, {}).setdefault(ev, {})[lv] = r_
    res["event_labels"] = {ev: {str(l): {"n_points": int(Y.size), "positives": int(Y.sum()), "rate": float(Y.mean())} for l, Y in d.items()} for ev, d in Yev.items()}
    res["event_onsets_in_bank"] = {ev: int(Bk[k].sum()) for ev, k in zip(C.EVENTS, ("ev_brake", "ev_turn", "ev_lead"))}
    log("event cells done")

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["seconds"] = round(time.time() - t_start, 1)
    np.savez_compressed(OUT_NPZ, **traces_out)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8", newline="\n"), default=jdefault)
    log(f"wrote {OUT_JSON} ({os.path.getsize(OUT_JSON)} bytes) and {OUT_NPZ}")


if __name__ == "__main__":
    arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    nw, out_dir = arg("--workers", 6), arg("--out-dir", HERE)
    max_ep, n_ut = arg("--max-episodes", 0), arg("--n-untrained", 16)
    rc = 0
    try:
        main(nw, out_dir, max_ep, n_ut)
    except BaseException:
        traceback.print_exc()
        rc = 1
    open(os.path.join(out_dir, "VOE1B_RUN.done"), "w").write(f"rc={rc}\n")
    sys.exit(rc)
