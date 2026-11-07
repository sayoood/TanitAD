"""PREREG_V8F_VOE1 step 2: the real run. Reads bank_voe1.pt (build_bank_voe1.py), scores the five manipulations, the real-vs-real control and the event alignment for
the six tags, writes RESULT_VOE1.json, traces_voe1.npz and VOE1_RUN.done (rc=<n>). Judges nothing: report_voe1.py checks the controls and writes RESULT_VOE1.md.
Command: see HANDOFF_VOE1.md. Needs WMACT_CODE_ROOT, CUDA_VISIBLE_DEVICES=-1."""
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import voe1_core as C  # noqa: E402

CARD = os.path.join(HERE, "..", "PREREG_V8F_VOE1.md")
CARD_PREFIX_BYTES, CARD_PREFIX_SHA = 5822, "555c5a3896d1ff0d745763b48a14ed7294b4ef814fab0fd5a31d44e2ccfbf69a"
OUT_JSON, OUT_NPZ, DONE = (os.path.join(HERE, n) for n in ("RESULT_VOE1.json", "traces_voe1.npz", "VOE1_RUN.done"))


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


def main(workers: int):
    import torch
    import voe1_real as R
    from voe1_real import W
    t_start = time.time()
    log = lambda *a: print(f"[{time.time() - t_start:6.0f}s]", *a, flush=True)

    # ---- inputs and their hashes
    assert open(CARD, "rb").read()[:CARD_PREFIX_BYTES] and hashlib.sha256(open(CARD, "rb").read()[:CARD_PREFIX_BYTES]).hexdigest() == CARD_PREFIX_SHA, "card prefix changed"
    impl = os.path.join(HERE, "IMPL_CHOICES_VOE1.md")
    impl_sha = sha_file(impl)
    assert open(os.path.join(HERE, "IMPL_CHOICES_VOE1.sha256")).read().split()[0] == impl_sha, "IMPL_CHOICES changed after its hash was recorded"
    bank_p = os.path.join(HERE, "bank_voe1.pt")
    Bk = torch.load(bank_p, weights_only=False)
    meta = {"card_prefix_sha256": CARD_PREFIX_SHA, "card_file_sha256": sha_file(CARD), "impl_sha256": impl_sha, "bank_sha256": sha_file(bank_p),
            "code_sha256": {f: sha_file(os.path.join(HERE, f)) for f in ("voe1_core.py", "voe1_real.py", "build_bank_voe1.py", "run_voe1.py", "test_voe1_synth.py")},
            "latents_pt_sha256": sha_file(os.path.join(HERE, "..", "wmact", "latents.pt")), "wmact_lib_sha256": sha_file(os.path.join(HERE, "..", "wmact", "wmact_lib.py")),
            "tanitad_file": W.tanitad.__file__, "workers": workers, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    chk = Bk["check"]
    assert chk["M2b_s0"]["max_abs_diff_z1"] < 1e-4 and chk["M2b_s1"]["max_abs_diff_z1"] < 1e-4 and chk["actions_max_abs_diff"] == 0.0, chk

    # ---- models, deterministic random-init controls (hash-checked against the bank build)
    trained = R.load_trained()
    wms = R.build_wms(trained)
    rep = R.reinit_report(trained, wms)
    for tag in R.CONTROL + R.BESIDE:
        assert rep[tag]["wm_state_sha256"] == Bk["reinit"][tag]["wm_state_sha256"], tag
    log("models built; random-init hashes equal the bank build's")

    # ---- windows
    w = Bk["win"]
    keep = np.asarray(w["keep"])
    eid = np.array(w["eid"])[keep]
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

    # ---- the manipulation rows, per tag
    res = {"meta": meta, "repro": chk, "reinit": rep, "window_table": {k: {"n_windows": int(q.sum()), "n_episodes": int(len(set(eid[q])))} for k, q in wt["q"].items()},
           "tags": {}, "events": {}}
    models, bank_z, bank_a = {}, {}, {}
    act_t = Bk["act"].float()
    traces_out = {}
    for tag in R.TAGS:
        wm = wms[tag]
        models[tag] = C.HJEPATraces(wm.stack)
        bank_z[tag] = Bk["z1"][tag].float()
        with torch.no_grad():
            bank_a[tag] = wm.scale_actions(act_t)
        t = time.time()
        r = C.eval_tag(models[tag], bank_z[tag], bank_a[tag], base, eid, wt, tag, cal=True, keep_traces=tag in R.DECIDING)
        tr = r.pop("_traces")
        if tr is not None:
            for name, d in tr.items():
                for k, v in d.items():
                    traces_out[f"{tag}/{name}/{k}"] = np.asarray(v, np.float32) if k != "q" else np.asarray(v, bool)
        res["tags"][tag] = r
        log(f"{tag}: manipulation rows done ({time.time() - t:.0f}s)")

    # ---- event alignment: unmodified 16-frame windows
    clip_lo = np.array([Bk["off"][e] for e in eid])
    clip_hi = clip_lo + T_clip
    fr16 = base[:, None] + np.arange(16)[None]
    ac16 = base[:, None] + np.arange(15)[None]
    Yev = {}
    for ev, key in zip(C.EVENTS, ("ev_brake", "ev_turn", "ev_lead")):
        Yev[ev] = {l: C.event_step_labels(Bk[key], base, clip_lo, clip_hi, l, {1: 15, 2: 7, 3: 3}[l]) for l in (1, 2, 3)}
    tasks, tr16 = [], {}
    for tag in R.TAGS:
        tr16[tag] = models[tag].traces(bank_z[tag][torch.from_numpy(fr16)], bank_a[tag][torch.from_numpy(ac16)])
        for ev in C.EVENTS:
            for l in (1, 2, 3):
                tasks.append((f"{tag}/{ev}/L{l}", tr16[tag][l]["sur"], tr16[tag][l]["null"], Yev[ev][l], ep, n_ep, tag in R.DECIDING))
    tasks.sort(key=lambda a: (-int(a[6]), a[1].shape[1] * -1))            # deciding cells first, level 1 first
    log(f"event cells: {len(tasks)} ({sum(1 for a in tasks if a[6])} with bootstrap and calibration); workers {workers}")
    if workers > 1:
        import multiprocessing as mp
        os.environ["OMP_NUM_THREADS"] = "1"
        with mp.get_context("spawn").Pool(workers) as pool:
            out = dict(pool.imap_unordered(ap_task, tasks))
    else:
        out = dict(ap_task(a) for a in tasks)
    for key, r in out.items():
        tag, ev, lv = key.split("/")
        res["events"].setdefault(tag, {}).setdefault(ev, {})[lv] = r
    res["event_labels"] = {ev: {str(l): {"n_points": int(Y.size), "positives": int(Y.sum()), "rate": float(Y.mean())} for l, Y in d.items()} for ev, d in Yev.items()}
    res["event_onsets_in_bank"] = {ev: int(Bk[k].sum()) for ev, k in zip(C.EVENTS, ("ev_brake", "ev_turn", "ev_lead"))}
    log("event cells done")

    # ---- two clips, whole-clip one-step surprise (figure)
    counts = json.load(open(os.path.join(HERE, "counts_voe1.json"), encoding="utf-8"))
    res["along_clip"] = {}
    for clip in counts["figure_clips"]:
        o, T = Bk["off"][clip], Bk["T"][clip]
        res["along_clip"][clip] = {"T": int(T), "onsets": {ev: np.flatnonzero(Bk[k][o:o + T]).tolist() for ev, k in zip(C.EVENTS, ("ev_brake", "ev_turn", "ev_lead"))}}
        for tag in R.DECIDING:
            tr = models[tag].traces(bank_z[tag][o:o + T][None], bank_a[tag][o:o + T - 1][None])
            med = res["tags"][tag]["median"]
            for l in (1, 2, 3):
                traces_out[f"clip/{clip}/{tag}/L{l}/sur"] = (tr[l]["sur"][0] / med[str(l)]["sur"]).astype(np.float32)
                traces_out[f"clip/{clip}/{tag}/L{l}/null"] = (tr[l]["null"][0] / med[str(l)]["null"]).astype(np.float32)
        traces_out[f"clip/{clip}/speed"] = spd[o:o + T].astype(np.float32)
        traces_out[f"clip/{clip}/act"] = act[o:o + T].astype(np.float32)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["seconds"] = round(time.time() - t_start, 1)
    np.savez_compressed(OUT_NPZ, **traces_out)
    json.dump(res, open(OUT_JSON, "w", encoding="utf-8", newline="\n"), default=jdefault)
    log(f"wrote {OUT_JSON} ({os.path.getsize(OUT_JSON)} bytes) and {OUT_NPZ}")


if __name__ == "__main__":
    nw = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    rc = 0
    try:
        main(nw)
    except BaseException:
        traceback.print_exc()
        rc = 1
    open(DONE, "w").write(f"rc={rc}\n")
    sys.exit(rc)
