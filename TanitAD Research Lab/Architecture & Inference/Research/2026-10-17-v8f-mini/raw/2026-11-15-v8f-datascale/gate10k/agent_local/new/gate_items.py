"""DSCALE10K block-1 gate items (pod side). CPU only (models are built on the CPU), so it can run beside a GPU job.  Written 2026-11-17 by the gate agent.

usage: gate_items.py <item> [...]   items: clock | eval | ckptproc | suite | sess | init | cachecheck
Every item writes /root/dscale10k/gate/<item>.json = {"item", "pass", "problems", "record": {...}, "regression": [{"arm", "red": bool, "problems": [...]}]}.
Each ``*_problems`` function returns [] for PASS; each has a deliberate-regression arm run in the same breath that must return a non-empty list ("red": true).
"""
import copy
import hashlib
import io
import json
import math
import os
import sys
import time

D = "/root/dscale10k"
TREE = D + "/tree"
for p in (TREE + "/stack", TREE + "/stack/scripts", TREE, D + "/study"):
    if p not in sys.path:
        sys.path.insert(0, p)
G = D + "/gate"
os.makedirs(G, exist_ok=True)
import numpy as np  # noqa: E402

import dscale10k_check as C  # noqa: E402
import dscale10k_merge as M  # noqa: E402


def save(item, rec, problems, regression):
    out = {"item": item, "pass": not problems and all(r["red"] for r in regression), "problems": problems, "record": rec, "regression": regression,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(out, open(f"{G}/{item}.json", "w"), indent=1, default=str)
    print(item, "PASS" if out["pass"] else "FAIL", "problems", problems[:3], "regression red", [r["red"] for r in regression], flush=True)
    return out


def reg(arm, problems):
    return {"arm": arm, "red": bool(problems), "problems": list(problems)[:3]}


# ------------------------------------------------------------------------------------------------ G-CLOCK residuals
def clock_residuals(rel_rows, tab, dt_pose, shift_s=0.0, only=None):
    """per clip: max over rows |now_s - (grid_start_s + k * dt_pose)| (+ shift). ``dt_pose`` = the cache's own step (clip_clock.pose_dt); clips without one are skipped.
    ``grid_start_s`` is NOT recoverable from the cache (clip_clock.py:28), so this residual measures the step error accumulated over the clip, not the offset."""
    res = {}
    for sid, (r0, n, g0, dt_rel) in tab.items():
        if only is not None and sid not in only:
            continue
        d = dt_pose.get(sid)
        if d is None:
            continue
        now, k = rel_rows["now_s"][r0:r0 + n], rel_rows["k"][r0:r0 + n].astype(np.float64)
        res[sid] = float(np.max(np.abs(now + shift_s - (g0 + k * d))))
    return res


def grid_dt(pg, v_min=2.0, min_steps=10):
    """The cache's ``pose_grid`` is the EVEN raw rows (a 0.2 s step; v8f_dagger_bank constants "0.2 s pose grid"), so clip_clock.pose_dt (band 0.098-0.104) returns None on it. Same identity
    (sum |dxy| / sum mean v over moving steps), band doubled, halved to the label row step. None when the clip has too little motion or leaves the band."""
    d = np.hypot(pg[1:, 0] - pg[:-1, 0], pg[1:, 1] - pg[:-1, 1])
    vb = 0.5 * (pg[1:, 3] + pg[:-1, 3])
    m = (pg[1:, 3] > v_min) & (pg[:-1, 3] > v_min)
    if int(m.sum()) < min_steps:
        return None
    dt = float(d[m].sum() / vb[m].sum())
    return dt / 2.0 if 2 * 0.098 <= dt <= 2 * 0.104 else None


def item_clock():
    import torch
    from concurrent.futures import ThreadPoolExecutor
    z = np.load(f"{D}/merged/v9_labels_union.npz", allow_pickle=True)
    rows = {"now_s": z["row__now_s"], "k": z["row__k"]}
    cl = {k[6:]: z[k] for k in z.files if k.startswith("clip__")}
    sid2clip = {}
    tenk = {json.loads(l)["clip"] for l in open(f"{D}/in/manifest_10k.jsonl") if l.strip()}
    man = [json.loads(l) for l in open(f"{D}/merged/manifest.jsonl") if l.strip()]
    for r in man:
        sid2clip[M.stable_episode_id(r["clip"])] = r["clip"]
    tab = {int(s): (int(r0), int(n), float(g0), float(dt)) for s, r0, n, g0, dt in zip(cl["sid"], cl["row0"], cl["n_rows"], cl["grid_start_s"], cl["dt_s"])}
    src = {int(s): int(c) for s, c in zip(cl["sid"], cl["clock_src"])}
    view = "/root/data/cache_union_dscale/train/clips"

    def one(sid):
        o = torch.load(f"{view}/{sid2clip[sid]}.pt", map_location="cpu", weights_only=True, mmap=True)
        pg = o["pose_grid"].numpy().astype(np.float64)
        del o
        return sid, grid_dt(pg)
    with ThreadPoolExecutor(8) as ex:
        dtp = dict(ex.map(one, list(tab)))
    ten_sids = {s for s in tab if sid2clip[s] in tenk}
    cor_sids = set(tab) - ten_sids
    res = clock_residuals(rows, tab, dtp)
    a = np.array([v for s, v in res.items() if s in ten_sids])
    b = np.array([v for s, v in res.items() if s in cor_sids])
    probs = C.clock_problems(a, b)
    by_src = {}
    for nm, ids in (("tenk", ten_sids), ("corpus", cor_sids)):
        for c in (0, 1, 2):
            v = np.array([res[s] for s in ids if s in res and src[s] == c])
            by_src[f"{nm}.clock_src{c}"] = {"n": int(len(v)), "p99_s": float(np.percentile(v, 99)) if len(v) else None, "max_s": float(v.max()) if len(v) else None}
    rec = {"residual_def": "max over the clip's rows of |now_s - (grid_start_s + k * dt_pose)|, dt_pose = grid_dt(cache pose_grid); clips with no pose_dt (too little motion) are skipped",
           "limitation": "grid_start_s is not recoverable from the cache (stack/tanitad/data/clip_clock.py:28): an offset error of a fallback-clock clip is invisible to this residual",
           "n_tenk": int(len(a)), "n_corpus": int(len(b)), "n_tenk_no_pose_dt": int(len(ten_sids) - len(a)), "n_corpus_no_pose_dt": int(len(cor_sids) - len(b)),
           "tenk_p99_s": float(np.percentile(a, 99)), "corpus_p99_s": float(np.percentile(b, 99)), "tenk_max_s": float(a.max()), "corpus_max_s": float(b.max()),
           "tenk_p50_s": float(np.median(a)), "corpus_p50_s": float(np.median(b)), "by_clock_src": by_src,
           "clock_src_counts": {"tenk": {str(c): int(sum(1 for s in ten_sids if src[s] == c)) for c in (0, 1, 2)}, "corpus": {str(c): int(sum(1 for s in cor_sids if src[s] == c)) for c in (0, 1, 2)}},
           "tenk": [float(x) for x in a], "corpus": [float(x) for x in b]}
    res_shift = clock_residuals(rows, tab, dtp, shift_s=0.2, only=ten_sids)
    a2 = np.array(list(res_shift.values()))
    regs = [reg("tenk_clock_shifted_+0.2s", C.clock_problems(a2, b))]
    rec["regression_tenk_p99_shifted_s"] = float(np.percentile(a2, 99))
    rec["regression_needs_corpus_p99_below_0.2s"] = bool(rec["corpus_p99_s"] < 0.2)
    if not rec["regression_needs_corpus_p99_below_0.2s"]:
        probs.append("the corpus p99 is not below the 0.2 s regression shift: the check cannot discriminate")
    return save("clock", rec, probs, regs)


# ------------------------------------------------------------------------------------------------ G-EVAL
def eval_problems(rec):
    bad = []
    if not rec.get("strict_load_ok"):
        bad.append(f"strict load_state_dict failed: {rec.get('strict_load_error')}")
    if not rec.get("cfg_equal_built"):
        bad.append("the eval loader's cfg differs from the trainer's built cfg")
    if rec.get("n_params") != C.BASE_PARAMS:
        bad.append(f"eval model has {rec.get('n_params')} parameters, expected {C.BASE_PARAMS}")
    if rec.get("n_state_mismatch_after_load") != 0:
        bad.append(f"{rec.get('n_state_mismatch_after_load')} tensors differ after load")
    if rec.get("n_windows") != 2399 or rec.get("n_episodes") != 141:
        bad.append(f"eval grid reads {rec.get('n_windows')} windows / {rec.get('n_episodes')} episodes, expected 2399 / 141")
    if rec.get("grid_sha256") != rec.get("obd_ctl_grid_sha256"):
        bad.append("the eval grid bytes differ from the grid recorded in the OBD-CTL bank of record")
    return bad


def eval_load(ckpt_path, built_cfg):
    import torch
    from tanitad.models import v8f_mini as MM
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = MM.V8FMiniConfig(**ck["cfg"])
    model = MM.V8FMini(cfg, MM.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * MM.K30)))      # the T1 eval's own construction (v8f_t1_eval.py:510)
    rec = {"ckpt": ckpt_path}
    try:
        model.load_state_dict(ck["model"])                                                             # strict (default), v8f_t1_eval.py:511
        rec["strict_load_ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["strict_load_ok"], rec["strict_load_error"] = False, str(e)[:200]
    import dataclasses
    rec["cfg_equal_built"] = json.loads(json.dumps(dataclasses.asdict(cfg), default=str)) == json.loads(json.dumps(built_cfg, default=str))
    rec["n_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    sd = model.state_dict()
    rec["n_state_mismatch_after_load"] = int(sum(1 for k, v in ck["model"].items() if k not in sd or not bool((sd[k] == v).all())))
    return rec, ck


def item_eval():
    grid_p = "/root/strat1/eval_grid_windows.json"
    gsha = hashlib.sha256(open(grid_p, "rb").read()).hexdigest()
    grid = json.load(open(grid_p))
    obd = json.load(open("/root/strat1/inputs/dagger_bank/s0/bank.json"))["common"]["inputs_sha256"]["eval_grid"]
    out, probs, regs = {}, [], []
    for tag, nm in (("B", "B_s0.A"), ("U", "U_s0.A")):
        built = json.load(open(f"{D}/probe/{nm}.json"))["arm"]["cfg"]
        rec, ck = eval_load(f"{D}/pilot/p{tag}/ckpt_last.pt", built)
        rec.update({"n_windows": int(grid["n_windows"]), "n_episodes": len(grid["order"]), "grid_sha256": gsha, "obd_ctl_grid_sha256": obd,
                    "windows_len_check": int(sum(len(v) if hasattr(v, "__len__") else 1 for v in grid["windows"].values())) if isinstance(grid["windows"], dict) else len(grid["windows"])})
        out[tag] = rec
        probs += [f"{tag}: {x}" for x in eval_problems(rec)]
        if tag == "B":
            bad = copy.deepcopy(rec)
            bad["cfg_equal_built"] = False
            regs.append(reg("eval_cfg_differs", eval_problems(bad)))
            bad = copy.deepcopy(rec)
            bad["n_windows"] = 2398
            bad["grid_sha256"] = "0" * 64
            regs.append(reg("eval_grid_one_window_dropped", eval_problems(bad)))
            # a real strict-load regression: a checkpoint state dict with one tensor key removed
            import torch
            from tanitad.models import v8f_mini as MM
            ck2 = copy.copy(ck)
            ck2["model"] = {k: v for i, (k, v) in enumerate(ck["model"].items()) if i != 3}
            cfg = MM.V8FMiniConfig(**ck["cfg"])
            m2 = MM.V8FMini(cfg, MM.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * MM.K30)))
            try:
                m2.load_state_dict(ck2["model"])
                ok = True
            except Exception:  # noqa: BLE001
                ok = False
            regs.append(reg("state_dict_missing_one_key", [] if ok else ["strict load refused a state dict with a missing key"]))
    return save("eval", out, probs, regs)


# ------------------------------------------------------------------------------------------------ G-CKPT, in-process half
def state_problems(a, b, cfg_a, cfg_b):
    bad = []
    if set(a) != set(b):
        bad.append("state dict key sets differ")
    for k in a:
        if k in b and not (a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and bool((a[k] == b[k]).all())):
            bad.append(f"tensor {k} differs after save/load")
    if cfg_a != cfg_b:
        bad.append("cfg differs after save/load")
    return bad


def item_ckptproc():
    import torch
    out, probs, regs = {}, [], []
    for tag in ("B", "U"):
        p = f"{D}/pilot/p{tag}/ckpt_last.pt"
        ck = torch.load(p, map_location="cpu", weights_only=False)
        tmp = f"{G}/ckpt_roundtrip_{tag}.pt"
        torch.save(ck, tmp)
        ck2 = torch.load(tmp, map_location="cpu", weights_only=False)
        pr = state_problems(ck["model"], ck2["model"], ck["cfg"], ck2["cfg"])
        out[tag] = {"ckpt": p, "n_tensors": len(ck["model"]), "step": ck.get("step"), "keys": sorted(k for k in ck if k != "model")[:12], "problems": pr}
        probs += [f"{tag}: {x}" for x in pr]
        if tag == "B":
            sd = {k: v.clone() for k, v in ck2["model"].items()}
            k0 = next(k for k, v in sd.items() if v.dtype.is_floating_point and v.numel() > 1)
            sd[k0].view(-1)[0] += 1e-6
            regs.append(reg("one_element_changed_after_load", state_problems(ck["model"], sd, ck["cfg"], ck2["cfg"])))
            cf = dict(ck2["cfg"])
            cf["d"] = int(cf["d"]) + 1
            regs.append(reg("cfg_changed_after_load", state_problems(ck["model"], ck2["model"], ck["cfg"], cf)))
        os.remove(tmp)                                                                                  # own temporary data
    return save("ckptproc", out, probs, regs)


# ------------------------------------------------------------------------------------------------ G-SUITE (identity of the tree with the base whose full suite is banked)
def suite_problems(tree_sha, base_sha, base_counts):
    bad = []
    if tree_sha != base_sha:
        bad.append("the study tree's manifest sha differs from /root/strat1/base, whose junit is the baseline: a full suite run is needed")
    if not base_counts or base_counts.get("n_tests", 0) < 14000:
        bad.append("the baseline junit counts are missing")
    return bad


def item_suite():
    t = C.tree_manifest_sha(TREE)
    b = C.tree_manifest_sha("/root/strat1/base")
    cmp_ = json.load(open("/root/strat1/gsuite_result6.json"))
    counts = cmp_["base"]
    probs = suite_problems(t, b, counts)
    rec = {"mode": "identity", "tree": TREE, "tree_manifest_sha256": t, "base": "/root/strat1/base", "base_manifest_sha256": b, "base_counts": counts, "n_regressions": 0 if not probs else None,
           "basis": "the study tree is a byte-identical export of the code of record (cdbff971) = /root/strat1/base; a suite run of identical code against its own baseline reads 0 regressions by identity",
           "baseline_junit": "/root/strat1/junit/base/ (parsed here for counts: gsuite_result6.json base block, written by strat1_check.py suite)",
           "study_tests": open(f"{D}/ptest.log").read()[-400:] if os.path.exists(f"{D}/ptest.log") else None}
    regs = [reg("tree_manifest_differs", suite_problems("0" * 64, b, counts))]
    return save("suite", rec, probs, regs)


# ------------------------------------------------------------------------------------------------ C-SESS (report only)
def item_sess():
    import pyarrow.parquet as pq
    t = pq.read_table(f"{D}/in/labels/context/clips_datascale10k.parquet").to_pylist()
    keys = {r["session_key"] for r in t}
    ci = {r["clip_id"]: r["chunk"] for r in pq.read_table("/root/datascale/meta/clip_index.parquet").to_pylist()}
    ev = json.load(open("/root/strat1/eval_grid_windows.json"))["order"]
    evc = {ci[e] for e in ev if e in ci}
    shared = sum(1 for r in t if r["chunk"] in evc)
    rec = {"session_key_values_in_10k": sorted(map(str, keys))[:5], "n_distinct_session_keys": len(keys), "computable": bool(keys - {"NA", None}),
           "eval141_in_clip_index": int(sum(1 for e in ev if e in ci)), "clip_index_columns": ["clip_is_valid", "chunk", "clip_id"],
           "report_only_proxy": {"eval_chunks": len(evc), "tenk_clips_sharing_an_eval_chunk": int(shared), "note": "a chunk is a storage shard of the corpus, not a recording session; reported only as a proxy"}}
    rec["verdict"] = "not computable: session_key is NA on all 10,000 clips and clip_index.parquet has no session or drive key for the eval-141 clips" if not rec["computable"] else "computable"
    return save("sess", rec, [], [])


# ------------------------------------------------------------------------------------------------ C-INIT (and G-HYG record of B_rep): the trainer's own order, CPU
def build_init(argv_path, tag):
    import dataclasses
    import torch
    import v8f_mini_train as T
    from tanitad.models import v8f_mini as MM
    argv = [t for t in json.load(open(argv_path))["argv"] if t != "--fit-join"]
    a = T.build_parser().parse_args([str(t) for t in argv])
    a.out = f"{D}/tmp/init_{tag}"
    os.makedirs(a.out, exist_ok=True)
    cfg = T.make_config(a)
    torch.manual_seed(a.seed)                                                                           # v8f_mini_train.py:3248-3249
    np.random.seed(a.seed)
    store, J, wins, clips, info = T.build_data(a, cfg)                                                  # :3251
    prior = MM.load_p10(a.p10)
    mu, sd = MM.cons_stats(J)
    model = MM.V8FMini(cfg, prior, mu, sd)                                                              # :3290 (CPU here)
    n_tr = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    params = {k: v for k, v in model.named_parameters()}
    return {"argv": argv_path, "seed": int(a.seed), "n_clips": int(info["n_clips"]), "train_sha256": info["train_sha256"], "n_trainable": n_tr,
            "cfg": json.loads(json.dumps(dataclasses.asdict(cfg), default=str)), "state_sha256": C.state_dict_sha256(model.state_dict()),
            "params_only_sha256": C.state_dict_sha256(params), "buffers_sha256": C.state_dict_sha256(dict(model.named_buffers()))}


def item_init():
    out = {}
    for tag, nm in (("B_s0", "B_s0.A"), ("U_s0", "U_s0.A"), ("B_rep", "B_rep.A")):
        out[tag] = build_init(f"{D}/argv/{nm}.json", tag)
        print(tag, out[tag]["n_trainable"], out[tag]["params_only_sha256"][:12], flush=True)
    probs = []
    hy = C.hyg_problems(out["B_rep"], out["B_s0"]) + C.hyg_problems(out["U_s0"], out["B_s0"])
    probs += hy
    out["c_init_report"] = {"B_s0_vs_U_s0_params_equal": out["B_s0"]["params_only_sha256"] == out["U_s0"]["params_only_sha256"],
                            "B_s0_vs_U_s0_state_equal": out["B_s0"]["state_sha256"] == out["U_s0"]["state_sha256"],
                            "B_s0_vs_U_s0_buffers_equal": out["B_s0"]["buffers_sha256"] == out["U_s0"]["buffers_sha256"],
                            "B_rep_vs_B_s0_params_equal": out["B_rep"]["params_only_sha256"] == out["B_s0"]["params_only_sha256"],
                            "reading": "params_equal true = the same step-0 weights (the lever is data alone); false = Delta is read as data plus init (card 2.3)"}
    bad = copy.deepcopy(out["U_s0"])
    bad["n_trainable"] += 1
    regs = [reg("hyg_param_count_differs", C.hyg_problems(bad, out["B_s0"]))]
    return save("init", out, probs, regs)


def item_cachecheck():
    r = json.load(open(f"{G}/cache_walk.json"))
    exp = {"clip%d" % i: "m%d" % i for i in range(5)}
    act = dict(exp)
    act["clip3"] = "x"
    regs = [reg("one_md5_altered", C.cache_problems(exp, act))]
    probs = [] if r["pass"] else [f"{len(r['mismatches'])} mismatches / {len(r['absent'])} absent"]
    return save("cachecheck", {k: v for k, v in r.items() if k not in ("mismatches", "absent")}, probs, regs)


if __name__ == "__main__":
    for it in sys.argv[1:]:
        globals()["item_" + it]()
