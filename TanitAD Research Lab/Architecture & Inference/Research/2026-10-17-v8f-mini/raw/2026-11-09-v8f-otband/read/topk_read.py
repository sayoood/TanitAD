"""topk_read.py -- the "trajectories, plural" beside-read of PREREG_V8F_OTBAND (section Read, "Beside the families (no bar)"; "What each outcome triggers", Top-K).  CPU, zero training.

For each arm: build every eval window's FIRST-TICK batch exactly as stack/scripts/v8f_t1_eval.py rollout() does at tick 0 (same make_states / ages_for / build_item / collate / MODEL_KEYS,
flags read from the dump's argv file), call model.forward_topk(batch, K=3), and score every one of the K plans (and forward_infer's own plan, traj30_top1) against the logged GT.
Rows per band (operative knots 0-9, tactical 10-29, full 0-29; read_lib.BANDS): top-1 ADE, plan-k0 ADE (O decoded on the hard top-ranked (lat, lon) pair), best-of-K ADE (per-window minimum
over the K plans), best-of-K minus top-1 (a gain when negative).  Then the share of windows whose GT manoeuvre pair is among the K codes.
GT manoeuvre label source: the v9 label join row of the window's T tick (batch["lab_t"]: lat_cls = lat_cls_a, lon_cls = lon_cls; the very classes T's lat/lon heads are trained on and topk_codes ranks);
label row = NOW row (tick 0 has T age 0).  Primary hit: lat_cls and lon_cls both labelled and the (lat, lon) pair equals one of the K codes.  Secondary hit: some code's lat is in lat_al and its lon is in lon_al
(the partial-label allowed sets T is trained against).
Controls: (1) max |traj30_top1 - dump ol| per arm; (2) K distinctness.  Estimator: taniteval.ci episode-cluster bootstrap (single and paired), B = 2000, seed 0.  It answers "another draw of episodes" only.
Usage: topk_read.py --tags OB-CTL_s0 OB-BAND_s0 [...] --out <json> [--smoke N]
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TREE = "F:/TanitAD-scratch/otband_read/tree"
T1 = "F:/TanitAD-scratch/otband_read/t1"
PULL = "C:/Users/Admin/t2v8f_pulls"
import numpy as np  # noqa: E402
import torch  # noqa: E402
import read_lib as BL  # noqa: E402  (puts TREE/stack and TREE/taniteval first on sys.path)
from read_lib import CI, BANDS  # noqa: E402

torch.set_num_threads(2)
import tanitad  # noqa: E402

assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(TREE), tanitad.__file__
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402

K = 3
NB, SEED = 2000, 0
BAND_NAMES = list(BANDS)


def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("l", ctypes.c_ulong), ("ml", ctypes.c_ulong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong), ("tpf", ctypes.c_ulonglong),
                    ("apf", ctypes.c_ulonglong), ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    s = MS()
    s.l = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ap / 2 ** 30


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_eval_module():
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_tree", TREE + "/stack/scripts/v8f_t1_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert os.path.abspath(mod.REPO).replace("\\", "/") == TREE, mod.REPO
    return mod


def parse_argv(path):
    toks = [t.strip() for t in open(path, encoding="utf-8").read().splitlines() if t.strip()]
    i = toks.index("run")
    a, j = {}, i + 1
    while j < len(toks):
        if toks[j].startswith("--"):
            if j + 1 < len(toks) and not toks[j + 1].startswith("--"):
                a[toks[j][2:]] = toks[j + 1]
                j += 2
            else:
                a[toks[j][2:]] = True
                j += 1
        else:
            j += 1
    return a


def tag_paths(tag):
    if tag.startswith("PARITY_"):          # method control on the parity dump: ckpt from that dump's argv file; the done marker is not required (checked by the dump's own log)
        c = parse_argv(f"{T1}/argv_{tag}_m10_causal_heads.txt")["ckpt"]
        return c, f"{T1}/dump_{tag}_m10_causal_heads", f"{T1}/log_{tag}_m10_causal_heads.txt"
    return f"{PULL}/otband_{tag}/ckpt_last.pt", f"{T1}/dump_{tag}_m10_causal_heads", f"{T1}/done_{tag}_m10_causal_heads.txt"


def run_arm(E, A, tag, smoke):
    ck_path, ddir, done = tag_paths(tag)
    got = sha256(ck_path)
    if os.path.exists(os.path.dirname(ck_path) + "/BANK_MANIFEST.json"):
        want = json.load(open(os.path.dirname(ck_path) + "/BANK_MANIFEST.json"))["files"]["ckpt_last.pt"]["sha256"]
        assert want == got, (tag, want, got)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = M.V8FMiniConfig(**ck["cfg"])
    model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
    model.load_state_dict(ck["model"])
    model.eval()
    tracks = D.load_native_tracks(A["tracks"])
    J = D.load_join(A["eval-join"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid=ck.get("grid", "pool4x10"), kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    if smoke:
        wins = [wins[i] for i in np.linspace(0, len(wins) - 1, smoke).round().astype(int)]
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    N = len(wins)
    out = {"traj": np.zeros((N, K, 30, 2)), "top1": np.zeros((N, 30, 2)), "prob": np.zeros((N, K)), "code_lat": np.zeros((N, K), np.int64),
           "code_lon": np.zeros((N, K), np.int64), "g": np.zeros((N, 30, 2)), "lat_cls": np.zeros(N, np.int64), "lon_cls": np.zeros(N, np.int64),
           "lat_al": np.zeros((N, 5), bool), "lon_al": np.zeros((N, 7), bool), "has": np.zeros(N, bool), "ci": np.zeros(N, np.int64),
           "ws": np.zeros(N, np.int64), "fresh_infer_diff": np.zeros(N)}
    a0, b0 = M.N_GOAL, M.N_GOAL + M.N_LAT
    t0 = time.time()
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        for _ in range(240):                 # a chunk waits (up to 20 min) while the other jobs hold memory
            if free_gb() >= 2.5:
                break
            time.sleep(5)
        assert free_gb() >= 1.5, "free physical memory below 1.5 GB: stop"
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        with torch.no_grad():
            r = model.forward_topk(batch, K)
            codes, _ = M.topk_codes(r["code"], K)
            fresh = model.forward_infer(batch)["traj30"]
        sl = slice(s, s + len(chunk))
        out["traj"][sl] = r["traj30"].float().numpy().astype(np.float64)
        out["top1"][sl] = r["traj30_top1"].float().numpy().astype(np.float64)
        out["fresh_infer_diff"][sl] = (fresh - r["traj30_top1"]).abs().amax((1, 2)).double().numpy()
        out["prob"][sl] = r["prob"].numpy()
        out["code_lat"][sl] = torch.stack([c[:, a0:b0].argmax(1) for c in codes], 1).numpy()
        out["code_lon"][sl] = torch.stack([c[:, b0:b0 + M.N_LON].argmax(1) for c in codes], 1).numpy()
        lt = full["lab_t"]
        out["lat_cls"][sl], out["lon_cls"][sl] = lt["lat_cls"].numpy(), lt["lon_cls"].numpy()
        out["lat_al"][sl], out["lon_al"][sl], out["has"][sl] = lt["lat_al"].numpy(), lt["lon_al"].numpy(), lt["has"].numpy()
        for i, (ci, tt, _, _) in enumerate(chunk):
            out["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            out["ci"][s + i], out["ws"][s + i] = ci, tt
        print(f"  {tag} [{s + len(chunk)}/{N}] {time.time() - t0:.0f}s free {free_gb():.1f} GB", flush=True)
    out["ckpt_sha256"] = got
    return out


def paired(a, b, eid):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.paired_episode_cluster_bootstrap(a[m], b[m], eid[m], n_boot=NB, seed=SEED)
    return {"delta": float(r["delta"]), "lo": float(r["lo"]), "hi": float(r["hi"]), "separated": bool(r["separated"]),
            "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def single(a, eid):
    a = np.asarray(a, np.float64)
    m = np.isfinite(a)
    if not m.any():
        return {"status": "NOT-APPLICABLE", "n_windows": 0}
    r = CI.episode_cluster_bootstrap(a[m], eid[m], reduce="mean", n_boot=NB, seed=SEED)
    return {"mean": float(a[m].mean()), "lo": float(r["lo"]), "hi": float(r["hi"]), "n_windows": int(m.sum()), "n_episodes": int(len(np.unique(eid[m])))}


def score(o):
    """Per-window vectors. ade[band][name] with name in top1, k0, best3."""
    res = {"band": {}, "hit": {}}
    for bn in BAND_NAMES:
        a, b = BANDS[bn]
        t1 = BL.band_rows(o["top1"], o["g"], a, b)["ade_m"]
        per_k = np.stack([BL.band_rows(o["traj"][:, k], o["g"], a, b)["ade_m"] for k in range(K)], 1)
        res["band"][bn] = {"top1": t1, "k0": per_k[:, 0], "best3": np.nanmin(per_k, 1), "per_k": per_k}
    lab = o["has"] & (o["lat_cls"] >= 0) & (o["lon_cls"] >= 0)
    match = (o["code_lat"] == o["lat_cls"][:, None]) & (o["code_lon"] == o["lon_cls"][:, None])
    nan = np.full(len(lab), np.nan)
    res["hit"]["pair_top1"] = np.where(lab, match[:, 0].astype(float), nan)
    res["hit"]["pair_topK"] = np.where(lab, match.any(1).astype(float), nan)
    al = np.take_along_axis(o["lat_al"], o["code_lat"], 1) & np.take_along_axis(o["lon_al"], o["code_lon"], 1)
    lab2 = o["has"] & (o["lat_al"].any(1)) & (o["lon_al"].any(1))
    res["hit"]["allowed_top1"] = np.where(lab2, al[:, 0].astype(float), nan)
    res["hit"]["allowed_topK"] = np.where(lab2, al.any(1).astype(float), nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", type=int, default=0, help="N evenly spaced windows, no dump needed (controls against fresh forward_infer only)")
    a = ap.parse_args()
    E = load_eval_module()
    res = {"tool": "topk_read.py", "K": K, "bands": {b: list(BANDS[b]) for b in BAND_NAMES}, "estimator": f"episode-cluster bootstrap, B={NB}, seed {SEED}, answers 'another draw of episodes' only",
           "tree": TREE, "smoke": a.smoke, "arms": {}, "controls": {}, "paired_arms": {}, "not_run": {}}
    store_vec = {}
    for tag in a.tags:
        ck_path, ddir, done = tag_paths(tag)
        if not os.path.exists(ck_path):
            res["not_run"][tag] = "checkpoint folder absent"
            continue
        if not a.smoke and not ((os.path.exists(done) or os.path.exists(f"{T1}/t1v8f_{tag}_m10_causal_heads.json")) and os.path.isdir(ddir)):
            res["not_run"][tag] = "dump not complete (no done marker)"
            continue
        for _ in range(120):                 # wait up to 20 min for the other jobs to leave 4 GB of physical memory
            if free_gb() >= 4.0:
                break
            time.sleep(10)
        assert free_gb() >= 3.0, f"free physical memory {free_gb():.1f} GB"
        A = parse_argv(f"{T1}/argv_{tag}_m10_causal_heads.txt") if os.path.exists(f"{T1}/argv_{tag}_m10_causal_heads.txt") else parse_argv(f"{T1}/argv_PARITY_DAGOACC_s0_m10_causal_heads.txt")
        print(f"arm {tag}", flush=True)
        o = run_arm(E, A, tag, a.smoke)
        ctl = {"ckpt_sha256_matches_manifest": o["ckpt_sha256"], "max_abs_forward_infer_repeat_diff_m": float(o["fresh_infer_diff"].max())}
        # control 2: K distinctness
        d01, d02, d12 = [np.abs(o["traj"][:, i] - o["traj"][:, j]).max((1, 2)) for i, j in ((0, 1), (0, 2), (1, 2))]
        mx = np.maximum(np.maximum(d01, d02), d12)
        mn = np.minimum(np.minimum(d01, d02), d12)
        ctl["k_distinct"] = {"threshold_m": 1e-4, "share_any_pair_differs": float((mx > 1e-4).mean()), "share_all_three_pairs_differ": float((mn > 1e-4).mean()),
                             "share_identical_plans": float((mx <= 1e-4).mean()), "median_max_pair_diff_m": float(np.median(mx)),
                             "share_codes_distinct_pairs": float(((o["code_lat"] * 7 + o["code_lon"])[:, 0] != (o["code_lat"] * 7 + o["code_lon"])[:, 1]).mean())}
        sc = score(o)
        # control 1: against the dump
        if not a.smoke:
            Z = BL.load_dir(ddir)
            key_d = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(Z["clip_index"], Z["ws"]))}
            key_m = [(int(c), int(w)) for c, w in zip(o["ci"], o["ws"])]
            assert set(key_d) == set(key_m) and len(key_d) == len(key_m), "window set differs from the dump"
            idx = np.array([key_d[k] for k in key_m])
            ol, gd = Z["ol"][idx], Z["g"][idx]
            ctl["top1_vs_dump_ol"] = {"max_abs_diff_m": float(np.abs(o["top1"] - ol).max()), "n_windows": int(len(idx)),
                                      "share_windows_over_1e-4": float((np.abs(o["top1"] - ol).max((1, 2)) > 1e-4).mean()),
                                      "max_abs_diff_g_m": float(np.abs(o["g"] - gd).max())}
            o["eid"] = Z["eid"][idx]
        else:
            o["eid"] = o["ci"]
        res["controls"][tag] = ctl
        arm = {"n_windows": int(len(o["ci"])), "bands": {}, "manoeuvre_hit": {}}
        for bn in BAND_NAMES:
            r = sc["band"][bn]
            arm["bands"][bn] = {"ade_top1": single(r["top1"], o["eid"]), "ade_k0_hard_top_pair": single(r["k0"], o["eid"]), "ade_best_of_3": single(r["best3"], o["eid"]),
                                "best_of_3_minus_top1": paired(r["best3"], r["top1"], o["eid"]),
                                "share_best_k_is_rank": [float((np.nanargmin(r["per_k"], 1) == k).mean()) for k in range(K)]}
        for h, v in sc["hit"].items():
            arm["manoeuvre_hit"][h] = single(v, o["eid"])
        res["arms"][tag] = arm
        store_vec[tag] = (o, sc)
    # paired arms: gain = best3 - top1 per window, BAND minus CTL of the same seed; rep minus s0
    def gain(tag, bn):
        o, sc = store_vec[tag]
        r = sc["band"][bn]
        return r["best3"] - r["top1"]
    pairs = [("OB-BAND_s0", "OB-CTL_s0"), ("OB-BAND_s1", "OB-CTL_s1"), ("OB-BAND_rep", "OB-BAND_s0")]
    for x, y in pairs:
        if x in store_vec and y in store_vec:
            ox, oy = store_vec[x][0], store_vec[y][0]
            if not (np.array_equal(ox["ci"], oy["ci"]) and np.array_equal(ox["ws"], oy["ws"])):
                res["paired_arms"][f"{x}_minus_{y}"] = "refused: window order differs"
                continue
            res["paired_arms"][f"{x}_minus_{y}"] = {bn: {"gain_best3_minus_top1_difference": paired(gain(x, bn), gain(y, bn), ox["eid"]),
                                                         "best3_difference": paired(store_vec[x][1]["band"][bn]["best3"], store_vec[y][1]["band"][bn]["best3"], ox["eid"]),
                                                         "top1_difference": paired(store_vec[x][1]["band"][bn]["top1"], store_vec[y][1]["band"][bn]["top1"], ox["eid"])}
                                                    for bn in BAND_NAMES}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"controls": res["controls"], "not_run": res["not_run"]}, indent=1))


if __name__ == "__main__":
    main()
