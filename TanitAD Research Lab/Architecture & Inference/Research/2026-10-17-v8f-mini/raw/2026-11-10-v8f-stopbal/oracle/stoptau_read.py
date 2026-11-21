"""stoptau_read.py -- PREREAD_V8F_STOPTAU (theory/stopbal/oracle/PREREAD_V8F_STOPTAU.md, frozen; sha256 in PREREAD_V8F_STOPTAU.sha256).  CPU, zero training.

A hardening threshold tau set on TRAINING windows only, then STOPHARD's transform with tau in place of 0.5 (goal) / argmax (lon).
Stages (each banks its output before the next):
  train     stream the 141 WMACT train clips ONE AT A TIME from box 2 (sftp get, read only; each clip deleted after use), build first-tick windows with the eval-141 window rule
            (t in range(3, n-30), t % 4 == 3), run forward_infer of the seven STOPBAL arms, bank per-window own codes + GT labels to scratch npz.
  selfcheck the same batch code on eval windows reproduces the STOPDEC npz probabilities of BAL-NULL_s0 (validity of the re-built loop).
  tau       tau per arm by the card's rule from the banked training npz -> TAU_STOPTAU.json
  run       eval-141 model pass (stoporc_read.run_arm, imported, NOT edited; its two hooks pointed at this file's transforms) -> scratch npz per arm
  analyse   cells, floors, flags, controls 1-3, outcome labels -> RESULT_STOPTAU.json
The pure logic (choose_tau, fires_*, tau_code, check_tau, outcome_lever) needs no torch: test_stoptau.py calls it directly.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stoporc_read as SO  # noqa: E402
import stophard_read as SH  # noqa: E402

THEORY, READ = SO.THEORY, SO.READ
SCRATCH = os.environ.get("STTAU_SCRATCH", "F:/TanitAD-scratch/stoptau_build")
CARD = os.path.join(HERE, "PREREAD_V8F_STOPTAU.md")
CARD_SHA = "e940a0a206afd9ea66a4f2006c9719e7381da4387b8bdde473d49bee28460cee"
CLIPS_JSON = os.path.join(THEORY, "wmact", "train_stream", "clips141.json")
TAU_JSON = os.path.join(HERE, "TAU_STOPTAU.json")
SFTP = "C:/Program Files/Git/usr/bin/sftp.exe"
REMOTE = "/D:/tanitad/cache_full/train/clips"
JOIN_TRAIN = os.path.join(SCRATCH, "join_train.npz")          # pulled read-only from box 2 (the file the STOPBAL argv names as --join-train)
MAX_PULLED_BYTES = 2 * 1024 ** 3

N_G, N_LON, CODE_DIM = SO.N_G, SO.N_LON, SO.CODE_DIM
STOP_GI, LON_STOP_I, LON_SL = SO.STOP_GI, SO.LON_STOP_I, SO.LON_SL
TAU_GRID = tuple(round(0.50 + 0.05 * k, 2) for k in range(10))          # 0.50 ... 0.95
TAU_FALLBACK = 0.95
SEEDS = SH.SEEDS
ALL_TAGS = SH.ALL_TAGS                                                   # NULL_s0, NULL_s1, NULL_rep, G_s0, G_s1, L_s0, L_s1
NULL_OF = {"BAL-NULL_s0": "BAL-NULL_s0", "BAL-NULL_s1": "BAL-NULL_s1", "BAL-NULL_rep": "BAL-NULL_rep",
           "BAL-G_s0": "BAL-NULL_s0", "BAL-G_s1": "BAL-NULL_s1", "BAL-L_s0": "BAL-NULL_s0", "BAL-L_s1": "BAL-NULL_s1"}   # "matching NULL arm" = same seed; a NULL arm is its own reference
DECIDE_TAGS = SH.DECIDE_TAGS
LEVER_DECISION = SH.LEVER_DECISION
LEVER_MODE = {"BAL-G": "t_goal", "BAL-L": "t_lon"}
MODES = ("unreach", "r_goal", "r_lon", "sh_lon", "t_goal", "t_lon")
VARIANTS = ("t_goal", "t_lon")
CELLS, S_CELL = SO.CELLS, SO.PRIMARY_CELL
EVAL_FAMILY_NOTE = SH.TACTICAL_FAMILY
STRATEGIC_FAMILY = SO.STRATEGIC_FAMILY


def sha256(p):
    return SO.sha256(p)


# ------------------------------------------------------------------------------------------------------------------ pure logic (numpy)
def fires_goal(own, tau):
    """[N] bool: own p(STOP_POINT) >= tau (float32 comparison, as STOPHARD's `own[:, STOP_GI] >= thr`)."""
    return np.asarray(own)[:, STOP_GI] >= np.float32(tau)


def fires_lon(own, tau):
    """[N] bool: tau None -> lon argmax == STOP (the unhardened rule); else own p(lon = STOP) >= tau."""
    lon = np.asarray(own)[:, LON_SL]
    if tau is None:
        return lon.argmax(1) == LON_STOP_I
    return lon[:, LON_STOP_I] >= np.float32(tau)


def tau_code(own, mode, goal_tau=None, lon_tau=None):
    """own [N, 60] float32 -> a copy with T's own decision written as a hard value at the given thresholds.  mode 'goal': goal[STOP_POINT] := 1.0 where p >= goal_tau.
    mode 'lon': lon part := one-hot STOP where (lon_tau None: argmax == STOP; else p(lon = STOP) >= lon_tau).  Uses no label and no other input."""
    assert mode in ("goal", "lon"), mode
    own = np.asarray(own)
    c = own.copy()
    if mode == "goal":
        c[fires_goal(own, goal_tau), STOP_GI] = 1.0
    else:
        f = fires_lon(own, lon_tau)
        oh = np.zeros((int(f.sum()), N_LON), own.dtype)
        oh[:, LON_STOP_I] = 1.0
        blk = c[:, LON_SL]
        blk[f] = oh
        c[:, LON_SL] = blk
    return c


def check_tau(own, code, mode, goal_tau=None, lon_tau=None, sel=None):
    """structural check, column by column, independent of tau_code's implementation.  -> (counts, problems)."""
    own, code = np.asarray(own), np.asarray(code)
    if sel is not None:
        own, code = own[sel], code[sel]
    prob = []
    if own.shape != code.shape or own.dtype != code.dtype:
        return {}, [f"{mode}: shape/dtype"]
    differ = own.view(np.uint32) != code.view(np.uint32)
    named = np.zeros(CODE_DIM, bool)
    if mode == "goal":
        named[STOP_GI] = True
        fire = own[:, STOP_GI] >= np.float32(goal_tau)
    else:
        named[LON_SL] = True
        lon = own[:, LON_SL]
        fire = (lon.argmax(1) == LON_STOP_I) if lon_tau is None else (lon[:, LON_STOP_I] >= np.float32(lon_tau))
    n_out = int(differ[:, ~named].sum())
    if n_out:
        prob.append(f"{mode}: {n_out} entries differ outside the named entries")
    if mode == "goal":
        if differ[~fire, STOP_GI].any():
            prob.append("goal entry changed where the decision does not fire")
        if fire.any() and not (code[fire, STOP_GI] == 1.0).all():
            prob.append("goal entry is not 1.0 where the decision fires")
    else:
        if differ[:, LON_SL][~fire].any():
            prob.append("lon part changed where the decision does not fire")
        if fire.any():
            want = np.zeros((int(fire.sum()), N_LON), own.dtype)
            want[:, LON_STOP_I] = 1.0
            if not np.array_equal(code[fire][:, LON_SL], want):
                prob.append("lon part is not one-hot STOP where the decision fires")
    if differ.any(1)[~fire].any():
        prob.append("a window with no firing decision changed")
    return {"n_windows": int(len(own)), "fires": int(fire.sum()), "windows_changed": int(differ.any(1).sum()), "entries_outside_named_changed": n_out}, prob


def choose_tau(counts_by_tau, ref_count, n_neg, grid=TAU_GRID, fallback=TAU_FALLBACK):
    """the card's rule: the smallest grid value at which the arm's training false-positive RATE is at most the reference rate (the matching NULL arm at the unhardened rule).
    counts_by_tau {tau: false-positive count on the n_neg non-stop-point windows}; ref_count the reference's count on the same n_neg windows.  -> (tau, fallback_used).
    Integer counts over one common denominator, so the comparison has no rounding."""
    for t in grid:
        if int(counts_by_tau[t]) <= int(ref_count):
            return t, False
    return fallback, True


def outcome_lever(per_seed):
    """per_seed {'s0': {'S_gain': bool, 'A_cost': bool}, 's1': {...}} -> label of the card's table, rows tried in the card's order:
    TAU-HELPS (S a gain at both seeds, A not a cost at either) / TAU-TRADES (A a cost at either seed) / TAU-NULL (S not a gain at either seed) / else SEEDS-DISAGREE.
    -> (label, list of every table row whose rule holds)."""
    if any(s not in per_seed or per_seed[s].get("S_gain") is None or per_seed[s].get("A_cost") is None for s in SEEDS):
        return "NOT-EVALUATED", []
    sg = [bool(per_seed[s]["S_gain"]) for s in SEEDS]
    ac = [bool(per_seed[s]["A_cost"]) for s in SEEDS]
    rows = []
    if all(sg) and not any(ac):
        rows.append("TAU-HELPS")
    if any(ac):
        rows.append("TAU-TRADES")
    if not any(sg):
        rows.append("TAU-NULL")
    return (rows[0] if rows else "SEEDS-DISAGREE"), rows


def clips_disjoint(train_clips, eval_clips):
    """control 2: -> sorted list of clip ids present in both."""
    return sorted(set(map(str, train_clips)) & set(map(str, eval_clips)))


def eval_rule_windows(n, first=3, stride=4, horizon=30):
    """the eval-141 window rule (verified equal to eval_grid_windows.json on all 141 eval clips): t in range(first, n - horizon) with t % stride == first % stride."""
    return [t for t in range(first, n - horizon) if t % stride == first % stride]


# ------------------------------------------------------------------------------------------------------------------ the model pass on arbitrary (store, tracks, join, windows)
def setup_env():
    import stopdec_arm as SA
    SA.install()
    sys.path.insert(0, os.path.join(THEORY, "stopdec"))
    sys.path.insert(0, os.path.join(THEORY, "speedsrc"))
    import speedsrc_read as SS
    import torch
    tree = SO.TREE
    import tanitad
    assert os.path.abspath(tanitad.__file__).replace("\\", "/").startswith(tree), tanitad.__file__
    return SS, torch


def load_models(SS, torch, tags):
    TK, M, D = SS.TK, SS.M, SS.D
    assert (M.N_GOAL, M.N_LAT, M.N_LON, M.N_CONS) == (N_G, SO.N_LAT, N_LON, SO.N_CONS) and D.GOAL22.index("STOP_POINT") == STOP_GI and D.LON9.index("STOP") == LON_STOP_I
    out, A0 = {}, None
    for tag in tags:
        A, ck_path, _ = SS.tag_paths(tag)
        got = sha256(ck_path)
        man = os.path.dirname(ck_path) + "/BANK_MANIFEST.json"
        man_ok = None
        if os.path.exists(man):
            man_ok = json.load(open(man, encoding="utf-8"))["files"]["ckpt_last.pt"]["sha256"] == got
            assert man_ok, (tag, "checkpoint sha256 differs from its bank manifest")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg = M.V8FMiniConfig(**ck["cfg"])
        model = M.V8FMini(cfg, M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        model.load_state_dict(ck["model"])
        model.eval()
        key = {k: A[k] for k in ("tracks", "nav", "scene", "limit", "schedule")}
        if A0 is None:
            A0 = dict(A, **{})
        assert all(A0[k] == A[k] for k in ("tracks", "nav", "scene", "limit", "schedule", "eval-cache", "eval-grid", "eval-join")), tag
        out[tag] = {"model": model, "cfg": cfg, "ckpt_sha256": got, "manifest_ok": bool(man_ok)}
    c0 = out[tags[0]]["cfg"]
    for t in tags:
        c = out[t]["cfg"]
        assert (c.hist, c.hmax, c.max_age_t, c.max_age_s, c.window_start) == (c0.hist, c0.hmax, c0.max_age_t, c0.max_age_s, c0.window_start), t
    return out, A0


def batch_codes(SS, torch, E, models, cfg, A, store, tracks, J, wins):
    """run forward_infer of every model on windows `wins` = [(ci, t, 0, 0)] of (store, tracks, J) with the eval pipeline's batch build (stoporc_read.run_arm's loop, same calls).
    -> dict of per-window arrays: code_<tag> [N, 60] float32 plus the label / geometry rows run_arm banks (g, v0, has, n_present, cons_gt, tl_hit, stop_hit, ci, ws)."""
    D = SS.D
    proxy = E._OneClipStore()
    ds = D.V8FMiniWindows(proxy, J, [], hist=cfg.hist, hmax=cfg.hmax, max_age_t=cfg.max_age_t, max_age_s=cfg.max_age_s, window_start=cfg.window_start)
    a_t, a_s = E.ages_for(0, A["schedule"])
    N = len(wins)
    o = {"g": np.zeros((N, 30, 2)), "ci": np.zeros(N, np.int64), "ws": np.zeros(N, np.int64), "has": np.zeros(N, bool), "v0": np.zeros(N), "n_present": np.zeros(N, np.int64),
         "cons_gt": np.full((N, SO.N_CONS), np.nan), "tl_hit": np.zeros(N, bool), "stop_hit": np.zeros(N, bool)}
    for tag in models:
        o[f"code_{tag}"] = np.zeros((N, CODE_DIM), np.float32)
    for s in range(0, N, 32):
        chunk = wins[s:s + 32]
        st = E.make_states(store, tracks, J, chunk, A["nav"])
        items = [E.build_item(ds, proxy, x, 0, a_t, a_s, A["scene"], A["limit"], None) for x in st]
        full = D.collate(items)
        batch = {k: full[k] for k in E.MODEL_KEYS}
        lt = full["lab_t"]
        has = lt["has"].bool()
        sl = slice(s, s + len(chunk))
        with torch.no_grad():
            for tag, mm in models.items():
                own = mm["model"].forward_infer(batch)["code"]
                assert own.shape[1] == CODE_DIM and own.dtype == torch.float32
                o[f"code_{tag}"][sl] = own.numpy()
            present = ~torch.isnan(lt["cons"])
            pm = (present[:, SS.SP_COLS] & has[:, None]).numpy()
            gy, gw = lt["goal_y"], lt["goal_w"]
            hit = ((gy > 0.5) & (gw > 0.5)).numpy() & has[:, None].numpy()
        o["has"][sl] = has.numpy()
        o["v0"][sl] = batch["p10f"][:, -1, 0].double().numpy()
        o["n_present"][sl] = pm.sum(1)
        o["cons_gt"][sl] = np.where(has.numpy()[:, None], lt["cons"].double().numpy(), np.nan)
        o["tl_hit"][sl] = hit[:, [SS.GI[g] for g in SS.TL_GOALS]].any(1)
        o["stop_hit"][sl] = hit[:, STOP_GI]
        for i, (ci, tt, _, _) in enumerate(chunk):
            o["g"][s + i] = store.get(str(J["clip"][ci]))["gt"][tt]
            o["ci"][s + i], o["ws"][s + i] = ci, tt
    return o


def stop_point_sets(SS, o):
    """ok (has and >= 1 present constraint cell), sp (GT STOP_POINT windows: STOPDEC's `tlstop == STOP_POINT` stratum), neg = ok minus sp."""
    ok = o["has"] & (o["n_present"] > 0)
    S, _ = SS.build_strata(o["g"], o["v0"], o["cons_gt"], o["tl_hit"], o["stop_hit"])
    sp = (S["tlstop"] == SO.TLSTOP_SP) & ok
    return ok, sp, ok & ~sp


# ------------------------------------------------------------------------------------------------------------------ stage: train windows
def pull_clip(clip, md5, dest):
    batch = os.path.join(SCRATCH, "_get_clip.txt")
    with open(batch, "w", newline="\n") as fh:
        fh.write(f"get {REMOTE}/{clip}.pt {dest}\nbye\n")
    for attempt in range(3):
        subprocess.run([SFTP, "-q", "-b", "_get_clip.txt", "tanitad-box2"], cwd=SCRATCH, check=True, timeout=300, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        h = hashlib.md5()
        with open(os.path.join(SCRATCH, dest), "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        if h.hexdigest() == md5:
            return
        print(f"  md5 mismatch {clip} attempt {attempt}", flush=True)
    raise RuntimeError(f"{clip}: md5 never matched box 2's manifest")


def dir_bytes(d):
    return sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d)) if os.path.isdir(d) else 0


def train_stage(tags, smoke):
    SS, torch = setup_env()
    D, TK = SS.D, SS.TK
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E = TK.load_eval_module()
    meta = json.load(open(CLIPS_JSON, encoding="utf-8"))
    clips = meta["clips"]
    assert hashlib.sha256("\n".join(clips).encode()).hexdigest() == meta["clips_sha256"]
    models, A = load_models(SS, torch, tags)
    cfg = models[tags[0]]["cfg"]
    Jt = D.load_join(JOIN_TRAIN)
    jc = [str(c) for c in Jt["clip"].tolist()]
    ci_of = {c: i for i, c in enumerate(jc)}
    Je = D.load_join(A["eval-join"])
    overlap = clips_disjoint(clips, [str(c) for c in Je["clip"].tolist()])
    assert not overlap, ("training clips share ids with eval-141", overlap[:3])          # control 2, asserted before any model pass
    tracks_all = D.load_native_tracks(A["tracks"])
    tracks = {c: tracks_all[c] for c in clips}
    del tracks_all
    cdir = os.path.join(SCRATCH, "cache", "train", "clips")
    os.makedirs(cdir, exist_ok=True)
    store = D.ClipStore(os.path.join(SCRATCH, "cache"), "train", lru=1, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    parts, peak = [], 0
    t0 = time.time()
    use = clips[:smoke] if smoke else clips
    for i, c in enumerate(use):
        dest = f"cache/train/clips/{c}.pt"
        pull_clip(c, meta["md5"][c], dest)
        peak = max(peak, dir_bytes(cdir))
        assert peak <= MAX_PULLED_BYTES
        try:
            store._c.clear()
            n = int(Jt["n"][ci_of[c]])
            assert n == meta["n_frames"][c], (c, n, meta["n_frames"][c])
            wins = [(ci_of[c], t, 0, 0) for t in eval_rule_windows(n)]
            o = batch_codes(SS, torch, E, models, cfg, A, store, tracks, Jt, wins)
        finally:
            store._c.clear()
            os.remove(os.path.join(SCRATCH, dest))                    # pulled copy: re-creatable (source on box 2), deleted after use
        o["clip"] = np.array([c] * len(wins))
        parts.append(o)
        print(f"clip {i + 1:3d}/{len(use)} {c[:8]} n={n} win={len(wins)} total={time.time() - t0:.0f}s", flush=True)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["ckpt_sha256"] = np.array([models[t]["ckpt_sha256"] for t in tags])
    out["tags"] = np.array(list(tags))
    out["tanitad_file"] = np.array(os.path.abspath(sys.modules["tanitad"].__file__))
    out["peak_pulled_bytes"] = np.array(peak)
    out["clips141_sha256"] = np.array(meta["clips_sha256"])
    np.savez(os.path.join(SCRATCH, ("smoke_" if smoke else "") + "train_windows.npz"), **out)
    print("banked train_windows.npz", {k: v.shape for k, v in out.items() if hasattr(v, "shape")}, f"peak pulled {peak} B", flush=True)


def selfcheck_stage(n_clips=3):
    """control: the batch code of this file, fed eval-141 (cache / join / tracks / grid), reproduces the STOPDEC npz probabilities."""
    SS, torch = setup_env()
    D, TK = SS.D, SS.TK
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    E = TK.load_eval_module()
    tag = "BAL-NULL_s0"
    models, A = load_models(SS, torch, [tag])
    cfg = models[tag]["cfg"]
    J = D.load_join(A["eval-join"])
    tracks = D.load_native_tracks(A["tracks"])
    store = D.ClipStore(A["eval-cache"], "eval", lru=32, grid="pool4x10", kin_mode="yaw", tracks=tracks)
    wins = D.eval_grid_windows(A["eval-grid"], J)
    first = list(dict.fromkeys(w[0] for w in wins))[:n_clips]
    wins = [w for w in wins if w[0] in first]
    o = batch_codes(SS, torch, E, models, cfg, A, store, tracks, J, wins)
    sd = dict(np.load(os.path.join(SO.STOPDEC_OUT, f"stopdec_{tag}.npz")))
    key = {(int(c), int(w)): i for i, (c, w) in enumerate(zip(sd["clip_index"], sd["ws"]))}
    idx = np.array([key[(int(c), int(w))] for c, w in zip(o["ci"], o["ws"])])
    code = o[f"code_{tag}"]
    dg = float(np.abs(code[:, :N_G].astype(np.float64) - sd["goal_prob"][idx]).max())
    dl = float(np.abs(code[:, LON_SL].astype(np.float64) - sd["lon_prob"][idx]).max())
    # the eval-window rule equals the eval grid json (all clips)
    g = json.load(open(A["eval-grid"], encoding="utf-8"))
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    rule_equal = all(g["windows"][c] == eval_rule_windows(int(J["n"][ci_of[c]])) for c in g["order"])
    res = {"tag": tag, "n_windows": int(len(wins)), "n_clips": n_clips, "goal_prob_max_abs_vs_stopdec": dg, "lon_prob_max_abs_vs_stopdec": dl, "window_rule_equals_eval_grid_all_141_clips": bool(rule_equal),
           "pass": bool(dg <= 1e-5 and dl <= 1e-5 and rule_equal)}
    json.dump(res, open(os.path.join(SCRATCH, "selfcheck.json"), "w"), indent=1)
    print(json.dumps(res))


# ------------------------------------------------------------------------------------------------------------------ stage: tau
def tau_stage():
    SS, _ = setup_env()
    z = dict(np.load(os.path.join(SCRATCH, "train_windows.npz")))
    ok, sp, neg = stop_point_sets(SS, z)
    n_neg, n_sp = int(neg.sum()), int(sp.sum())
    tags = [str(t) for t in z["tags"]]
    rate = {}
    for tag in tags:
        own = z[f"code_{tag}"]
        rate[tag] = {"goal": {str(t): int(fires_goal(own, t)[neg].sum()) for t in TAU_GRID}, "lon": {str(t): int(fires_lon(own, t)[neg].sum()) for t in TAU_GRID}}
        rate[tag]["goal_unhardened_0.5"] = int(fires_goal(own, 0.5)[neg].sum())
        rate[tag]["lon_unhardened_argmax"] = int(fires_lon(own, None)[neg].sum())
        rate[tag]["tp_goal_0.5_on_stop_point"] = int(fires_goal(own, 0.5)[sp].sum())
        rate[tag]["tp_lon_argmax_on_stop_point"] = int(fires_lon(own, None)[sp].sum())
    arms = {}
    for tag in ALL_TAGS:
        ref = NULL_OF[tag]
        a = {"null_reference_arm": ref, "n_non_stop_point_windows": n_neg, "n_stop_point_windows": n_sp, "false_positive_counts": {"goal": rate[tag]["goal"], "lon": rate[tag]["lon"]},
             "false_positive_rates": {d: {t: c / n_neg for t, c in rate[tag][d].items()} for d in ("goal", "lon")},
             "reference_counts": {"goal_unhardened_0.5": rate[ref]["goal_unhardened_0.5"], "lon_unhardened_argmax": rate[ref]["lon_unhardened_argmax"]},
             "own_unhardened": {"goal_fp": rate[tag]["goal_unhardened_0.5"], "lon_fp": rate[tag]["lon_unhardened_argmax"], "goal_tp": rate[tag]["tp_goal_0.5_on_stop_point"], "lon_tp": rate[tag]["tp_lon_argmax_on_stop_point"]}}
        tg, fg = choose_tau({float(t): c for t, c in rate[tag]["goal"].items()}, rate[ref]["goal_unhardened_0.5"], n_neg)
        tl, fl = choose_tau({float(t): c for t, c in rate[tag]["lon"].items()}, rate[ref]["lon_unhardened_argmax"], n_neg)
        a["tau_goal"], a["tau_goal_fallback_no_grid_value_meets_rule"] = tg, fg
        a["tau_lon"], a["tau_lon_fallback_no_grid_value_meets_rule"] = tl, fl
        a["tp_at_tau_on_stop_point"] = {"goal": int(fires_goal(z[f"code_{tag}"], tg)[sp].sum()), "lon": int(fires_lon(z[f"code_{tag}"], tl)[sp].sum())}
        arms[tag] = a
    Je_clips = None
    res = {"tool": "stoptau_read.py --stage tau", "card": "theory/stopbal/oracle/PREREAD_V8F_STOPTAU.md", "card_sha256": CARD_SHA, "evidence_class": "MEASURED (this file on the banked training-window npz; sha256 below)",
           "rule": "tau = smallest value of the grid 0.50, 0.55, ..., 0.95 at which the arm's training false-positive rate (firing on training windows that are not stop-point windows) is at most the matching NULL arm's rate at the unhardened rule (goal p(STOP_POINT) >= 0.5; lon argmax == STOP); none meets it -> 0.95 and reported; matching NULL = same seed, a NULL arm is its own reference",
           "tau_grid": list(TAU_GRID), "window_rule": "t in range(3, n-30), t % 4 == 3 per clip (equal to eval_grid_windows.json on all 141 eval clips, see selfcheck)", "windows": {"n_windows": int(len(ok)), "n_ok_has_and_present": int(ok.sum()), "n_stop_point": n_sp, "n_non_stop_point": n_neg},
           "stop_point_definition": "GT STOP_POINT window = STOPDEC's `tlstop == STOP_POINT` stratum (goal_y/goal_w STOP_POINT hit, no traffic-light goal hit) from the join_train labels, among has-and-present windows; the same definition as the eval S set",
           "train_clips": {"n": int(len(set(z['clip'].tolist()))), "clips141_sha256": str(z["clips141_sha256"]), "disjoint_from_eval141": True, "peak_pulled_bytes_on_disk": int(z["peak_pulled_bytes"])},
           "ckpt_sha256": dict(zip(tags, [str(x) for x in z["ckpt_sha256"]])), "banked_train_windows_npz_sha256": sha256(os.path.join(SCRATCH, "train_windows.npz")), "arms": arms}
    json.dump(res, open(TAU_JSON, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({t: {"tau_goal": a["tau_goal"], "tau_lon": a["tau_lon"]} for t, a in arms.items()}))


# ------------------------------------------------------------------------------------------------------------------ stage: eval run (stoporc_read.run_arm, unedited)
CUR = {"goal": None, "lon": None}


def _hook(own, lon_al, mode):
    if mode == "unreach":
        return SH.hard_code(own, "unreach")
    if mode == "sh_lon":
        return SH.hard_code(own, "hlon")
    if mode == "r_goal":
        return tau_code(own, "goal", 0.5)
    if mode == "r_lon":
        return tau_code(own, "lon", None, None)
    if mode == "t_goal":
        return tau_code(own, "goal", CUR["goal"])
    if mode == "t_lon":
        return tau_code(own, "lon", None, CUR["lon"])
    raise AssertionError(mode)


def run_stage(tags, smoke):
    os.makedirs(SCRATCH, exist_ok=True)
    taus = json.load(open(TAU_JSON, encoding="utf-8"))["arms"]
    SO.MODES = MODES
    SO.force_codes = _hook
    for tag in tags:
        CUR["goal"], CUR["lon"] = taus[tag]["tau_goal"], taus[tag]["tau_lon"]
        print(f"arm {tag} tau_goal {CUR['goal']} tau_lon {CUR['lon']}", flush=True)
        o = SO.run_arm(tag, smoke)
        o["tau_goal"], o["tau_lon"] = np.array(CUR["goal"]), np.array(CUR["lon"])
        np.savez(os.path.join(SCRATCH, f"{'smoke_' if smoke else ''}stoptau_{tag}.npz"), **{k: np.asarray(v) for k, v in o.items()})
        print(f"banked {tag}", flush=True)


# ------------------------------------------------------------------------------------------------------------------ stage: analyse
def analyse_arm(tag, z, CI, BL, sh_arm):
    ok, sp, ctl, S = SO.window_sets(z)
    own = z["own_code"]
    tg, tl = float(z["tau_goal"]), float(z["tau_lon"])
    sd = dict(np.load(os.path.join(SO.STOPDEC_OUT, f"stopdec_{tag}.npz")))
    cx = {"windows_equal_stopdec": bool(np.array_equal(sd["clip_index"], z["ci"]) and np.array_equal(sd["ws"], z["ws"]) and np.array_equal(sd["has"], z["has"])),
          "own_goal_prob_max_abs_vs_stopdec": float(np.abs(own[:, :N_G].astype(np.float64) - sd["goal_prob"]).max()),
          "own_lon_prob_max_abs_vs_stopdec": float(np.abs(own[:, LON_SL].astype(np.float64) - sd["lon_prob"]).max())}
    own_plan = z["plan_own"].astype(np.float64)
    ctrl = {"n_windows": int(len(z["ci"])), "n_stop_point": int(sp.sum()), "n_control_set": int(ctl.sum()), "n_all_has_present": int(ok.sum()), "ckpt_sha256": str(z["ckpt_sha256"]),
            "manifest_ok": bool(z["manifest_ok"]), "tanitad_file": str(z["tanitad_file"]), "own_plan_vs_dump_ol_max_abs_m": float(z["ol_vs_own_max_abs_m"]), "stopdec_cross_check": cx, "problems": []}
    P = ctrl["problems"]
    cells = {}
    for v in ("own",) + MODES:
        plan = z[f"plan_{v}" if v != "own" else "plan_own"].astype(np.float64)
        cells[v] = {}
        for bn, (a, b) in SO.BANDS.items():
            r = BL.band_rows(plan, z["g"], a, b)
            for m in SO.METRICS:
                cells[v][f"{bn}:{m}"] = r[m]
    # base checks (as STOPHARD)
    if (ctrl["n_stop_point"], ctrl["n_control_set"]) != (SO.EXPECT["n_stop"], SO.EXPECT["n_ctl"]):
        P.append(f"window counts {ctrl['n_stop_point']} / {ctrl['n_control_set']}")
    if ctrl["own_plan_vs_dump_ol_max_abs_m"] > 1e-4:
        P.append(f"own plan vs dump ol {ctrl['own_plan_vs_dump_ol_max_abs_m']}")
    if not cx["windows_equal_stopdec"] or cx["own_goal_prob_max_abs_vs_stopdec"] > 1e-5 or cx["own_lon_prob_max_abs_vs_stopdec"] > 1e-5:
        P.append("windows or own probabilities differ from the STOPDEC npz")
    # control 1: replay at tau 0.5 (goal) / argmax (lon) reproduces STOPHARD
    c1 = {"unreach_code_bit_equal_own": bool(np.array_equal(z["code_unreach"].view(np.uint32), own.view(np.uint32))),
          "unreach_plan_vs_own_max_abs_m": float(np.abs(z["plan_unreach"].astype(np.float64) - own_plan).max())}
    sh_goal_code = SH.hard_code(own, "hgoal")                      # STOPHARD's own function on this arm's own codes (pure numpy)
    c1["r_goal_code_bit_equal_STOPHARD_transform"] = bool(np.array_equal(z["code_r_goal"].view(np.uint32), sh_goal_code.view(np.uint32)))
    c1["r_goal_plan_vs_STOPHARD_transform_plan_max_abs_m_window_by_window"] = "not run (goal plans are checked through the code bits and the cells against RESULT_STOPHARD.json)"
    c1["r_lon_code_bit_equal_STOPHARD_transform"] = bool(np.array_equal(z["code_r_lon"].view(np.uint32), z["code_sh_lon"].view(np.uint32)))
    c1["r_lon_plan_vs_STOPHARD_transform_plan_max_abs_m_window_by_window"] = float(np.abs(z["plan_r_lon"].astype(np.float64) - z["plan_sh_lon"].astype(np.float64)).max())
    sets = {"S_stop_point": sp, "A_all": ok}
    delta = {}
    for v in ("r_goal", "r_lon", "t_goal", "t_lon"):
        delta[v] = {sn: {c: SO.paired(CI, cells[v][c][np.where(m)[0]], cells["own"][c][np.where(m)[0]], np.asarray(z["eid"])[np.where(m)[0]]) for c in CELLS} for sn, m in sets.items()}
    shd = sh_arm["delta_variant_minus_own"]
    mism, maxd = [], 0.0
    for v, hv in (("r_goal", "hgoal"), ("r_lon", "hlon")):
        for sn in sets:
            for c in CELLS:
                a, b = delta[v][sn][c], shd[hv][sn][c]
                for k in ("point", "lo", "hi", "mean_own", "mean_forced"):
                    if k in a or k in b:
                        d = abs(a[k] - b[k])
                        maxd = max(maxd, d)
                        if d > 1e-9:
                            mism.append(f"{v}:{sn}:{c}:{k}")
                if a.get("separated") != b.get("separated"):
                    mism.append(f"{v}:{sn}:{c}:separated")
    c1["replay_cells_vs_RESULT_STOPHARD_json"] = {"n_cell_comparisons": 2 * 2 * len(CELLS), "max_abs_diff": maxd, "mismatches": mism[:10], "n_mismatches": len(mism)}
    ctrl["control1_replay"] = c1
    if not (c1["unreach_code_bit_equal_own"] and c1["unreach_plan_vs_own_max_abs_m"] == 0.0 and c1["r_goal_code_bit_equal_STOPHARD_transform"] and c1["r_lon_code_bit_equal_STOPHARD_transform"]
            and c1["r_lon_plan_vs_STOPHARD_transform_plan_max_abs_m_window_by_window"] == 0.0 and not mism):
        P.append("control 1: replay differs from STOPHARD")
    # structural checks of the tau transforms
    ctrl["tau_code_checks"] = {}
    for m, (mode, gt, lt_) in {"t_goal": ("goal", tg, None), "t_lon": ("lon", None, tl), "r_goal": ("goal", 0.5, None), "r_lon": ("lon", None, None)}.items():
        for sn, msk in (("stop", sp), ("ctl", ctl)):
            cnt, pr = check_tau(own, z[f"code_{m}"], mode, gt, lt_, msk)
            ctrl["tau_code_checks"][f"{m}:{sn}"] = cnt
            P.extend(f"structure [{sn}] {m}: {p}" for p in pr)
        same = (z[f"code_{m}"].view(np.uint32) == own.view(np.uint32)).all(1)
        pm = float(np.abs(z[f"plan_{m}"].astype(np.float64)[same] - own_plan[same]).max()) if same.any() else 0.0
        ctrl["tau_code_checks"][f"{m}:unchanged_windows_plan_max_abs_m"] = {"n_unchanged_windows": int(same.sum()), "max_abs_m": pm}
        if pm != 0.0:
            P.append(f"{m}: unchanged-code window with plan != OWN ({pm})")
    ctrl["decision_counts_at_tau"] = {"goal": {"tau": tg, "stop_point": int(fires_goal(own, tg)[sp].sum()), "control": int(fires_goal(own, tg)[ctl].sum())},
                                      "lon": {"tau": tl, "stop_point": int(fires_lon(own, tl)[sp].sum()), "control": int(fires_lon(own, tl)[ctl].sum())},
                                      "goal_at_0.5": {"stop_point": int(fires_goal(own, 0.5)[sp].sum()), "control": int(fires_goal(own, 0.5)[ctl].sum())},
                                      "lon_argmax": {"stop_point": int(fires_lon(own, None)[sp].sum()), "control": int(fires_lon(own, None)[ctl].sum())}}
    ctrl["pass"] = not P
    return {"controls": ctrl, "tau_goal": tg, "tau_lon": tl, "delta_variant_minus_own": delta, "own_tactical_ade_mean": {"S_stop_point": float(np.nanmean(cells["own"][S_CELL][sp])), "A_all": float(np.nanmean(cells["own"][S_CELL][ok]))}}


def floors_from(arms):
    out = {}
    if SH.FLOOR_ARM not in arms or SH.NULL_TAGS[0] not in arms:
        return out
    for v in VARIANTS:
        out[v] = {}
        for sn in ("S_stop_point", "A_all"):
            out[v][sn] = {}
            for c in CELLS:
                a, b = arms[SH.FLOOR_ARM]["delta_variant_minus_own"][v][sn][c], arms[SH.NULL_TAGS[0]]["delta_variant_minus_own"][v][sn][c]
                out[v][sn][c] = float(abs(a["point"] - b["point"])) if "point" in a and "point" in b else None
    return out


def flags_for(arm, floors):
    out = {}
    for v in VARIANTS:
        out[v] = {}
        for sn in ("S_stop_point", "A_all"):
            out[v][sn] = {}
            for c in CELLS:
                d = arm["delta_variant_minus_own"][v][sn][c]
                fl = floors.get(v, {}).get(sn, {}).get(c)
                out[v][sn][c] = {"gain": SO.is_gain(d, fl), "cost": SH.is_cost(d, fl), "floor_m": fl}
    return out


def analyse_stage(out_json, tags):
    SS, _ = setup_env()
    BL = SS.BL
    CI = BL.CI
    sh = json.load(open(os.path.join(HERE, "RESULT_STOPHARD.json"), encoding="utf-8"))
    tj = json.load(open(TAU_JSON, encoding="utf-8"))
    arms, banked = {}, {}
    for tag in tags:
        p = os.path.join(SCRATCH, f"stoptau_{tag}.npz")
        if not os.path.exists(p):
            banked[tag] = "per-arm npz absent"
            continue
        z = dict(np.load(p, allow_pickle=False))
        assert float(z["tau_goal"]) == tj["arms"][tag]["tau_goal"] and float(z["tau_lon"]) == tj["arms"][tag]["tau_lon"]
        arms[tag] = analyse_arm(tag, z, CI, BL, sh["arms"][tag])
        banked[tag] = {"scratch_npz_sha256": sha256(p)}
        print(f"analysed {tag}: controls pass {arms[tag]['controls']['pass']}", flush=True)
    floors = floors_from(arms)
    ctl_ok = all(a["controls"]["pass"] for a in arms.values()) and len(arms) == len(tags)
    ctl_ok = ctl_ok and tj["train_clips"]["disjoint_from_eval141"]
    flags = {t: flags_for(a, floors) for t, a in arms.items()}
    outcomes = {}
    for lever, v in LEVER_MODE.items():
        per_seed = {}
        for s, tag in zip(SEEDS, DECIDE_TAGS[lever]):
            if tag in flags:
                per_seed[s] = {"S_gain": flags[tag][v]["S_stop_point"][S_CELL]["gain"], "A_cost": flags[tag][v]["A_all"][S_CELL]["cost"], "arm": tag,
                               "tau": tj["arms"][tag]["tau_goal" if v == "t_goal" else "tau_lon"],
                               "tau_fallback": tj["arms"][tag]["tau_goal_fallback_no_grid_value_meets_rule" if v == "t_goal" else "tau_lon_fallback_no_grid_value_meets_rule"],
                               "S_delta": arms[tag]["delta_variant_minus_own"][v]["S_stop_point"][S_CELL], "A_delta": arms[tag]["delta_variant_minus_own"][v]["A_all"][S_CELL]}
        o, rows = outcome_lever(per_seed)
        outcomes[lever] = {"variant_as_decision": v, "outcome_if_controls_held": o, "table_rows_whose_rule_holds": rows, "outcome": "INVALID-CONTROLS" if not ctl_ok else o, "per_seed": per_seed}
    res = {"tool": "stoptau_read.py", "card": "theory/stopbal/oracle/PREREAD_V8F_STOPTAU.md", "card_sha256_expected": CARD_SHA,
           "tier": "T1-kin FIRST TICK, zero training, inference-time transform of T's own output; never driving performance",
           "estimator": f"taniteval.ci paired episode-cluster bootstrap, B={SO.NB}, seed {SO.SEED}; answers 'another draw of episodes' only (not another training run, not another inference run: the planner is deterministic)",
           "evidence_class": "MEASURED (this file; tau from TAU_STOPTAU.json on training windows; per-arm eval rows banked to scratch npz, sha256 in `banked`)",
           "definitions": {"transform": "goal entry 1.0 where p(STOP_POINT) >= tau; lon part one-hot STOP where p(lon = STOP) >= tau (tau from TAU_STOPTAU.json per arm); r_goal / r_lon = replay at 0.5 / argmax; sh_goal / sh_lon = STOPHARD's own transform",
                           "S": f"{S_CELL} on the 117 stop-point windows", "A": f"{S_CELL} on all 2,351 has-and-present windows", "delta": "variant minus OWN (negative = lower error)",
                           "floor": "per variant, set and cell: |delta(NULL_rep) - delta(NULL_s0)| (point) with each NULL arm at its own tau, read at seed 0, applied to both seeds", "lever_to_variant": LEVER_MODE},
           "families": {"PRIMARY/ADE-FDE, LONGITUDINAL, LATERAL": "in delta_variant_minus_own per cell and set", "TACTICAL": EVAL_FAMILY_NOTE, "STRATEGIC": STRATEGIC_FAMILY},
           "banked": banked, "controls_ok": ctl_ok, "outcome": outcomes, "floors_m": floors, "flags": flags, "arms": arms}
    json.dump(res, open(out_json, "w", encoding="utf-8"), indent=1, default=float)
    print(json.dumps({"outcome": {k: (v["outcome"], v["table_rows_whose_rule_holds"]) for k, v in outcomes.items()}, "controls_ok": ctl_ok}, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "selfcheck", "tau", "run", "analyse"), required=True)
    ap.add_argument("--tags", nargs="+", default=list(ALL_TAGS))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "RESULT_STOPTAU.json"))
    a = ap.parse_args()
    assert sha256(CARD) == CARD_SHA, "PREREAD_V8F_STOPTAU.md hash differs from the frozen sha256"
    os.makedirs(SCRATCH, exist_ok=True)
    if a.stage == "train":
        train_stage(a.tags, a.smoke)
    elif a.stage == "selfcheck":
        selfcheck_stage()
    elif a.stage == "tau":
        tau_stage()
    elif a.stage == "run":
        run_stage(a.tags, a.smoke)
    else:
        analyse_stage(a.out, a.tags)


if __name__ == "__main__":
    sys.path.insert(0, READ)
    main()
