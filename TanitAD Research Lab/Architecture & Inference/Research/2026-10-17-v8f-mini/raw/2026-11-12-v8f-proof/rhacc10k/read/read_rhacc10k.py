"""read_rhacc10k.py -- card PREREG_V8F_PROOF_ADD139 (RHACC10K) with amendments 1 and 2: the read.
Modelled on integ10k/read_headbound10k.py. Two stages:
  engines   (default; --skip-engines reads the files an earlier call wrote) runs the landed readers UNCHANGED on the run folders of the pull
            headway  headway_read_r2.py read   (subprocess)  -> <read-dir>/hw/RESULT_HEADWAY.json
            families q3_read_r3.py read        (subprocess)  -> <read-dir>/q3/RESULT_Q3.json
            route goal   strat_roll_read.py (card ADD66), driven in memory with its run registry, seeds and file layout pointed at these runs -> <read-dir>/strat/STRAT_READ.json
            C5       read_longroll10k_n4.py (reader of record of pre-rolled runs): its read_grid, engines and controls driven in memory on the 1,271 windows -> <read-dir>/c5/C5_READ.json
  evaluate  controls k1 to k6 and the reader controls first; names, levels and pairs ONLY if every control passed, else the result file carries names null.
Never run on data by the tool builder. Tier: T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance.

Runs (arm names as in box2_rhacc_queue.py; run folder <pull>/<ARM>_s<m>): H_R0K, H_R0A, H_RLcA, H_RQ50A, U_RLcK, U_RQ50K, H_RLcA_P30, H_RLcA_P0; model seeds 0, 1, 2 (amendment 1).
Pairs: C1 = H_R0A - H_R0K; C2 label = H_RLcA - U_RLcK; C2 reader = H_RQ50A - U_RQ50K; C3 = route goal of H_RLcA - U_RLcK at 30/50/80 m (third tick); C5 = H_RLcA_P30 - H_RLcA_P0.
STATED CHOICES (the card leaves them open):
  spread arm: the family bar and the headway bar are "the spread of the convention over the model seeds": the convention is the line model under readout K, so the arm U_RLcK on eval-141 (stated in the output).
              On the 1,271-window grid of C5 the only arms are the two H runs; the spread arm there is H_RLcA_P0 (stated in the output).
  family reading: as in read_headbound10k.py (BETTER if every named cell BETTER, WORSE if every one WORSE, NOT-SEPARATED if none either, else MIXED); a family counts as WORSE only when it reads WORSE.
                  STRATEGIC = the three arcs 30, 50, 80 m of the pos cell at the third tick, same rule.
  C2 "passed" inside C3: the label form of C2.
  C5: every family reads NOT-SEPARATED or BETTER (MIXED is neither, so it fails) and U reads NOT-SEPARATED or LOWER.
Usage: python read_rhacc10k.py --pull PULL --read-dir D --out OUT.json [--tracks native_tracks.pt] [--skip-engines]"""
import argparse
import contextlib
import glob
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V8 = "F:/TanitAD-scratch/v8proof"
RH10 = V8 + "/rhacc10k"
INTEG = V8 + "/integ10k"

CARD = "PREREG_V8F_PROOF_ADD139 (+ AM1, AM2)"
PRE = "RHACC10K"
TIER_LINE = "TIER: T1-kin/hb0.2 (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance"
SEEDS = (0, 1, 2)
SPREAD_ARM = "U_RLcK"
SPREAD_ARM_C5 = "H_RLcA_P0"
WRAPPER_SHA_PREFIX = "373ffe72f2655c35"
WRAPPER_PRE_SHA_PREFIX = "92a8034cc0f7b0d2"                                                       # rzgfhq, rows g and h (amendment 3)
K4_U, K4_C, K4_TOL = 0.0391, 0.0, 0.010
FAMS = ("LONGITUDINAL", "LATERAL", "TACTICAL", "STRATEGIC")
NAMED = {"LONGITUDINAL": ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m"], "LATERAL": ["LATERAL:heading_mae_deg", "LATERAL:yaw_rate_mae_degps"], "TACTICAL": ["TACTICAL:ade_tac_m"]}
ARMS = ("H_R0K", "H_R0A", "H_RLcA", "H_RQ50A", "U_RLcK", "U_RQ50K", "H_RLcA_P30", "H_RLcA_P0")
P_ARMS = ("H_RLcA_P30", "H_RLcA_P0")
# pair name -> (A, B)
PAIRS = {"C1": ("H_R0A", "H_R0K"), "C2_LABEL": ("H_RLcA", "U_RLcK"), "C2_READER": ("H_RQ50A", "U_RQ50K")}
C5_PAIR = ("H_RLcA_P30", "H_RLcA_P0")
C3_PAIR = PAIRS["C2_LABEL"]
# k5, recorded finding (determined from source; the file:line list is the evidence). The reader form takes the cached encoder tokens (eval cache), not the model's latent.
K5_FINDING = {
    "reader_reads_model_latent": False,
    "input": "batch['feats'] = the logged eval-cache encoder tokens of cache steps j-4..j; the token reader reads them and no model weight (the argument `model` is not read)",
    "evidence": [
        "integ10k/v8f_t1_eval_reader_rule_rzgfh.py:73 (module text: the reader reads batch[\"feats\"][:, -5:], the cached encoder tokens ..., no model weight is read)",
        "integ10k/v8f_t1_eval_reader_rule_rzgfh.py:188-192 (head_values_reader: `Takes tokens only; model is not read`; f = feats[:, -rd.hist:]; p, g = rd.read(f))",
        "integ10k/v8f_t1_eval_reader_rule_rzgfh.py:171 and :476 (later looks: toks.append(s.d[\"feats\"][e - rd.hist + 1:e + 1]); rd.read at :173 and :479)",
        "rhacc10k/tree/stack/scripts/v8f_t1_eval.py:12 and :166 (feats / frame_ok stay shared and logged camera input) and :244-245",
    ],
    "wrapper_sha256_prefix": WRAPPER_SHA_PREFIX,
    "status": "determined from source by the reader's builder; the file at --wrapper must carry the sha256 prefix above or k5 fails",
}
CARD_CKPT_PREFIX = {"RH_s0": "351bb5f2", "RH_s1": "c387f4ce", "RH_s2": "0ba5d78e", "UE_s0": "2fd4b747", "UE_s1": "f13df8ce", "UE_s2": "6d434525"}
DEFAULT_QUEUE = RH10 + "/kit/box2_rhacc_queue.py"
DEFAULT_INPUTS = RH10 + "/inputs/INPUTS.json"


# ============================================================================================ small helpers
def jl(p):
    try:
        return json.load(io.open(p, encoding="utf-8")) if p and os.path.isfile(p) else None
    except Exception:                                                                           # noqa: BLE001
        return None


def sha(p):
    if not p or not os.path.isfile(p):
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def txt(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.isfile(p) else ""


def conv(o):
    if isinstance(o, dict):
        return {(k if isinstance(k, str) else str(k)): conv(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [conv(v) for v in o]
    if isinstance(o, np.ndarray):
        return conv(o.tolist())
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def key_of(arm, m):
    return "%s_s%d" % (arm, m)


def run_dir(pull, arm, m):
    return os.path.join(pull, key_of(arm, m))


def dump_dir(pull, arm, m):
    return os.path.join(run_dir(pull, arm, m), "dump_%s_m10_causal_heads" % key_of(arm, m))


def dump_files(pull, arm, m):
    return sorted(glob.glob(os.path.join(dump_dir(pull, arm, m), "ep*.npz")))


def load_queue(path):
    spec = importlib.util.spec_from_file_location("box2_rhacc_queue_landed", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ============================================================================================ family rule and verdict helpers (as read_headbound10k.py)
def family_of(vs):
    if vs and all(v == "BETTER" for v in vs):
        return "BETTER"
    if vs and all(v == "WORSE" for v in vs):
        return "WORSE"
    if not any(v in ("BETTER", "WORSE") for v in vs):
        return "NOT-SEPARATED"
    return "MIXED"


def reading(cells, fam):
    return family_of([cells[c]["verdict"] for c in NAMED[fam]])


def below(d):
    return bool(all(x < 0 for x in d["per_seed"]) and d["interval"][1] < 0)


def above(d):
    return bool(all(x > 0 for x in d["per_seed"]) and d["interval"][0] > 0)


def u_name(u):
    return "LOWER" if below(u) else ("HIGHER" if above(u) else "NOT-SEPARATED")


def cells_of(pair_families):
    return {c: {"D": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)], "per_seed": [round(float(x), 4) for x in v["d_s"]], "verdict": v["verdict"]}
            for c, v in pair_families.items()}


def ucf(pair_cells):
    out = {}
    for c in ("U", "C", "F"):
        v = pair_cells[c]
        out[c] = {"per_seed": [round(float(x), 4) for x in v["d_s"]], "pooled": round(float(v["D"]), 4), "interval": [round(float(v["A_lo"]), 4), round(float(v["A_hi"]), 4)]}
    return out


def strat_reading(pos_cells):
    return family_of([c["verdict"] for c in pos_cells])


# ============================================================================================ controls k1 to k6 (on the run folders and the recorded inputs)
def expected_files(arm, a, grid_n):
    return grid_n if arm in P_ARMS else a.expect_files


def k1_complete(a, seeds, grid_n):
    per, ok = {}, True
    for m in seeds:
        for arm in ARMS:
            key = key_of(arm, m)
            rd = run_dir(a.pull, arm, m)
            done = jl(os.path.join(rd, "RUN_DONE.json"))
            files = dump_files(a.pull, arm, m)
            want = expected_files(arm, a, grid_n)
            bad_nf, why = 0, []
            for f in files:
                try:
                    with np.load(f) as z:
                        if "cl" not in z.files or not np.isfinite(np.asarray(z["cl"], np.float64)).all():
                            bad_nf += 1
                except Exception:                                                               # noqa: BLE001
                    bad_nf += 1
            form_r0 = arm in ("H_R0K", "H_R0A")
            rule_log = os.path.isfile(os.path.join(rd, "ticks", "rule_log_%s.npz" % key))
            good = bool(done is not None and done.get("rc") == 0 and len(files) == want and bad_nf == 0 and (form_r0 or rule_log))
            if done is None or done.get("rc") != 0:
                why.append("done file missing or rc != 0")
            if len(files) != want:
                why.append("dump files %d, expected %d" % (len(files), want))
            if bad_nf:
                why.append("%d files with a non-finite or unreadable plan" % bad_nf)
            if not form_r0 and not rule_log:
                why.append("rule log missing")
            per[key] = {"done": done is not None, "dump_files": len(files), "expected": want, "files_with_nonfinite_plan": bad_nf, "rule_log": rule_log if not form_r0 else None, "PASS": good, "why": why}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def argv_flag(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else None


def k2_models(a, seeds, queue, inputs):
    per, ok = {}, True
    table = getattr(queue, "CKPT_SHA", {})
    for m in seeds:
        for arm in ARMS:
            key = key_of(arm, m)
            rec = jl(os.path.join(run_dir(a.pull, arm, m), "argv.json"))
            ck = argv_flag(rec["argv"], "--ckpt") if rec and rec.get("argv") else None
            fam = "RH" if arm.startswith("H_") else "UE"
            want_key = "%s_s%d" % (fam, m)
            got_key = os.path.basename(os.path.dirname(ck.replace("\\", "/"))) if ck else None
            ent = (inputs or {}).get("inputs/%s/ckpt_last.pt" % want_key) or {}
            rec_sha = ent.get("sha256")
            rr = txt(os.path.join(run_dir(a.pull, arm, m), "ticks", "run_%s.txt" % key))
            mm = re.search(r"ckpt sha256 ([0-9a-f]{64})", rr)
            good = bool(rec_sha and got_key == want_key and rec_sha == table.get(want_key) and rec_sha.startswith(CARD_CKPT_PREFIX[want_key])
                        and (mm is None or mm.group(1) == rec_sha))
            per[key] = {"ckpt_key_in_argv": got_key, "expected_key": want_key, "recorded_sha256_prefix": (rec_sha or "")[:16], "run_record_sha_checked": bool(mm), "PASS": good}
            ok &= good
    return {"per_run": per, "note": "hashes compared as recorded (INPUTS.json vs the queue table and the card prefixes); a checkpoint is not hashed here", "PASS": bool(ok and per)}


def k3_argv(a, seeds, queue):
    full = {r["arm"]: r for r in queue.build_runs()}
    P = queue.paths()
    per, ok = {}, True
    for m in seeds:
        for arm in ARMS:
            key = key_of(arm, m)
            rec = jl(os.path.join(run_dir(a.pull, arm, m), "argv.json"))
            row = full.get(key)
            argv = rec.get("argv") if rec else None
            want_w = P[row.get("wrapper", "wrapper")] if row else None       # amendment 3: rows g and h carry the pre-roll wrapper rzgfhq, rows a to f rzgfh
            base = os.path.basename(want_w or "")
            w_ok = bool(argv and len(argv) > 1 and argv[1] == want_w and base == ("v8f_t1_eval_reader_rule_rzgfhq.py" if arm in P_ARMS else "v8f_t1_eval_reader_rule_rzgfh.py"))
            good = bool(row and argv == row["argv"] and w_ok)
            per[key] = {"recorded": rec is not None, "row_in_queue": row is not None, "wrapper_in_argv": argv[1] if argv and len(argv) > 1 else None, "wrapper_ok": w_ok, "PASS": good}
            ok &= good
    return {"per_run": per, "PASS": bool(ok and per)}


def k4_environment(hw):
    lv = ((hw or {}).get("levels") or {}).get("U_RLcK") or {}
    try:
        u0, c0 = float(lv["cells"]["U"]["per_seed"][0]), float(lv["cells"]["C"]["per_seed"][0])
    except Exception:                                                                           # noqa: BLE001
        return {"PASS": False, "why": "level of U_RLcK seed 0 not in the headway result"}
    return {"U_seed0": round(u0, 4), "C_seed0": round(c0, 4), "banked_U": K4_U, "banked_C": K4_C, "tolerance": K4_TOL,
            "PASS": bool(abs(u0 - K4_U) <= K4_TOL and abs(c0 - K4_C) <= K4_TOL)}


def k5_reader_input(a):
    w = sha(a.wrapper)
    wp = sha(a.wrapper_pre)
    return dict(K5_FINDING, wrapper_file=os.path.basename(a.wrapper or ""), wrapper_sha256=w[:16],
                reader_input_is_eval_cache_tokens=bool(K5_FINDING["reader_reads_model_latent"] is False),
                wrapper_pre_sha256=wp[:16], PASS=bool(w.startswith(WRAPPER_SHA_PREFIX) and wp.startswith(WRAPPER_PRE_SHA_PREFIX)))


def k6_identity(a, seeds, inputs):
    per, ok = {}, True
    for m in seeds:
        rh = ((inputs or {}).get("inputs/RH_s%d/ckpt_last.pt" % m) or {}).get("sha256")
        ue = ((inputs or {}).get("inputs/UE_s%d/ckpt_last.pt" % m) or {}).get("sha256")
        fa, fe = dump_files(a.pull, "H_R0K", m), dump_files(a.pull, "U_RLcK", m)
        ha = {os.path.basename(f): sha(f) for f in fa}
        he = {os.path.basename(f): sha(f) for f in fe}
        identical = bool(ha and ha == he)
        good = bool(rh and ue and rh != ue and fa and fe and not identical)
        per["s%d" % m] = {"ckpt_sha_differ": bool(rh and ue and rh != ue), "dump_files_a_e": [len(fa), len(fe)], "dumps_byte_identical": identical, "PASS": good}
        ok &= good
    return {"per_seed": per, "note": "run a = H_R0K, run e = U_RLcK; every episode dump file hashed", "PASS": bool(ok and per)}


# ============================================================================================ engines
def sub_read(a, script_dir, script, out_dir, pairs, log_name, extra=()):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(HEADWAY_Q3DIR=a.q3_dir, HIERREP_STAGEB=a.stageb_dir, TANITEVAL_ROOT=a.tev_dir, HEADWAY_JOIN=a.join, HEADWAY_BANK=a.bank,
               CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2", PYTHONIOENCODING="utf-8", PYTHONDONTWRITEBYTECODE="1")
    seeds = ",".join(str(s) for s in a.seeds)
    cmd = [sys.executable, script, "read", "--t1-root", a.pull, "--pairs", pairs, "--seeds", seeds, "--spread-arm", SPREAD_ARM, "--spread-seeds", seeds, "--out-dir", out_dir] + list(extra)
    with open(os.path.join(os.path.dirname(out_dir), log_name), "wb") as lf:
        p = subprocess.run(cmd, cwd=script_dir, env=env, stdout=lf, stderr=subprocess.STDOUT)
    return p.returncode, cmd


def strat_cells_pair(S, cells, A, B, eid):
    pr = S.pair_read(cells, A, B, eid, ticks=(2,))
    pos = [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed"), "per_seed": b.get("per_seed")}
           for arc, b in zip(S.ARCS, pr["tick2"]["pos"])]
    return {"pos_30_50_80m_third_tick": pos, "reading": strat_reading(pos)}


def episode_map_for(S, pull, arm, m, T):
    key = {}
    for f in dump_files(pull, arm, m):
        with np.load(f, allow_pickle=True) as z:
            for e, w, c in zip(z["eid"], z["ws"], z["clip_index"]):
                key[(int(c), int(w))] = int(e)
    o = S.window_order(T)[0]
    return np.array([key.get((int(T["ci"][r]), int(T["t0"][r])), -1) for r in o], np.int64), len(key)


def strategic_engine(a, runs, pairs, s3_arm, p_of, keys=None, log=print):
    """the instrument of card ADD66 (strat_roll_read.py, unchanged), pointed at the runs of this card. runs: arm names; p_of: arm -> pre-roll steps (0 = none);
    keys: [N,2] (clip index, t0) of a restricted grid or None for the full grid. Returns dict (controls, pair reads)."""
    sys.path.insert(0, INTEG)
    import strat_roll_read as S
    R = None
    if keys is not None:
        import read_longroll10k as R
    S.SEEDS = tuple(a.seeds)
    S.tick_path = lambda W, run, k: os.path.join(a.pull, key_of(run, k), "ticks", "ticks_%s_m10_causal_heads.npz" % key_of(run, k))
    if keys is not None:
        S.N_WIN, S.N_ROWS, S.N_EP = int(len(keys)), 3 * int(len(keys)), int(len(np.unique(keys[:, 0])))
        stats = {}
        orig_fields = S.FIELDS

        def load_sel(path):
            z = np.load(path, allow_pickle=True)
            T = {k: z[k] for k in orig_fields}
            T2, st = R.restrict_ticks(T, keys)
            stats[os.path.basename(os.path.dirname(os.path.dirname(path)))] = st
            return T2
        S.load_ticks = load_sel
    sa = argparse.Namespace(w=a.pull, tracks=a.tracks, join=a.join, tree=a.tree_ov)
    ctx = S.Ctx(sa)
    ctx.load_static()
    C = {"s4": S.run_s4()}
    s1 = {"max_dxy_m": 0.0, "max_dpsi_rad": 0.0, "per_file": {}}
    s2 = {"max_first_tick_anchor_dist_m": 0.0, "n_rows_anchor_not_t0": 0, "files": []}
    s6 = {"per_file": {}, "pass": True}
    inputs, missing, ref = {}, [], None
    for run in runs:
        for k in S.SEEDS:
            nm = key_of(run, k)
            p = S.tick_path(a.pull, run, k)
            if not os.path.isfile(p):
                missing.append(nm)
                continue
            inputs[nm] = {"path": p, "sha256": sha(p)}
            T = S.load_ticks(p)
            rbt = S.window_order(T)
            if ref is None:
                ref = {"ci": T["ci"][rbt[0]], "t0": T["t0"][rbt[0]]}
            ok6, _ = S.structure(T, nm, ref)
            exact = True
            if keys is not None and p_of.get(run, 0) > 0:
                exact = bool(int(len(np.load(p, allow_pickle=True)["tick"])) == 3 * len(keys))
            s6["per_file"][nm] = bool(ok6 and exact)
            s6["pass"] &= bool(ok6 and exact)
            dxy, dps = S.s1_row_check(T)
            s1["per_file"][nm] = [dxy, dps]
            s1["max_dxy_m"], s1["max_dpsi_rad"] = max(s1["max_dxy_m"], dxy), max(s1["max_dpsi_rad"], dps)
            if p_of.get(run, 0) == 0:
                st, ds = S.anchors_for(ctx, T, rbt, ticks=(0,))[0]
                s2["n_rows_anchor_not_t0"] += int((st != T["t0"][rbt[0]]).sum())
                s2["max_first_tick_anchor_dist_m"] = max(s2["max_first_tick_anchor_dist_m"], float(ds.max()))
                s2["files"].append(nm)
    s1["pass"] = bool(s1["max_dxy_m"] <= 1e-3 and s1["max_dpsi_rad"] <= 1e-3 and s1["per_file"])
    s2["pass"] = bool(s2["max_first_tick_anchor_dist_m"] <= 1e-6 and s2["n_rows_anchor_not_t0"] == 0 and s2["files"])
    C["s1"], C["s2"], C["s6"] = s1, s2, s6
    C["s9"] = {"pass": bool(inputs) and not missing, "n_files": len(inputs), "missing": missing}
    nm3 = key_of(s3_arm, 0)
    if nm3 in inputs:
        T0 = S.load_ticks(inputs[nm3]["path"])
        C["s3"] = S.s3_check(ctx, T0, S.window_order(T0), tree_ov=a.tree_ov)
    else:
        C["s3"] = {"pass": False, "computable": False, "reason": "%s tick file absent" % nm3}
    stage_a = bool(C["s4"]["pass"] and s1["pass"] and s2["pass"] and s6["pass"] and C["s9"]["pass"] and C["s3"].get("pass") is True)
    out = {"controls": C, "inputs": inputs, "stage_A_pass": stage_a, "runs": list(runs), "seeds": list(S.SEEDS), "windows": None if keys is None else int(len(keys)), "pairs": {}}
    if not stage_a:
        out["ALL_PASS"] = False
        return out
    cells = {}
    for run in runs:
        for k in S.SEEDS:
            T = S.load_ticks(S.tick_path(a.pull, run, k))
            c, _st, _ds = S.compute_cells(ctx, T)
            cells[(run, k)] = c
            log("cells %s done" % key_of(run, k))
    ref_arm = s3_arm
    eid, n_map = episode_map_for(S, a.pull, ref_arm, 0, S.load_ticks(S.tick_path(a.pull, ref_arm, 0)))
    C["episode_map"] = {"all_windows_mapped": bool((eid >= 0).all()), "n_episodes": int(len(np.unique(eid))), "n_windows": int(len(eid))}
    # s5: the first tick is the logged state, so runs of one model that start from it read the same first-tick cells whatever their readout or rule
    groups = {}
    for run in runs:
        if p_of.get(run, 0) == 0:
            groups.setdefault(run[0], []).append(run)
    d5, n5 = 0.0, 0
    for fam, rs in groups.items():
        for run in rs[1:]:
            for k in S.SEEDS:
                for cell in S.CELLS:
                    x, y = cells[(run, k)][cell][0], cells[(rs[0], k)][cell][0]
                    n5 += 1
                    if not (np.isnan(x) == np.isnan(y)).all():
                        d5 = float("inf")
                    else:
                        ok = np.isfinite(x)
                        d5 = max(d5, float(np.abs(x[ok] - y[ok]).max()) if ok.any() else 0.0)
    C["s5"] = {"pass": bool(d5 == 0.0), "largest_difference": d5, "n_comparisons": n5,
               "note": "first-tick cells equal across the runs of one model that start from the logged state (pre-rolled runs excluded); n_comparisons 0 means no group had two such runs (not judged)"}
    out["stage_B_pass"] = bool(C["s5"]["pass"] and C["episode_map"]["all_windows_mapped"] and C["episode_map"]["n_episodes"] == S.N_EP)
    out["ALL_PASS"] = out["stage_B_pass"]
    if out["stage_B_pass"]:
        for nm, (A, B) in pairs.items():
            out["pairs"][nm] = strat_cells_pair(S, cells, A, B, eid)
    return out


def c5_engine(a, log=print):
    """the 1,271-window grid: headway and four-family engines, controls and tick-file controls of read_longroll10k_n4.py (landed, unchanged), driven in memory on the arms of this card"""
    sys.path.insert(0, INTEG)
    env = {"HEADWAY_Q3DIR": a.q3_dir, "HIERREP_STAGEB": a.stageb_dir, "TANITEVAL_ROOT": a.tev_dir, "HEADWAY_JOIN": a.join, "HEADWAY_BANK": a.bank}
    for k, v in env.items():
        os.environ[k] = v
    for d in (a.headway_dir, a.q3_dir, a.stageb_dir):
        if d not in sys.path:
            sys.path.insert(0, d)
    import read_longroll10k_n4 as N4
    R = N4.R
    R.SEEDS = tuple(a.seeds)
    R.key_of = key_of
    for arm in P_ARMS:
        for m in a.seeds:
            R.ALL_RUNS[key_of(arm, m)] = ("RLc", "A", 30 if arm.endswith("P30") else 0, "deadreckon", "30", "plan", 5, 0.2)
    ns = argparse.Namespace(pull=a.pull, code=a.code, n_boot=2000, n_boot_control=300, w=a.pull, tree=a.tree_ov)
    E = R.engine(ns)
    grid = json.load(io.open(a.grid_p30, encoding="utf-8"))
    clip_names = [str(c) for c in np.load(a.join, allow_pickle=True)["clip"]]
    keys = R.grid_keys(grid, clip_names)
    out = R.read_grid(ns, E, "30", keys, list(P_ARMS), SPREAD_ARM_C5, {"C5": C5_PAIR}, {}, log)
    cn = N4.controls_n4()
    res = {"restriction": out.get("restriction"), "counts": out.get("counts"), "controls_engines": out.get("controls"), "controls_n4": cn, "spread_arm": SPREAD_ARM_C5,
           "levels": out.get("levels"), "pair": out.get("pairs", {}).get("C5"), "windows": int(len(keys))}
    eng_ok = bool((out.get("controls") or {}).get("ALL_PASS") is True and all(v.get("PASS") is True for v in cn.values()))
    res["engine_controls_pass"] = eng_ok
    sres = strategic_engine(a, list(P_ARMS), {"C5": C5_PAIR}, "H_RLcA_P0", {"H_RLcA_P30": 30, "H_RLcA_P0": 0}, keys=keys, log=log)
    res["strategic"] = sres
    res["ALL_PASS"] = bool(eng_ok and sres.get("ALL_PASS") is True)
    return res


def run_engines(a):
    """runs the landed readers; every failure is recorded in the files and withholds names later"""
    os.makedirs(a.read_dir, exist_ok=True)
    pairs = ",".join("%s:%s" % (A, B) for A, B in PAIRS.values())
    status = {}
    rc, cmd = sub_read(a, a.headway_dir, "headway_read_r2.py", os.path.join(a.read_dir, "hw"), pairs, "hw.log")
    status["headway"] = {"rc": rc, "cmd": cmd}
    rc, cmd = sub_read(a, a.q3_dir, "q3_read_r3.py", os.path.join(a.read_dir, "q3"), pairs, "q3.log", extra=("--same-sign",))
    status["families"] = {"rc": rc, "cmd": cmd}
    log = lambda s: print(s, flush=True)                                                        # noqa: E731
    os.makedirs(os.path.join(a.read_dir, "strat"), exist_ok=True)
    os.makedirs(os.path.join(a.read_dir, "c5"), exist_ok=True)
    for nm, fn, folder, fname in (("route_goal", lambda: strategic_engine(a, ["H_R0K", "H_R0A", "H_RLcA", "H_RQ50A", "U_RLcK", "U_RQ50K"], dict(PAIRS), "U_RLcK", {}, None, log), "strat", "STRAT_READ.json"),
                                  ("c5", lambda: c5_engine(a, log), "c5", "C5_READ.json")):
        try:
            res = fn()
        except BaseException as ex:                                                             # noqa: BLE001
            res = {"ALL_PASS": False, "error": "%s: %s" % (type(ex).__name__, str(ex)[:400])}
        json.dump(conv(res), io.open(os.path.join(a.read_dir, folder, fname), "w", encoding="utf-8", newline="\n"), indent=1)
        status[nm] = {"ALL_PASS": res.get("ALL_PASS")}
    json.dump(conv(status), io.open(os.path.join(a.read_dir, "ENGINES_STATUS.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    return status


# ============================================================================================ evaluate
def pair_block(hw, q3, st, A, B, nm):
    tag = "%s_minus_%s" % (A, B)
    hc = ucf(hw["pairs"][tag]["cells"])
    fc = cells_of(q3["pairs"][tag]["families"])
    fams = {f: reading(fc, f) for f in NAMED}
    sp = st["pairs"][nm]
    fams["STRATEGIC"] = sp["reading"]
    return {"A": A, "B": B, "headway": hc, "family_cells": fc, "strategic": sp, "families": fams}


def judge(blocks, k5_ok, c5blk):
    """checks and names from the pair blocks; pure function of the numbers (the mutation test flips a sign here)"""
    out, names, failed = {}, {}, []
    b = blocks["C1"]
    worse = [f for f in FAMS if b["families"][f] == "WORSE"]
    c1 = bool(not worse and not above(b["headway"]["F"]))
    out["C1"] = {"families_WORSE": worse, "F_higher": above(b["headway"]["F"]), "PASS": c1}
    names["C1"] = "%s-C1-%s" % (PRE, "PASS" if c1 else "FAIL")
    for tag, nm, key in (("LABEL", "C2_LABEL", "C2_LABEL"), ("READER", "C2_READER", "C2_READER")):
        if tag == "READER" and not k5_ok:
            names[key] = "%s-C2-READER-NOT-READ" % PRE
            out[key] = {"PASS": None, "why": "k5: the reader form reads the model latent; rows d and f are not read; C2 is read on the label form alone"}
            continue
        b = blocks[nm]
        worse = [f for f in FAMS if b["families"][f] == "WORSE"]
        ok = bool(not worse and not above(b["headway"]["U"]))
        out[key] = {"families_WORSE": worse, "U_higher": above(b["headway"]["U"]), "PASS": ok}
        names[key] = "%s-C2-%s-%s" % (PRE, tag, "PASS" if ok else "FAIL")
    sreading = blocks["C2_LABEL"]["strategic"]["reading"]
    c3 = bool(sreading == "BETTER" or (sreading == "NOT-SEPARATED" and out["C1"]["PASS"] and out["C2_LABEL"]["PASS"]))
    out["C3"] = {"route_goal_reading_30_50_80m_third_tick": sreading, "arcs": [(x["arc_m"], x["verdict"]) for x in blocks["C2_LABEL"]["strategic"]["pos_30_50_80m_third_tick"]], "PASS": c3,
                 "rule": "BETTER, or NOT-SEPARATED with C1 and C2 (label form) passed"}
    names["C3"] = "%s-C3-%s" % (PRE, "PASS" if c3 else "FAIL")
    fam5 = c5blk["families"]
    u5 = u_name(c5blk["headway"]["U"])
    bad5 = [f for f in FAMS if fam5[f] not in ("BETTER", "NOT-SEPARATED")]
    c5 = bool(not bad5 and u5 in ("NOT-SEPARATED", "LOWER"))
    out["C5"] = {"families": fam5, "families_not_better_or_not_separated": bad5, "U": u5, "PASS": c5}
    names["C5"] = "%s-C5-%s" % (PRE, "PASS" if c5 else "FAIL")
    for chk in ("C1", "C2_LABEL", "C3", "C5"):
        if out[chk]["PASS"] is not True:
            failed.append(chk)
    if out["C2_READER"]["PASS"] is False:
        failed.append("C2_READER")
    overall = bool(not [c for c in ("C1", "C2_LABEL", "C3", "C5") if out[c]["PASS"] is not True])
    names["OVERALL"] = "%s-SCALE-CANDIDATE" % PRE if overall else "%s-NOT-YET" % PRE
    out["OVERALL"] = {"name": names["OVERALL"], "failed_checks": failed, "next_lever": None if overall else
                      "per Rule Zero the lead writes the next lever for each failed check (%s) from the per-family cells; this reader does not invent one" % ", ".join(failed)}
    return out, names


def evaluate(a):
    seeds = tuple(a.seeds)
    queue = load_queue(a.queue)
    inputs = jl(a.inputs)
    grid_n = len(json.load(io.open(a.grid_p30, encoding="utf-8"))["order"]) if os.path.isfile(a.grid_p30) else a.expect_files
    hw = jl(os.path.join(a.read_dir, "hw", "RESULT_HEADWAY.json"))
    q3 = jl(os.path.join(a.read_dir, "q3", "RESULT_Q3.json"))
    st = jl(os.path.join(a.read_dir, "strat", "STRAT_READ.json"))
    c5 = jl(os.path.join(a.read_dir, "c5", "C5_READ.json"))
    C = {}
    C["k1_complete"] = k1_complete(a, seeds, grid_n)
    C["k2_models"] = k2_models(a, seeds, queue, inputs)
    C["k3_argv"] = k3_argv(a, seeds, queue)
    C["k4_environment"] = k4_environment(hw)
    C["k5_reader_input"] = k5_reader_input(a)
    C["k6_identity"] = k6_identity(a, seeds, inputs)
    cnt = (hw or {}).get("counts", {})
    C["e_reader_controls"] = {"headway": bool(hw and (hw.get("controls") or {}).get("ALL_PASS")), "families": bool(q3 and (q3.get("controls") or {}).get("ALL_PASS")),
                              "route_goal": bool(st and st.get("ALL_PASS") is True), "c5_grid_engines_and_tick_controls": bool(c5 and c5.get("ALL_PASS") is True),
                              "headway_counts": {"n_windows_total": cnt.get("n_windows_total"), "n_lead_windows": cnt.get("n_lead_windows"), "PASS": cnt.get("PASS")}}
    C["e_reader_controls"]["PASS"] = bool(C["e_reader_controls"]["headway"] and C["e_reader_controls"]["families"] and C["e_reader_controls"]["route_goal"]
                                          and C["e_reader_controls"]["c5_grid_engines_and_tick_controls"] and cnt.get("PASS") is True and cnt.get("n_windows_total") == 2399)
    ok = all(v.get("PASS") is True for v in C.values())
    res = {"card": CARD, "tier": TIER_LINE, "seeds": list(seeds), "spread_arm": SPREAD_ARM, "spread_arm_c5": SPREAD_ARM_C5,
           "spread_arm_note": "STATED CHOICE: the convention = the line model under readout K on eval-141 (U_RLcK); on the C5 grid the arm H_RLcA_P0",
           "c2_reader_note": "rows d and f of model seed 2 use the reader TK16_s0 and the map QMAP_LR16_s0_q50 (AM2): they share their reader with model seed 0",
           "k5_finding": K5_FINDING, "controls": C, "controls_all_pass": bool(ok), "names": None, "checks": None, "pairs": None, "levels": None, "c5": None, "line_model_reference": None}
    print(TIER_LINE)
    print("== controls (k1 to k6 and the reader controls)")
    for k, v in C.items():
        print("   %s %s %s" % (k, "PASS" if v.get("PASS") is True else "FAIL", json.dumps(conv({q: w for q, w in v.items() if q != "per_run"}))[:500]))
    if ok:
        blocks = {nm: pair_block(hw, q3, st, A, B, nm) for nm, (A, B) in PAIRS.items()}
        pf, ph = c5["pair"], c5["pair"]
        c5blk = {"A": C5_PAIR[0], "B": C5_PAIR[1], "headway": ucf(ph["headway"]), "family_cells": cells_of(pf["family"]), "strategic": c5["strategic"]["pairs"]["C5"]}
        c5blk["families"] = {f: reading(c5blk["family_cells"], f) for f in NAMED}
        c5blk["families"]["STRATEGIC"] = c5blk["strategic"]["reading"]
        checks, names = judge(blocks, C["k5_reader_input"]["reader_input_is_eval_cache_tokens"], c5blk)
        lv = hw["levels"]
        res.update(names=names, checks=checks, pairs=blocks, c5={"pair": c5blk, "windows": c5.get("windows"), "levels": c5.get("levels")},
                   levels={arm: {c: [round(float(x), 4) for x in lv[arm]["cells"][c]["per_seed"]] for c in ("U", "C", "F")} for arm in lv})
        ref = jl(a.line_ref) if a.line_ref else None
        res["line_model_reference"] = ref if ref is not None else "not supplied to the reader (section 5 of CANDIDATE_SPEC_V8F_10K_2026-11-20.md); the lead puts the line-model cells beside the C5 cells"
        print("== names")
        for k, v in names.items():
            print("   %-10s %s" % (k, v))
        for nm, b in blocks.items():
            print("== %s: %s minus %s" % (nm, b["A"], b["B"]))
            print("   U %s  C %s  F %s" % (b["headway"]["U"], b["headway"]["C"], b["headway"]["F"]))
            print("   families %s" % b["families"])
        print("== C5 families %s U %s" % (c5blk["families"], u_name(c5blk["headway"]["U"])))
    else:
        print("== a control did not pass: no name, no level, no difference; the read is reported as written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(conv(res), io.open(a.out, "w", encoding="utf-8", newline="\n"), indent=1)
    return 0 if ok else 2


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull", required=True)
    ap.add_argument("--read-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-engines", action="store_true")
    ap.add_argument("--expect-files", type=int, default=141)
    ap.add_argument("--queue", default=DEFAULT_QUEUE)
    ap.add_argument("--inputs", default=DEFAULT_INPUTS)
    ap.add_argument("--wrapper", default=INTEG + "/v8f_t1_eval_reader_rule_rzgfh.py")
    ap.add_argument("--wrapper-pre", default=INTEG + "/v8f_t1_eval_reader_rule_rzgfhq.py")
    ap.add_argument("--grid-p30", default=RH10 + "/pins/grid_p30.json")
    ap.add_argument("--join", default=RH10 + "/pins/join_eval141.npz")
    ap.add_argument("--bank", default=V8 + "/res/cache/frames.npz")
    ap.add_argument("--tracks", default="")
    ap.add_argument("--tree-ov", default=RH10 + "/tree")
    ap.add_argument("--code", default="")
    ap.add_argument("--headway-dir", default=V8 + "/headway")
    ap.add_argument("--q3-dir", default=V8 + "/spatial/q3")
    ap.add_argument("--stageb-dir", default=V8 + "/pod/stageB")
    ap.add_argument("--tev-dir", default=RH10 + "/tree/taniteval")
    ap.add_argument("--line-ref", default="")
    a = ap.parse_args(argv)
    a.seeds = tuple(int(x) for x in a.seeds.split(",") if x != "")
    if not a.code:                                                                              # read_longroll10k.engine wants <code>/{headway,q3,stageB,tev}
        a.code = os.path.join(a.read_dir, "_code")
    return a


def main(argv=None):
    a = parse(argv)
    if not a.skip_engines:
        run_engines(a)
    return evaluate(a)


if __name__ == "__main__":
    sys.exit(main())
