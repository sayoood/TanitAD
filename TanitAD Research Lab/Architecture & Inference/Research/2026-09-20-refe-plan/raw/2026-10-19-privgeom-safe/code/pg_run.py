"""PRIVGEOM-SAFE stages after the (label-free) extraction:  join -> gate -> fit -> analyze.

  join     LABEL-FREE.  scalars/<log>.npz -> run/join/{arrays.npz, feat_sub.npy, join_report.json}; the census / pickrule alignment proofs that need no label.
  gate     FIRST stage that opens a label (census meta `labels6`, only the retained sets).  C-AGREE, the label-identity rates, C-REACH on the real arrays (planted label), the synthetic
           test suites (RED mutations, C-ID-L, C-CONV, C-NULL false-fire) -> run/gate_token.json bound to the prereg, amendment 1, the code sha256s, the subset and the join.
  fit      the OOF fits (resumable; `--shard i --nshards n` splits the task list between at most 2 processes); refuses without a matching token.
  analyze  AUCs, Delta, calls, controls, the pick-level read -> run/results/{results.json, RESULT_tables.md}.
Python: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore ; CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_lib as P                                                                       # noqa: E402

FL = P.FL
ROOT = os.path.normpath(os.path.join(HERE, ".."))
SMOKE = bool(os.environ.get("PG_SMOKE"))            # infrastructure smoke on the logs finished so far; labels := the RECOMPUTED verdicts (never the stored labels)
RUN = os.environ.get("PG_RUN") or f"{ROOT}/run"
META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
FEAT = "C:/Users/Admin/par18/fastneg/run/inputs/feat_f10.npy"
PICK_ARR = "C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz"
EP_COLS = "C:/Users/Admin/par18/pickrule/run/stage/ep_columns.npz"
PREREG = f"{ROOT}/PREREG_PRIVGEOM_SAFE.md"
AMEND1 = f"{ROOT}/PREREG_PRIVGEOM_SAFE_AMENDMENT1.md"
CODE_FILES = ("pg_geom.py", "pg_extract.py", "pg_select.py", "pg_lib.py", "pg_run.py", "test_pg_geom.py", "test_pg_lib.py")
REF_FILES = ("fastneg_lib.py", "pickrule_lib.py", "epr1.py")
NC_COLS, DAC_COLS = list(range(11, 15)), list(range(0, 11))
LABEL_COL = {"nc": 0, "dac": 1}


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(path: str) -> str:
    return FL.sha256_file(path)


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def paths():
    return {"scal": os.environ.get("PG_SCAL") or f"{ROOT}/run/scalars", "join": f"{RUN}/join", "arrays": f"{RUN}/join/arrays.npz", "feat": f"{RUN}/join/feat_sub.npy", "labels": f"{RUN}/join/labels_sub.npy",
            "report": f"{RUN}/join/join_report.json", "gate_report": f"{RUN}/gate_report.json", "token": f"{RUN}/gate_token.json", "fits": f"{RUN}/fits", "results": f"{RUN}/results", "subset": f"{ROOT}/run/subset.json"}


def require_prereg() -> dict:
    h = sha(PREREG)
    if h != P.PREREG_SHA256:
        raise P.PgError(f"PREREG sha256 {h} != {P.PREREG_SHA256}: every stage refuses")
    out = {"prereg": h}
    if os.path.exists(AMEND1):
        rec = open(AMEND1.replace(".md", ".sha256"), encoding="utf-8").read().split()[0]
        if sha(AMEND1) != rec:
            raise P.PgError("Amendment 1 does not hash to its recorded sha256")
        out["amend1"] = rec
    return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_join(a) -> int:
    pins = require_prereg()
    Pth = paths()
    os.makedirs(Pth["join"], exist_ok=True)
    sub = json.load(open(Pth["subset"], encoding="utf-8"))
    z = np.load(META, allow_pickle=False)
    if SMOKE:
        have = [n for n in sub["logs_in_order"] if os.path.exists(f"{Pth['scal']}/{n}.npz")]
        kl = z["key_log"]
        sub = dict(sub, logs_in_order=have, census_idx=[i for i in sub["census_idx"] if str(kl[i]) in set(have)])
    M = {k: z[k] for k in ("log_idx", "fold", "code", "z6", "key_log", "key_token", "key_step", "key_rank", "key_ckpt_step", "log_names")}     # never 'labels6'
    where = {(str(l), str(t), int(s), int(r), int(c)): i for i, (l, t, s, r, c) in enumerate(zip(M["key_log"], M["key_token"], M["key_step"], M["key_rank"], M["key_ckpt_step"]))}
    got_g, got_rl, lost, per_log = {}, {}, [], {}
    for name in sub["logs_in_order"]:
        p = f"{Pth['scal']}/{name}.npz"
        if not os.path.exists(p):
            raise P.PgError(f"extraction incomplete: {p} missing")
        d = np.load(p, allow_pickle=False)
        arr = {k: d[k] for k in d.files}
        n = len(arr["token"])
        if arr["g"].shape != (n, 64, 15) or arr["rl6"].shape != (n, 64, 6) or arr["ok"].shape != (n,):
            raise P.PgError(f"{p}: bad shapes {arr['g'].shape}")
        per_log[name] = {"n": int(n), "ok": int(arr["ok"].sum())}
        for i in range(n):
            key = (name, str(arr["token"][i]), int(arr["step"][i]), int(arr["rank"][i]), int(arr["ckpt_step"][i]))
            ci = where[key]
            if arr["ok"][i]:
                got_g[ci], got_rl[ci] = arr["g"][i], arr["rl6"][i]
            else:
                lost.append(ci)
    census_idx = np.array(sub["census_idx"], np.int64)
    keep = np.array([i for i in census_idx if i in got_g], np.int64)
    n_expect, n_keep = len(census_idx), len(keep)
    if len(set(keep.tolist())) != n_keep:
        raise P.PgError("duplicate census index in the join")
    g = np.stack([got_g[i] for i in keep])
    rl6 = np.stack([got_rl[i] for i in keep])
    pk = np.load(PICK_ARR, allow_pickle=False)
    z6_pick, li_pick = pk["z6"][keep], pk["log_idx"][keep]                                  # NOT 'lab'
    ep = np.load(EP_COLS, allow_pickle=False)
    ep0, ep1 = ep["ep_OOF_s0"][keep], ep["ep_OOF_s1"][keep]
    z6 = M["z6"][keep]
    ok_z = bool(np.array_equal(z6_pick, z6)) and bool(np.array_equal(li_pick, M["log_idx"][keep])) and bool(np.array_equal(ep["log_idx"][keep], M["log_idx"][keep]))
    if not ok_z:
        raise P.PgError("C-JOIN FAILED: the pickrule package's z6 / log_idx differ from the census's on the retained sets")
    glob_li = M["log_idx"][keep]
    uniq = np.unique(glob_li)
    local = np.searchsorted(uniq, glob_li)
    fm = np.load(FEAT, mmap_mode="r")
    np.save(Pth["feat"], np.ascontiguousarray(fm[keep]))
    np.savez(Pth["arrays"], idx=keep, g=g, rl6=rl6, z6=z6, fold=M["fold"][keep], code=M["code"][keep], log_idx=local, glob_log_idx=glob_li, ep_s0=ep0, ep_s1=ep1,
             p2_W6=z["p2_W6"], p2_b6=z["p2_b6"], log_names=M["log_names"][uniq])
    rep = {"created": now(), "ok": True, "pins": pins, "n_expected": n_expect, "n_retained": n_keep, "n_lost": int(n_expect - n_keep), "lost_frac": float((n_expect - n_keep) / n_expect), "n_logs": int(len(uniq)),
           "per_log": per_log, "z6_equals_pickrule_z6": ok_z, "arrays_sha256": sha(Pth["arrays"]), "feat_sha256": sha(Pth["feat"]), "subset_sha256": sha(Pth["subset"]),
           "lost_disclose_gt_5pct": bool((n_expect - n_keep) / n_expect > P.LOSS_FRAC_DISCLOSE)}
    jdump(rep, Pth["report"])
    log(f"join OK: {n_keep}/{n_expect} sets in {len(uniq)} logs")
    return 0


def load_join(labels: bool = False) -> dict:
    Pth = paths()
    z = np.load(Pth["arrays"], allow_pickle=False)
    A = {k: z[k] for k in z.files}
    A["feat"] = np.load(Pth["feat"], mmap_mode="r")
    A["n_logs"] = int(A["log_idx"].max()) + 1
    A["N"] = len(A["fold"])
    if labels:
        A["lab"] = np.load(Pth["labels"], allow_pickle=False)
    return A


def g_for(A: dict, row: str, kind: str, rep: int = 0):
    """The G block [N,64,k] float32 of a task kind: 'G' (primary), 'GV' (the registered no-contact companion), 'plantlabel', 'plantweak', 'null'."""
    cols = NC_COLS if row == "nc" else DAC_COLS
    import pg_geom as G
    g = A["g"][..., cols].astype(np.float64)
    if kind == "G":
        return g.astype(np.float32)
    if kind == "GV":
        return (G.floored(g) if row == "nc" else G.unsigned(A["g"][..., :11].astype(np.float64))).astype(np.float32)
    lab = A["lab"][..., LABEL_COL[row]]
    if kind == "plantlabel":
        return lab[..., None].astype(np.float32)
    if kind == "plantweak":
        return P.weak_plant(lab, LABEL_COL[row])
    if kind == "null":
        return P.permute_within_set(g.astype(np.float32), P.NULL_SEED_BASE + P.NULL_SALT * rep)
    raise P.PgError(kind)


def task_list() -> list:
    T = []
    for row in P.ROWS:
        for s in P.SEEDS:
            T += [dict(name=f"L_{row}_s{s}", arm="L", row=row, seed=s, kind=None), dict(name=f"LG_{row}_s{s}", arm="LG", row=row, seed=s, kind="G"), dict(name=f"G_{row}_s{s}", arm="G", row=row, seed=s, kind="G"),
                  dict(name=f"LGV_{row}_s{s}", arm="LG", row=row, seed=s, kind="GV"), dict(name=f"plantweak_{row}_s{s}", arm="LG", row=row, seed=s, kind="plantweak")]
            T += [dict(name=f"null{r}_{row}_s{s}", arm="LG", row=row, seed=s, kind="null", rep=r) for r in range(P.NULL_REPS)]
    return T


def plant_tasks() -> list:
    return [dict(name=f"plantlabel_{arm}_{row}_s{s}", arm=arm, row=row, seed=s, kind="plantlabel") for row in P.ROWS for s in P.SEEDS for arm in ("LG", "G")]


def run_task(A: dict, t: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{t['name']}.npy"
    if os.path.exists(out) and os.path.exists(out.replace(".npy", ".json")):
        return {"skipped": True}
    col = LABEL_COL[t["row"]]
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(A["lab"][..., col])
    g = None if t["kind"] is None else g_for(A, t["row"], t["kind"], t.get("rep", 0))
    t0 = time.time()
    trace: list = []
    oof = P.oof_logits(t["arm"], A["feat"], z_st, y, g, A["fold"], t["seed"], A["p2_W6"][col], float(A["p2_b6"][col]), ram_guard=lambda: FL.wait_for_ram(3.0, log=log), trace=trace)
    np.save(out, oof)
    jdump({**{k: v for k, v in t.items()}, "seconds": time.time() - t0, "oof_sha256": sha(out), "k": 0 if g is None else int(g.shape[-1]), "folds": trace}, out.replace(".npy", ".json"))
    return {"skipped": False, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def code_pins() -> dict:
    return {"code": {n: sha(f"{HERE}/{n}") for n in CODE_FILES if os.path.exists(f"{HERE}/{n}")}, "ref": {n: sha(f"{ROOT}/ref/{n}") for n in REF_FILES}}


def stage_gate(a) -> int:
    pins = require_prereg()
    if "amend1" not in pins and not SMOKE:
        raise P.PgError("Amendment 1 (the code pins) is not sealed: the gate is the first stage that opens a label and refuses")
    Pth = paths()
    rep_join = json.load(open(Pth["report"], encoding="utf-8"))
    if sha(Pth["arrays"]) != rep_join["arrays_sha256"]:
        raise P.PgError("join arrays changed since join")
    A = load_join()
    z = np.load(META, allow_pickle=False)
    if SMOKE:
        lab = np.clip(A["rl6"], 0.0, 1.0).astype(np.float32)                                  # smoke: the recomputed verdicts, NOT the stored labels
    else:
        lab = np.ascontiguousarray(z["labels6"][A["idx"]])                                  # THE FIRST LABEL READ (retained sets only)
    np.save(Pth["labels"], lab)
    A["lab"] = lab
    log(f"gate: labels opened for {len(lab)} retained sets")
    G = {}
    # ---- C-AGREE
    rl = np.clip(A["rl6"], 0.0, 1.0)
    agree = {}
    for r, c in (("nc", 0), ("dac", 1)):
        agree[r] = {"exact_frac": float((rl[..., c] == lab[..., c]).mean()), "n_prop": int(lab[..., c].size), "pass": bool((rl[..., c] == lab[..., c]).mean() >= P.AGREE_MIN)}
    G["C-AGREE"] = agree
    # ---- label-identity rates of s3.4 (descriptive)
    g = A["g"]
    dac0, nc0 = lab[..., 1] == 0.0, lab[..., 0] == 0.0
    G["label_identity"] = {"P(dac_frac_out>0 | DAC=0)": float((g[..., 1][dac0] > 0).mean()), "P(dac_frac_out=0 | DAC=1)": float((g[..., 1][~dac0] == 0).mean()),
                           "P(dac_clear_min<0 | DAC=0)": float((g[..., 0][dac0] < 0).mean()), "P(dac_clear_min>=0 | DAC=1)": float((g[..., 0][~dac0] >= 0).mean()),
                           "P(nc_min_dist=0 | NC=0)": float((g[..., 11][nc0] == 0).mean()), "P(nc_min_dist>0 | NC=1)": float((g[..., 11][lab[..., 0] == 1.0] > 0).mean()),
                           "n_dac0": int(dac0.sum()), "n_nc0": int(nc0.sum())}
    # ---- the synthetic suites (RED mutations, C-ID-L, C-CONV, C-NULL)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1")
    suites = {}
    interp = {"test_pg_geom.py": sys.executable, "test_pg_lib.py": sys.executable}                                    # every stage runs in driverl-eval (shapely + torch)
    for tf in ("test_pg_geom.py", "test_pg_lib.py"):
        r = subprocess.run([interp[tf], "-B", "-W", "ignore", f"{HERE}/{tf}"], capture_output=True, text=True, env=env, cwd=HERE)
        suites[tf] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-600:]}
    G["suites"] = suites
    # ---- C-REACH on the real arrays (planted label through the real fit / score code)
    an = P.Analyzer(lab, A["code"], A["log_idx"], A["n_logs"])
    reach = {}
    for t in plant_tasks():
        run_task(A, t, Pth["fits"])
        oof = np.load(f"{Pth['fits']}/{t['name']}.npy")
        reach[t["name"]] = float(an.auc(an.per_log_stats(oof, t["row"]), "Q4")["point"])
    G["C-REACH"] = {"F_auc": reach, "min": P.REACH_MIN_AUC, "pass": bool(all(v >= P.REACH_MIN_AUC for v in reach.values()))}
    G["C-RED_branch_table_failures"] = P.branch_table_failures()
    gate_pass = bool(G["C-REACH"]["pass"] and all(s["rc"] == 0 for s in suites.values()) and not G["C-RED_branch_table_failures"] and rep_join["z6_equals_pickrule_z6"]) or SMOKE
    token = {"created": now(), "pass": gate_pass, "pins": pins, "code_pins": code_pins(), "subset_sha256": sha(Pth["subset"]), "arrays_sha256": rep_join["arrays_sha256"], "labels_sub_sha256": sha(Pth["labels"]), "gate": G}
    jdump(G, Pth["gate_report"])
    jdump(token, Pth["token"])
    log(f"gate {'PASS' if gate_pass else 'FAIL'}: agree {agree}  reach {reach}  suites { {k: v['rc'] for k, v in suites.items()} }")
    return 0 if gate_pass else 5


def require_token() -> dict:
    Pth = paths()
    tok = json.load(open(Pth["token"], encoding="utf-8"))
    if not tok["pass"]:
        raise P.PgError("gate token is a FAIL")
    now_pins = code_pins()
    if now_pins != tok["code_pins"] or sha(Pth["subset"]) != tok["subset_sha256"] or sha(Pth["arrays"]) != tok["arrays_sha256"]:
        raise P.PgError("the code / subset / join differ from what the gate token binds: refused")
    return tok


def stage_fit(a) -> int:
    require_prereg()
    require_token()
    Pth = paths()
    os.makedirs(Pth["fits"], exist_ok=True)
    A = load_join(labels=True)
    T = task_list()[a.shard::a.nshards]
    log(f"fit: shard {a.shard}/{a.nshards}: {len(T)} tasks")
    for t in T:
        r = run_task(A, t, Pth["fits"])
        log(f"  {t['name']}: {'skipped' if r['skipped'] else format(r['seconds'], '.0f') + ' s'}")
    log(f"FIT_SHARD_DONE {a.shard}")
    open(f"{Pth['fits']}/FIT_SHARD_DONE_{a.shard}.txt", "w").write(now())
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_analyze(a) -> int:
    require_prereg()
    tok = require_token()
    import pickrule_lib as PK
    Pth = paths()
    os.makedirs(Pth["results"], exist_ok=True)
    A = load_join(labels=True)
    gate = json.load(open(Pth["gate_report"], encoding="utf-8"))
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    oof = lambda name: np.load(f"{Pth['fits']}/{name}.npy")                              # noqa: E731
    st_cache: dict = {}

    def st(name, row):
        if (name, row) not in st_cache:
            st_cache[(name, row)] = an.per_log_stats(oof(name), row)
        return st_cache[(name, row)]
    R = {"schema": P.SCHEMA, "created": now(), "n_sets": int(A["N"]), "n_logs": int(A["n_logs"]), "rows": {}, "controls": {"gate": gate}, "pins": tok["pins"]}
    for row in P.ROWS:
        rr = {"auc": {}, "delta_F": {}, "delta_Q123": {}, "calls": {}, "variant": {}, "null": {}, "power": {}}
        void_inputs = not gate["C-AGREE"][row]["pass"]
        for s in P.SEEDS:
            for arm in ("L", "LG", "G", "LGV"):
                nm = f"{arm}_{row}_s{s}"
                rr["auc"][nm] = {"F": an.auc(st(nm, row), "Q4"), "Q123": an.auc(st(nm, row), "Q123")}
            dF = an.diff(st(f"LG_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q4")
            dQ = an.diff(st(f"LG_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q123")
            vF = an.diff(st(f"LGV_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q4")
            gF = an.diff(st(f"G_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q4")
            rr["delta_F"][f"s{s}"], rr["delta_Q123"][f"s{s}"] = dF, dQ
            rr["variant"][f"s{s}"] = {"delta_F": vF, "call": P.seed_call(vF)}
            rr["calls"][f"s{s}"] = P.seed_call(dF)
            rr["g_only_minus_L_F"] = rr.get("g_only_minus_L_F", {})
            rr["g_only_minus_L_F"][f"s{s}"] = gF
        call = P.row_call([rr["calls"][f"s{s}"] for s in P.SEEDS])
        rr["variant"]["row_call"] = P.row_call([rr["variant"][f"s{s}"]["call"] for s in P.SEEDS])
        # C-NULL
        fires = []
        for r in range(P.NULL_REPS):
            cis = [an.diff(st(f"null{r}_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q4") for s in P.SEEDS]
            fires.append(P.null_fires(cis))
            rr["null"][f"rep{r}"] = {"delta_F_by_seed": cis, "fires": fires[-1]}
        void_null = P.null_void(fires)
        # C-POWER
        pcis = [an.diff(st(f"plantweak_{row}_s{s}", row), st(f"L_{row}_s{s}", row), "Q4") for s in P.SEEDS]
        pw = P.power_ok(pcis)
        rr["power"] = {"delta_F_by_seed": pcis, "fires_on_both_seeds": pw}
        rr["raw_call"], rr["void_null"], rr["null_fires"], rr["void_inputs"] = call, void_null, fires, void_inputs
        rr["final_call"] = P.final_call(call, void_null, void_inputs, pw)
        rr["variant"]["final_call"] = P.final_call(rr["variant"]["row_call"], void_null, void_inputs, pw)
        R["rows"][row] = rr
    # ---- the pick-level read (DESCRIPTIVE)
    S = PK.Stats(A["log_idx"], an.draws, FL, n_logs=A["n_logs"])
    lab = A["lab"]
    pdms = PK.pdms_from_labels(lab)
    ones = np.ones(A["N"])
    pick: dict = {}
    for s in P.SEEDS:
        ep = A[f"ep_s{s}"]
        z_ep = A["z6"].copy()
        z_ep[..., 2] = ep
        comps = {"P2": A["z6"], "EP": z_ep}
        for arm in ("L", "LG", "G"):
            zc = z_ep.copy()
            zc[..., 0], zc[..., 1] = oof(f"{arm}_nc_s{s}"), oof(f"{arm}_dac_s{s}")
            comps[f"EP[{arm}]"] = zc
            if arm == "LG":
                zn, zd = z_ep.copy(), z_ep.copy()
                zn[..., 0] = oof(f"LG_nc_s{s}")
                zd[..., 1] = oof(f"LG_dac_s{s}")
                comps["EP[LG nc only]"], comps["EP[LG dac only]"] = zn, zd
        fl = {k: PK.pick_flags(lab, pdms, PK.rule_pick(z)) for k, z in comps.items()}
        out = {}
        for k, f in fl.items():
            out[k] = {"nc0_rate": S.rate(f["nc0"]), "dac0_rate": S.rate(f["dac0"]), "unsafe_epr1_rate": S.rate(f["unsafe_epr1"]), "pdms_x100": PK.scale_ci(S.mean(f["pd"]), 100.0)}
        base = fl["P2"]
        for k, f in fl.items():
            if k == "P2":
                continue
            out[k]["vs_P2"] = {"d_nc0": S.rate_diff(f["nc0"], base["nc0"]), "d_dac0": S.rate_diff(f["dac0"], base["dac0"]), "d_unsafe_epr1": S.rate_diff(f["unsafe_epr1"], base["unsafe_epr1"]),
                               "d_pdms_x100": PK.scale_ci(S.ratio_diff(f["pd"], ones, base["pd"], ones), 100.0)}
        for a_, b_ in (("EP[LG]", "EP"), ("EP[LG]", "EP[L]")):
            fa, fb = fl[a_], fl[b_]
            out[f"{a_} minus {b_}"] = {"d_nc0": S.rate_diff(fa["nc0"], fb["nc0"]), "d_dac0": S.rate_diff(fa["dac0"], fb["dac0"]), "d_unsafe_epr1": S.rate_diff(fa["unsafe_epr1"], fb["unsafe_epr1"]),
                                       "d_pdms_x100": PK.scale_ci(S.ratio_diff(fa["pd"], ones, fb["pd"], ones), 100.0)}
        pick[f"s{s}"] = out
    R["pick_level"] = pick
    R["pairs"] = {row: {k: float(v[1].sum()) for k, v in st("L_" + row + "_s0", row).items()} for row in P.ROWS}
    jdump(R, f"{Pth['results']}/results.json")
    write_tables(R, f"{Pth['results']}/RESULT_tables.md")
    log("analyze OK: " + "; ".join(f"{r}: {R['rows'][r]['final_call']} (raw {R['rows'][r]['raw_call']}, variant {R['rows'][r]['variant']['final_call']})" for r in P.ROWS))
    return 0


def fmt(ci: dict, k: int = 4) -> str:
    return f"{ci['point']:+.{k}f} [{ci['lo']:+.{k}f}, {ci['hi']:+.{k}f}]"


def fmtu(ci: dict, k: int = 4) -> str:
    return f"{ci['point']:.{k}f} [{ci['lo']:.{k}f}, {ci['hi']:.{k}f}]"


def write_tables(R: dict, path: str) -> None:
    L = [f"# PRIVGEOM-SAFE tables (generated {R['created']}; {R['n_sets']} sets, {R['n_logs']} logs; paired log-cluster bootstrap, B = {P.BOOT_B}, seed {P.BOOT_SEED})", ""]
    for row in P.ROWS:
        rr = R["rows"][row]
        L += [f"## Row {row.upper()}: final **{rr['final_call']}** (raw {rr['raw_call']}; null fires {rr['null_fires']}; void_null {rr['void_null']}; void_inputs {rr['void_inputs']}; power fires {rr['power']['fires_on_both_seeds']})", "",
              "| seed | F-AUC L | F-AUC L+G | F-AUC G-only | Delta_F (L+G - L) | call | Delta_Q1-3 | G-only - L (F) | companion Delta_F | companion call |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in P.SEEDS:
            a = rr["auc"]
            L.append(f"| {s} | {fmtu(a[f'L_{row}_s{s}']['F'])} | {fmtu(a[f'LG_{row}_s{s}']['F'])} | {fmtu(a[f'G_{row}_s{s}']['F'])} | {fmt(rr['delta_F'][f's{s}'])} | {rr['calls'][f's{s}']} | {fmt(rr['delta_Q123'][f's{s}'])} | "
                     f"{fmt(rr['g_only_minus_L_F'][f's{s}'])} | {fmt(rr['variant'][f's{s}']['delta_F'])} | {rr['variant'][f's{s}']['call']} |")
        L += ["", "Q1-3 AUC: " + "; ".join(f"{k}: {fmtu(v['Q123'])}" for k, v in rr["auc"].items()), "", "C-NULL (Delta_F of the permuted block, seed 0 / seed 1; fires = lower CI > +0.02 on both):", ""]
        for k, v in rr["null"].items():
            L.append(f"- {k}: {fmt(v['delta_F_by_seed'][0])} / {fmt(v['delta_F_by_seed'][1])} fires={v['fires']}")
        L += ["", "C-POWER (weak plant, Delta_F): " + " / ".join(fmt(c) for c in rr["power"]["delta_F_by_seed"]), ""]
    L += ["## Pick-level read (DESCRIPTIVE, no authority)", ""]
    for sk, out in R["pick_level"].items():
        L += [f"### seed {sk}", "", "| composite | NC-unsafe pick | DAC-unsafe pick | unsafe (EP-R1) | PDMS x100 | dPDMS x100 vs P2 | d unsafe(EP-R1) vs P2 |", "|---|---|---|---|---|---|---|"]
        for k, v in out.items():
            if "vs_P2" in v and "nc0_rate" in v:
                L.append(f"| {k} | {fmtu(v['nc0_rate'])} | {fmtu(v['dac0_rate'])} | {fmtu(v['unsafe_epr1_rate'])} | {fmtu(v['pdms_x100'], 3)} | {fmt(v['vs_P2']['d_pdms_x100'], 3)} | {fmt(v['vs_P2']['d_unsafe_epr1'])} |")
            elif "nc0_rate" in v:
                L.append(f"| {k} | {fmtu(v['nc0_rate'])} | {fmtu(v['dac0_rate'])} | {fmtu(v['unsafe_epr1_rate'])} | {fmtu(v['pdms_x100'], 3)} | - | - |")
        for k, v in out.items():
            if " minus " in k:
                L.append(f"\n{k}: d_nc0 {fmt(v['d_nc0'])}, d_dac0 {fmt(v['d_dac0'])}, d_unsafe(EP-R1) {fmt(v['d_unsafe_epr1'])}, dPDMS x100 {fmt(v['d_pdms_x100'], 3)}")
        L.append("")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("join", "gate", "fit", "analyze"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    return {"join": stage_join, "gate": stage_gate, "fit": stage_fit, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
