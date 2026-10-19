"""PRIVGEOM-CLEAN stages:  select -> (frozen `pg_extract.py work` on the 76 logs) -> join -> gate -> fit -> analyze.

  seal-check LABEL-FREE.  Verifies the seal (pgc_seal.check_seal, with the input files): the ONLY thing it does.  Exit 0 iff the seal holds.
  preflight  LABEL-FREE, BEFORE the extraction.  seal-check + the branch table + select; any failure STOPS the run before a CPU-hour is spent.
  select   LABEL-FREE.  coverage.json -> run_clean/subset_clean.json (the 76 unseen logs; the file the FROZEN pg_extract.py `work --subset` reads).
  join     LABEL-FREE.  scalars/<log>.npz -> run/join/{arrays.npz, join_report.json}; keys matched 1:1 to the census bank; the 256-d tokens are read through a row view of the census `feat_f10.npy` (no 356 MB copy).
  gate     PHASE A (LABEL-FREE; a failure STOPS here, no label has been opened): C-JOIN, C-ID-EXTRACT, the synthetic suites (C-GRAD synthetic, C-RED), the branch table, C-GRAD-REAL (one step on the REAL
           centred block with a seeded synthetic label).  PHASE B = the FIRST stage that opens a label (census meta `labels6`, retained sets only): C-AGREE, label-identity rates, C-ADMIT (the planted
           admissibility check, its RED arm), C-REACH on the real arrays.  -> run_clean/gate_token.json bound to the prereg, the code sha256s, the coverage and the join.
  fit      the OOF fits (resumable; `--shard i --nshards n`); refuses without a matching token.
  analyze  Delta_F, calls, controls, final call -> run/results/{results.json, RESULT_tables.md}.
Python: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore ; CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8.
Outputs of the READ go to run_clean/ (the defaults; nothing is overridden by the environment).
Env overrides (rehearsal on the banked SEEN arrays; NEVER the read): PGC_RUN (output dir), PGC_JOIN_DIR (a join dir), PGC_SCAL (scalars dir), PGC_SUBSET / PGC_COVERAGE (the subset / coverage json), PGC_REHEARSAL=1 (skip the prereg-seal check; stamps every output REHEARSAL;
`seal-check` and `preflight` REFUSE to run under it, and the read chain refuses to start with it set).
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pgc_admit as AD                                                                   # noqa: E402
import pgc_lib as C                                                                      # noqa: E402
import pgc_seal as SL                                                                    # noqa: E402

P, FL = C.P, C.FL
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REHEARSAL = bool(os.environ.get("PGC_REHEARSAL"))
RUN = os.environ.get("PGC_RUN") or f"{ROOT}/run_clean"
META = "C:/Users/Admin/par18/fastneg/run/inputs/meta.npz"
FEAT = "C:/Users/Admin/par18/fastneg/run/inputs/feat_f10.npy"
PREREG = f"{ROOT}/PREREG_PRIVGEOM_CLEAN.md"
CODE_FILES = ("pg_geom.py", "pg_extract.py", "pg_lib.py", "pgc_lib.py", "pgc_admit.py", "pgc_run.py", "pgc_coverage.py", "pgc_seal.py", "test_pgc_lib.py", "test_pgc_admit.py", "test_pgc_seal.py")
FROZEN_SUITES = ("test_pg_geom.py", "test_pg_lib.py")
REF_FILES = ("fastneg_lib.py",)
SEAL_REQUIRED = tuple(f"code/{n}" for n in CODE_FILES + FROZEN_SUITES + ("pgc_read_chain.py", "PINS.json")) + tuple(f"ref/{n}" for n in REF_FILES)     # the seal must pin everything the read runs
SEAL_INPUTS = {"census_meta": META, "census_tokens": FEAT, "coverage": f"{ROOT}/run/coverage.json", "plant_calibration": f"{ROOT}/run/plant_calibration.json", "glive_registration": f"{ROOT}/run/glive/glive.json"}
MIN_LOGS_RETAINED = 60                               # of 76: fewer retained logs is a VOID-INPUTS (prereg s7 C-JOIN)


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
    jd = os.environ.get("PGC_JOIN_DIR") or f"{RUN}/join"
    return {"scal": os.environ.get("PGC_SCAL") or f"{ROOT}/run_clean/scalars", "join": jd, "arrays": f"{jd}/arrays.npz", "feat_sub": f"{jd}/feat_sub.npy", "labels": f"{RUN}/join_labels_sub.npy",
            "report": f"{jd}/join_report.json", "gate_report": f"{RUN}/gate_report.json", "token": f"{RUN}/gate_token.json", "fits": f"{RUN}/fits", "results": f"{RUN}/results",
            "coverage": os.environ.get("PGC_COVERAGE") or f"{ROOT}/run/coverage.json", "subset": os.environ.get("PGC_SUBSET") or f"{ROOT}/run_clean/subset_clean.json", "plant": f"{ROOT}/run/plant_calibration.json"}


def require_prereg(inputs: bool = False) -> dict:
    """Every stage refuses unless the seal holds (pgc_seal.check_seal: prereg hash, SEALED status, pinned code, the section-13 table, Revision 1 verbatim, amendments, and with inputs=True the input files)."""
    if REHEARSAL:
        return {"prereg": "REHEARSAL (not sealed)"}
    try:
        return SL.check_seal(ROOT, required=SEAL_REQUIRED, inputs=inputs)["pins"]
    except SL.SealError as e:
        raise C.P.PgError(str(e)) from None


def plant_sigma() -> float:
    d = json.load(open(paths()["plant"], encoding="utf-8"))
    return float(d["sigma_star"])


class RowView:
    """feat[bi] for local set indices through the census `feat_f10.npy` memmap (no copy of the subset)."""

    def __init__(self, mm, idx):
        self.mm, self.idx = mm, np.asarray(idx, np.int64)

    def __getitem__(self, bi):
        bi = np.asarray(bi)
        gi = self.idx[bi]
        order = np.argsort(gi, kind="stable")
        out = np.empty((len(gi),) + self.mm.shape[1:], self.mm.dtype)
        out[order] = self.mm[gi[order]]
        return out


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_seal_check(a) -> int:
    """LABEL-FREE and nothing else: verify the seal (prereg hash + SEALED status + pinned code + table + Revision 1 + amendments + the label-free input files).  Opens no label, no array, no scalar store."""
    if REHEARSAL:
        log("seal-check REFUSES under PGC_REHEARSAL (the rehearsal bypass is not a seal)")
        return 3
    info = require_prereg(inputs=True)
    log(f"SEAL HOLDS: {json.dumps(info)}")
    return 0


def stage_preflight(a) -> int:
    """LABEL-FREE, BEFORE the extraction: the seal, the registered branch table, then select.  A failure stops the run before any CPU-hour or any label."""
    if stage_seal_check(a) != 0:
        return 3
    bad = C.branch_table_failures()
    if bad:
        raise C.P.PgError(f"branch table failures: {bad}")
    log("preflight: branch table clean")
    return stage_select(a)


def stage_select(a) -> int:
    require_prereg()
    Pth = paths()
    cov = json.load(open(Pth["coverage"], encoding="utf-8"))
    doc = {"schema": "privgeom-clean-subset-v1", "from_coverage_sha256": sha(Pth["coverage"]), "n_logs": cov["n_unseen_resolved"], "logs_in_order": cov["logs_in_order"], "n_sets": cov["n_sets"], "n_tokens": cov["n_tokens"],
           "census_idx": cov["census_idx"], "excluded_seen_by_safe": cov["seen_subset_sha256"]}
    jdump(doc, Pth["subset"])
    log(f"select OK: {doc['n_logs']} logs, {doc['n_sets']} sets, {doc['n_tokens']} tokens -> {Pth['subset']}")
    return 0


def stage_join(a) -> int:
    pins = require_prereg()
    Pth = paths()
    os.makedirs(Pth["join"], exist_ok=True)
    sub = json.load(open(Pth["subset"], encoding="utf-8"))
    z = np.load(META, allow_pickle=False)
    M = {k: z[k] for k in ("log_idx", "fold", "code", "z6", "key_log", "key_token", "key_step", "key_rank", "key_ckpt_step", "log_names")}     # never 'labels6'
    where = {(str(l), str(t), int(s), int(r), int(c)): i for i, (l, t, s, r, c) in enumerate(zip(M["key_log"], M["key_token"], M["key_step"], M["key_rank"], M["key_ckpt_step"]))}
    got_g, got_rl, lost, per_log = {}, {}, [], {}
    for name in sub["logs_in_order"]:
        p = f"{Pth['scal']}/{name}.npz"
        if not os.path.exists(p):
            raise C.P.PgError(f"extraction incomplete: {p} missing")
        d = np.load(p, allow_pickle=False)
        arr = {k: d[k] for k in d.files}
        n = len(arr["token"])
        if arr["g"].shape != (n, 64, 15) or arr["rl6"].shape != (n, 64, 6) or arr["ok"].shape != (n,):
            raise C.P.PgError(f"{p}: bad shapes {arr['g'].shape}")
        per_log[name] = {"n": int(n), "ok": int(arr["ok"].sum())}
        for i in range(n):
            ci = where[(name, str(arr["token"][i]), int(arr["step"][i]), int(arr["rank"][i]), int(arr["ckpt_step"][i]))]
            if arr["ok"][i]:
                got_g[ci], got_rl[ci] = arr["g"][i], arr["rl6"][i]
            else:
                lost.append(ci)
    census_idx = np.array(sub["census_idx"], np.int64)
    keep = np.array([i for i in census_idx if i in got_g], np.int64)
    n_expect, n_keep = len(census_idx), len(keep)
    if len(set(keep.tolist())) != n_keep:
        raise C.P.PgError("duplicate census index in the join")
    g = np.stack([got_g[i] for i in keep])
    rl6 = np.stack([got_rl[i] for i in keep])
    glob_li = M["log_idx"][keep]
    uniq = np.unique(glob_li)
    local = np.searchsorted(uniq, glob_li)
    np.savez(Pth["arrays"], idx=keep, g=g, rl6=rl6, z6=M["z6"][keep], fold=M["fold"][keep], code=M["code"][keep], log_idx=local, glob_log_idx=glob_li, p2_W6=z["p2_W6"], p2_b6=z["p2_b6"], log_names=M["log_names"][uniq])
    rep = {"created": now(), "ok": True, "pins": pins, "n_expected": n_expect, "n_retained": n_keep, "n_lost": int(n_expect - n_keep), "lost_frac": float((n_expect - n_keep) / n_expect), "n_logs": int(len(uniq)),
           "n_logs_expected": len(sub["logs_in_order"]), "per_log": per_log, "arrays_sha256": sha(Pth["arrays"]), "subset_sha256": sha(Pth["subset"]), "lost_disclose_gt_5pct": bool((n_expect - n_keep) / n_expect > C.LOSS_FRAC_DISCLOSE),
           "folds_logs": [int(len({int(x) for x in local[M["fold"][keep] == f]})) for f in range(5)]}
    jdump(rep, Pth["report"])
    log(f"join OK: {n_keep}/{n_expect} sets in {len(uniq)} logs")
    return 0


def load_join(labels: bool = False) -> dict:
    Pth = paths()
    z = np.load(Pth["arrays"], allow_pickle=False)
    A = {k: z[k] for k in z.files}
    if os.path.exists(Pth["feat_sub"]):
        A["feat"] = np.load(Pth["feat_sub"], mmap_mode="r")
    else:
        A["feat"] = RowView(np.load(FEAT, mmap_mode="r"), A["idx"])
    A["n_logs"] = int(A["log_idx"].max()) + 1
    A["N"] = len(A["fold"])
    if labels:
        lab = np.load(Pth["labels"], allow_pickle=False) if os.path.exists(Pth["labels"]) else np.load(f"{Pth['join']}/labels_sub.npy", allow_pickle=False)
        A["lab"] = lab
    return A


def block_for(A: dict, kind: str, rep: int = 0) -> np.ndarray:
    """The G block [N,64,k] float32 of a task kind: 'Gc' (primary, within-set centred), 'Gu' (companion, uncentred = PRIVGEOM-SAFE's `drop0`), 'plantweak', 'plantlabel', 'null' (within-set permuted, centred)."""
    blk = C.clean_block(A["g"])
    if kind == "Gc":
        return C.prepared(blk, True)
    if kind == "Gu":
        return C.prepared(blk, False)
    lab = A["lab"][..., C.ROW_COL]
    if kind == "plantlabel":
        return C.prepared(C.label_plant(lab), True)
    if kind == "plantweak":
        return C.prepared(C.weak_plant(lab, plant_sigma()), True)
    if kind == "null":
        return C.prepared(C.null_block(blk, rep), True)
    raise C.P.PgError(kind)


def task_list() -> list:
    T = []
    for s in C.SEEDS:
        T += [dict(name=f"L_s{s}", arm="L", seed=s, kind=None), dict(name=f"LGc_s{s}", arm="LG", seed=s, kind="Gc"), dict(name=f"LGu_s{s}", arm="LG", seed=s, kind="Gu"),
              dict(name=f"Gc_s{s}", arm="G", seed=s, kind="Gc"), dict(name=f"plantweak_s{s}", arm="LG", seed=s, kind="plantweak")]
    T += [dict(name=f"null{r}_s{r % len(C.SEEDS)}", arm="LG", seed=r % len(C.SEEDS), kind="null", rep=r) for r in range(C.NULL_REPS)]
    return T


def plant_tasks() -> list:
    return [dict(name=f"plantlabel_{arm}_s{s}", arm=arm, seed=s, kind="plantlabel") for s in C.SEEDS for arm in ("LG", "G")]


def run_task(A: dict, t: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    out = f"{out_dir}/{t['name']}.npy"
    if os.path.exists(out) and os.path.exists(out.replace(".npy", ".json")):
        return {"skipped": True}
    col = C.ROW_COL
    z_st, y = np.ascontiguousarray(A["z6"][..., col]), np.ascontiguousarray(A["lab"][..., col])
    g = None if t["kind"] is None else block_for(A, t["kind"], t.get("rep", 0))
    t0 = time.time()
    trace: list = []
    oof = P.oof_logits(t["arm"], A["feat"], z_st, y, g, A["fold"], t["seed"], A["p2_W6"][col], float(A["p2_b6"][col]), ram_guard=lambda: FL.wait_for_ram(3.0, log=log), trace=trace)
    np.save(out, oof)
    jdump({**{k: v for k, v in t.items()}, "seconds": time.time() - t0, "oof_sha256": sha(out), "k": 0 if g is None else int(g.shape[-1]), "folds": trace}, out.replace(".npy", ".json"))
    return {"skipped": False, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def code_pins() -> dict:
    return {"code": {n: sha(f"{HERE}/{n}") for n in CODE_FILES + FROZEN_SUITES if os.path.exists(f"{HERE}/{n}")}, "ref": {n: sha(f"{ROOT}/ref/{n}") for n in REF_FILES}}


def grad_probe_real(A: dict) -> dict:
    """C-GRAD-REAL (LABEL-FREE).  One registered step of the L+G fit path on the REAL centred block with a SEEDED SYNTHETIC label: no stored label and no recomputed verdict is read.  Every parameter group moves;
    every column of the block that varies moves `u`; a CONSTANT column (the start clearance, exactly 0 after centring) reads EXACTLY 0.0; and an all-zero block reads EXACTLY 0.0 (the control that proves the probe sees a zero)."""
    import torch
    blk = C.prepared(C.clean_block(A["g"]), True)
    n = int(min(256, blk.shape[0]))
    idx, col = np.arange(n), C.ROW_COL
    y = torch.as_tensor((np.random.default_rng(C.PLANT_SEED + 17).random((n, 64)) < 0.5).astype(np.float32))
    feat = torch.as_tensor(np.ascontiguousarray(A["feat"][idx]))
    z_st = torch.as_tensor(np.ascontiguousarray(A["z6"][idx][..., col]).astype(np.float32))

    def step(g_np):
        head = P.Head("LG", A["p2_W6"][col], float(A["p2_b6"][col]), g_np.shape[-1])
        w0, b0 = head.w.detach().clone(), head.b.detach().clone()
        head.step(feat, z_st, y, torch.as_tensor(P.standardise(g_np, idx)[idx]))
        return float((head.w - w0).abs().sum()), float((head.b - b0).abs().sum()), head.u.detach().abs().numpy().astype(np.float64)
    dw, db, du = step(blk)
    sd = blk[idx].astype(np.float64).reshape(-1, blk.shape[-1]).std(axis=0)
    varies = sd >= 1e-8
    dw0, db0, du0 = step(np.zeros_like(blk))
    ok = bool(dw > 0.0 and db > 0.0 and varies.sum() >= 1 and (du[varies] > 0.0).all() and (du[~varies] == 0.0).all() and float(du0.max()) == 0.0 and dw0 > 0.0)
    return {"pass": ok, "n_sets": n, "token_w_moved": dw, "token_b_moved": db, "u_moved": du.tolist(), "column_varies": varies.tolist(), "zero_block_u_max": float(du0.max()), "zero_block_token_w_moved": dw0}


def _write_gate(Gt: dict, token: dict, Pth: dict) -> None:
    jdump(Gt, Pth["gate_report"])
    jdump(token, Pth["token"])


def stage_gate(a) -> int:
    pins = require_prereg()
    Pth = paths()
    os.makedirs(RUN, exist_ok=True)
    rep_join = json.load(open(Pth["report"], encoding="utf-8"))
    if sha(Pth["arrays"]) != rep_join["arrays_sha256"]:
        raise C.P.PgError("join arrays changed since join")
    A = load_join()                                                                      # arrays.npz carries NO stored label
    Gt = {"primary_arm": C.PRIMARY_ARM, "labels_opened": False}
    # ================= PHASE A: LABEL-FREE.  A failure here STOPS the run before any label is opened. =================
    n_exp_logs = rep_join.get("n_logs_expected", rep_join["n_logs"])
    folds_logs = rep_join.get("folds_logs") or [int(len({int(x) for x in A["log_idx"][A["fold"] == f]})) for f in range(5)]
    min_logs = 1 if REHEARSAL else MIN_LOGS_RETAINED
    Gt["C-JOIN"] = {"n_retained": rep_join["n_retained"], "n_expected": rep_join["n_expected"], "n_logs": rep_join["n_logs"], "n_logs_expected": n_exp_logs, "lost_frac": rep_join["lost_frac"], "folds_logs": folds_logs,
                    "pass": bool(rep_join["n_logs"] >= min_logs and min(folds_logs) >= 1)}
    glp = f"{ROOT}/run/glive/glive.json"                                                 # C-ID-EXTRACT: the full extraction of the two G-LIVE logs reproduces the block sha256 registered pre-seal
    rows = {}
    if os.path.exists(glp):
        for r in json.load(open(glp, encoding="utf-8"))["records"]:
            p = f"{Pth['scal']}/{r['log']}.npz"
            if os.path.exists(p):
                d = np.load(p, allow_pickle=False)
                h = C.block_hash(d["g"], d["ok"])
                rows[r["log"]] = {"registered": r["block_sha256_ok_sets"], "recomputed": h, "match": bool(h == r["block_sha256_ok_sets"])}
            else:
                rows[r["log"]] = {"registered": r["block_sha256_ok_sets"], "recomputed": None, "match": None}
    Gt["C-ID-EXTRACT"] = {"rows": rows, "pass": bool(REHEARSAL or (len(rows) > 0 and all(v["match"] is True for v in rows.values())))}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8", PYTHONDONTWRITEBYTECODE="1")
    suites = {}
    for tf in FROZEN_SUITES + ("test_pgc_lib.py", "test_pgc_admit.py", "test_pgc_seal.py"):
        r = subprocess.run([sys.executable, "-B", "-W", "ignore", f"{HERE}/{tf}"], capture_output=True, text=True, env=env, cwd=HERE)
        suites[tf] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-500:]}
    Gt["suites"] = suites
    Gt["C-RED_branch_table_failures"] = C.branch_table_failures()
    Gt["C-GRAD-REAL"] = grad_probe_real(A)
    free_ok = bool(Gt["C-JOIN"]["pass"] and Gt["C-ID-EXTRACT"]["pass"] and all(s["rc"] == 0 for s in suites.values()) and not Gt["C-RED_branch_table_failures"] and Gt["C-GRAD-REAL"]["pass"])
    Gt["phase_A_label_free_pass"] = free_ok
    log(f"gate PHASE A (label-free): {'PASS' if free_ok else 'FAIL'}  join {Gt['C-JOIN']['pass']}  id-extract {Gt['C-ID-EXTRACT']['pass']}  suites { {k: v['rc'] for k, v in suites.items()} }  branch-table {not Gt['C-RED_branch_table_failures']}  grad-real {Gt['C-GRAD-REAL']['pass']}")
    if not free_ok:
        Gt["gate_pass"], Gt["stopped_before_labels"] = False, True
        _write_gate(Gt, {"created": now(), "pass": False, "rehearsal": REHEARSAL, "pins": pins, "code_pins": code_pins(), "arrays_sha256": rep_join["arrays_sha256"], "labels_sub_sha256": None, "gate": Gt}, Pth)
        log("gate FAIL in phase A: NO LABEL WAS OPENED")
        return 5
    # ================= PHASE B: the FIRST label read (retained sets only). =================
    if REHEARSAL:
        lab = np.load(f"{Pth['join']}/labels_sub.npy", allow_pickle=False)
    else:
        z = np.load(META, allow_pickle=False)
        lab = np.ascontiguousarray(z["labels6"][A["idx"]])                              # THE FIRST LABEL READ
    np.save(Pth["labels"], lab)
    A["lab"] = lab
    Gt["labels_opened"] = True
    log(f"gate: labels opened for {len(lab)} retained sets" + (" (REHEARSAL: the SEEN banked labels)" if REHEARSAL else ""))
    rl = np.clip(A["rl6"], 0.0, 1.0)
    agree = {r: {"exact_frac": float((rl[..., c] == lab[..., c]).mean()), "n_prop": int(lab[..., c].size)} for r, c in (("nc", 0), ("dac", 1))}
    agree["dac"]["pass"] = bool(agree["dac"]["exact_frac"] >= C.AGREE_MIN)             # the NC column is REPORTED (the NC row is not read here)
    Gt["C-AGREE"] = agree
    g = A["g"]
    dac0 = lab[..., 1] == 0.0
    Gt["label_identity"] = {"P(dac_frac_out>0 | DAC=0)": float((g[..., 1][dac0] > 0).mean()), "P(dac_clear_min<0 | DAC=0)": float((g[..., 0][dac0] < 0).mean()), "P(dac_clear_min>=0 | DAC=1)": float((g[..., 0][~dac0] >= 0).mean()),
                            "n_dac0": int(dac0.sum()), "n_prop": int(dac0.size)}
    # ---- C-ADMIT: the planted admissibility check on the REAL arrays, then its RED arm (the disabled check must let the planted leaks through)
    import pg_geom as G
    block, leaky10, labd = C.clean_block(g), G.unsigned(np.asarray(g, np.float64)[..., :11]), lab[..., C.ROW_COL]
    t0 = time.time()
    adm = AD.a_test(block, labd, leaky10)
    red = AD.a_test(block, labd, leaky10, cfg=AD.AdmitConfig(disabled=True))
    Gt["C-ADMIT"] = {"admissible": adm["admissible"], "controls_ok": adm["controls_ok"], "controls": adm["controls"], "candidate": {k: adm["candidate"][k] for k in ("flagged", "hits_by_family", "best_unsafe_proof_by_family", "n_rules", "n_unsafe", "n_proposals", "top")},
                     "red_arm_controls_ok": red["controls_ok"], "red_arm_leaks_through": sorted(k for k, v in red["controls"].items() if v["expected_flagged"] and not v["flagged"]), "seconds": time.time() - t0,
                     "pass": bool(adm["admissible"] and not red["controls_ok"])}
    log(f"C-ADMIT: admissible {adm['admissible']} controls_ok {adm['controls_ok']} red-arm controls_ok {red['controls_ok']} ({time.time() - t0:.0f} s)")
    # ---- C-REACH on the real arrays (planted label through the real fit / score code)
    an = P.Analyzer(lab, A["code"], A["log_idx"], A["n_logs"])
    reach = {}
    for t in plant_tasks():
        run_task(A, t, Pth["fits"])
        oof = np.load(f"{Pth['fits']}/{t['name']}.npy")
        reach[t["name"]] = float(an.auc(an.per_log_stats(oof, C.ROW), "Q4")["point"])
    Gt["C-REACH"] = {"F_auc": reach, "min": C.REACH_MIN_AUC, "pass": bool(all(v >= C.REACH_MIN_AUC for v in reach.values()))}
    gate_pass = bool(Gt["C-REACH"]["pass"] and Gt["C-ADMIT"]["pass"])                  # phase A already passed; a REFUSED block = NO-CALL(ADMIT): no fit starts.  A C-AGREE failure does not stop the fits: it VOIDs the call
    Gt["gate_pass"] = gate_pass
    token = {"created": now(), "pass": gate_pass, "rehearsal": REHEARSAL, "pins": pins, "code_pins": code_pins(), "arrays_sha256": rep_join["arrays_sha256"], "labels_sub_sha256": sha(Pth["labels"]), "gate": Gt}
    _write_gate(Gt, token, Pth)
    log(f"gate {'PASS' if gate_pass else 'FAIL'}: agree {agree['dac']}  reach {reach}  admit {Gt['C-ADMIT']['pass']}  suites { {k: v['rc'] for k, v in suites.items()} }")
    return 0 if gate_pass else 5


def require_token() -> dict:
    Pth = paths()
    tok = json.load(open(Pth["token"], encoding="utf-8"))
    if not tok["pass"]:
        raise C.P.PgError("gate token is a FAIL")
    if code_pins() != tok["code_pins"] or sha(Pth["arrays"]) != tok["arrays_sha256"]:
        raise C.P.PgError("the code / join differ from what the gate token binds: refused")
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
    open(f"{Pth['fits']}/FIT_SHARD_DONE_{a.shard}.txt", "w").write(now())
    log(f"FIT_SHARD_DONE {a.shard}")
    return 0


def stage_analyze(a) -> int:
    require_prereg()
    tok = require_token()
    Pth = paths()
    os.makedirs(Pth["results"], exist_ok=True)
    A = load_join(labels=True)
    gate = json.load(open(Pth["gate_report"], encoding="utf-8"))
    an = P.Analyzer(A["lab"], A["code"], A["log_idx"], A["n_logs"])
    oof = lambda name: np.load(f"{Pth['fits']}/{name}.npy")                              # noqa: E731
    cache: dict = {}

    def st(name):
        if name not in cache:
            cache[name] = an.per_log_stats(oof(name), C.ROW)
        return cache[name]
    R = {"schema": C.SCHEMA, "created": now(), "rehearsal": REHEARSAL, "n_sets": int(A["N"]), "n_logs": int(A["n_logs"]), "controls": {"gate": gate}, "pins": tok["pins"], "auc": {}, "delta_F": {}, "delta_Q123": {}, "calls": {},
         "companion": {}, "g_only_minus_L_F": {}, "null": {}, "power": {}}
    prim, comp = C.PRIMARY_ARM, C.COMPANION_ARM
    R["primary_arm"], R["companion_arm"] = prim, comp
    for s in C.SEEDS:
        for arm in ("L", "LGc", "LGu", "Gc"):
            R["auc"][f"{arm}_s{s}"] = {"F": an.auc(st(f"{arm}_s{s}"), "Q4"), "Q123": an.auc(st(f"{arm}_s{s}"), "Q123")}
        dF = an.diff(st(f"{prim}_s{s}"), st(f"L_s{s}"), "Q4")
        R["delta_F"][f"s{s}"] = dF
        R["delta_Q123"][f"s{s}"] = an.diff(st(f"{prim}_s{s}"), st(f"L_s{s}"), "Q123")
        R["calls"][f"s{s}"] = C.seed_call(dF)
        cF = an.diff(st(f"{comp}_s{s}"), st(f"L_s{s}"), "Q4")
        R["companion"][f"s{s}"] = {"delta_F": cF, "call": C.seed_call(cF)}
        R["g_only_minus_L_F"][f"s{s}"] = an.diff(st(f"Gc_s{s}"), st(f"L_s{s}"), "Q4")
    call = C.row_call([R["calls"][f"s{s}"] for s in C.SEEDS])
    R["companion"]["row_call"] = C.row_call([R["companion"][f"s{s}"]["call"] for s in C.SEEDS])
    R["small_reading"] = bool(call in ("INCONCLUSIVE", "NULL") and C.small_reading([R["delta_F"][f"s{s}"] for s in C.SEEDS]))
    fires = []
    for r in range(C.NULL_REPS):
        s = r % len(C.SEEDS)
        ci = an.diff(st(f"null{r}_s{s}"), st(f"L_s{s}"), "Q4")
        fires.append(C.null_fires(ci))
        R["null"][f"rep{r}"] = {"seed": s, "delta_F": ci, "fires": fires[-1]}
    void_null = C.null_void(fires)
    pcis = [an.diff(st(f"plantweak_s{s}"), st(f"L_s{s}"), "Q4") for s in C.SEEDS]
    pw = C.power_ok(pcis)
    R["power"] = {"sigma": plant_sigma(), "delta_F_by_seed": pcis, "fires_on_both_seeds": pw}
    void_inputs = not (gate["C-AGREE"]["dac"]["pass"] and gate["C-JOIN"]["pass"])
    admit_ok = bool(gate["C-ADMIT"]["pass"])
    R["raw_call"], R["void_null"], R["null_fires"], R["void_inputs"], R["admit_ok"] = call, void_null, fires, void_inputs, admit_ok
    R["final_call"] = C.final_call(call, admit_ok, void_inputs, void_null, pw)
    R["pairs_F"] = float(st("L_s0")["Q4"][1].sum())
    jdump(R, f"{Pth['results']}/results.json")
    write_tables(R, f"{Pth['results']}/RESULT_tables.md")
    log(f"analyze OK: final {R['final_call']} (raw {R['raw_call']}; companion {R['companion']['row_call']}; void_null {void_null}; power {pw})")
    return 0


def fmt(ci: dict, k: int = 4) -> str:
    return f"{ci['point']:+.{k}f} [{ci['lo']:+.{k}f}, {ci['hi']:+.{k}f}]"


def fmtu(ci: dict, k: int = 4) -> str:
    return f"{ci['point']:.{k}f} [{ci['lo']:.{k}f}, {ci['hi']:.{k}f}]"


def write_tables(R: dict, path: str) -> None:
    L = [f"# PRIVGEOM-CLEAN tables (generated {R['created']}{' REHEARSAL, NOT A RESULT' if R['rehearsal'] else ''}; {R['n_sets']} sets, {R['n_logs']} logs; paired log-cluster bootstrap, B = {C.BOOT_B}, seed {C.BOOT_SEED})", "",
         f"## DAC, sign-clean 9-column block: final **{R['final_call']}** (raw {R['raw_call']}; companion {R['companion']['row_call']}; null fires {R['null_fires']}; void_null {R['void_null']}; void_inputs {R['void_inputs']}; admit_ok {R['admit_ok']}; power fires {R['power']['fires_on_both_seeds']}; small-reading {R['small_reading']})", "",
         f"| seed | F-AUC L | F-AUC L+G (primary {R['primary_arm']}) | F-AUC G-only (centred) | Delta_F primary | call | Delta_Q1-3 | G-only - L (F) | companion ({R['companion_arm']}) Delta_F | companion call |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in C.SEEDS:
        a = R["auc"]
        pk = R["primary_arm"] + f"_s{s}"
        L.append(f"| {s} | {fmtu(a[f'L_s{s}']['F'])} | {fmtu(a[pk]['F'])} | {fmtu(a[f'Gc_s{s}']['F'])} | {fmt(R['delta_F'][f's{s}'])} | {R['calls'][f's{s}']} | {fmt(R['delta_Q123'][f's{s}'])} | "
                 f"{fmt(R['g_only_minus_L_F'][f's{s}'])} | {fmt(R['companion'][f's{s}']['delta_F'])} | {R['companion'][f's{s}']['call']} |")
    L += ["", "C-NULL (Delta_F of the within-set-permuted, centred block; fires iff the CALL's own rule holds on the replicate's seed):", ""]
    for k, v in R["null"].items():
        L.append(f"- {k} (seed {v['seed']}): {fmt(v['delta_F'])} fires={v['fires']}")
    L += ["", f"C-POWER (plant sigma {R['power']['sigma']}, Delta_F by seed): " + " / ".join(fmt(c) for c in R["power"]["delta_F_by_seed"]), "",
          f"F pairs: {R['pairs_F']:.0f}", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("seal-check", "preflight", "select", "join", "gate", "fit", "analyze"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args(argv)
    return {"seal-check": stage_seal_check, "preflight": stage_preflight, "select": stage_select, "join": stage_join, "gate": stage_gate, "fit": stage_fit, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
