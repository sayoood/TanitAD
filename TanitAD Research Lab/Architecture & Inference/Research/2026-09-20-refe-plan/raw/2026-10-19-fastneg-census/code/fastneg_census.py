"""FASTNEG-CENSUS label pass: inputs -> gate -> fit -> analyze.  REFUSES to start any stage unless `strata.json` is present AT ITS RECORDED SHA256 (passed with --strata-sha and also equal to
`strata.json.sha256`); the strata script is label-free and ran first.  CPU only.

REGISTERED SOURCE: PREREG_FASTNEG_CENSUS.md + PREREG_FASTNEG_CENSUS_AMENDMENT1.md (A1.1 to A1.8).
    fastneg_census.py inputs  --strata-sha H
    fastneg_census.py gate    --strata-sha H
    fastneg_census.py fit     --strata-sha H --groups primary,placebo,cold      (or: null)
    fastneg_census.py analyze --strata-sha H
"""
from __future__ import annotations

import argparse
import datetime
import inspect
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fastneg_lib as L                                                               # noqa: E402

RUN = "C:/Users/Admin/par18/fastneg/run"
DEFAULT_BANK = "C:/Users/Admin/srttc_s0/work/bank"
DEFAULT_MANIFEST = DEFAULT_BANK + "/MANIFEST.sha256.json"
DEFAULT_RANKED = "C:/Users/Admin/srttc_s0/run/plan/ranked_fit_logs.tsv"
DEFAULT_STORES = "C:/Users/Admin/srttc_s0/work/sets"
DEFAULT_P2 = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
DEFAULT_PROBE_OOF = ("C:/Users/Admin/par18/probe/out/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-19-headrow-probe/raw/f10/oof_proposals_f10.npz")
CODE_FILES = ("fastneg_lib.py", "fastneg_strata.py", "fastneg_census.py", "test_fastneg.py", "test_fastneg_more.py")      # fastneg_analyze.py launches no fit: not bound
CONFIG = {"lr": L.LR, "weight_decay": L.WD, "betas": list(L.BETAS), "eps": L.EPS, "epochs": L.EPOCHS, "batch_sets": L.BATCH_SETS, "seeds": list(L.SEEDS), "sizes": list(L.SIZES),
          "null_reps": L.NULL_REPS, "null_salt": L.NULL_SALT, "sub_seed_base": L.SUB_SEED_BASE, "sub_seed_mult": L.SUB_SEED_MULT, "n_folds": L.N_FOLDS, "boot_B": L.BOOT_B, "boot_seed": L.BOOT_SEED,
          "rows": list(L.ROWS), "init": "P2 rows 0 (NC) and 1 (DAC) of score_head", "feature_dtype": "fp16 -> fp32, detached", "deficit_margin": L.DEFICIT_MARGIN, "slope_margin": L.SLOPE_MARGIN,
          "min_avail_gb": L.MIN_AVAIL_GB}


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def paths(a) -> dict:
    run = a.run_dir
    return {"run": run, "strata": f"{run}/strata", "inputs": f"{run}/inputs", "fits": f"{run}/fits", "results": f"{run}/results",
            "feat": f"{run}/inputs/feat_f10.npy", "meta": f"{run}/inputs/meta.npz", "inputs_report": f"{run}/inputs/inputs_report.json", "token": f"{run}/gate_token.json"}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# the strata precondition: every stage calls this FIRST
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def require_strata(a) -> dict:
    P = paths(a)
    sj = f"{P['strata']}/strata.json"
    if not a.strata_sha:
        raise L.CensusError("--strata-sha is required: the label pass refuses to start without strata.json at its recorded sha256")
    if not os.path.exists(sj):
        raise L.CensusError(f"{sj} does not exist: run fastneg_strata.py first (the label pass refuses to start)")
    with open(sj, "rb") as f:
        raw = f.read()
    sha = L.sha256_bytes(raw)
    with open(sj + ".sha256", "r", encoding="utf-8") as f:
        rec = f.read().split()[0]
    if sha != a.strata_sha or sha != rec:
        raise L.CensusError(f"strata.json sha256 {sha} != --strata-sha {a.strata_sha} / recorded {rec}: the label pass refuses to start")
    doc = json.loads(raw)
    sp = f"{P['strata']}/{doc['speeds_npz']['file']}"
    if L.sha256_file(sp) != doc["speeds_npz"]["sha256"]:
        raise L.CensusError("speeds.npz does not match the sha256 recorded in strata.json")
    z = np.load(sp, allow_pickle=False)
    arrs = {k: z[k] for k in z.files}
    if L.arrays_digest(arrs) != doc["speeds_npz"]["content_digest"]:
        raise L.CensusError("speeds.npz content digest differs from strata.json")
    if doc["prereg_sha256"] != L.PREREG_SHA256 or doc["amendment1_sha256"] != L.AMENDMENT1_SHA256:
        raise L.CensusError("strata.json was written under a different registration")
    return {"doc": doc, "sha": sha, "arrays": arrs}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stage: inputs  (bank sha256-verified BEFORE np.load; features -> a read-only fp16 memmap; labels; the alignment and the join proofs)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_inputs(a) -> int:
    S = require_strata(a)
    P = paths(a)
    E = L.load_epr1(a.epr1)
    os.makedirs(P["inputs"], exist_ok=True)
    guard = E.PathGuard(read_dirs=[a.bank_dir, a.stores_dir, P["inputs"]], read_files=[a.manifest, a.ranked, a.p2_ckpt, a.probe_oof], write_dirs=[P["inputs"]])
    manifest, man_sha = E.load_manifest(guard, a.manifest)
    if man_sha != S["doc"]["bank_manifest_sha256"]:
        raise L.CensusError("bank manifest differs from the one the strata were computed under")
    rows, ranked_sha = E.load_ranked_logs(guard, a.ranked, (a.ranked_sha or None), (a.ranked_n or None))
    selected, _ = E.select_logs(rows, 10, None)
    sel_pos = {r["log"]: i for i, r in enumerate(selected)}
    ranks = np.array([r["rank"] for r in selected], np.int64)
    log_fold = np.array([E.fold_of_rank(r) for r in ranks], np.int64)
    W6, b6 = E.load_p2_row(guard, a.p2_ckpt)
    units = [n for n in E._io_listdir(guard, a.bank_dir) if n.endswith(".sets.npz")]
    unit_sha, masks, stats = {}, {}, {}
    ckpts, rules = set(), set()
    for n in units:                                                                       # pass 1: sha256 against the manifest BEFORE np.load
        p = os.path.join(a.bank_dir, n)
        if n not in manifest:
            raise L.CensusError(f"unit {n} is not in the manifest (refused: unverified)")
        sha = E._io_sha256(guard, p)
        if sha != manifest[n]:
            raise L.CensusError(f"unit {n}: sha256 {sha} != manifest {manifest[n]}")
        unit_sha[n] = sha
        stats[n] = E._io_stat(guard, p)
        z = E._io_npz(guard, p)
        if [str(x) for x in z["label_order"]] != list(E.LABEL_ORDER):
            raise L.CensusError(f"unit {n}: label_order {list(z['label_order'])}")
        rules.add(str(z["rule"]))
        ckpts.add(str(z["ckpt"]))
        masks[n] = np.array([str(x) in sel_pos for x in z["log"]], dtype=bool)
        z.close()
    if rules != {E.RULE} or len(ckpts) != 1:
        raise L.CensusError(f"bank rule {rules} / ckpts {ckpts}")
    N = int(sum(m.sum() for m in masks.values()))
    log(f"inputs: {len(units)} units sha256-verified; {N} sets")
    feat_mm = np.lib.format.open_memmap(P["feat"], mode="w+", dtype=np.float16, shape=(N, L.N_PROP, L.FEAT_DIM))
    z6 = np.empty((N, L.N_PROP, 6), np.float32)
    lab = np.empty((N, L.N_PROP, 6), np.float32)
    top8 = np.empty((N, 8), np.int64)
    keys = {k: [] for k in ("log", "token", "step", "rank", "ckpt_step")}
    n_ties, off = 0, 0
    for n in units:                                                                       # pass 2: arrays, with epr1.load_fit_data's integrity checks
        mask = masks[n]
        if not mask.any():
            continue
        p = os.path.join(a.bank_dir, n)
        if E._io_stat(guard, p) != stats[n]:
            raise L.CensusError(f"unit {n} changed on disk between verification and load")
        z = E._io_npz(guard, p)
        f, lo, tp, lb = z["feat"][mask], z["logits"][mask], z["top8"][mask], z["labels6"][mask]
        for k in keys:
            keys[k].append(z[k][mask])
        z.close()
        Sn = len(lo)
        if f.dtype != np.float16 or f.shape[1:] != (L.N_PROP, L.FEAT_DIM) or lo.shape[1:] != (L.N_PROP, 6) or tp.shape[1:] != (8,) or lb.shape[1:] != (L.N_PROP, 6):
            raise L.CensusError(f"unit {n}: shapes/dtypes {f.dtype}{f.shape} {lo.shape} {tp.shape} {lb.shape}")
        if not (np.isfinite(lo).all() and np.isfinite(lb).all() and np.isfinite(f.astype(np.float32)).all()):
            raise L.CensusError(f"unit {n}: non-finite feature / logit / label")
        if lb.min() < 0.0 or lb.max() > 1.0:
            raise L.CensusError(f"unit {n}: a label outside [0,1]")
        again = np.argsort(-E.aggregate_np(lo), axis=1, kind="stable")[:, :8]
        bad = np.nonzero((again != tp.astype(np.int64)).any(axis=1))[0]
        for i in bad:
            if not np.array_equal(E.top8_np(lo[i]), tp[i]):
                raise L.CensusError(f"unit {n}: stored top8 of set {int(i)} is not the frozen rule applied to its stored logits")
            n_ties += 1
        feat_mm[off:off + Sn] = f
        z6[off:off + Sn], lab[off:off + Sn], top8[off:off + Sn] = lo, lb, tp.astype(np.int64)
        off += Sn
    assert off == N
    feat_mm.flush()
    del feat_mm
    key = {k: np.concatenate(v) for k, v in keys.items()}
    log_idx = np.array([sel_pos[str(x)] for x in key["log"]], np.int64)
    fold = log_fold[log_idx]
    # ---- alignment with the label-free strata: the same sets, the same order, the same frozen-rule pick
    A = S["arrays"]
    ok_keys = (np.array_equal(A["key_log"].astype("U"), key["log"].astype("U")) and np.array_equal(A["key_token"].astype("U"), key["token"].astype("U"))
               and np.array_equal(A["key_step"], key["step"].astype(np.int64)) and np.array_equal(A["key_rank"], key["rank"].astype(np.int64))
               and np.array_equal(A["key_ckpt_step"], key["ckpt_step"].astype(np.int64)))
    pick = E.picks_np(z6)
    ok_pick = np.array_equal(pick, A["pick"]) and np.array_equal(pick, top8[:, 0])
    if not (ok_keys and ok_pick):
        raise L.CensusError(f"the label pass's sets differ from the strata's (keys equal: {ok_keys}, pick equal: {ok_pick}); strata are void")
    # ---- P2-row consistency for the two rows (documentation of which head produced the stored logits)
    step = max(1, N // 2048)
    ii = np.arange(0, N, step)[:2048]
    fm = np.load(P["feat"], mmap_mode="r")
    dev = {}
    for c in (0, 1):
        chk = np.asarray(fm[ii], np.float32) @ np.asarray(W6[c], np.float32) + float(b6[c])
        dev[c] = float(np.abs(chk - z6[ii, :, c]).max())
    if not all(v <= E.ROW_CONSISTENCY_TOL for v in dev.values()):
        raise L.CensusError(f"P2 rows do not reproduce the stored NC / DAC logits: {dev}")
    # ---- the content proof of the join: the store's labels6 equal the bank's EXACTLY on every set
    lg_arr = np.array([str(x) for x in key["log"]])
    store_sha, n_join, n_bad = {}, 0, 0
    for r in selected:
        name = r["log"]
        sp = os.path.join(a.stores_dir, name + ".npz")
        guard.check_read(sp)
        store_sha[name] = L.sha256_file(sp)
        zs = np.load(sp, allow_pickle=False)
        where = {(str(t), int(st), int(rk)): j for j, (t, st, rk) in enumerate(zip(zs["token"], zs["step"], zs["rank"]))}
        slab = zs["labels6"]
        for i in np.nonzero(lg_arr == name)[0]:
            j = where[(str(key["token"][i]), int(key["step"][i]), int(key["rank"][i]))]
            n_join += 1
            if not np.array_equal(slab[j], lab[i]):
                n_bad += 1
    if n_bad:
        raise L.CensusError(f"{n_bad} of {n_join} store rows have labels6 different from the bank's: the trajectory join is wrong; strata are VOID")
    nc, dac = lab[..., 0], lab[..., 1]
    vals = lambda x: {str(k): int(v) for k, v in zip(*np.unique(x, return_counts=True))}                 # noqa: E731
    meta = dict(log_idx=log_idx, fold=fold, z6=z6, labels6=lab, top8=top8, code=A["code"], pick=pick, p2_W6=np.asarray(W6, np.float32), p2_b6=np.asarray(b6, np.float32),
                key_log=key["log"].astype("U"), key_token=key["token"].astype("U"), key_step=key["step"].astype(np.int64), key_rank=key["rank"].astype(np.int64),
                key_ckpt_step=key["ckpt_step"].astype(np.int64), v=A["v"], v_b=A["v_b"], r=A["r"], log_rank=ranks, log_names=np.array([r["log"] for r in selected]))
    np.savez(P["meta"], **meta)
    rep = {"schema": L.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "ok": True, "n_sets": int(N), "n_logs": len(selected), "n_units": len(units),
           "strata_json_sha256": S["sha"], "bank_manifest_sha256": man_sha, "ranked_logs_sha256": ranked_sha, "p2_ckpt": a.p2_ckpt, "bank_ckpt": sorted(ckpts)[0],
           "unit_sha256_verified": unit_sha, "store_sha256": store_sha, "store_sha256_combined": L.sha256_bytes("".join(f"{k}:{store_sha[k]}\n" for k in sorted(store_sha)).encode()),
           "alignment_with_strata": {"keys_equal": bool(ok_keys), "pick_equal": bool(ok_pick)}, "join_proof": {"store_rows_compared": n_join, "labels6_mismatches": n_bad},
           "p2_row_consistency_max_abs": dev, "top8_vectorised_tie_rechecked_sets": n_ties,
           "label_value_counts": {"nc": vals(nc), "dac": vals(dac)}, "fold_sets": [int((fold == k).sum()) for k in range(L.N_FOLDS)],
           "fold_logs": [int((log_fold == k).sum()) for k in range(L.N_FOLDS)], "feat_npy_sha256": L.sha256_file(P["feat"]), "meta_npz_sha256": L.sha256_file(P["meta"])}
    with open(P["inputs_report"], "w", encoding="utf-8", newline="\n") as f:
        f.write(L.canon_json(rep).decode())
    log(f"inputs OK: {N} sets, join proof {n_join} rows / {n_bad} mismatches, row consistency {dev}")
    return 0


def load_ctx(a, need_feat: bool = True) -> dict:
    P = paths(a)
    z = np.load(P["meta"], allow_pickle=False)
    ctx = {k: z[k] for k in z.files}
    ctx["P"] = P
    if need_feat:
        ctx["feat"] = np.load(P["feat"], mmap_mode="r")
    ctx["n_logs"] = int(len(ctx["log_names"]))
    ctx["N"] = int(len(ctx["fold"]))
    ctx["z_st"] = np.ascontiguousarray(ctx["z6"][..., :2])
    ctx["y"] = np.ascontiguousarray(ctx["labels6"][..., :2])
    return ctx


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stage: gate  (package-level launch gate: G-BANK, G-DVB, G-LIVE, G-SUITE; mints a token bound to the code, config, bank, strata and inputs)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def gate_bindings(a, S: dict) -> dict:
    P = paths(a)
    rep_sha = L.sha256_file(P["inputs_report"]) if os.path.exists(P["inputs_report"]) else None
    return {"code_sha256": {n: L.sha256_file(os.path.join(HERE, n)) for n in CODE_FILES}, "epr1_sha256": L.EPR1_SHA256, "strata_json_sha256": S["sha"], "inputs_report_sha256": rep_sha,
            "bank_manifest_sha256": S["doc"]["bank_manifest_sha256"], "prereg_sha256": L.PREREG_SHA256, "amendment1_sha256": L.AMENDMENT1_SHA256, "config": CONFIG}


def stage_gate(a) -> int:
    import torch
    S = require_strata(a)
    P = paths(a)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(a.threads))
    checks, ok = {}, True

    def rec(name, passed, detail):
        nonlocal ok
        checks[name] = {"pass": bool(passed), "detail": detail}
        ok = ok and bool(passed)
        log(f"gate {name}: {'PASS' if passed else 'FAIL'} {detail}")
    with open(P["inputs_report"], "r", encoding="utf-8") as f:
        rep = json.load(f)
    rec("G-BANK", rep["ok"] and rep["alignment_with_strata"]["keys_equal"] and rep["alignment_with_strata"]["pick_equal"] and rep["join_proof"]["labels6_mismatches"] == 0
        and rep["n_sets"] == S["doc"]["n_sets"] and rep["n_logs"] == S["doc"]["n_logs"], f"units sha256-verified {rep['n_units']}; join mismatches {rep['join_proof']['labels6_mismatches']}; sets {rep['n_sets']}")
    ctx = load_ctx(a)
    p2_w, p2_b = ctx["p2_W6"], ctx["p2_b6"]
    # ---- G-DVB: every lever of the declared config equals the value the BUILT optimiser / fit function reports
    h = L.LRows(p2_w, p2_b, cols=(0, 1))
    built = [{"lr": o.param_groups[0]["lr"], "weight_decay": o.param_groups[0]["weight_decay"], "betas": list(o.param_groups[0]["betas"]), "eps": o.param_groups[0]["eps"]} for o in h.opts]
    want = {"lr": CONFIG["lr"], "weight_decay": CONFIG["weight_decay"], "betas": CONFIG["betas"], "eps": CONFIG["eps"]}
    sig = inspect.signature(L.fit_rows).parameters
    fit_defaults = {"epochs": sig["epochs"].default, "batch_sets": sig["batch_sets"].default}
    init_equal = all(np.array_equal(h.w[r].detach().numpy(), p2_w[c]) and np.array_equal(h.b[r].detach().numpy(), p2_b[c:c + 1]) for r, c in enumerate((0, 1)))
    cold = L.LRows(p2_w, p2_b, cols=(0, 1), cold=True)
    cold_zero = all(float(w.abs().sum()) == 0.0 for w in cold.w)
    dvb = all(b == want for b in built) and fit_defaults == {"epochs": CONFIG["epochs"], "batch_sets": CONFIG["batch_sets"]} and init_equal and cold_zero
    rec("G-DVB", dvb, f"built optimiser {built[0]} (want {want}); fit defaults {fit_defaults}; P2-row init bit-equal {init_equal}; cold init zero {cold_zero}")
    # ---- G-LIVE: the reach proof on a real batch
    n = min(256, ctx["N"])
    rng = np.random.default_rng(0)
    bi = np.sort(rng.choice(ctx["N"], n, replace=False))
    f16 = torch.from_numpy(np.ascontiguousarray(ctx["feat"][bi]))
    zs, yy = torch.from_numpy(ctx["z_st"][bi]), torch.from_numpy(ctx["y"][bi])
    code = ctx["code"][bi]
    g = h.grads(f16, zs, yy, None)
    nonzero = all(float(g[r][0].abs().sum()) > 0 and float(g[r][1].abs().sum()) > 0 for r in (0, 1))
    g0 = h.grads(f16, zs, yy, torch.zeros(n, 64))
    zero_ctl = all(float(g0[r][0].abs().sum()) == 0.0 and float(g0[r][1].abs().sum()) == 0.0 for r in (0, 1))
    fl = f16.clone().float().requires_grad_(True)
    h.grads(fl.half(), zs, yy, None)
    detached = fl.grad is None
    incl0 = L.make_incl(code, np.full(code.shape, 0.99, np.float32), 0.0)
    gp = h.grads(f16, zs, yy, torch.from_numpy(incl0))
    keep = torch.from_numpy(incl0).double()
    ident = True
    for r in (0, 1):
        w = torch.nn.Parameter(h.p2_w[r].double().clone())
        b = torch.nn.Parameter(h.p2_b[r].double().clone())
        z = zs[..., r].double() + (f16.float().double() @ (w - h.p2_w[r].double()) + (b - h.p2_b[r].double()))
        l = torch.nn.functional.binary_cross_entropy_with_logits(z, yy[..., r].double(), reduction="none")
        ((l * keep).sum() / keep.sum()).backward()
        ident = ident and bool(torch.allclose(gp[r][0].double(), w.grad, rtol=1e-4, atol=1e-6))
    u = L.subsample_priority(code.shape, 0)
    q = L.make_incl(code, u, 0.25)
    nq4, nin = int((code == 4).sum()), int(((code == 4) & (q == 1.0)).sum())
    partial = 0 < nin < nq4
    cid = np.array_equal(L.score_rows(ctx["feat"], ctx["z_st"], bi, p2_w[:2], p2_b[:2], p2_w, p2_b), ctx["z_st"][bi])
    rec("G-LIVE", nonzero and zero_ctl and detached and ident and partial and cid,
        f"grads nonzero {nonzero}; zero-mask control reads exactly 0.0 {zero_ctl}; features detached {detached}; F-masked gradient = hand-computed {ident}; 25 % mask partial ({nin}/{nq4}) {partial}; "
        f"scoring at P2's rows == stored logits bit for bit {cid}")
    # ---- G-SUITE: the synthetic tests (including the RED mutation arms) pass
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="8")
    outs = {}
    for t in ("test_fastneg.py", "test_fastneg_more.py"):
        r = subprocess.run([sys.executable, "-W", "ignore", "-I", os.path.join(HERE, t)], capture_output=True, text=True, env=env, cwd=HERE)
        outs[t] = (r.returncode, [ln for ln in r.stderr.strip().splitlines() if ln.startswith(("Ran ", "OK", "FAILED"))])
    rec("G-SUITE", all(v[0] == 0 for v in outs.values()), str(outs))
    tok = {"schema": L.SCHEMA, "kind": "gate_token", "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "pass": bool(ok), "checks": checks,
           "bindings": gate_bindings(a, S)}
    tok["bindings_sha256"] = L.sha256_bytes(L.canon_json(tok["bindings"]))
    with open(P["token"], "w", encoding="utf-8", newline="\n") as f:
        f.write(L.canon_json(tok).decode())
    log(f"GATE {'PASS' if ok else 'FAIL'}; token bindings sha256 {tok['bindings_sha256']}")
    return 0 if ok else 1


def require_token(a, S: dict) -> dict:
    P = paths(a)
    if not os.path.exists(P["token"]):
        raise L.CensusError("no gate token: every fitting stage REFUSES to start without a PASS token (run `gate`)")
    with open(P["token"], "r", encoding="utf-8") as f:
        tok = json.load(f)
    if not tok.get("pass"):
        raise L.CensusError("the gate token is not a PASS")
    now = gate_bindings(a, S)
    if L.canon_json(now) != L.canon_json(tok["bindings"]):
        diff = [k for k in now if L.canon_json(now[k]) != L.canon_json(tok["bindings"].get(k))]
        raise L.CensusError(f"the gate token's bindings differ from the current code / config / data: {diff}; re-run `gate`")
    return tok


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# stage: fit
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def task_list(group: str) -> list:
    out = []
    if group == "primary":
        for size in (100, 25, 50):
            for s in L.SEEDS:
                out.append({"id": f"L_f{size}_s{s}", "seed": s, "size": size, "cold": False, "gen_salt": 0, "placebo": None})
    elif group == "null":
        for r in range(1, L.NULL_REPS + 1):
            for s in L.SEEDS:
                for side, k in (("A", 2 * r - 1), ("B", 2 * r)):
                    out.append({"id": f"null_r{r}{side}_s{s}", "seed": s, "size": 100, "cold": False, "gen_salt": L.NULL_SALT * k, "placebo": None})
    elif group == "placebo":
        for s in L.SEEDS:
            out.append({"id": f"placebo_q2_25_s{s}", "seed": s, "size": 100, "cold": False, "gen_salt": 0, "placebo": {"stratum": 2, "frac": 0.25}})
    elif group == "cold":
        for size in (100, 25, 50):
            for s in L.SEEDS:
                out.append({"id": f"cold_f{size}_s{s}", "seed": s, "size": size, "cold": True, "gen_salt": 0, "placebo": None})
    else:
        raise L.CensusError(f"unknown group {group}")
    return out


def run_task(ctx: dict, t: dict, P: dict, ram_guard) -> dict:
    N, fold, code = ctx["N"], ctx["fold"], ctx["code"]
    u = L.subsample_priority(code.shape, t["seed"])
    if t["placebo"]:
        incl = L.make_incl(code, u, t["placebo"]["frac"], stratum=t["placebo"]["stratum"])
    elif t["size"] < 100:
        incl = L.make_incl(code, u, t["size"] / 100.0)
    else:
        incl = None
    table = np.empty((2, N, L.N_PROP), np.float32)
    t0 = time.time()
    folds_s = []
    for k in range(L.N_FOLDS):
        tf = time.time()
        tr = np.nonzero(fold != k)[0]
        oof = np.nonzero(fold == k)[0]
        tr_trace: list = []
        w, b = L.fit_rows(ctx["feat"], ctx["z_st"], ctx["y"], tr, ctx["p2_W6"], ctx["p2_b6"], t["seed"], incl=incl, gen_salt=t["gen_salt"], cold=t["cold"], ram_guard=ram_guard, trace=tr_trace)
        sc = L.score_rows(ctx["feat"], ctx["z_st"], oof, w, b, ctx["p2_W6"], ctx["p2_b6"], cold=t["cold"])
        table[0, oof], table[1, oof] = sc[..., 0], sc[..., 1]
        per_epoch = len(tr_trace) // L.EPOCHS
        tr_arr = np.array(tr_trace)
        folds_s.append({"fold": k, "train_sets": int(len(tr)), "oof_sets": int(len(oof)), "seconds": round(time.time() - tf, 1),
                        "loss_first_epoch": [float(x) for x in tr_arr[:per_epoch].mean(0)], "loss_last_epoch": [float(x) for x in tr_arr[-per_epoch:].mean(0)],
                        "w_sha256": L.sha256_bytes(np.ascontiguousarray(w).tobytes())})
        log(f"  {t['id']} fold {k}: {folds_s[-1]['seconds']} s, loss {folds_s[-1]['loss_first_epoch']} -> {folds_s[-1]['loss_last_epoch']}")
    if not np.isfinite(table).all():
        raise L.CensusError(f"{t['id']}: non-finite OOF logits")
    return {"table": table, "seconds": round(time.time() - t0, 1), "folds": folds_s}


def stage_fit(a) -> int:
    import torch
    S = require_strata(a)
    tok = require_token(a, S)
    P = paths(a)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(a.threads))
    os.makedirs(P["fits"], exist_ok=True)
    ctx = load_ctx(a)
    for group in [g for g in a.groups.split(",") if g]:
        for t in task_list(group):
            outp, done = f"{P['fits']}/{t['id']}.npy", f"{P['fits']}/{t['id']}.json"
            if os.path.exists(done) and os.path.exists(outp):
                log(f"skip {t['id']} (done)")
                continue
            L.wait_for_ram(log=log)
            log(f"fit {t['id']} (group {group}) {json.dumps({k: v for k, v in t.items() if k != 'id'})}")
            res = run_task(ctx, t, P, ram_guard=lambda: L.wait_for_ram(log=log))
            tmp = outp + ".tmp.npy"
            np.save(tmp, res["table"])
            os.replace(tmp, outp)
            meta = {"task": t, "table_sha256": L.sha256_file(outp), "seconds": res["seconds"], "folds": res["folds"], "gate_bindings_sha256": tok["bindings_sha256"],
                    "finished": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
            with open(done, "w", encoding="utf-8", newline="\n") as f:
                f.write(L.canon_json(meta).decode())
            log(f"done {t['id']} in {res['seconds']} s")
    log("FIT_GROUPS_DONE " + a.groups)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("inputs", "gate", "fit", "analyze"))
    ap.add_argument("--strata-sha", default="")
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--bank-dir", default=DEFAULT_BANK)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--ranked", default=DEFAULT_RANKED)
    ap.add_argument("--stores-dir", default=DEFAULT_STORES)
    ap.add_argument("--p2-ckpt", default=DEFAULT_P2)
    ap.add_argument("--probe-oof", default=DEFAULT_PROBE_OOF)
    ap.add_argument("--epr1", default=None)
    ap.add_argument("--ranked-sha", default="a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1", help="pinned sha256 of ranked_fit_logs.tsv (tests pass an empty string)")
    ap.add_argument("--ranked-n", type=int, default=1167, help="pinned number of ranked fit logs (tests pass 0)")
    ap.add_argument("--groups", default="primary")
    ap.add_argument("--threads", type=int, default=8)
    return ap


def stage_analyze(a) -> int:
    import fastneg_analyze as AN
    S = require_strata(a)
    P = paths(a)
    return AN.write_analysis(a, S, P, load_ctx(a, need_feat=False))


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"inputs": stage_inputs, "gate": stage_gate, "fit": stage_fit, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
