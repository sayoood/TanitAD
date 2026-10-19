"""PICKRULE-AUDIT-NC runner.  CPU only.  REGISTERED SOURCE: PREREG_PICKRULE_AUDIT_NC.md (sha256 pinned in pickrule_lib.PREREG_SHA256; every stage refuses on a mismatch).

    pickrule_run.py prepare   # S0 bank -> sha256-verified arrays (+ the labels6 join proof), the OOF EP-R1 rows at c* for seeds 0 / 1 (landed epr1.fit_row), the banked final rows; writes run/prepare/
    pickrule_run.py analyze   # gate (C-ID1, C-ID2, C-ID3, C-COL, C-PDMS) -> refuses on a red control -> Q1 / Q2 / Q3, the decision rule, C-NULL; writes run/results/result.json

Env: CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8.  No git.  Stage, never push.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pickrule_lib as P                                                                  # noqa: E402

ROOT = "C:/Users/Admin/par18/pickrule"
RUN = f"{ROOT}/run"
EPR1_RESULT = f"{P.REF}/epr1/epr1_result.json"
EPR1_ROWS = f"{P.REF}/epr1"                                      # ep_row_seed{0,1}.npy exported byte-exact from the tip (raw/2026-10-18-ep-r1/raw/f10/)
PLANNER_COPY = f"{P.REF}/planner.py"
THREADS = 8


def log(msg: str) -> None:
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_ram(min_gb: float = P.MIN_AVAIL_GB, poll_s: float = 20.0, max_wait_s: float | None = None) -> float:
    import psutil
    t0 = time.time()
    while True:
        av = psutil.virtual_memory().available / 2 ** 30
        if av >= min_gb:
            return av
        if max_wait_s is not None and time.time() - t0 > max_wait_s:
            raise P.PickruleError(f"available RAM {av:.2f} GB < {min_gb} GB after {max_wait_s} s")
        log(f"[ram] available {av:.2f} GB < {min_gb} GB; waiting {poll_s:.0f} s")
        time.sleep(poll_s)


def paths(a) -> dict:
    run = a.run_dir
    return {"run": run, "prepare": f"{run}/prepare", "results": f"{run}/results", "arrays": f"{run}/prepare/arrays.npz", "report": f"{run}/prepare/prepare_report.json",
            "ctx": f"{run}/epr1_ctx"}


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_prepare(a) -> int:
    import torch
    prereg_sha = P.require_prereg(a.prereg)
    Pth = paths(a)
    os.makedirs(Pth["prepare"], exist_ok=True)
    if os.path.exists(Pth["arrays"]) and not a.force:
        raise P.PickruleError(f"{Pth['arrays']} exists: prepare is run once (--force to redo)")
    avail = wait_for_ram()
    E = P.load_epr1(a.epr1)
    F = P.load_fastneg(a.fastneg)
    torch.set_num_threads(THREADS)
    opts = E.Options(fraction=10, out_dir=Pth["ctx"], threads=THREADS)
    t0 = time.time()
    log(f"prepare: RAM available {avail:.2f} GB; loading S0 through the landed epr1 loaders")
    guard, out, manifest, man_sha, ranked_sha, selected, repl_rep, per_fold_logs, W6, b6, data = E._setup(opts)
    N = data.n_sets
    log(f"epr1 FitData: {N} sets, {data.n_logs} logs, per-fold logs {per_fold_logs}, provenance row consistency {data.provenance['row_consistency_max_abs']:.3e}")
    if N != 15808 or data.n_logs != P.N_LOGS:
        raise P.PickruleError(f"unexpected size: {N} sets / {data.n_logs} logs (want 15808 / 117)")
    # ---- the full labels6 from the SAME verified units, same order, same masks (C-JOIN)
    sel_pos = {r["log"]: i for i, r in enumerate(selected)}
    units = [n for n in E._io_listdir(guard, opts.bank_dir) if n.endswith(".sets.npz")]
    lab = np.empty((N, 64, 6), np.float32)
    z6b = np.empty((N, 64, 6), np.float32)
    top8 = np.empty((N, 8), np.int64)
    logv = np.empty(N, dtype=object)
    unit_sha, off = {}, 0
    for n in units:
        p = os.path.join(opts.bank_dir, n)
        sha = E._io_sha256(guard, p)
        if n not in manifest or sha != manifest[n]:
            raise P.PickruleError(f"unit {n}: sha256 {sha} != manifest")
        unit_sha[n] = sha
        z = E._io_npz(guard, p)
        mask = np.array([str(x) in sel_pos for x in z["log"]], dtype=bool)
        if not mask.any():
            z.close()
            continue
        lb, lo, tp, lg = z["labels6"][mask], z["logits"][mask], z["top8"][mask], z["log"][mask]
        z.close()
        S = len(lg)
        lab[off:off + S], z6b[off:off + S], top8[off:off + S] = lb, lo, tp.astype(np.int64)
        logv[off:off + S] = [str(x) for x in lg]
        off += S
    if off != N:
        raise P.PickruleError(f"labels pass read {off} sets, FitData has {N}")
    log_idx_lab = np.array([sel_pos[x] for x in logv], np.int64)
    join = {"ep_column_equals_fitdata_y": bool(np.array_equal(lab[..., P.EP], data.y.numpy())),
            "safe_np_equals_fitdata_safe": bool(np.array_equal(E.safe_np(lab), data.safe.numpy())),
            "pdms_np_equals_fitdata_pdms": bool(np.array_equal(E.pdms_np(lab), data.pdms.numpy())),
            "logits_equal_fitdata_z6": bool(np.array_equal(z6b, data.z6.numpy())),
            "top8_equal_fitdata_top8": bool(np.array_equal(top8, data.top8.numpy())),
            "log_idx_equal": bool(np.array_equal(log_idx_lab, data.log_idx)),
            "labels_value_counts_nc": {str(k): int(v) for k, v in zip(*np.unique(lab[..., 0], return_counts=True))},
            "labels_value_counts_dac": {str(k): int(v) for k, v in zip(*np.unique(lab[..., 1], return_counts=True))}}
    if not all(v for k, v in join.items() if isinstance(v, bool)):
        raise P.PickruleError(f"C-JOIN FAILED: {join}")
    log(f"C-JOIN PASS: {join}")
    # ---- the OOF EP rows at c* (the landed epr1.fit_row / apply_row, exactly as epr1.oof_run does), seeds 0 / 1
    cfg = E.Cfg(lr=P.CSTAR["lr"], anchor=P.CSTAR["anchor"], epochs=P.CSTAR["epochs"], batch_sets=P.CSTAR["batch_sets"])
    if cfg.id() != P.CSTAR["id"] or cfg.sha256() != P.CSTAR["config_sha256"]:
        raise P.PickruleError(f"c* config {cfg.id()} / {cfg.sha256()} != the registered {P.CSTAR}")
    p2_w, p2_b = np.asarray(W6, np.float32)[P.EP], np.asarray(b6, np.float32)[P.EP:P.EP + 1]
    arrays = {"z6": z6b, "lab": lab, "top8": top8, "log_idx": data.log_idx.astype(np.int64), "fold": data.fold.astype(np.int64), "log_rank": data.log_rank.astype(np.int64),
              "log_fold": data.log_fold.astype(np.int64), "log_names": np.array(data.log_names)}
    fit_times, oof_rows = {}, {}
    for seed in P.SEEDS:
        ep = np.empty((N, 64), np.float32)
        rows_k = []
        for k in range(E.N_FOLDS):
            tf = time.time()
            oof = np.nonzero(data.fold == k)[0]
            tr_logs = [i for i in np.argsort(data.log_rank, kind="stable") if data.log_fold[i] != k]
            tr = np.nonzero(np.isin(data.log_idx, np.array(tr_logs, np.int64)))[0]
            wait_for_ram()
            w, b = E.fit_row(data, tr, cfg, seed, W6, b6)
            rows_k.append(np.concatenate([w, b]).astype(np.float32))
            for s in range(0, len(oof), 2048):
                ii = oof[s:s + 2048]
                zz = E.apply_row(data.z6[torch.as_tensor(ii)], data.feat[torch.as_tensor(ii)], w, b, p2_w, p2_b)
                ep[ii] = zz[..., P.EP].numpy()
            fit_times[f"seed{seed}_fold{k}"] = round(time.time() - tf, 1)
            log(f"OOF EP row seed {seed} fold {k}: train sets {len(tr)}, held-out sets {len(oof)}, {fit_times[f'seed{seed}_fold{k}']} s")
        arrays[f"ep_OOF_s{seed}"] = ep
        oof_rows[seed] = np.stack(rows_k)
        np.save(f"{Pth['prepare']}/oof_rows_seed{seed}.npy", oof_rows[seed])
    # ---- the banked FINAL rows (SECONDARY, in-sample, zero fits), sha256-checked against the EP-R1 result
    with open(EPR1_RESULT, "r", encoding="utf-8") as f:
        epr1_result = json.load(f)
    banked = {}
    for seed in P.SEEDS:
        rp = f"{EPR1_ROWS}/ep_row_seed{seed}.npy"
        sha = P.sha256_file(rp)
        want = epr1_result["final_rows"][str(seed)]["sha256"]
        if sha != want:
            raise P.PickruleError(f"banked row {rp} sha256 {sha} != the EP-R1 result's {want}")
        r = np.load(rp)
        if r.shape != (257,) or r.dtype != np.float32:
            raise P.PickruleError(f"banked row {rp}: {r.dtype}{r.shape}, want float32[257]")
        w, b = r[:256].copy(), r[256:257].copy()
        ep = np.empty((N, 64), np.float32)
        for s in range(0, N, 2048):
            ii = np.arange(s, min(N, s + 2048))
            zz = E.apply_row(data.z6[torch.as_tensor(ii)], data.feat[torch.as_tensor(ii)], w, b, p2_w, p2_b)
            ep[ii] = zz[..., P.EP].numpy()
        arrays[f"ep_BANKED_s{seed}"] = ep
        banked[str(seed)] = {"file": os.path.basename(rp), "sha256": sha, "matches_epr1_result": True}
    np.savez(Pth["arrays"], **arrays)
    p2_sha = P.sha256_file(opts.p2_ckpt)
    rep = {"schema": P.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "prereg_sha256": prereg_sha, "n_sets": int(N), "n_logs": int(data.n_logs),
           "per_fold_logs": per_fold_logs, "fold_sets": [int((data.fold == k).sum()) for k in range(E.N_FOLDS)], "bank_manifest_sha256": man_sha, "ranked_logs_sha256": ranked_sha,
           "p2_ckpt": opts.p2_ckpt, "p2_ckpt_sha256": p2_sha, "bank_ckpt": data.provenance["bank_ckpt"], "unit_sha256_verified": unit_sha, "epr1_provenance": {k: v for k, v in data.provenance.items() if k != "unit_sha256"},
           "c_join": join, "cstar": P.CSTAR, "banked_rows": banked, "epr1_result_file_sha256": P.sha256_file(EPR1_RESULT), "epr1_sha256": P.EPR1_SHA256, "fastneg_lib_sha256": P.FASTNEG_LIB_SHA256,
           "code_sha256": {n: P.sha256_file(os.path.join(HERE, n)) for n in ("pickrule_lib.py", "pickrule_run.py")}, "fit_times_s": fit_times, "torch": torch.__version__, "numpy": np.__version__,
           "threads": THREADS, "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()), "arrays_npz_sha256": P.sha256_file(Pth["arrays"]),
           "elapsed_s": round(time.time() - t0, 1)}
    with open(Pth["report"], "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(rep), indent=1, sort_keys=True))
    log(f"PREPARE_DONE in {rep['elapsed_s']} s -> {Pth['arrays']}")
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def stage_analyze(a) -> int:
    prereg_sha = P.require_prereg(a.prereg)
    Pth = paths(a)
    os.makedirs(Pth["results"], exist_ok=True)
    E = P.load_epr1(a.epr1)
    F = P.load_fastneg(a.fastneg)
    z = np.load(Pth["arrays"], allow_pickle=False)
    arr = {k: z[k] for k in z.files}
    with open(Pth["report"], "r", encoding="utf-8") as f:
        prep = json.load(f)
    if prep["arrays_npz_sha256"] != P.sha256_file(Pth["arrays"]):
        raise P.PickruleError("arrays.npz does not match the sha256 recorded by prepare")
    if prep["prereg_sha256"] != prereg_sha:
        raise P.PickruleError("prepare ran under a different prereg")
    ep_cols = {(src, sd): arr[f"ep_{src}_s{sd}"] for src in P.SOURCES for sd in P.SEEDS}
    with open(EPR1_RESULT, "r", encoding="utf-8") as f:
        epr1_result = json.load(f)
    row = [r for r in epr1_result["grid"] if r["id"] == P.CSTAR["id"]][0]
    ref_pl = {sd: {**row["per_seed"][str(sd)]["pick_level"], "oof_picks_changed": row["per_seed"][str(sd)]["oof_picks_changed"]} for sd in P.SEEDS}
    gate = P.gate_checks(arr, ep_cols, E, ref_pl, a.planner)
    gate["prereg_sha256"] = prereg_sha
    gate["c_join"] = prep["c_join"]
    with open(f"{Pth['results']}/gate_report.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(gate), indent=1, sort_keys=True))
    for k, v in gate["checks"].items():
        log(f"gate {k}: {'PASS' if v['pass'] else 'FAIL'}")
    if not gate["ok"]:
        log("GATE FAILED: no quantity is computed (write an amendment before the number it governs)")
        return 3
    draws = F.boot_draws(P.N_LOGS, P.BOOT_B, P.BOOT_SEED)
    if not np.array_equal(draws, E.boot_draws(P.N_LOGS, P.BOOT_B, P.BOOT_SEED)):
        raise P.PickruleError("fastneg_lib.boot_draws differs from epr1.boot_draws")
    S = P.Stats(arr["log_idx"], draws, F)
    if len(np.unique(arr["log_idx"])) != P.N_LOGS:
        raise P.PickruleError("log assignment does not cover 117 logs")
    log("gate PASS: computing Q1 / Q2 / Q3")
    res = P.run_analysis(arr, ep_cols, S)
    log("computing C-NULL")
    null = P.run_null(arr, ep_cols, S)
    primary = res["by_source"][P.PRIMARY]
    final = "VOID (C-NULL fired on >= 2 of 5 replicates)" if null["void"] else primary["outcome"]
    out = {"schema": P.SCHEMA, "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "prereg_sha256": prereg_sha, "n_sets": int(len(arr["z6"])), "n_logs": P.N_LOGS,
           "bootstrap": {"B": P.BOOT_B, "seed": P.BOOT_SEED, "unit": "log", "percentiles": [2.5, 97.5]}, "gate": gate, "analysis": res, "null": null, "final_outcome": final,
           "primary_source": P.PRIMARY, "source_sensitive": bool(res["by_source"]["OOF"]["outcome"] != res["by_source"]["BANKED"]["outcome"]),
           "next_arm": ("none: the null fired (VOID)" if null["void"] else primary["next_arm"]), "veto_recovers_primary": primary["veto_recovers"], "recovering_taus_primary": primary["recovering_taus"],
           "code_sha256": {n: P.sha256_file(os.path.join(HERE, n)) for n in ("pickrule_lib.py", "pickrule_run.py")}}
    with open(f"{Pth['results']}/result.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(P.jsonable(out), indent=1, sort_keys=True))
    log(f"ANALYZE_DONE: outcome {final}; source_sensitive {out['source_sensitive']}; veto_recovers {out['veto_recovers_primary']} {out['recovering_taus_primary']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "analyze"])
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--prereg", default=os.path.join(ROOT, "PREREG_PICKRULE_AUDIT_NC.md"))
    ap.add_argument("--epr1", default=None)
    ap.add_argument("--fastneg", default=None)
    ap.add_argument("--planner", default=PLANNER_COPY)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"prepare": stage_prepare, "analyze": stage_analyze}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
