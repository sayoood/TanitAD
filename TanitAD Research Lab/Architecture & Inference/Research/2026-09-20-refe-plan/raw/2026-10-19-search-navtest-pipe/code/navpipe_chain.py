#!/usr/bin/env python3
"""SEARCH-NAVTEST-PIPE chain: the chunked navtest pipeline for SEARCH-P2-CONFIRM READ C2.  NOT LAUNCHED by the build.

    c2 mode      G-SEAL + the C1 LICENCE + the dose result + the CONFIG (C8 or O4: exactly the one `c2_config` returns) + the FIXTURE RECORD (reads IDENTICAL, code shas == the sealed pins) (first, always; MM_RULING_C2_LICENCE.md,
                 MM_RULING_C2AMEND.md) -> pins vs files -> the ONE-RUN marker (one C2 run per H-SEARCH-P2-1) -> disk -> per chunk [ bundle(GPU, lock) -> G-IDENT -> G-REACH -> search P2HEAD x3 seeds -> T1 re-score of P2's PICK
                 (DQ6 (b); cross-checked against the banked CSV, a mismatch STOPS) + T1 score of the 3 seeds + T1 planted identity -> bank -> delete ctx ] -> merge (sorted-token order) -> analysis (DQ3 terms; a safety miss in
                 the headline line, DQ5).   Reads no label until the seal passes.
    s0bank mode  (MM scope addition 2026-10-19; TOKEN-LIST GENERIC) the same pipeline on ANY pinned token list, scored by the label-only PDM route (`label_stepB`, NOT the T1 harness): G-SEAL(S0) -> pins -> G-NOT-NAVTEST ->
                 per chunk [ bundle(GPU, lock) -> fan-label join -> G-IDENT vs P2's own S0 fan files -> G-REACH -> search -> label_stepB + its planted identity -> bank -> delete ctx ] -> merge.  Output = the Step B / C1
                 npz triple that `epadm_searchbank.py build` reads.  The seal is a SEALED S0-bank amendment sha (no admission record: EP-ADMISSION is READ on this bank).
    fixture mode the SAME stage functions on carve B's ALREADY-READ bundle as the stand-in chunks (bundle slicer instead of the GPU, `label_stepB` instead of the T1 harness, no seal: it refuses any navtest token).
                 It must reproduce Step B's banked P2HEAD read EXACTLY.

Resumable: a stage whose output validates is skipped; an output that does not validate after its child ran is a STOP, never a pass.  The per-chunk state file is `<work>/navpipe_state.json`.
Markers ZZNAVPIPE_* start their line.  Exit: 0 done, 3 gate, 4 timeout, 5 seal / data refusal, 6 search, 7 score, 8 analysis, 9 crash.   python navpipe_chain.py --mode {navtest|c2,s0bank,fixture} [--tokens FILE] ... [--dry-run]
  navtest: --amendment F --amendment-sha256 H --c1-result F --c1-result-sha256 H --dose-result F --dose-result-sha256 H --config F --fixture-record F [--table-ref F]   (no default config)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
from navpipe_core import Refusal  # noqa: E402
NC.add_code_paths()                                    # Step B's and L1's code: the repo-relative sibling packages (navpipe_core.STEPB_CODE / L1_CODE); SPC_STEPB_CODE / L1_CODE override
STEPB_CODE, L1_CODE = NC.STEPB_CODE, NC.L1_CODE

PY_CPU = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_GPU = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
PY_LABEL = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
REFE_DIR = "C:/Users/Admin/par18/l1run/tree_hybrid/refe"
CONFIG = NC.CONFIG_C8                                                                              # the C8 file: the s0bank / fixture default.  The navtest (C2) mode takes --config (C8 or O4) and refuses a choice `c2_config` did not make.
HEAD_S0 = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads/final_head_s0.npz"
LAUNCHER = NC.find_launcher()
NAVTEST_TOKENS = "D:/Projects/TanitAD/data/refe_navtest/navtest_full_tokens.json"
BANKED_SEAM = "D:/Projects/TanitAD/data/refe_navtest/seams/stage2_p2_full.npz"
BANKED_CSV = "D:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p2_full/refe_stage2_p2_full.csv"
WORK_C2 = "D:/Projects/TanitAD-artifacts/navpipe_c2"
HARNESS = os.path.join(PKG, "ref", "score_navtest_refe.py")
HARNESS_SHA256_LF = "94f7b024ab2a375a62cf7268e85b0d245ac81ff7b69f2f0c4a9295050ab962b7"          # eval/score_navtest_refe.py on the mirror tip fb7ad9a2 (NT-FIX1: loads W3's run_v1 through run_v1_fix1.load())
T1ID_K = 24                                                                                       # tokens per chunk in the T1 planted identity (2 extra harness calls of K tokens)
PLAN_LABEL = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"                                  # fixture: carve B's token -> log / DB
CARVE_B = "C:/Users/Admin/par18/l2fresh/raw/fresh_sample_token_disjoint.json"
CARVE_B_SHA = "0629811ff5d05879eb62abc5726005ccc4924f03539423003503a099ba29577c"
SRC_BUNDLE = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle"
BANKED_SEARCH = "C:/Users/Admin/par18/l2stepb/run/search_P2HEAD.npz"
BANKED_LABELS = "C:/Users/Admin/par18/l2stepb/run/labels_P2HEAD.npz"
BANKED_RESULT = "C:/Users/Admin/par18/l2stepb/run/RESULT_STEPB.json"
TIME_FIELDS = ("t_search_s", "t_token_s")
CRASH_CODES = (3221225477, 3221226505, -1073741819, -1073740791)                                  # 0xC0000005 access violation, 0xC0000409 stack buffer overrun (as unsigned / signed)
CRASH_RETRIES = 2


def fixture_slice(src_dir: str, tokens: list, out_dir: str) -> dict:
    """FIXTURE ONLY (stands in for the GPU stage): the chunk's rows of carve B's banked bundle -> a chunk bundle.npz + ctx.npy + ctx.json, exactly the files the real builder writes.  Row order = sorted tokens."""
    src = {k: v for k, v in np.load(os.path.join(src_dir, "bundle.npz"), allow_pickle=False).items()}
    ctx = np.load(os.path.join(src_dir, "ctx.npy"), mmap_mode="r")
    ix = {str(t): i for i, t in enumerate(src["token"])}
    rows = [ix[t] for t in tokens]
    os.makedirs(out_dir, exist_ok=True)
    out = {k: (v[rows] if getattr(v, "ndim", 0) >= 1 and v.shape[0] == len(ix) else v) for k, v in src.items()}
    np.savez(os.path.join(out_dir, "bundle.npz"), **out)
    mm = np.lib.format.open_memmap(os.path.join(out_dir, "ctx.npy"), mode="w+", dtype=ctx.dtype, shape=(len(rows),) + ctx.shape[1:])
    for j, r in enumerate(rows):
        mm[j] = ctx[r]
    mm.flush()
    del mm
    cj = {"n": len(rows), "ctx_file": "ctx.npy", "dtype": str(ctx.dtype), "shape": [len(rows)] + list(ctx.shape[1:]), "ctx_sha256": NC.sha256_file(os.path.join(out_dir, "ctx.npy")), "tokens_sha256": NC.tokens_sha256(tokens),
          "source": "FIXTURE slice of " + src_dir}
    json.dump(cj, open(os.path.join(out_dir, "ctx.json"), "w"), indent=1)
    return cj


class Cfg:
    def __init__(self, work, mode="c2", amendment=None, amendment_sha256=None, c1_result=None, c1_result_sha256=None, cap=1000, procs=1, prefetch=0, min_free_gib=NC.MIN_FREE_GIB, min_ram_gb=3.0, max_chunks=None,
                 token_file=NAVTEST_TOKENS, banked_seam=BANKED_SEAM, banked_csv=BANKED_CSV, ckpt=CKPT, config=CONFIG, head_s0=HEAD_S0, refe_dir=REFE_DIR, launcher=LAUNCHER, py_cpu=PY_CPU, py_gpu=PY_GPU,
                 py_label=PY_LABEL, harness=HARNESS, plan_label=PLAN_LABEL, src_bundle=SRC_BUNDLE, code_dir=HERE, threads=4, hash_ckpt=True, t1id_k=T1ID_K, label_script=None, harness_py=None, expect_tokens=(12146, 136), reference_kind=None, reference_path=None, table_ref=None, fan_labels=None, navtest_tokens=NAVTEST_TOKENS,
                 dose_result=None, dose_result_sha256=None, dose_chain=None, fixture_record=None, one_run_marker=NC.ONE_RUN_MARKER):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.work = os.path.abspath(work)
        self.reference_kind = reference_kind or ("s0fan" if mode == "s0bank" else "seam")
        self.reference_path = reference_path or (None if mode == "s0bank" else banked_seam)
        self.label_script = label_script or os.path.join(STEPB_CODE, "label_stepB.py")
        self.harness_py = harness_py or py_gpu

    def d(self, *p):
        return os.path.join(self.work, *p)

    def cdir(self, cid, *p):
        return os.path.join(self.work, "chunks", f"c{int(cid):02d}", *p)


def child_env(label=False):
    import stepB_chain as CH
    return CH.child_env(label)


def real_runner(argv, env, log_path, cwd=None):
    import stepB_chain as CH
    return CH.real_runner(argv, env, log_path, cwd)


class Chain:
    def __init__(self, cfg, runner=real_runner, out=None, ram_fn=None, free_fn=None, harness_fn=None):
        self.cfg, self.run_child_fn, self.out = cfg, runner, out or sys.stdout
        self.ram_fn = ram_fn or (lambda: __import__("stepB_chain").avail_gb())
        self.free_fn = free_fn
        self.harness_fn = harness_fn                      # test seam: (label, seam_path, tokens_file) -> csv path
        self.retry_sleep_s = 30
        self.state = None
        self.seal = None
        self.token_doc = None
        self.plan = None
        self.pins = {}

    # ------------------------------------------------------------------------------------------------ plumbing
    def emit(self, marker, *parts):
        print("ZZNAVPIPE_" + marker + (" " + " ".join(str(p) for p in parts) if parts else ""), file=self.out, flush=True)

    def child(self, argv, log_name, label=False, env_extra=None, cuda=False, cwd=None):
        env = dict(child_env(label))
        if not cuda:
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        env.update(env_extra or {})
        os.makedirs(self.cfg.d("logs"), exist_ok=True)
        for att in range(1 + CRASH_RETRIES):
            rc = self.run_child_fn(argv, env, self.cfg.d("logs", log_name), cwd)
            if rc not in CRASH_CODES or att == CRASH_RETRIES:                                      # @GUARD:C-CRASH-RETRY
                return rc
            self.emit("CHILD_CRASH_RETRY", log_name, f"rc={rc}", f"attempt {att + 1}")             # MEASURED 2026-10-19: a gate child died with 0xC0000005 under memory pressure; the same command then passed
            time.sleep(self.retry_sleep_s)
        return rc

    def disk(self, need_bytes=0):
        os.makedirs(self.cfg.work, exist_ok=True)
        return NC.check_disk(self.cfg.work, self.cfg.min_free_gib, need_bytes, self.free_fn)

    # ------------------------------------------------------------------------------------------------ gates before any chunk
    def check_c2(self):
        c = self.cfg
        self.seal = NC.check_c2_seal(c.amendment, c.amendment_sha256, c.c1_result, c.c1_result_sha256, c.code_dir, config=c.config, dose_result=c.dose_result, dose_result_sha256=c.dose_result_sha256, fixture_record=c.fixture_record, dose_chain=c.dose_chain)   # G-SEAL + the C1 LICENCE + the CONFIG (C8 or O4, == c2_config) + the FIXTURE RECORD: FIRST, before any data is opened   @GUARD:C-SEAL-FIRST
        pins = self.seal["pins"]
        lic, sel = self.seal["licence"], self.seal["selection"]
        self.emit("SEAL_OK", self.seal["amendment_sha256"][:12])
        self.emit("LICENCE_OK", lic["outcome"], "safety", json.dumps(lic["safety_points"]), "moved", f"{lic['moved_share_inert_frac_moved']:.4f}", "c1_result", lic["c1_result_sha256"][:12], "licensing_read", lic["licensing_read"])
        self.emit("CONFIG_OK", sel["config"], f"row {sel['row']}", "config", sel["config_sha256"][:12], "dose_result", sel["dose_result_sha256"][:12], "dose_chain", sel["dose_chain_sha256"][:12])
        self.emit("FIXTURE_RECORD_OK", self.seal["fixture"]["verdict"], self.seal["fixture"]["fixture_record_sha256"][:12])
        bad = NC.file_pin_mismatches(pins, {"token_file_sha256": c.token_file, "ckpt_sha256": c.ckpt if c.hash_ckpt else None, "config_sha256": c.config, "head_s0_sha256": c.head_s0, "banked_seam_sha256": c.banked_seam,
                                            "banked_csv_sha256": c.banked_csv, "harness_script_sha256": c.harness, "seam_module_sha256": os.path.join(PKG, "ref", "refe_navtest_seam.py"),
                                            "dose_chain_sha256": c.dose_chain or NC.DOSE_CHAIN})
        if NC.sha_lf(c.harness) != HARNESS_SHA256_LF:
            bad.append("harness_script_sha256 (this code's constant)")
        if bad:
            raise Refusal("seal", "SEAL_PIN_MISMATCH", ";".join(bad))
        ident = {"amendment_sha256": self.seal["amendment_sha256"], "config_sha256": sel["config_sha256"], "c1_result_sha256": sel["c1_result_sha256"], "dose_result_sha256": sel["dose_result_sha256"], "work": NC.norm_work(c.work)}
        self.one_run_ident, self.one_run = ident, NC.one_run_check(c.one_run_marker, ident)        # ONE look per H-SEARCH-P2-1: a consumed or foreign marker refuses here, before any data is opened
        self.emit("ONE_RUN_" + self.one_run, NC.HYPOTHESIS, sel["config"])
        self.token_doc = json.load(open(c.token_file, encoding="utf-8"))
        plan = NC.plan_doc(self.token_doc, pins["token_file_sha256"], c.cap)
        if plan["plan_sha256"] != pins["chunk_plan_sha256"] or (plan["n_tokens"], plan["n_logs"]) != tuple(c.expect_tokens):
            raise Refusal("seal", "PLAN_NOT_THE_PINNED_ONE", f"recomputed {plan['plan_sha256'][:12]}, {plan['n_tokens']} tokens / {plan['n_logs']} logs")
        self.plan = plan
        os.makedirs(c.work, exist_ok=True)
        json.dump(plan, open(c.d("chunk_plan.json"), "w"), indent=1)
        json.dump({"amendment_sha256": self.seal["amendment_sha256"], "licence": lic, "selection": sel, "fixture": self.seal["fixture"]}, open(c.d("licence_c1.json"), "w"), indent=1)       # the licence and the config choice as READ BY CODE, banked beside the plan
        self.emit("DATA_OK", plan["n_tokens"], "tokens", plan["n_logs"], "logs", len(plan["chunks"]), "chunks", plan["plan_sha256"][:12])

    def check_s0bank(self):
        """G-SEAL(S0) FIRST (before any data is opened), then every pin against the file it names, then the G-NOT-NAVTEST guard, then the chunk plan against its pin"""
        c = self.cfg
        self.seal = NC.check_s0_seal(c.amendment, c.amendment_sha256, c.code_dir)                              # @GUARD:C-S0-SEAL-FIRST
        pins = self.seal["pins"]
        self.emit("SEAL_OK", self.seal["amendment_sha256"][:12], "s0bank", "no admission record")
        if not c.reference_path or not os.path.exists(c.reference_path):
            raise Refusal("seal", "S0_REFERENCE_MISSING", str(c.reference_path))
        if not c.fan_labels or not os.path.isfile(c.fan_labels):
            raise Refusal("seal", "S0_FAN_LABELS_MISSING", str(c.fan_labels))
        if not c.plan_label or not os.path.isfile(c.plan_label):
            raise Refusal("seal", "S0_PLAN_MISSING", str(c.plan_label))
        ref_sha = NC.fan_dir_sha256(c.reference_path) if os.path.isdir(c.reference_path) else NC.sha256_file(c.reference_path)
        bad = []
        if NC.sha256_file(c.token_file) != pins["token_file_sha256"]:
            bad.append("token_file_sha256")
        if NC.sha256_file(c.plan_label) != pins["plan_json_sha256"]:
            bad.append("plan_json_sha256")
        if c.hash_ckpt and NC.sha256_file(c.ckpt) != pins["ckpt_sha256"]:
            bad.append("ckpt_sha256")
        if NC.sha_lf(c.config) != pins["config_sha256"]:
            bad.append("config_sha256")
        if NC.sha256_file(c.head_s0) != pins["head_s0_sha256"]:
            bad.append("head_s0_sha256")
        if ref_sha != pins["reference_sha256"] or pins["reference_kind"] != c.reference_kind:
            bad.append("reference_sha256/reference_kind")
        if NC.sha256_file(c.fan_labels) != pins["fan_labels_sha256"]:
            bad.append("fan_labels_sha256")
        if bad:
            raise Refusal("seal", "SEAL_PIN_MISMATCH", ";".join(bad))
        self.token_doc = json.load(open(c.token_file, encoding="utf-8"))
        NC.refuse_navtest_tokens(self.token_doc["tokens"], json.load(open(c.navtest_tokens, encoding="utf-8"))["tokens"])
        plan = NC.plan_doc(self.token_doc, pins["token_file_sha256"], c.cap)
        if plan["plan_sha256"] != pins["chunk_plan_sha256"] or (plan["n_tokens"], plan["n_logs"]) != (int(pins["n_tokens"]), int(pins["n_logs"])):
            raise Refusal("seal", "PLAN_NOT_THE_PINNED_ONE", f"recomputed {plan['plan_sha256'][:12]}, {plan['n_tokens']} tokens / {plan['n_logs']} logs")
        self.plan = plan
        os.makedirs(c.work, exist_ok=True)
        json.dump(plan, open(c.d("chunk_plan.json"), "w"), indent=1)
        self.emit("DATA_OK", plan["n_tokens"], "S0 tokens", plan["n_logs"], "logs", len(plan["chunks"]), "chunks", plan["plan_sha256"][:12])

    def check_fixture(self):
        c = self.cfg
        cb = json.load(open(CARVE_B, encoding="utf-8"))
        if NC.tokens_sha256(sorted(cb["tokens"])) != CARVE_B_SHA:
            raise Refusal("seal", "FIXTURE_DATA_REFUSED", "carve B's token list does not hash to its sealed sha")
        nav = set(json.load(open(NAVTEST_TOKENS, encoding="utf-8"))["tokens"])
        hit = nav & set(cb["tokens"])
        if hit:                                                                              # @GUARD:C-FIXTURE-NOT-NAVTEST
            raise Refusal("seal", "FIXTURE_REFUSED_NAVTEST_TOKEN", f"{len(hit)} fixture tokens are navtest tokens")
        pl = json.load(open(c.plan_label, encoding="utf-8"))
        self.token_doc = {"tokens": sorted(cb["tokens"]), "token_log": {t: pl["token_log"][t] for t in cb["tokens"]}}
        self.plan = NC.plan_doc(self.token_doc, CARVE_B_SHA, c.cap)
        os.makedirs(c.work, exist_ok=True)
        json.dump(self.plan, open(c.d("chunk_plan.json"), "w"), indent=1)
        self.emit("DATA_OK", self.plan["n_tokens"], "FIXTURE tokens", self.plan["n_logs"], "logs", len(self.plan["chunks"]), "chunks", self.plan["plan_sha256"][:12])

    # ------------------------------------------------------------------------------------------------ stage: bundle
    def sample_file(self, cid, toks):
        p = self.cfg.cdir(cid, "sample.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"n": len(toks), "tokens": toks, "tokens_sha256": NC.tokens_sha256(toks)}, open(p, "w"))
        return p

    def bundle_valid(self, cid, chunk):
        d = self.cfg.cdir(cid, "bundle")
        try:
            toks = [str(t) for t in np.load(os.path.join(d, "bundle.npz"), allow_pickle=False)["token"]]
            cj = json.load(open(os.path.join(d, "ctx.json")))
        except (OSError, KeyError, ValueError) as e:
            return False, f"unreadable ({type(e).__name__})"
        if len(toks) != chunk["n"] or NC.tokens_sha256(toks) != chunk["tokens_sha256"]:              # @GUARD:C-BUNDLE-TOKENS
            return False, "bundle tokens differ from the chunk plan"
        if cj.get("tokens_sha256") != chunk["tokens_sha256"] or cj.get("n") != chunk["n"]:
            return False, "ctx.json differs from the chunk plan"
        p = os.path.join(d, cj.get("ctx_file", "ctx.npy"))
        if not os.path.isfile(p) or NC.sha256_file(p) != cj.get("ctx_sha256"):                      # @GUARD:C-BUNDLE-CTX-SHA
            return False, "ctx.npy missing or its sha256 differs from ctx.json"
        return True, ""

    def stage_bundle(self, cid):
        c, chunk = self.cfg, self.plan["chunks"][cid]
        toks = NC.chunk_tokens(self.token_doc, chunk)
        ok, _ = self.bundle_valid(cid, chunk)
        if not ok:
            self.disk(chunk["n"] * NC.CTX_BYTES_PER_TOKEN)                                            # G-DISK before the ctx exists
            if c.mode == "fixture":
                fixture_slice(c.src_bundle, toks, c.cdir(cid, "bundle"))
            else:
                sample = self.sample_file(cid, toks)
                plan_json = c.plan_label if c.mode == "s0bank" else c.d("navtest_plan.json")
                argv = [c.py_cpu, "-u", c.launcher, "run", "--max-wait", "21600", "--", c.py_gpu, "-u", os.path.join(c.code_dir, "navpipe_gpu_stage.py"), "--out-dir", c.cdir(cid), "--plan", plan_json,
                        "--chunk-sample", sample, "--n-expected", str(chunk["n"]), "--reference-kind", c.reference_kind, "--reference-path", c.reference_path, "--ckpt", c.ckpt]
                if cid == 0:
                    argv += ["--bridge-sample", c.d("bridge3_sample.json")]
                env = dict(child_env(False))
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env["L1_TREE"] = "C:/Users/Admin/par18/l1run/tree_hybrid"
                os.makedirs(c.d("logs"), exist_ok=True)
                rc = self.run_child_fn(argv, env, c.d("logs", f"gpu_c{cid:02d}.log"), None)
                log = open(c.d("logs", f"gpu_c{cid:02d}.log"), encoding="utf-8", errors="replace").read() if os.path.isfile(c.d("logs", f"gpu_c{cid:02d}.log")) else ""
                if rc != 0 or re.search(r"^ZZSTAGE_FAIL", log, re.M) or not re.search(rf"^\s*ZZBUNDLE_OK {chunk['n']}(\s|$)", log, re.M):
                    raise Refusal("gate", "G_GPU_STAGE_FAILED", f"chunk {cid} rc={rc}; see {c.d('logs', f'gpu_c{cid:02d}.log')}")
            ok, why = self.bundle_valid(cid, chunk)
            if not ok:
                raise Refusal("gate", "G_BUNDLE_INVALID", f"chunk {cid}: {why}")
        if c.mode == "s0bank":
            self.join_fan(cid)                                                                       # idempotent; BEFORE the gate (its stamp is bound to the bundle's sha256)
        return toks

    def join_fan(self, cid):
        """s0bank: add P2's labelled fan (sub / pdms / valid, a PINNED INPUT of the amendment, never computed here) to the chunk bundle by token"""
        c = self.cfg
        p = c.cdir(cid, "bundle", "bundle.npz")
        z = {k: v for k, v in np.load(p, allow_pickle=False).items()}
        if all(k in z for k in ("sub", "pdms", "valid")):
            return
        fl = getattr(self, "_fan_labels", None)
        if fl is None:
            fl = self._fan_labels = {k: v for k, v in np.load(c.fan_labels, allow_pickle=False).items()}
        out = NC.join_fan_labels(z, fl)
        tmp = p + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, p)

    def identity(self, cid, toks):
        """G-IDENT (c2) from the WRITTEN bundle, independent of the builder's own gate; fixture: the rows are copies of the banked bundle (the navtest identity does not apply)"""
        if self.cfg.mode == "fixture":
            return {"ok": True, "n": len(toks), "note": "FIXTURE: rows sliced from carve B's banked bundle; the navtest-seam identity is C2-only"}
        b = np.load(self.cfg.cdir(cid, "bundle", "bundle.npz"), allow_pickle=False)
        toks_b = [str(t) for t in b["token"]]
        if self.cfg.mode == "s0bank":
            ref = getattr(self, "_ref", None) or NC.make_reference(self.cfg.reference_kind, self.cfg.reference_path)
            self._ref = ref
            rec = NC.check_reference_identity(ref, toks_b, b["traj_raw"], b["logits"], b["pick"])
        else:
            banked = getattr(self, "_banked", None) or NC.load_banked_seam(self.cfg.banked_seam)
            self._banked = banked
            rec = NC.check_pick_identity(toks_b, b["traj_raw"], b["pick"], banked)
            if rec["ok"] and self.cfg.table_ref:                                                # the optional STRONGER navtest identity: all 64 proposals, the six logits and the pick vs P2's banked table
                tref = getattr(self, "_tref", None) or NC.make_reference("table", self.cfg.table_ref)
                self._tref = tref
                rt = NC.check_reference_identity(tref, toks_b, b["traj_raw"], b["logits"], b["pick"])
                rec = {"ok": bool(rec["ok"] and rt["ok"]), "table": rt, **{k: v for k, v in rec.items() if k != "ok"}}
        if not rec["ok"]:                                                                      # @GUARD:C-IDENT-STOPS
            raise Refusal("gate", "G_IDENT_NOT_PASSED", json.dumps(rec)[:400])
        return rec

    # ------------------------------------------------------------------------------------------------ stage: G-REACH stamp
    def stamp_ok(self, cid):
        g = glob.glob(self.cfg.cdir(cid, "gate", "G-REACH-STEPB_PASS_*.json"))
        if not g:
            return None
        try:
            st = json.load(open(g[0]))
        except (OSError, ValueError):
            return None
        pins = self.pins
        good = st.get("schema") == "tanitad.g_reach_stepb/1" and st.get("verdict") == "PASS" and st.get("bundle_sha256") == NC.sha256_file(self.cfg.cdir(cid, "bundle", "bundle.npz")) \
            and st.get("ckpt_sha256") == pins["ckpt_sha256"] and st.get("config_sha256") == pins["config_sha256"]
        return g[0] if good else None

    def common(self, cid):
        c, d = self.cfg, self.cfg.cdir(cid, "bundle")
        return ["--plan-file", c.d("chunk_plan.json"), "--plan-sha256", self.plan["plan_sha256"], "--chunk", str(cid), "--bundle", os.path.join(d, "bundle.npz"), "--ctx", os.path.join(d, "ctx.npy"),
                "--ckpt", c.ckpt, "--ckpt-sha256", self.pins["ckpt_sha256"], "--refe-dir", c.refe_dir, "--config", c.config, "--config-sha256", self.pins["config_sha256"]]

    def stage_gate(self, cid):
        c = self.cfg
        if not self.stamp_ok(cid):
            argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "navpipe_launch.py"), "gate", *self.common(cid), "--head", c.head_s0, "--head-sha256", self.pins["head_s0_sha256"], "--out-dir", c.cdir(cid, "gate")]
            rc = self.child(argv, f"gate_c{cid:02d}.log")
            if rc != 0 or not self.stamp_ok(cid):                                              # @GUARD:C-GATE-STAMP
                raise Refusal("gate", "G_REACH_NOT_PASSED", f"chunk {cid} rc={rc}")
        return self.stamp_ok(cid)

    # ------------------------------------------------------------------------------------------------ stage: search
    def merged_valid(self, cid, path):
        chunk = self.plan["chunks"][cid]
        if not os.path.isfile(path):
            return False, "missing"
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            toks = [str(t) for t in z["token"]]
        except (OSError, KeyError, ValueError):
            return False, "unreadable"
        if len(toks) != chunk["n"] or NC.tokens_sha256(sorted(toks)) != chunk["tokens_sha256"]:
            return False, "search tokens differ from the chunk"
        if meta.get("arm") != "P2HEAD" or meta.get("head_sha256") is not None:                      # @GUARD:C-NOHEAD
            return False, "G-NOHEAD: arm/head in the search meta"
        if meta.get("ckpt_sha256") != self.pins["ckpt_sha256"] or meta.get("config_sha256") != self.pins["config_sha256"]:
            return False, "ckpt/config sha in the search meta"
        if (meta.get("data_clause") or {}).get("carve_sha256") != chunk["tokens_sha256"]:           # @GUARD:C-CLAUSE-IN-META
            return False, "the data clause in the search meta is not this chunk's"
        return True, ""

    def stage_search(self, cid):
        c, chunk = self.cfg, self.plan["chunks"][cid]
        out_dir, merged = c.cdir(cid, "search"), c.cdir(cid, "search_P2HEAD.npz")
        ok, _ = self.merged_valid(cid, merged)
        if ok:
            return merged
        stamp = self.stage_gate(cid)
        n, shard = chunk["n"], 20
        ranges = search_ranges(n, c.procs, shard)
        t0 = time.time()

        def one(r):
            argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "navpipe_launch.py"), "search", *self.common(cid), "--arm", "P2HEAD", "--gate", stamp, "--out-dir", out_dir, "--threads", str(c.threads),
                    "--start", str(r[0]), "--end", str(r[1]), "--shard", str(shard)]
            return self.child(argv, f"search_c{cid:02d}_{r[0]:05d}.log")

        with cf.ThreadPoolExecutor(len(ranges)) as ex:
            rcs = list(ex.map(one, ranges))
        if any(rcs):
            raise Refusal("search", "SEARCH_FAILED", f"chunk {cid} rcs={rcs}")
        rc = self.child([c.py_cpu, "-u", os.path.join(c.code_dir, "navpipe_launch.py"), "merge", "--out-dir", out_dir, "--arm", "P2HEAD", "--out", merged, "--n", str(n)], f"merge_c{cid:02d}.log")
        ok, why = self.merged_valid(cid, merged)
        if rc != 0 or not ok:
            raise Refusal("search", "SEARCH_MERGE_INVALID", f"chunk {cid}: {why}")
        self.state.chunk(cid)["seconds"]["search"] = round(time.time() - t0, 1)
        return merged

    # ------------------------------------------------------------------------------------------------ stage: score (c2: seam -> W3 harness; fixture: label_stepB)
    def scores_valid(self, path, search_path):
        if not (os.path.isfile(path) and os.path.isfile(search_path)):
            return False, "missing"
        try:
            lab, so = np.load(path, allow_pickle=False), np.load(search_path, allow_pickle=False)
            ident = str(lab["identity"])
            toks_ok = [str(t) for t in lab["token"]] == [str(t) for t in so["token"]]
            fin = bool(np.isfinite(lab["pdms"]).all() and np.isfinite(lab["sub"]).all())
            n_ = len(lab["token"])
            pick_ok = bool("pick_sub" in lab.files and "pick_pdms" in lab.files and lab["pick_sub"].shape == (n_, 6) and lab["pick_pdms"].shape == (n_,) and np.isfinite(lab["pick_sub"]).all() and np.isfinite(lab["pick_pdms"]).all())
        except (OSError, KeyError, ValueError):
            return False, "unreadable"
        if self.cfg.mode == "c2" and not pick_ok:                                                   # @GUARD:C-SCORES-PICK
            return False, "DQ6(b): the scores carry no (finite) RE-SCORED picks; a file written before the re-score existed is not accepted"
        m = re.search(r"\{.*\}", ident)
        try:
            j = json.loads(m.group(0)) if m else {}
        except ValueError:
            j = {}
        if not (j.get("passed") is True and j.get("control_ok") is True):                          # @GUARD:C-SCORES-IDENTITY
            return False, "G-T1ID: the planted identity / its control did not pass"
        if not toks_ok or not fin:                                                                  # @GUARD:C-SCORES-ALIGNED
            return False, "score tokens differ from the search tokens, or non-finite scores"
        return True, ""

    def run_harness(self, label, seam_path, tokens_file):
        """one W3 harness call (a child); returns the per-token CSV path.  PASS is read from the harness's OWN counts.json, never inferred from a CSV existing."""
        if self.harness_fn:
            return self.harness_fn(label, seam_path, tokens_file)
        c = self.cfg
        out = c.d("t1")
        last = None
        for att in range(1, 4):
            waited = 0
            while self.ram_fn() < 4.0:                                                              # the harness' own RAM guard (stage2_sft waits at 4000 MB too), capped at 2 h
                time.sleep(60)
                waited += 60
                if waited > 7200:
                    raise Refusal("timeout", "HARNESS_RAM_WAIT", f"{label}: < 4 GiB free for 2 h")
            argv = [c.harness_py, c.harness, "--label", label, "--seam", seam_path, "--tokens", tokens_file, "--out", out]
            rc = self.child(argv, f"harness_{label}.log", env_extra={"PYTHONIOENCODING": "utf-8"}, cwd=os.path.dirname(c.harness))
            cj = os.path.join(out, label, f"{label}.counts.json")
            csvp = os.path.join(out, label, f"{label}.csv")
            st = json.load(open(cj, encoding="utf-8")) if os.path.isfile(cj) else {}
            last = (rc, st.get("status"), st.get("failures"))
            if rc == 0 and st.get("status") == "PASS" and os.path.isfile(csvp):                    # @GUARD:C-HARNESS-PASS
                return csvp
            time.sleep(120)
        raise Refusal("score", "HARNESS_NOT_PASS", f"{label}: rc/status/failures {last}")

    def stage_score_c2(self, cid, toks, merged, scores):
        """T1 scoring of one chunk.  DQ6 (b) (MM_RULING_C2AMEND.md): P2's PICKS are RE-SCORED through the identical route as the search outputs (the pick's `pick_exec` -> `exec_to_poses` -> seam npz -> W3 harness -> CSV), so the paired
        delta's two arms are scored by the same code on the same harness by construction.  The BANKED CSV stays only as a CROSS-CHECK: the re-scored rows must equal it for every token of the chunk, else the chain STOPS with
        ZZNAVPIPE_PICK_RESCORE_MISMATCH.  The pick is scored FIRST, so a route that is not the one P2's 88.82 was measured on stops the chain before the three seed calls are spent."""
        c = self.cfg
        so = {k: v for k, v in np.load(merged, allow_pickle=False).items()}
        if "pick_exec" not in so:
            raise Refusal("score", "PICK_EXEC_MISSING", f"chunk {cid}: the search output carries no `pick_exec` (P2's pick plan, which DQ6 (b) re-scores)")
        banked = getattr(self, "_banked", None) or NC.load_banked_seam(c.banked_seam)
        self._banked = banked
        tl = self.token_doc["token_log"]
        n, S = so["exec"].shape[:2]
        fps = banked["fp"]
        sub = np.zeros((n, S, 6))
        pdms = np.zeros((n, S))
        seam_dir = c.cdir(cid, "t1")
        os.makedirs(seam_dir, exist_ok=True)
        tf = NC.write_tokens_file(os.path.join(seam_dir, "tokens.json"), toks, tl)
        extra = {}
        # ---- DQ6 (b): P2's pick, re-scored through the same route
        pick_poses = NC.exec_to_poses(so["pick_exec"])
        pp = NC.check_pick_plan_is_banked(pick_poses, toks, banked)
        if not pp["ok"]:                                                                            # @GUARD:C-PICKPLAN-STOPS
            self.emit("PICK_PLAN_NOT_BANKED", cid, json.dumps(pp)[:300])
            raise Refusal("score", "G_PICK_PLAN_NOT_BANKED", json.dumps(pp)[:300])
        sp0 = NC.write_seam(os.path.join(seam_dir, "seam_pick.npz"), toks, pick_poses, fps, f"NAVPIPE_c{cid:02d}_pick")
        csv0 = self.run_harness(f"refe_navpipe_c{cid:02d}_pick", sp0, tf)
        psub, ppdms = NC.read_harness_csv(csv0, toks)
        extra["seam_pick"], extra["csv_pick"] = sp0, csv0
        xc = NC.crosscheck_rescored_picks(toks, (psub, ppdms), NC.read_harness_csv(c.banked_csv, toks))
        if not xc["ok"]:                                                                            # @GUARD:C-PICKX-STOPS
            self.emit("PICK_RESCORE_MISMATCH", cid, f"{xc['n_bad']} of {xc['n']} tokens differ from the banked CSV; max |diff| {xc['max_abs_diff']}; e.g. {xc['bad_examples'][:2]}")
            raise Refusal("score", "G_PICK_RESCORE_DIFFERS_FROM_BANKED", json.dumps(xc)[:300])
        self.emit("PICK_RESCORE_OK", cid, f"{xc['n']} tokens equal the banked CSV; max |diff| {xc['max_abs_diff']:.3e}")
        # ---- the search outputs, 3 seeds, the same route
        for s in range(S):
            poses = NC.exec_to_poses(so["exec"][:, s])
            sp = NC.write_seam(os.path.join(seam_dir, f"seam_s{s}.npz"), toks, poses, fps, f"NAVPIPE_c{cid:02d}_s{s}")
            csvp = self.run_harness(f"refe_navpipe_c{cid:02d}_s{s}", sp, tf)
            sub[:, s], pdms[:, s] = NC.read_harness_csv(csvp, toks)
            extra[f"seam_s{s}"], extra[f"csv_s{s}"] = sp, csvp
        # ---- the planted identity: K tokens' re-scored picks (out of the call above) vs P2's banked row, plus the +0.5 m control through the same route
        K = min(c.t1id_k, n)
        order = sorted(range(n), key=lambda i: __import__("hashlib").sha1(toks[i].encode()).hexdigest())[:K]
        idt = [toks[i] for i in sorted(order)]
        ix = {t: i for i, t in enumerate(toks)}
        rows = [ix[t] for t in idt]
        tfk = NC.write_tokens_file(os.path.join(seam_dir, "tokens_id.json"), idt, tl)

        def run_scores(poses):
            sp = NC.write_seam(os.path.join(seam_dir, "seam_ctl.npz"), idt, poses, fps, f"NAVPIPE_c{cid:02d}_ctl")
            return NC.read_harness_csv(self.run_harness(f"refe_navpipe_c{cid:02d}_ctl", sp, tfk), idt)

        banked_rows = NC.read_harness_csv(c.banked_csv, idt)
        rec = NC.t1_planted_identity(idt, pick_poses[rows], (psub[rows], ppdms[rows]), banked_rows, run_scores)
        rec = {**rec, "passed": rec["identity_ok"], "control_ok": rec["control_ok"], "pick_crosscheck": xc, "pick_plan": pp}
        np.savez(scores, sub=sub, pdms=pdms, pick_sub=psub, pick_pdms=ppdms, token=np.array(toks), identity=np.array(json.dumps(rec)), sub_names=np.array(NC.SUBS))
        return rec, extra

    def stage_score_fixture(self, cid, toks, merged, scores):
        c = self.cfg
        if self.ram_fn() < c.min_ram_gb:
            raise Refusal("score", "RAM_LOW", f"{self.ram_fn():.1f} GiB < {c.min_ram_gb}")
        argv = [c.py_label, "-u", c.label_script, "--bundle", c.cdir(cid, "bundle", "bundle.npz"), "--search", merged, "--plan", c.plan_label, "--out", scores]
        rc = self.child(argv, f"label_c{cid:02d}.log", label=True)
        ok, why = self.scores_valid(scores, merged)
        if rc != 0 or not ok:
            raise Refusal("score", "G_LABEL_NOT_PASSED", f"chunk {cid} rc={rc}; {why}")
        rec = json.loads(re.search(r"\{.*\}", str(np.load(scores, allow_pickle=False)["identity"])).group(0))
        return {**rec, "ok": bool(rec.get("passed") and rec.get("control_ok"))}, {}

    # ------------------------------------------------------------------------------------------------ one chunk
    def run_chunk(self, cid, prefetched=None):
        c, st, chunk = self.cfg, self.state, self.plan["chunks"][cid]
        if st.status(cid) == "DELETED":
            return
        t0 = time.time()
        if STATUS_IDX(st.status(cid)) < STATUS_IDX("BUNDLED"):
            toks = prefetched if prefetched is not None else self.stage_bundle(cid)
            idn = self.identity(cid, toks)
            json.dump(idn, open(c.cdir(cid, "identity.json"), "w"), indent=1)
            st.advance(cid, "BUNDLED", identity_ok=bool(idn["ok"]))
            self.emit("BUNDLE_OK", cid, chunk["n"], "identity ok")
        else:
            toks = NC.chunk_tokens(self.token_doc, chunk)
        self.disk(0)
        merged = c.cdir(cid, "search_P2HEAD.npz")
        if STATUS_IDX(st.status(cid)) < STATUS_IDX("SEARCHED"):
            merged = self.stage_search(cid)
            if STATUS_IDX(st.status(cid)) < STATUS_IDX("GATED"):
                st.advance(cid, "GATED", gate_stamp=self.stamp_ok(cid))
            st.advance(cid, "SEARCHED", search_sha256=NC.sha256_file(merged))
            self.emit("SEARCH_DONE", cid)
        self.disk(0)
        scores = c.cdir(cid, "scores.npz")
        extra = {}
        ok, _ = self.scores_valid(scores, merged)
        if ok:
            rec = json.loads(re.search(r"\{.*\}", str(np.load(scores, allow_pickle=False)["identity"])).group(0))
        else:
            t1 = time.time()
            fn = self.stage_score_c2 if c.mode == "c2" else self.stage_score_fixture
            rec, extra = fn(cid, toks, merged, scores)
            st.chunk(cid)["seconds"]["score"] = round(time.time() - t1, 1)
            if not (rec.get("passed") and rec.get("control_ok")):                                     # @GUARD:C-T1ID-STOPS
                raise Refusal("score", "G_T1ID_NOT_PASSED", json.dumps(rec)[:300])
        if STATUS_IDX(st.status(cid)) < STATUS_IDX("SCORED"):
            st.advance(cid, "SCORED", t1id_ok=bool(rec.get("passed") and rec.get("control_ok")), scores_sha256=NC.sha256_file(scores))
            self.emit("SCORE_DONE", cid)
        files = {"bundle": c.cdir(cid, "bundle", "bundle.npz"), "search": merged, "scores": scores, "identity": c.cdir(cid, "identity.json")}
        files.update({k: v for k, v in extra.items() if os.path.isfile(v)})
        if STATUS_IDX(st.status(cid)) < STATUS_IDX("BANKED"):
            NC.bank_chunk(st, cid, files, c.cdir(cid, "bundle", "ctx.json"), c.cdir(cid, "bank_manifest.json"))
            st.advance(cid, "BANKED")
            self.emit("BANKED", cid, NC.sha256_file(c.cdir(cid, "bank_manifest.json"))[:12])
        NC.delete_ctx(st, cid, c.cdir(cid, "bundle", "ctx.npy"))                                    # G-BANK: only after BANKED and re-verified from disk   @GUARD:C-DELETE-CALLED
        st.chunk(cid)["seconds"]["total"] = round(time.time() - t0, 1)
        st.save()
        self.emit("CHUNK_DONE", cid, f"{time.time() - t0:.0f}s", f"free {NC.free_gib(c.work):.1f}GiB")

    # ------------------------------------------------------------------------------------------------ merge + analysis
    def merge_all(self):
        c, n = self.cfg, len(self.plan["chunks"])
        B, S, L = [], [], []
        keep = None
        for cid in range(n):
            b = {k: v for k, v in np.load(c.cdir(cid, "bundle", "bundle.npz"), allow_pickle=False).items()}
            keep = [k for k in b if k not in ("traj_raw",)] if keep is None else keep
            B.append({k: b[k] for k in keep})
            S.append({k: v for k, v in np.load(c.cdir(cid, "search_P2HEAD.npz"), allow_pickle=False).items()})
            L.append({k: v for k, v in np.load(c.cdir(cid, "scores.npz"), allow_pickle=False).items()})
        mb = NC.canonical_merge(B, skip=())
        ms = NC.canonical_merge(S)
        ml = NC.canonical_merge(L)
        N = len(mb["token"])
        if not ([str(t) for t in mb["token"]] == [str(t) for t in ms["token"]] == [str(t) for t in ml["token"]]):
            raise Refusal("analysis", "MERGE_ORDER", "bundle, search and score tokens differ after the canonical merge")
        ms["bundle_index"] = np.arange(N, dtype=np.int64)
        ms["meta"] = np.array(json.dumps({**json.loads(str(S[0]["meta"])), "merged_from_chunks": n, "n_tokens": int(N)}))
        ml["bundle_index"] = ms["bundle_index"]
        fin = c.d("final")
        os.makedirs(fin, exist_ok=True)
        if c.mode == "c2":                                  # T1 labels the PICK only (its RE-SCORED row, DQ6 (b)): the shim bundle has ONE column (the pick's), pick := 0 (memory-light, same arithmetic)
            sub, pdms = ml["pick_sub"], ml["pick_pdms"]                                    # DQ6 (b): the baseline is P2's pick RE-SCORED through the same route as the search outputs, not the banked CSV   @GUARD:C-MERGE-RESCORED
            mb = {"token": mb["token"], "log": mb["log"], "pick_real": np.asarray(mb["pick"], np.int64), "pick": np.zeros(N, np.int64), "sub": sub[:, None, :], "pdms": pdms[:, None]}
            ms["arm_argmax_real"] = np.asarray(ms["arm_argmax"], np.int64)
            ms["arm_argmax"] = np.zeros(N, np.int64)
        np.savez(os.path.join(fin, "bundle_merged.npz"), **mb)
        np.savez(os.path.join(fin, "search_merged.npz"), **ms)
        np.savez(os.path.join(fin, "scores_merged.npz"), **ml)
        return fin

    def analyse(self, fin):
        c = self.cfg
        out = os.path.join(fin, "RESULT_NAVPIPE.json")
        argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "navpipe_analyse.py"), "--bundle", os.path.join(fin, "bundle_merged.npz"), "--search", os.path.join(fin, "search_merged.npz"),
                "--labels", os.path.join(fin, "scores_merged.npz"), "--rule", "c1" if c.mode == "fixture" else "c2", "--out", out]
        if c.mode == "c2":
            bars = {"bar_dpdms_points": float(self.pins["bar_dpdms_points"]), "bar_safety_points": float(self.pins["bar_safety_points"])}
            json.dump(bars, open(os.path.join(fin, "bars.json"), "w"))
            argv += ["--bars-json", os.path.join(fin, "bars.json")]
        rc = self.child(argv, "analysis.log")
        if rc != 0 or not os.path.isfile(out):
            raise Refusal("analysis", "ANALYSIS_FAILED", f"rc={rc}")
        res = json.load(open(out))
        if c.mode == "c2":
            NC.one_run_finish(c.one_run_marker, NC.sha256_file(out))                                # the endpoint now exists: the ONE look is consumed (a second run of any config is refused)   @GUARD:C-ONERUN-FINISH
        self.emit("ANALYSIS_DONE")
        e = res["endpoint_dPDMS_points"]
        head = res.get("outcome", "RULE_MET" if res.get("rule_met") else "RULE_NOT_MET")
        if res.get("negative"):
            head += " NEGATIVE"
        if res.get("safety_miss"):                                                                  # DQ5 (MM_RULING_C2AMEND.md): a safety-family miss is printed IN THE HEADLINE LINE, whatever the outcome   @GUARD:C-HEADLINE-SAFETY
            head += " SAFETY_MISS " + json.dumps(res["safety_miss"], sort_keys=True)
        self.emit("DONE", head)
        self.emit("ENDPOINT", f"dPDMS {e['point']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] points; seed range {res['per_seed_points']['range']:.4f}; points {[round(x, 4) for x in res['per_seed_points']['points']]}")
        for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
            self.emit("FAMILY", fam, json.dumps(res["families"][fam])[:400])
        self.emit("FAMILY", "STRATEGIC", res["STRATEGIC"])
        return res

    def fixture_identity(self, res):
        """FIXTURE: the pipeline's analysis equals Step B's banked P2HEAD read EXACTLY (same floats), and every chunk row of search / labels equals the banked row bit for bit"""
        banked = json.load(open(BANKED_RESULT))["arms"]["P2HEAD"]
        diffs = [k for k in ("endpoint_dPDMS_points", "per_seed_points", "sub_score_deltas_points", "families", "inert", "moved_share") if json.dumps(res[k], sort_keys=True) != json.dumps(banked[k], sort_keys=True)]
        fin = self.cfg.d("final")
        rows = fixture_row_compare(os.path.join(fin, "search_merged.npz"), os.path.join(fin, "scores_merged.npz"), BANKED_SEARCH, BANKED_LABELS)
        ok = bool(not diffs and rows["ok"])
        self.emit("FIXTURE_IDENTICAL" if ok else "FIXTURE_DIFFERS", json.dumps({"analysis_blocks_differing": diffs, "rows": rows}))
        self.write_fixture_record(ok, {"analysis_blocks_differing": diffs, "rows": rows})
        return ok

    def write_fixture_record(self, ok, detail):
        """DQ7 (b): the fixture chain's own RECORD (verdict, the sha256 of every code file it ran, its token list's sha), `<work>/FIXTURE_RECORD.json`.  C2 refuses unless this reads IDENTICAL and its code shas equal the sealed pins.
        A file edited while the fixture ran makes the verdict DIFFERS (the code that ran is not the code on disk now)."""
        c = self.cfg
        detail = dict(detail or {})
        verdict = "DIFFERS"
        if ok:                                                                                      # @GUARD:C-FIXREC-VERDICT
            verdict = "IDENTICAL"
        if verdict == "IDENTICAL" and NC.code_shas(c.code_dir) != self.code_at_start:               # @GUARD:C-FIXREC-CODE-STABLE
            verdict = "DIFFERS"
            detail["code_changed_during_the_run"] = True
        res_p = c.d("final", "RESULT_NAVPIPE.json")
        rec = NC.make_fixture_record(verdict, c.code_dir, NC.tokens_sha256(sorted(self.token_doc["tokens"])), self.plan["plan_sha256"], NC.sha256_file(res_p) if os.path.isfile(res_p) else None, self.pins.get("ckpt_sha256"),
                                     self.pins["config_sha256"], detail)
        path = c.d("FIXTURE_RECORD.json")
        json.dump(rec, open(path, "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
        self.emit("FIXTURE_RECORD", verdict, NC.sha256_file(path))
        return path

    # ------------------------------------------------------------------------------------------------ the run
    def run(self):
        c = self.cfg
        try:
            self.emit("START", c.mode)
            self.code_at_start = NC.code_shas(c.code_dir)
            if c.mode == "c2":
                self.check_c2()
                self.pins = self.seal["pins"]
                if self.one_run == "NEW":                                                           # the marker is written before the first navtest token is touched; a RESUME of this very run keeps its own
                    NC.one_run_start(c.one_run_marker, self.one_run_ident, self.seal["selection"]["config"])
                from make_navtest_bundle import make_plan
                json.dump(make_plan(self.token_doc), open(c.d("navtest_plan.json"), "w"))
                self.make_bridge_sample()
            elif c.mode == "s0bank":
                self.check_s0bank()
                self.pins = self.seal["pins"]
                self.make_bridge_sample()
            else:
                self.check_fixture()
                import hashlib
                self.pins = {"ckpt_sha256": NC.sha256_file(c.ckpt) if c.hash_ckpt else "x", "config_sha256": NC.sha_lf(c.config), "head_s0_sha256": NC.sha256_file(c.head_s0)}
            self.disk(0)
            self.emit("DISK_OK", f"{NC.free_gib(c.work):.1f}GiB")
            self.state = NC.State(c.d("navpipe_state.json"), self.plan["plan_sha256"])
            self.state.save()
            ids = list(range(len(self.plan["chunks"])))
            if c.max_chunks:
                ids = ids[:c.max_chunks]
            pre = None
            with cf.ThreadPoolExecutor(1) as ex:
                for k, cid in enumerate(ids):
                    pre_toks = None
                    if pre is not None:
                        pre_toks = pre.result()
                        pre = None
                    if c.prefetch and k + 1 < len(ids) and self.state.status(ids[k + 1]) == "NEW" and c.mode in ("c2", "s0bank"):
                        try:
                            self.disk(self.plan["chunks"][ids[k + 1]]["n"] * NC.CTX_BYTES_PER_TOKEN + self.plan["chunks"][cid]["n"] * NC.CTX_BYTES_PER_TOKEN)
                            pre = ex.submit(self.stage_bundle, ids[k + 1])
                        except Refusal:
                            pre = None                                                   # no room for a second ctx: run sequentially
                    self.run_chunk(cid, pre_toks if pre_toks is not None else None)
            if c.max_chunks and len(ids) < len(self.plan["chunks"]):
                self.emit("PARTIAL", f"{len(ids)} of {len(self.plan['chunks'])} chunks; no analysis")
                if c.mode == "fixture":
                    self.partial_row_compare(ids)
                return 0
            fin = self.merge_all()
            if c.mode == "s0bank":
                self.finish_s0bank(fin)
                return 0
            res = self.analyse(fin)
            if c.mode == "fixture":
                self.fixture_identity(res)
            else:
                self.emit("LICENCE_BASIS", "C1's licence (MM_RULING_C2_LICENCE.md), not EP-ADMISSION; the look read the T1 endpoint only")
            return 0
        except Refusal as e:
            self.emit("STOP", e.reason, e.detail)
            return e.code
        except Exception as e:  # a crash is a stop, not a pass
            self.emit("STOP", "CRASH", f"{type(e).__name__}: {e}")
            return 9

    def finish_s0bank(self, fin):
        """no analysis here: EP-ADMISSION is READ downstream, on this bank, by `epadm_searchbank.py build` under its own sealed amendment.  Write the manifest of the triple that adapter reads."""
        files = {n: os.path.join(fin, n) for n in ("bundle_merged.npz", "search_merged.npz", "scores_merged.npz")}
        man = {"mode": "s0bank", "plan_sha256": self.plan["plan_sha256"], "n_tokens": self.plan["n_tokens"], "n_logs": self.plan["n_logs"], "amendment_sha256": self.seal["amendment_sha256"],
               "files": {k: {"path": v, "sha256": NC.sha256_file(v)} for k, v in files.items()},
               "next": "python epadm_searchbank.py build --bundle bundle_merged.npz --search search_merged.npz --labels scores_merged.npz --out-dir <bank dir>   (the adapter's own sealed amendment)",
               "stamps": "label-only PDM route (NOT T1, NOT navtest); S0 in-sample tokens; one training seed; 3 search seeds; STRATEGIC PAUSED"}
        json.dump(man, open(os.path.join(fin, "S0BANK_MANIFEST.json"), "w"), indent=1)
        self.emit("S0BANK_MERGED", self.plan["n_tokens"], "tokens", NC.sha256_file(files["search_merged.npz"])[:12])
        self.emit("ANALYSIS_NOT_RUN", "EP-ADMISSION is read downstream by the bank adapter under its own sealed amendment")

    def make_bridge_sample(self):
        """the 3 bridge tokens: the first 3 of chunk 0's sorted tokens (deterministic, pinned by the plan)"""
        toks = NC.chunk_tokens(self.token_doc, self.plan["chunks"][0])[:3]
        json.dump({"n": 3, "tokens": toks}, open(self.cfg.d("bridge3_sample.json"), "w"))

    def partial_row_compare(self, ids):
        """FIXTURE, first chunks only: each finished chunk's search rows and labels equal the banked rows bit for bit (the chunked run is the unchunked run)"""
        c = self.cfg
        tot = {"ok": True, "chunks": []}
        for cid in ids:
            r = fixture_row_compare(c.cdir(cid, "search_P2HEAD.npz"), c.cdir(cid, "scores.npz"), BANKED_SEARCH, BANKED_LABELS)
            tot["ok"] &= r["ok"]
            tot["chunks"].append({"chunk": cid, **r})
        self.emit("FIXTURE_ROWS_IDENTICAL" if tot["ok"] else "FIXTURE_ROWS_DIFFER", json.dumps(tot)[:900])


def search_ranges(n: int, procs: int, shard: int = 20) -> list:
    """the token ranges of the concurrent search processes; every boundary on a shard multiple so the processes write DIFFERENT shard files (shard_P2HEAD_<lo>.npz)"""
    mid = min(n, int(math.ceil(n / max(procs, 1) / shard)) * shard)
    return [(0, n)] if procs <= 1 or mid >= n else [(0, mid), (mid, n)]


def STATUS_IDX(s):
    return NC.STATUS_ORDER.index(s)


def fixture_row_compare(search_path, labels_path, banked_search, banked_labels) -> dict:
    """every token of `search_path` / `labels_path` against the BANKED Step B rows (looked up by token): each array equal BYTE FOR BYTE except the wall-clock fields; labels `sub` / `pdms` equal exactly"""
    so, bs = np.load(search_path, allow_pickle=False), np.load(banked_search, allow_pickle=False)
    lb, bl = np.load(labels_path, allow_pickle=False), np.load(banked_labels, allow_pickle=False)
    bi = {str(t): i for i, t in enumerate(bs["token"])}
    li = {str(t): i for i, t in enumerate(bl["token"])}
    rows = [bi[str(t)] for t in so["token"]]
    lrows = [li[str(t)] for t in lb["token"]]
    bad = {}
    for k in so.files:
        if k in TIME_FIELDS or k in ("meta", "seeds", "bundle_index", "token"):
            continue
        a = np.asarray(so[k])
        b = np.asarray(bs[k])[rows] if np.asarray(bs[k]).ndim >= 1 and np.asarray(bs[k]).shape[0] == len(bi) else np.asarray(bs[k])
        if a.tobytes() != b.tobytes():
            bad[k] = int(np.sum(np.any((a != b).reshape(len(a), -1), 1))) if a.shape == b.shape else "shape"
    for k in ("sub", "pdms"):
        if np.asarray(lb[k]).tobytes() != np.asarray(bl[k])[lrows].tobytes():
            bad["labels." + k] = "differs"
    return {"ok": not bad, "n_tokens": len(rows), "arrays_differing": bad}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("navtest", "c2", "s0bank", "fixture"), help="navtest (alias c2): the T1 route on the navtest token file; s0bank: the label-only route on --tokens; fixture: carve B")
    ap.add_argument("--tokens", default=None, help="the pinned token-list json ({'tokens': [...], 'token_log': {...}}); default: the navtest file for navtest, required for s0bank")
    ap.add_argument("--reference-path", default=None, help="s0bank: the directory of P2's own `*.fan.npz` files (or one file)")
    ap.add_argument("--table-ref", default=None, help="navtest, optional: P2's banked 64-proposal table (token/proposals/logits/pick are read; the label arrays never)")
    ap.add_argument("--fan-labels", default=None, help="s0bank: the pinned npz with P2's labelled fan (token, sub, pdms, valid)")
    ap.add_argument("--plan-label", default=None, help="s0bank: the pinned plan json (token -> log / DB / frames) used by the GPU builder and the label route")
    ap.add_argument("--expect-tokens", default=None, help="navtest only: 'N,L' (default 12146,136)")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--amendment", default=None)
    ap.add_argument("--amendment-sha256", default=None)
    ap.add_argument("--c1-result", default=None, help="navtest: C1's RESULT_SPC.json, with its chain_state.json beside it (the C1 licence, MM_RULING_C2_LICENCE.md)")
    ap.add_argument("--c1-result-sha256", default=None, help="navtest: the sha256 of that RESULT_SPC.json, typed by the operator from the register (the pin)")
    ap.add_argument("--config", default=None, help="navtest: the search config file, stepB_config.json (C8) or stepB_config_O4.json (O4); it must be the one `c2_config(c1_read, dose_read)` returns")
    ap.add_argument("--dose-result", default=None, help="navtest: the dose arm's RESULT_DOSE.json (with its chain_state.json beside it); needed for either config, because c2_config takes both reads")
    ap.add_argument("--dose-result-sha256", default=None, help="navtest: the sha256 of that RESULT_DOSE.json, typed by the operator from the register")
    ap.add_argument("--fixture-record", default=None, help="navtest: the fixture chain's FIXTURE_RECORD.json (verdict IDENTICAL, code shas == the sealed pins); the amendment pins its sha256")
    ap.add_argument("--procs", type=int, default=1, choices=(1, 2))
    ap.add_argument("--prefetch", type=int, default=0, choices=(0, 1))
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--min-ram-gb", type=float, default=3.0)
    ap.add_argument("--min-free-gib", type=float, default=NC.MIN_FREE_GIB, help="the disk guard's floor (6 GiB for the C2 run; the fixture may run on another volume with a lower floor)")
    ap.add_argument("--dry-run", action="store_true", help="seal + data + disk checks only; no child process")
    a = ap.parse_args(argv)
    mode = "c2" if a.mode in ("navtest", "c2") else a.mode
    work = a.work or (WORK_C2 if mode == "c2" else None)
    if not work:
        raise SystemExit("--work is required in fixture and s0bank mode")
    if mode == "s0bank" and not a.tokens:
        raise SystemExit("--tokens is required in s0bank mode")
    kw = {}
    if a.tokens:
        kw["token_file"] = a.tokens
    if mode == "s0bank":
        kw.update(reference_path=a.reference_path, fan_labels=a.fan_labels, plan_label=a.plan_label)
    if a.table_ref:
        kw["table_ref"] = a.table_ref
    if a.expect_tokens:
        kw["expect_tokens"] = tuple(int(x) for x in a.expect_tokens.split(","))
    if mode == "c2":
        kw.update(config=a.config, dose_result=a.dose_result, dose_result_sha256=a.dose_result_sha256, fixture_record=a.fixture_record)       # no default config: C8 or O4 is named, and checked against c2_config
    cfg = Cfg(work, mode=mode, amendment=a.amendment, amendment_sha256=a.amendment_sha256, c1_result=a.c1_result, c1_result_sha256=a.c1_result_sha256, cap=a.cap or (110 if mode == "fixture" else 1000), procs=a.procs,
              prefetch=a.prefetch, max_chunks=a.max_chunks, min_ram_gb=a.min_ram_gb, min_free_gib=a.min_free_gib, **kw)
    ch = Chain(cfg)
    if a.dry_run:
        try:
            ch.emit("START", mode)
            {"c2": ch.check_c2, "s0bank": ch.check_s0bank, "fixture": ch.check_fixture}[mode]()
            ch.disk(0)
            ch.emit("DRYRUN_OK")
            return 0
        except Refusal as e:
            ch.emit("STOP", e.reason, e.detail)
            return e.code
    return ch.run()


if __name__ == "__main__":
    sys.exit(main())
