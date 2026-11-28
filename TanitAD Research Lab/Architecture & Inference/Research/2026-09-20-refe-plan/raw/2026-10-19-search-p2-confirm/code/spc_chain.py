#!/usr/bin/env python3
"""SEARCH-P2-CONFIRM chain (NOT LAUNCHED by the build).  ONE detached process; REFUSES (exit 5) without the seal.

    seal guard -> data clause -> disk -> [GPU stage under the dev-box lock: bridge, C-ID bundle, B-REST bundle] -> bundle checks -> G-REACH stamps (CID, C1) -> C-ID (CPU search on 16 carve-B tokens, bit for bit)
    -> search P2HEAD x 3 seeds on B-REST -> merge -> labels (+ planted identity) -> READ C1 analysis -> ZZSPC_DONE <outcome>

Every CPU child runs with CUDA_VISIBLE_DEVICES=-1.  The GPU stage runs ONLY through L1's `launch_stepA_ctype.py run` (it waits for `chain_active`, >= 8 GiB RAM and a free GPU, and never touches another owner's lock).
Resumable: a stage whose output already validates is skipped; an output that does not validate after its child ran is a STOP, never a pass.  The EP-ADMISSION gate is NEVER run here (DQ6): the chain banks its inputs only.
Markers ZZSPC_* start their line.  Exit: 0 done, 3 gate, 4 timeout, 5 seal / data refusal, 6 search, 7 label / RAM, 8 analysis, 9 crash.
    python spc_chain.py --pkg <package dir holding the sealed text> [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPB_CODE = os.environ.get("SPC_STEPB_CODE", "C:/Users/Admin/par18/l2stepb/code")
L1_CODE = os.environ.get("L1_CODE", "C:/Users/Admin/par18/l1run/code")
for _p in (HERE, STEPB_CODE, L1_CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import spc_launch as SL  # noqa: E402
import spc_cid as CID  # noqa: E402
import b_rest as BR  # noqa: E402

SEALED_NAME = "PREREG_SEARCH_P2_CONFIRM.md"
OPERATIVE = ("spc_launch.py", "spc_chain.py", "spc_gpu_stage.py", "spc_cid.py", "spc_analyse.py", "b_rest.py")
STEPB_OPERATIVE = ("stepB.py", "l2_scorer.py", "g_reach_stepb.py", "label_stepB.py", "analyse_stepB.py", "amend1_concordance.py", "stepB_chain.py")
REQUIRED_PINS = ("b_rest_tokens_sha256", "carve_b_tokens_sha256", "step_a_tokens_sha256", "ckpt_sha256", "config_sha256", "head_s0_sha256", "bridge_bundle_script_sha256_lf")
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
PY_CPU = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_GPU = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
PY_LABEL = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
REFE_DIR = "C:/Users/Admin/par18/l1run/tree_hybrid/refe"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
HEAD_S0 = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads/final_head_s0.npz"
CONFIG = "C:/Users/Admin/par18/l2stepb/raw/stepB_config.json"
LAUNCHER = os.path.join(L1_CODE, "launch_stepA_ctype.py")
WORK = "D:/Projects/TanitAD-artifacts/spc"
BANKED_SEARCH = "C:/Users/Admin/par18/l2stepb/run/search_P2HEAD.npz"
BANKED_BUNDLE_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint/bundle"


class StopChain(Exception):
    def __init__(self, code, reason, detail=""):
        super().__init__(f"{reason} {detail}")
        self.code, self.reason, self.detail = code, reason, detail


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def parse_pins(text):
    if "<!-- PINS-BEGIN -->" not in text or "<!-- PINS-END -->" not in text:
        raise StopChain(5, "SEAL_NO_PINS")
    blk = text.split("<!-- PINS-BEGIN -->")[1].split("<!-- PINS-END -->")[0]
    pins = {}
    for line in blk.splitlines():
        line = line.strip()
        if not line or line.startswith("`") or line.startswith("#"):
            continue
        k, _, v = line.partition(" ")
        pins[k] = v.strip()
    return pins


class Cfg:
    def __init__(self, pkg, work=WORK, code_dir=HERE, stepb_code=STEPB_CODE, l1_code=L1_CODE, py_cpu=PY_CPU, py_gpu=PY_GPU, py_label=PY_LABEL, ckpt=CKPT, refe_dir=REFE_DIR, plan=PLAN,
                 head_s0=HEAD_S0, config=CONFIG, launcher=LAUNCHER, banked_search=BANKED_SEARCH, banked_bundle_dir=BANKED_BUNDLE_DIR, min_free_gib=12.0, min_ram_gb=8.0, hash_ckpt=True,
                 stepa_sample=None, carveb_sample=None):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})
        self.pkg = os.path.abspath(pkg)
        self.prereg = os.path.join(self.pkg, SEALED_NAME)
        self.sha_file = os.path.join(self.pkg, "PREREG_SEARCH_P2_CONFIRM.sha256")
        self.b_rest_json = os.path.join(self.pkg, "raw", "b_rest.json")
        self.cid_json = os.path.join(self.pkg, "raw", "cid_sample.json")
        self.stepa_sample = stepa_sample or f"{BR.W}/sample.json"
        self.carveb_sample = carveb_sample or BR.CARVE_B

    def d(self, *p):
        return os.path.join(self.work, *p)


def child_env(label=False):
    import stepB_chain as CH
    return CH.child_env(label)


def real_runner(argv, env, log_path, cwd=None):
    import stepB_chain as CH
    return CH.real_runner(argv, env, log_path, cwd)


class Chain:
    def __init__(self, cfg, runner=real_runner, out=None, ram_fn=None, free_fn=None):
        self.cfg, self.run_child_fn, self.out = cfg, runner, out or sys.stdout
        self.ram_fn = ram_fn or (lambda: __import__("stepB_chain").avail_gb())
        self.free_fn = free_fn or (lambda p: __import__("shutil").disk_usage(p).free / 2 ** 30)
        self.st = {"stage": "NEW", "done": []}

    # ------------------------------------------------------------------------------------------------ plumbing
    def emit(self, marker, *parts):
        print(("ZZSPC_" + marker + (" " + " ".join(str(p) for p in parts) if parts else "")), file=self.out, flush=True)

    def stop(self, code, reason, detail=""):
        raise StopChain(code, reason, detail)

    def state(self, stage, **kw):
        self.st["stage"] = stage
        if stage not in self.st["done"] and not stage.endswith("_RUNNING"):
            self.st["done"].append(stage)
        self.st.update(kw)
        os.makedirs(self.cfg.work, exist_ok=True)
        json.dump(self.st, open(self.cfg.d("chain_state.json"), "w"), indent=1)

    def child(self, argv, log_name, label=False):
        env = dict(child_env(label))
        env["CUDA_VISIBLE_DEVICES"] = "-1"
        os.makedirs(self.cfg.d("logs"), exist_ok=True)
        return self.run_child_fn(argv, env, self.cfg.d("logs", log_name))

    # ------------------------------------------------------------------------------------------------ G-SEAL
    def check_seal(self):
        c = self.cfg
        if not (os.path.isfile(c.prereg) and os.path.isfile(c.sha_file)):
            self.stop(5, "SEAL_MISSING", f"{c.prereg} / {c.sha_file}")
        text = open(c.prereg, "rb").read().replace(b"\r\n", b"\n")
        want = open(c.sha_file, encoding="utf-8").read().split()[0].strip().lower()
        if hashlib.sha256(text).hexdigest() != want:
            self.stop(5, "SEAL_SHA_MISMATCH", "the text differs from its .sha256")
        t = text.decode("utf-8")
        m = re.search(r"^Status: .*$", t, re.M)                      # the FIRST Status line is the document's status (the body may quote the words elsewhere)
        if not m or not m.group(0).startswith("Status: **SEALED"):
            self.stop(5, "SEAL_NOT_SEALED", f"first status line: {(m.group(0)[:40] if m else None)!r}")
        pins = parse_pins(t)
        miss = [k for k in REQUIRED_PINS if k not in pins]
        if miss:
            self.stop(5, "SEAL_PIN_MISSING", ",".join(miss))
        bad = []
        if pins["b_rest_tokens_sha256"] != SL.B_REST_SHA256:
            bad.append("b_rest_tokens_sha256 vs spc_launch.B_REST_SHA256")
        if pins["config_sha256"] != sha_lf(c.config):
            bad.append("config_sha256")
        if c.hash_ckpt and pins["ckpt_sha256"] != sha256_file(c.ckpt):
            bad.append("ckpt_sha256")
        if pins["head_s0_sha256"] != sha256_file(c.head_s0):
            bad.append("head_s0_sha256")
        if pins["bridge_bundle_script_sha256_lf"] != sha_lf(os.path.join(c.l1_code, "make_stepA_bundle.py")):
            bad.append("bridge_bundle_script_sha256_lf")
        for k, v in pins.items():
            if k.startswith("code:"):
                f = k[5:]
                p = next((q for q in (os.path.join(c.code_dir, f), os.path.join(c.stepb_code, f)) if os.path.isfile(q)), None)
                if p is None or sha_lf(p) != v:
                    bad.append(k)
            elif k.startswith("l1:"):
                p = os.path.join(c.l1_code, k[3:])
                if not os.path.isfile(p) or sha_lf(p) != v:
                    bad.append(k)
            elif k.startswith("raw:"):
                p = os.path.join(c.pkg, "raw", k[4:])
                if not os.path.isfile(p) or sha_lf(p) != v:
                    bad.append(k)
        for f in OPERATIVE + STEPB_OPERATIVE:
            if f"code:{f}" not in pins:
                bad.append(f"operative file not pinned: {f}")
        if bad:
            self.stop(5, "SEAL_PIN_MISMATCH", ";".join(bad))
        self.emit("SEAL_OK", want[:12])
        self.state("SEAL_OK", prereg_sha256=want)
        return pins

    # ------------------------------------------------------------------------------------------------ G-DATA
    def check_data(self, pins):
        c = self.cfg
        try:
            plan, sA, cb = json.load(open(c.plan)), json.load(open(c.stepa_sample)), json.load(open(c.carveb_sample, encoding="utf-8"))
            rest, _ = BR.b_rest(plan["tokens"], plan["token_log"], sA["tokens"], cb["tokens"])
            rec = json.load(open(c.b_rest_json, encoding="utf-8"))
            cid = json.load(open(c.cid_json, encoding="utf-8"))
        except (AssertionError, OSError, KeyError, ValueError) as e:
            self.stop(5, "DATA_REFUSED", str(e))
        probs = []
        if rec["tokens"] != rest or BR.sha_tokens(rest) != pins["b_rest_tokens_sha256"]:       # ONE check: the file equals the recomputation AND the recomputation hashes to the pin
            probs.append("raw/b_rest.json is not the recomputed B-REST list, or the recomputation does not hash to the pinned sha")
        if len(rest) != 905 or len(set(plan["token_log"][t] for t in rest)) != 24:
            probs.append(f"B-REST has {len(rest)} tokens / {len(set(plan['token_log'][t] for t in rest))} logs, not 905 / 24")
        if BR.sha_tokens(cb["tokens"]) != pins["carve_b_tokens_sha256"] or BR.sha_tokens(sA["tokens"]) != pins["step_a_tokens_sha256"]:
            probs.append("Step A / carve B token lists do not hash to their pins")
        if BR.sha_tokens(cid["tokens"]) != SL.CID_SHA256 or len(cid["tokens"]) != 16 or not set(cid["tokens"]) <= set(cb["tokens"]):      # the pinned sha implies the subset; the subset is kept as the stated reason
            probs.append("the C-ID sample is not the pinned list of 16 carve-B tokens")
        if probs:
            self.stop(5, "DATA_REFUSED", "; ".join(probs))
        self.emit("DATA_OK", len(rest), "tokens", len(set(rec["logs"])), "logs", pins["b_rest_tokens_sha256"][:12])
        self.state("DATA_OK")

    def check_disk(self):
        os.makedirs(self.cfg.work, exist_ok=True)
        free = self.free_fn(self.cfg.work)
        if free < self.cfg.min_free_gib:
            self.stop(7, "DISK_LOW", f"{free:.1f} GiB free on the work volume < {self.cfg.min_free_gib}")
        self.emit("DISK_OK", f"{free:.1f}GiB")

    # ------------------------------------------------------------------------------------------------ GPU stage + G-BUNDLE
    def bundle_dir(self, which):
        return self.cfg.d("B_REST" if which == "C1" else "cid", "bundle")

    def bundle_valid(self, which, expect_n):
        """-> (ok, why).  Token list of bundle.npz and ctx.json must hash to the pinned read sha; ctx.npy must hash to ctx.json's sha."""
        import numpy as np
        d = self.bundle_dir(which)
        want = SL.READS[which]
        try:
            toks = [str(t) for t in np.load(os.path.join(d, "bundle.npz"), allow_pickle=False)["token"]]
            cj = json.load(open(os.path.join(d, "ctx.json")))
        except (OSError, KeyError, ValueError) as e:
            return False, f"unreadable ({type(e).__name__})"
        if len(toks) != expect_n or BR.sha_tokens(toks) != want:
            return False, f"bundle tokens: n={len(toks)}, sha {BR.sha_tokens(toks)[:12]} != {want[:12]}"
        if cj.get("tokens_sha256") != want or cj.get("n") != expect_n:
            return False, "ctx.json tokens_sha256 / n differ from the pinned read"
        p = os.path.join(d, cj.get("ctx_file", "ctx.npy"))
        if not os.path.isfile(p) or sha256_file(p) != cj.get("ctx_sha256"):
            return False, "ctx.npy missing or its sha256 differs from ctx.json"
        return True, ""

    def gpu_stage(self):
        c = self.cfg
        log = c.d("logs", "gpu_stage.log")
        ok16, _ = self.bundle_valid("CID", 16)
        ok905, _ = self.bundle_valid("C1", 905)
        if not (ok16 and ok905):
            argv = [c.py_cpu, "-u", c.launcher, "run", "--max-wait", "21600", "--", c.py_gpu, "-u", os.path.join(c.code_dir, "spc_gpu_stage.py"), "--work", c.work, "--cid-sample", c.cid_json, "--brest-sample", c.b_rest_json]
            env_unset_cuda = True
            self.emit("GPU_STAGE_START", "(waits for the dev-box lock; does not take or touch a foreign holder's)")
            env = dict(child_env(False))
            env.pop("CUDA_VISIBLE_DEVICES", None)
            env["L1_TREE"] = "C:/Users/Admin/par18/l1run/tree_hybrid"
            os.makedirs(c.d("logs"), exist_ok=True)
            rc = self.run_child_fn(argv, env, log)
            if rc != 0:
                self.stop(3, "G_GPU_STAGE_FAILED", f"rc={rc}; see {log}")
        text = open(log, encoding="utf-8", errors="replace").read() if os.path.isfile(log) else ""
        if re.search(r"^ZZSTAGE_FAIL", text, re.M):
            self.stop(3, "G_GPU_STAGE_FAILED", "ZZSTAGE_FAIL in the stage log")
        for name, pat in (("G-BRIDGE", r"^ZZBRIDGE_PASS"), ("G-BUNDLE(16)", r"^\s*ZZBUNDLE_OK 16(\s|$)"), ("G-BUNDLE(905)", r"^\s*ZZBUNDLE_OK 905(\s|$)")):
            if not re.search(pat, text, re.M):
                self.stop(3, "G_BUNDLE_NOT_PASSED", f"{name}: marker not found in {log}")
        for which, n in (("CID", 16), ("C1", 905)):
            ok, why = self.bundle_valid(which, n)
            if not ok:
                self.stop(3, "G_BUNDLE_INVALID", f"{which}: {why}")
        self.emit("GPU_STAGE_OK")
        self.state("GPU_STAGE_OK")

    # ------------------------------------------------------------------------------------------------ G-REACH
    def common(self, which, pins):
        c = self.cfg
        d = self.bundle_dir(which)
        return ["--bundle", os.path.join(d, "bundle.npz"), "--ctx", os.path.join(d, "ctx.npy"), "--ckpt", c.ckpt, "--ckpt-sha256", pins["ckpt_sha256"], "--refe-dir", c.refe_dir,
                "--config", c.config, "--config-sha256", pins["config_sha256"]]

    def stamp_for(self, which):
        g = glob.glob(self.cfg.d(f"gate_{which}", "G-REACH-STEPB_PASS_*.json"))
        return g[0] if g else None

    def stamp_ok(self, which, pins):
        p = self.stamp_for(which)
        if not p:
            return False
        try:
            st = json.load(open(p))
        except (OSError, ValueError):
            return False
        return st.get("schema") == "tanitad.g_reach_stepb/1" and st.get("verdict") == "PASS" and st.get("bundle_sha256") == sha256_file(os.path.join(self.bundle_dir(which), "bundle.npz")) \
            and st.get("ckpt_sha256") == pins["ckpt_sha256"] and st.get("config_sha256") == pins["config_sha256"]

    def mint_gates(self, pins):
        c = self.cfg
        for which in ("CID", "C1"):
            if not self.stamp_ok(which, pins):
                argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "spc_launch.py"), "gate", "--read", which, "--head", c.head_s0, "--head-sha256", pins["head_s0_sha256"], *self.common(which, pins)[0:], "--out-dir", c.d(f"gate_{which}")]
                rc = self.child(argv, f"gate_{which}.log")
                if rc != 0 or not self.stamp_ok(which, pins):
                    self.stop(3, "G_REACH_NOT_PASSED", f"{which}: rc={rc}")
            self.emit("REACH_PASS", which)
        self.state("REACH_PASS")

    # ------------------------------------------------------------------------------------------------ search / C-ID
    def search_args(self, which, pins, out_dir):
        c = self.cfg
        return [c.py_cpu, "-u", os.path.join(c.code_dir, "spc_launch.py"), "search", "--read", which, "--arm", "P2HEAD", *self.common(which, pins), "--gate", self.stamp_for(which), "--out-dir", out_dir, "--threads", "4"]

    def merged_valid(self, path, which, n, pins):
        import numpy as np
        if not os.path.isfile(path):
            return False, "missing"
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
            toks = [str(t) for t in z["token"]]
        except (OSError, KeyError, ValueError):
            return False, "unreadable"
        if len(toks) != n or BR.sha_tokens(sorted(toks)) != SL.READS[which]:
            return False, "tokens differ from the pinned read"
        if meta.get("arm") != "P2HEAD" or meta.get("head_sha256") is not None:
            return False, "G-NOHEAD: arm/head in the search meta"
        if meta.get("ckpt_sha256") != pins["ckpt_sha256"] or meta.get("config_sha256") != pins["config_sha256"]:
            return False, "ckpt/config sha in the search meta"
        if (meta.get("data_clause") or {}).get("carve_sha256") != SL.READS[which]:
            return False, "data clause in the search meta"
        return True, ""

    def run_search(self, which, pins, n):
        c = self.cfg
        out_dir = c.d(f"search_{which}")
        merged = c.d(f"search_{which}_P2HEAD.npz")
        ok, _ = self.merged_valid(merged, which, n, pins)
        if not ok:
            rc = self.child(self.search_args(which, pins, out_dir), f"search_{which}.log")
            if rc != 0:
                self.stop(6, "SEARCH_FAILED", f"{which} rc={rc}")
            rc = self.child([c.py_cpu, "-u", os.path.join(c.code_dir, "spc_launch.py"), "merge", "--out-dir", out_dir, "--arm", "P2HEAD", "--out", merged, "--n", str(n)], f"merge_{which}.log")
            ok, why = self.merged_valid(merged, which, n, pins)
            if rc != 0 or not ok:
                self.stop(6, "SEARCH_MERGE_INVALID", f"{which}: {why}")
        return merged

    def cid_gate(self, pins):
        import numpy as np
        c = self.cfg
        merged = self.run_search("CID", pins, 16)
        ld = lambda p: {k: v for k, v in np.load(p, allow_pickle=False).items()}   # noqa: E731
        d_new, d_old = self.bundle_dir("CID"), c.banked_bundle_dir
        res = CID.check(ld(c.banked_search), ld(merged), ld(os.path.join(d_old, "bundle.npz")), ld(os.path.join(d_new, "bundle.npz")), np.load(os.path.join(d_old, "ctx.npy"), mmap_mode="r"),
                        np.load(os.path.join(d_new, "ctx.npy"), mmap_mode="r"))
        json.dump(res, open(c.d("cid.json"), "w"), indent=1)
        if not res["pass"]:
            self.stop(3, "G_CID_NOT_PASSED", "; ".join(res["failures"])[:400])
        self.emit("CID_PASS", res["n_tokens"], "tokens", res["non_vacuity"])
        self.state("CID_PASS")

    # ------------------------------------------------------------------------------------------------ labels + G-LABEL
    def labels_valid(self, path, search_path):
        import numpy as np
        if not (os.path.isfile(path) and os.path.isfile(search_path)):
            return False, "missing"
        try:
            lab, so = np.load(path, allow_pickle=False), np.load(search_path, allow_pickle=False)
            ident = str(lab["identity"])
            toks_ok = [str(t) for t in lab["token"]] == [str(t) for t in so["token"]]
            fin = bool(np.isfinite(lab["pdms"]).all() and np.isfinite(lab["sub"]).all())
        except (OSError, KeyError, ValueError):
            return False, "unreadable"
        m = re.search(r"\{.*\}", ident)
        try:
            j = json.loads(m.group(0)) if m else {}
        except ValueError:
            j = {}
        if not (j.get("passed") is True and j.get("control_ok") is True):
            return False, "G-LABEL: the planted identity / its control did not pass"
        if not toks_ok or not fin:
            return False, "label tokens differ from the search tokens, or non-finite labels"
        return True, ""

    def run_labels(self, merged):
        c = self.cfg
        labels = c.d("labels_C1_P2HEAD.npz")
        ok, _ = self.labels_valid(labels, merged)
        if not ok:
            if self.ram_fn() < c.min_ram_gb:
                self.stop(7, "RAM_LOW", f"{self.ram_fn():.1f} GiB < {c.min_ram_gb}")
            argv = [c.py_label, "-u", os.path.join(c.stepb_code, "label_stepB.py"), "--bundle", os.path.join(self.bundle_dir("C1"), "bundle.npz"), "--search", merged, "--plan", c.plan, "--out", labels]
            rc = self.child(argv, "label_C1.log", label=True)
            ok, why = self.labels_valid(labels, merged)
            if rc != 0 or not ok:
                self.stop(7, "G_LABEL_NOT_PASSED", f"rc={rc}; {why}")
        self.emit("LABEL_DONE")
        self.state("LABEL_DONE")
        return labels

    # ------------------------------------------------------------------------------------------------ analysis + G-ORDER
    def run_analysis(self, merged, labels, pins):
        c = self.cfg
        need = ["SEAL_OK", "DATA_OK", "GPU_STAGE_OK", "REACH_PASS", "CID_PASS", "LABEL_DONE"]
        miss = [s for s in need if s not in self.st["done"]]
        if miss:
            self.stop(8, "G_ORDER", f"analysis refused: stages not done: {miss}")
        out = c.d("RESULT_SPC.json")
        rc = self.child([c.py_cpu, "-u", os.path.join(c.code_dir, "spc_analyse.py"), "--bundle", os.path.join(self.bundle_dir("C1"), "bundle.npz"), "--search", merged, "--labels", labels, "--out", out], "analysis.log")
        if rc != 0 or not os.path.isfile(out):
            self.stop(8, "ANALYSIS_FAILED", f"rc={rc}")
        res = json.load(open(out))
        self.emit("ANALYSIS_DONE")
        self.emit("DONE", res["outcome"])
        e = res["endpoint_dPDMS_points"]
        self.emit("ENDPOINT", f"dPDMS {e['point']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}] points; seed range {res['per_seed_points']['range']:.4f}; safety {json.dumps(res['safety_points'])}")
        for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL"):
            self.emit("FAMILY", fam, json.dumps(res["families"][fam])[:600])
        self.emit("FAMILY", "STRATEGIC", res["STRATEGIC"])
        self.emit("NEXT", res["next_arm"])
        self.emit("ADMISSION_NOT_RUN", "EP-ADMISSION is never run by this chain (DQ6: bank geometry); the merged search file and the labels are the bank's inputs")
        self.emit("STAMPS", res["stamps"])

    def run(self):
        try:
            self.emit("START")
            pins = self.check_seal()
            self.check_data(pins)
            self.check_disk()
            self.gpu_stage()
            self.mint_gates(pins)
            self.cid_gate(pins)
            merged = self.run_search("C1", pins, 905)
            self.emit("SEARCH_DONE")
            self.state("SEARCH_DONE")
            labels = self.run_labels(merged)
            self.run_analysis(merged, labels, pins)
            return 0
        except StopChain as e:
            self.emit("STOP", e.reason, e.detail)
            return e.code
        except Exception as e:  # a crash is a stop, not a pass
            self.emit("STOP", "CRASH", f"{type(e).__name__}: {e}")
            return 9


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default=os.path.dirname(HERE))
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--dry-run", action="store_true", help="seal + data + disk checks only; no child process")
    a = ap.parse_args(argv)
    ch = Chain(Cfg(a.pkg, work=a.work))
    if a.dry_run:
        try:
            pins = ch.check_seal()
            ch.check_data(pins)
            ch.check_disk()
            ch.emit("DRYRUN_OK")
            return 0
        except StopChain as e:
            ch.emit("STOP", e.reason, e.detail)
            return e.code
    return ch.run()


if __name__ == "__main__":
    sys.exit(main())
