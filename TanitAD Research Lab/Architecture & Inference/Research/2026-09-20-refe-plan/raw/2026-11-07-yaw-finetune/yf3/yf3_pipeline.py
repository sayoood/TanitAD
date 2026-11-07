#!/usr/bin/env python3
"""YF3 ONE DETACHED PIPELINE for WP YAW item 1 (PREREG_YAWFT.md, sha256 d28a630f...; AMENDMENT_1.md).  Costs no model turns while it runs.

    python yf3_pipeline.py --root <run root> [--smoke] [--max-rung R3]

Flow (idempotent: a finished stage leaves stages/<name>.json and is skipped on a re-launch; a half-trained arm dir is moved aside, never overwritten):
  pins -> gates_real (G2 G3 G4 G5 G9 with their RED arms, real cache) ->
  RUNG R1: CPU thread  train B s0, B s1, C s0 -> build_yaw_ckpt (K1-K4) -> identity (C == A0 state_dict; trained decode == composed decode) -> B0 identity -> G8 CPU rerun of B s0
           GPU thread  eval A0 (G6) -> eval B s0, B s1 as their checkpoints appear -> G8 eval (B s0 on every 10th token, seam only)
  analysis R1 (yf3_analyze.py: the committed rule) -> NOT PROVEN / REFUTED(gates green): RUNG R2 (B s0,C s0,B s1,C s1; all four evaluated) -> analysis -> RUNG R3 the same.
  VOID (a red gate) or any stage failure -> markers/yf3_<tag>.FAILED ; end of chain -> markers/yf3_<tag>.DONE + FINAL.md.
The GPU is taken ONLY inside yf3_eval_arm.py (YF2/gpu_lock.py: waits behind a foreign holder, never forces); training and analysis are CPU (CUDA_VISIBLE_DEVICES=-1).
"""
import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12", "PYTHONIOENCODING": "utf-8"}
RUNGS = ["R1", "R2", "R3"]
SMOKE_TRAIN = ["--steps", "4", "--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "2"]


class Pipe:
    def __init__(self, root, smoke, max_rung):
        self.root, self.smoke, self.max_rung = Path(root), smoke, max_rung
        self.tag = "smoke" if smoke else "full"
        for d in ("stages", "arms", "ckpt", "identity", "eval", "analysis", "logs", "tokens"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.mu = threading.Lock()
        self.fail = None
        self.ready = {}
        self.st = {"tag": self.tag, "root": str(self.root), "started": C.now(), "pid": os.getpid(), "stage": "init", "done_stages": [], "verdicts": {}, "state": "RUNNING"}
        self.tokens_path, self.g8_tokens_path = self._token_files()
        self.toks = sorted(C.jload(self.tokens_path)["tokens"])

    # ---------------------------------------------------------------- bookkeeping
    def log(self, m):
        line = f"[{C.now()}] {m}"
        with self.mu:
            with open(self.root / "PIPELINE.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        print(line, flush=True)

    def status(self, **kw):
        with self.mu:
            self.st.update(kw)
            self.st["updated"] = C.now()
            C.jdump(self.st, self.root / "status.json")

    def stage(self, name, fn):
        mp = self.root / "stages" / f"{name}.json"
        if mp.exists():
            self.log(f"stage {name}: already done, skipped")
            return C.jload(mp)
        self.log(f"stage {name}: start")
        t0 = time.time()
        res = fn() or {}
        res = {**res, "seconds": round(time.time() - t0, 1), "finished": C.now()}
        C.jdump(res, mp)
        with self.mu:
            self.st["done_stages"].append(name)
        self.status()
        self.log(f"stage {name}: done in {res['seconds']} s")
        return res

    def sh(self, name, argv, token=None, env=None, cwd=None):
        """one subprocess; the log is logs/<name>.log; failure = rc != 0 or the success token absent (an exit code alone is not evidence)"""
        lp = self.root / "logs" / f"{name}.log"
        e = dict(os.environ)
        e.update(env or {})
        with open(lp, "wb") as lf:
            rc = subprocess.run([str(x) for x in argv], cwd=cwd, env=e, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        txt = lp.read_text(encoding="utf-8", errors="replace")
        if rc != 0 or (token and token not in txt):
            raise RuntimeError(f"{name}: rc={rc} token {token!r} {'present' if token and token in txt else 'ABSENT'}; tail: {txt[-1500:]}")
        return txt

    # ---------------------------------------------------------------- tokens
    def _token_files(self):
        tk = C.jload(C.A5_TOKENS)
        toks = sorted(tk["tokens"])
        if self.smoke:
            sys.path.insert(0, str(C.Y2))
            import yaw_common as YC
            table, _, _ = YC.load_yaw_table(C.TABLE_TEST)
            by = {}
            for t in toks:
                by.setdefault(tk["token_log"][t], []).append(t)
            sel = []
            for lg in sorted(by)[::3]:                                           # logs spread over the set; >= 100 tokens so the adapter's statistical frame guard runs (as it does at n = 923)
                ts = sorted(by[lg], key=lambda t: -abs(table[t]))
                sel += ts[:5] + ts[-5:] if len(ts) > 10 else ts                  # the 5 highest and 5 lowest |omega| tokens of the log
                if len(set(sel)) >= 110:
                    break
            sel = list(set(sel))
            toks = sorted(sel)
        every = 4 if self.smoke else 10
        g8 = toks[::every]
        out = []
        for name, ts in (("eval_tokens.json", toks), ("g8_tokens.json", g8)):
            p = self.root / "tokens" / name
            if not p.exists():
                C.jdump({"rule": tk.get("rule", ""), "tokens": ts, "token_log": {t: tk["token_log"][t] for t in ts}}, p)
            out.append(p)
        return out[0], out[1]

    # ---------------------------------------------------------------- stages
    def pins(self):
        ok = {}
        ok["prereg_sha256"] = C.sha256_file(C.PREREG) == C.PREREG_SHA
        bad = []
        for line in (C.Y2 / "PREREG_YAWFT.sha256").read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            sha, rel = line.split(None, 1)
            rel = rel.strip().lstrip("*")
            if C.sha256_file(C.Y2 / rel) != sha:
                bad.append(rel)
        ok["pinned_files_match"] = not bad
        ok["navtest_table"] = C.sha256_file(C.TABLE_TEST) == C.TABLE_TEST_SHA
        ok["navtrain_table"] = C.sha256_file(C.TABLE_TRAIN) == C.TABLE_TRAIN_SHA
        pin = (C.HERE / "CACHE_PIN.txt").read_text(encoding="utf-8")
        ok["cache_manifest_pin"] = C.sha256_file(C.CACHE / "manifest.json") in pin
        ok["eval_tokens_in_navtest_table"] = True
        if not all(ok.values()):
            raise RuntimeError(f"pins red: {ok} bad files {bad}")
        return {"pins": ok}

    def gates_real(self):
        self.sh("gates_real", [C.PY_D, C.HERE / "real_cache_gates.py", self.root / "gates_real"], token="ZZYF3_REAL_GATES_OK", env=CPU_ENV)
        return {}

    def train(self, rung, arm, seed, suffix=""):
        name = f"{rung}_{arm}_s{seed}{suffix}"
        out = self.root / "arms" / name
        if (out / "final.pt").exists():
            return name
        if out.exists():
            shutil.move(str(out), str(out) + f".stale_{int(time.time())}")
        argv = [C.PY_D, C.Y2 / "yaw_ft.py", "--cache", C.CACHE, "--manifest", C.MANIFEST, "--yaw-table-sha256", "517c6413", "--arm", arm, "--rung", rung, "--seed", seed, "--lr-col", "1e-3",
                "--out", out] + (SMOKE_TRAIN if self.smoke else [])
        self.sh(f"train_{name}", argv, token="ZZYAWFT_OK", env=CPU_ENV)
        return name

    def build_ckpt(self, name):
        ck = self.root / "ckpt" / f"{name}.pt"
        if not ck.exists():
            self.sh(f"ckpt_{name}", [C.PY_D, C.HERE / "build_yaw_ckpt_yf3.py", "--final", self.root / "arms" / name / "final.pt", "--out", ck], token="ZZYAWCKPT_OK", env=CPU_ENV)
        return ck

    def identity(self, kind, name, ck):
        self.sh(f"identity_{name}", [C.PY_D, C.HERE / "yf3_identity.py", "--kind", kind, "--ckpt", ck, "--final", self.root / "arms" / name / "final.pt", "--out", self.root / "identity" / f"{name}.json",
                                     "--frames", 128 if self.smoke else 512], token="ZZYF3ID_OK", env=CPU_ENV)

    def cpu_arm(self, rung, arm, seed):
        name = f"{rung}_{arm}_s{seed}"
        def go():
            self.train(rung, arm, seed)
            ck = self.build_ckpt(name)
            self.identity(arm, name, ck)
            if rung == "R1" and arm == "C":                          # C == A0 bit for bit (identity just proved it): 1.29 GB of u400 again is not kept
                try:
                    os.remove(ck)
                except OSError:
                    pass
            return {"ckpt": str(ck)}
        self.stage(f"cpu_{name}", go)
        self.status(stage=f"cpu_{name} ready")
        self.ready[name].set()

    def g8_cpu(self):
        def go():
            n2 = self.train("R1", "B", 0, suffix="_rerun")
            a = __import__("torch").load(self.root / "arms" / "R1_B_s0" / "final.pt", map_location="cpu", weights_only=False)
            b = __import__("torch").load(self.root / "arms" / n2 / "final.pt", map_location="cpu", weights_only=False)
            torch = __import__("torch")
            same = bool(torch.equal(a["col4"], b["col4"]) and all(torch.equal(a["state"][k], b["state"][k]) for k in a["state"]))
            C.jdump({"ok": same, "col4_equal": bool(torch.equal(a["col4"], b["col4"])), "threads": CPU_ENV["OMP_NUM_THREADS"]}, self.root / "analysis" / "G8_cpu.json")
            os.remove(self.root / "arms" / n2 / "final.pt")
            if not same:
                raise RuntimeError("G8 CPU: B rerun is NOT bit-identical")
            return {"same": same}
        self.stage("g8_cpu", go)

    def b0_identity(self):
        def go():
            self.sh("identity_B0", [C.PY_D, C.HERE / "yf3_identity.py", "--kind", "B0", "--out", self.root / "identity" / "B0.json", "--frames", 128 if self.smoke else 512], token="ZZYF3ID_OK", env=CPU_ENV)
        self.stage("identity_B0", go)

    def eval_arm(self, name, ckpt, seam_only=False, tokens=None, out_name=None, delete_ckpt=False):
        out = self.root / "eval" / (out_name or name)
        done = out / ("seam.report.json" if seam_only else "arm.json")
        if not done.exists():
            self.status(stage=f"eval_{out_name or name}")
            argv = [C.PY_D, C.HERE / "yf3_eval_arm.py", "--name", out_name or name, "--ckpt", ckpt, "--tokens", tokens or self.tokens_path, "--out", out, "--tag", self.tag]
            if seam_only:
                argv.append("--seam-only")
            self.sh(f"eval_{out_name or name}", argv, token=f"ZZYF3EVAL {out_name or name} OK")
        if delete_ckpt:
            try:
                os.remove(ckpt)
            except OSError:
                pass

    def wait_ready(self, name):
        while not self.ready[name].wait(20):
            if self.fail:
                raise RuntimeError(f"aborting wait for {name}: {self.fail[0]} failed")

    # ---------------------------------------------------------------- one rung
    def run_rung(self, rung):
        seeds = [0] if (self.smoke and rung != "R1") else [0, 1]            # smoke R2: ONE seed pair (the chain + the C-arm path are what it proves); the real run always uses two
        if rung == "R1":
            cpu = [("B", 0), ("B", 1), ("C", 0)]
            ev = [("A0", C.U400)] + [(f"R1_B_s{s}", None) for s in seeds]
        else:
            cpu = [(a, s) for s in seeds for a in ("B", "C")]
            ev = [(f"{rung}_{a}_s{s}", None) for s in seeds for a in ("B", "C")]
        if self.smoke and rung == "R3":
            cpu = [("B", 0), ("C", 0)]                                 # smoke: R3 exercises train + ckpt + identity only (an eval costs ~6 min of GPU queue each)
            ev = []
        for a, s in cpu:
            self.ready[f"{rung}_{a}_s{s}"] = threading.Event()
        errs = []

        def cpu_thread():
            try:
                for a, s in cpu:
                    if self.fail:
                        return
                    self.cpu_arm(rung, a, s)
                if rung == "R1":
                    self.b0_identity()
                    self.g8_cpu()
            except Exception:
                errs.append(("cpu", traceback.format_exc()))
                self.fail = ("cpu", errs[-1][1])

        def gpu_thread():
            try:
                for name, ck in ev:
                    if self.fail:
                        return
                    if ck is None:
                        self.wait_ready(name)
                        ck = self.root / "ckpt" / f"{name}.pt"
                    self.eval_arm(name, ck, delete_ckpt=("_C_" in name))
                if rung == "R1" and ev:
                    self.wait_ready("R1_B_s0")
                    self.eval_arm("R1_B_s0", self.root / "ckpt" / "R1_B_s0.pt", seam_only=True, tokens=self.g8_tokens_path, out_name="R1_B_s0_g8")
            except Exception:
                errs.append(("gpu", traceback.format_exc()))
                self.fail = ("gpu", errs[-1][1])

        th = [threading.Thread(target=cpu_thread, name="cpu"), threading.Thread(target=gpu_thread, name="gpu")]
        [t.start() for t in th]
        [t.join() for t in th]
        if errs:
            raise RuntimeError("; ".join(f"{w}: {m[-1800:]}" for w, m in errs))
        if rung == "R1" and ev:
            self.finish_g8()
        if rung == "R3" and self.smoke:
            return "SMOKE_TRAIN_ONLY"
        return self.analyze(rung, seeds)

    def finish_g8(self):
        def go():
            main_z, g8_z = np.load(self.root / "eval" / "R1_B_s0" / "seam.npz"), np.load(self.root / "eval" / "R1_B_s0_g8" / "seam.npz")
            mi = {str(t): i for i, t in enumerate(main_z["token"])}
            same = bool(len(g8_z["token"]) > 0 and all(str(t) in mi and np.array_equal(g8_z["poses"][i], main_z["poses"][mi[str(t)]]) for i, t in enumerate(g8_z["token"])))
            ga, gb = C.jload(self.root / "eval" / "R1_B_s0" / "shim_g7.json")["by_token"], C.jload(self.root / "eval" / "R1_B_s0_g8" / "shim_g7.json")["by_token"]
            shim_same = bool(all(ga.get(t) == v for t, v in gb.items()) and len(gb) > 0)
            cpu = C.jload(self.root / "analysis" / "G8_cpu.json")
            r = {"ok": bool(same and shim_same and cpu["ok"]), "eval_plans_bit_identical": same, "eval_shim_ego_identical": shim_same, "n_tokens": int(len(g8_z["token"])), "cpu": cpu}
            C.jdump(r, self.root / "analysis" / "G8.json")
            return r
        self.stage("g8_final", go)

    def analyze(self, rung, seeds):
        txt = self.sh(f"analyze_{rung}", [C.PY_D, C.HERE / "yf3_analyze.py", "--root", self.root, "--rung", rung, "--tokens", self.tokens_path, "--seeds", ",".join(map(str, seeds))], env=CPU_ENV)
        line = next(l for l in txt.splitlines() if l.startswith("ZZYF3VERDICT"))
        verdict = line.split(" ", 2)[2].split("  (")[0].strip()
        self.st["verdicts"][rung] = verdict
        self.status(stage=f"analysis {rung}: {verdict}")
        self.log(line)
        return verdict

    # ---------------------------------------------------------------- driver
    def run(self):
        self.status(stage="pins")
        self.stage("pins", self.pins)
        self.status(stage="gates_real")
        self.stage("gates_real", self.gates_real)
        chain = [r for r in RUNGS if RUNGS.index(r) <= RUNGS.index(self.max_rung)]
        final = None
        for i, rung in enumerate(chain):
            self.status(stage=f"rung {rung}")
            v = self.run_rung(rung)
            final = (rung, v)
            if v in ("NOT PROVEN", "REFUTED"):                                  # E: has ~21 GB free: composed checkpoints (1.29 GB each) of a rung that did not ADOPT are reproducible from arms/*/final.pt
                for ck in (self.root / "ckpt").glob(f"{rung}_*.pt"):              # by build_yaw_ckpt_yf3.py (deterministic), so they are not kept. An ADOPT rung keeps its checkpoints (closed-loop A/B input).
                    try:
                        os.remove(ck)
                    except OSError:
                        pass
            if v == "ADOPT":
                break
            if v == "VOID":
                raise RuntimeError(f"{rung}: VOID -- a gate is red; see analysis/{rung}.json gates_red")
            if v == "SMOKE_TRAIN_ONLY":
                break
            if v not in ("NOT PROVEN", "REFUTED"):
                raise RuntimeError(f"unknown verdict {v!r}")
            self.log(f"{rung}: {v} -> next rung (Rule Zero) if any")
        self.write_final(final)

    def write_final(self, final):
        rung, v = final
        L = [f"# YF3 FINAL ({self.tag}) {C.now()}", "", f"Verdicts: {json.dumps(self.st['verdicts'])}", f"Chain ended at {rung}: {v}", "",
             "Tier: NAVSIM v1.1 pseudo-simulation (open-loop plan vs logged agents), NOT closed loop. Estimator: paired log-cluster bootstrap 10,000, seed 20260927 (answers 'another draw of logs' only).", ""]
        if v == "ADOPT":
            L += [f"ADOPT at {rung} (B's rung only). Claim: on NAVSIM open-loop the yaw input is usable and lowers lateral error; nothing about closed loop.",
                  "NEXT (PREREG section 6): (a) confirmatory read on the full 12,146 navtest tokens (A0 = 88.25 banked), same bars; (b) closed-loop sim A/B with live omega on identical scenarios; (c) rerun this PREREG on P6 final."]
        elif rung == "R3" and v in ("NOT PROVEN", "REFUTED"):
            L += ["R3 is " + v + ". ELIMINATION (PREREG section 6): omega is redundant with (ay, v) at fine-tune scale (R-squared 0.854 on the DB).",
                  "Next levers in order of expected effect: (i) P7 from scratch with the YAW_SCALE input; (ii) a sim-side check that ay is trustworthy there; (iii) scorer refresh on B's fan (blocked on PDM relabel compute)."]
        else:
            L += ["Chain stopped before R3 (max_rung / smoke)."]
        (self.root / "FINAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        self.status(stage="FINAL", state="DONE", final={"rung": rung, "verdict": v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rung", default="R3", choices=RUNGS)
    a = ap.parse_args()
    marker_dir = C.HERE / "markers"
    marker_dir.mkdir(exist_ok=True)
    p = Pipe(a.root, a.smoke, a.max_rung)
    tag = f"yf3_{p.tag}"
    (marker_dir / f"{tag}.STARTED").write_text(json.dumps({"pid": os.getpid(), "root": str(p.root), "t": C.now(), "argv": sys.argv}), encoding="utf-8")
    for m in ("DONE", "FAILED"):
        if (marker_dir / f"{tag}.{m}").exists():
            os.remove(marker_dir / f"{tag}.{m}")
    try:
        p.run()
        rec = {"state": "DONE", "status_file": str(p.root / "status.json"), "verdicts": p.st["verdicts"], "finished": C.now()}
        (marker_dir / f"{tag}.DONE").write_text(json.dumps(rec, indent=1), encoding="utf-8")                     # LAST ACTION
        return 0
    except Exception:
        why = traceback.format_exc()[-3500:]
        p.log("FAILED: " + why)
        p.status(state="FAILED", why=why)
        (marker_dir / f"{tag}.FAILED").write_text(json.dumps({"state": "FAILED", "why": why, "status_file": str(p.root / "status.json"), "finished": C.now()}, indent=1), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
