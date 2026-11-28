#!/usr/bin/env python3
"""YAX ONE DETACHED PIPELINE for the AXOFF / AXON experiment (PREREG_AXOFF.md).  Costs no model turns while it runs.   python ax_pipeline.py --root <run root> [--smoke]
Flow (idempotent; a finished stage leaves stages/<name>.json and is skipped on a re-launch; a half-trained arm dir is moved aside, never overwritten):
  pins -> gates_real (real cache: AXOFF and AXON gates green + the three deliberate-regression arms RED) -> g8_mini -> A0_banked per-window ->
  select (AXOFF s0 under K1 and K2 in parallel; lower cache-VAL loss, ties within 1 % -> K1) ->
  RUNG X1 (selected recipe): CPU thread trains AXON s0, AXOFF s1, AXON s1 -> compose ckpt (K1-K5) -> identity (GA-GAIN / G-CKPT);  GPU thread evals AXOFF s0, AXON s0, AXOFF s1, AXON s1 as they appear
  (shared lock, YF2 protocol, waits behind YF3, never forces) -> ax_analyze.py (the committed rule) -> CANDIDATE: stop.  NOT PROVEN / REFUTED (gates green): RUNG X2 = the OTHER recipe, same rule.  VOID: fail.
Markers (last action): markers/yax_<full|smoke>.DONE or .FAILED ; FINAL.md.   Training/analysis are CPU (CUDA_VISIBLE_DEVICES=-1); the GPU is taken only inside ax_eval_arm.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ax_common as C  # noqa: E402

CPU_ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "10", "MKL_NUM_THREADS": "10", "PYTHONIOENCODING": "utf-8"}
CANDS = {"K1": (200, "5e-5"), "K2": (400, "1e-4")}
SMOKE = ["--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "2"]
CODE_FILES = ["ax_ft.py", "ax_ckpt_tools.py", "ax_analyze.py", "ax_stats.py", "ax_perwindow.py", "ax_eval_arm.py", "ax_common.py", "ax_pipeline.py", "yaw_ft_ref.py", "yaw_common.py", "gpu_lock.py"]
POL = {"AXOFF": "zero", "AXON": "real"}


class Pipe:
    def __init__(self, root, smoke):
        self.root, self.smoke = Path(root), smoke
        self.tag = "smoke" if smoke else "full"
        for d in ("stages", "arms", "ckpt", "identity", "eval", "analysis", "logs", "tokens"):
            (self.root / d).mkdir(parents=True, exist_ok=True)
        self.mu, self.fail, self.ready = threading.Lock(), None, {}
        self.st = {"tag": self.tag, "root": str(self.root), "started": C.now(), "pid": os.getpid(), "stage": "init", "done_stages": [], "verdicts": {}, "state": "RUNNING"}
        self.tokens_path = self._tokens()
        self.seeds = [0] if smoke else [0, 1]

    def log(self, m):
        line = f"[{C.now()}] {m}"
        with self.mu, open(self.root / "PIPELINE.log", "a", encoding="utf-8") as f:
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
            return C.jload(mp)
        self.log(f"stage {name}: start")
        t0 = time.time()
        res = {**(fn() or {}), "seconds": round(time.time() - t0, 1), "finished": C.now()}
        C.jdump(res, mp)
        with self.mu:
            self.st["done_stages"].append(name)
        self.status()
        self.log(f"stage {name}: done in {res['seconds']} s")
        return res

    def sh(self, name, argv, token=None, env=None, expect_rc=0):
        lp = self.root / "logs" / f"{name}.log"
        e = dict(os.environ)
        e.update(env or {})
        with open(lp, "wb") as lf:
            rc = subprocess.run([str(x) for x in argv], env=e, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
        txt = lp.read_text(encoding="utf-8", errors="replace")
        if rc != expect_rc or (token and token not in txt):
            raise RuntimeError(f"{name}: rc={rc} (want {expect_rc}) token {token!r} {'present' if token and token in txt else 'ABSENT'}; tail: {txt[-1500:]}")
        return txt

    def _tokens(self):
        tk = C.jload(C.A5_TOKENS)
        toks = sorted(tk["tokens"])
        if self.smoke:
            by = {}
            for t in toks:
                by.setdefault(tk["token_log"][t], []).append(t)
            sel = []
            for lg in sorted(by)[::3]:
                sel += by[lg][:6]
                if len(sel) >= 110:
                    break
            toks = sorted(set(sel))
        p = self.root / "tokens" / "eval_tokens.json"
        if not p.exists():
            C.jdump({"rule": tk.get("rule", ""), "tokens": toks, "token_log": {t: tk["token_log"][t] for t in toks}}, p)
        return p

    # ---------------------------------------------------------------- stages
    def pins(self):
        ok = {"prereg_sha256": C.sha256_file(C.PREREG) == C.PREREG_SHA, "prereg_sha_file": C.PREREG_SHA in (HERE / "PREREG_AXOFF.sha256").read_text(encoding="utf-8"),
              "cache_manifest": C.sha256_file(C.CACHE / "manifest.json") == C.CACHE_MANIFEST_SHA, "frame_manifest": C.sha256_file(C.MANIFEST) == C.MANIFEST_SHA, "base_u400": C.sha256_file(C.U400) == C.U400_SHA}
        bad = []
        pin = HERE / "CODE_PIN.sha256"
        if pin.exists():
            for line in pin.read_text(encoding="utf-8").splitlines():
                if line.strip() and not line.startswith("#"):
                    sha, rel = line.split(None, 1)
                    if C.sha256_file(HERE / rel.strip().lstrip("*")) != sha:
                        bad.append(rel.strip())
        ok["code_pin_present_and_matches"] = bool(pin.exists() and not bad)
        if not all(ok.values()):
            raise RuntimeError(f"pins red: {ok} bad code files {bad}")
        return {"pins": ok}

    def gates_real(self):
        a = ["--cache", C.CACHE, "--manifest", C.MANIFEST, "--steps", 200, "--lr", "5e-5", "--gates-only"] + (SMOKE if self.smoke else [])
        res = {}
        for nm, pol, reg, want in (("axoff", "zero", "none", 0), ("axon", "real", "none", 0), ("RED_fed_in_zero", "zero", "ax_fed_in_zero", 3), ("RED_not_fed", "real", "ax_not_fed", 3), ("RED_keep_col", "zero", "keep_ax_col", 3)):
            out = self.root / "gates_real" / nm
            if out.exists():
                shutil.rmtree(out)
            self.sh(f"gates_real_{nm}", [C.PY_D, HERE / "ax_ft.py", "--ax-policy", pol, "--regress", reg, "--out", out] + a, token="ZZAXFT_GATES_OK" if want == 0 else "ZZAXFT_GATE_RED", env=CPU_ENV, expect_rc=want)
            res[nm] = {"rc": want, "gates": C.jload(out / "gates.json")}
        ok = bool(res["axoff"]["gates"]["pass"] and res["axon"]["gates"]["pass"] and not res["RED_fed_in_zero"]["gates"]["GA_FEED"] and not res["RED_not_fed"]["gates"]["GB_STEP0"] and not res["RED_keep_col"]["gates"]["GA_COL"])
        C.jdump({"pass": ok, **res}, self.root / "analysis" / "gates_real.json")
        if not ok:
            raise RuntimeError("gates_real: a gate is red or a deliberate-regression arm did not read RED")
        return {"pass": ok}

    def g8_mini(self):
        import torch
        a = ["--cache", C.CACHE, "--manifest", C.MANIFEST, "--ax-policy", "zero", "--steps", 5, "--lr", "5e-5", "--max-frames", "400", "--batch", "16", "--accum", "2", "--gate-frames", "64", "--val-every", "5"]
        fs = []
        for k in ("a", "b"):
            out = self.root / "arms" / f"_g8mini_{k}"
            if out.exists():
                shutil.rmtree(out)
            self.sh(f"g8mini_{k}", [C.PY_D, HERE / "ax_ft.py", "--out", out, "--seed", 0] + a, token="ZZAXFT_OK", env=CPU_ENV)
            fs.append(torch.load(out / "final.pt", map_location="cpu", weights_only=False))
        same = all(torch.equal(fs[0]["state"][k], fs[1]["state"][k]) for k in fs[0]["state"])
        C.jdump({"ok": bool(same), "note": "two 5-step runs, same arm+seed, CPU: weights torch.equal (full-run rerun NOT done)"}, self.root / "analysis" / "G8_mini.json")
        if not same:
            raise RuntimeError("G8-mini: reruns differ")
        return {"same": same}

    def a0_banked(self):
        d = self.root / "eval" / "A0_banked"
        d.mkdir(parents=True, exist_ok=True)
        self.sh("perwindow_A0_banked", [C.PY_T, HERE / "ax_perwindow.py", "--seam", C.BANKED_SEAM, "--tokens", self.tokens_path, "--out", d / "perwindow.npz"], env={"PYTHONIOENCODING": "utf-8"})
        return {}

    def train(self, cand, arm, seed):
        name = f"{cand}_{arm}_s{seed}"
        out = self.root / "arms" / name
        if (out / "final.pt").exists():
            return name
        if out.exists():
            shutil.move(str(out), str(out) + f".stale_{int(time.time())}")
        steps, lr = CANDS[cand]
        argv = [C.PY_D, HERE / "ax_ft.py", "--cache", C.CACHE, "--manifest", C.MANIFEST, "--ax-policy", POL[arm], "--seed", seed, "--steps", 4 if self.smoke else steps, "--lr", lr, "--out", out] + (SMOKE if self.smoke else [])
        self.sh(f"train_{name}", argv, token="ZZAXFT_OK", env=CPU_ENV)
        return name

    def select(self):
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(2) as ex:
            list(ex.map(lambda c: self.train(c, "AXOFF", 0), ("K1", "K2")))
        v = {c: C.jload(self.root / "arms" / f"{c}_AXOFF_s0" / "summary.json")["val_final"]["loss"] for c in CANDS}
        chosen = "K2" if (v["K2"] < v["K1"] and (v["K1"] - v["K2"]) / v["K1"] >= 0.01) else "K1"
        r = {"val_final_loss": v, "chosen_X1": chosen, "other_X2": "K2" if chosen == "K1" else "K1", "rule": "lower cache-VAL loss, relative gap < 1 % -> K1 (PREREG section 2)"}
        self.log(f"select: {r}")
        return r

    def free_gb(self):
        return shutil.disk_usage(str(self.root)).free / 1e9

    def build_ckpt(self, rung, cand, arm, seed):
        ck = self.root / "ckpt" / f"{rung}_{arm}_s{seed}.pt"
        if not ck.exists():
            t0 = time.time()
            while self.free_gb() < 4.0 and time.time() - t0 < 7200:
                time.sleep(60)
            self.sh(f"ckpt_{rung}_{arm}_s{seed}", [C.PY_D, HERE / "ax_ckpt_tools.py", "build", "--final", self.root / "arms" / f"{cand}_{arm}_s{seed}" / "final.pt", "--out", ck], token="ZZAXCKPT_OK", env=CPU_ENV)
        return ck

    def cpu_arm(self, rung, cand, arm, seed):
        name = f"{rung}_{arm}_s{seed}"

        def go():
            self.train(cand, arm, seed)
            ck = self.build_ckpt(rung, cand, arm, seed)
            self.sh(f"identity_{name}", [C.PY_D, HERE / "ax_ckpt_tools.py", "identity", "--ckpt", ck, "--final", self.root / "arms" / f"{cand}_{arm}_s{seed}" / "final.pt", "--out", self.root / "identity" / f"{name}.json",
                                         "--frames", 128 if self.smoke else 512], token="ZZAXID_OK", env=CPU_ENV)
            return {"ckpt": str(ck)}
        self.stage(f"cpu_{name}", go)
        self.status(stage=f"cpu_{name} ready")
        self.ready[name].set()

    def eval_arm(self, name):
        out = self.root / "eval" / name
        if not (out / "arm.json").exists():
            self.status(stage=f"eval_{name}")
            self.sh(f"eval_{name}", [C.PY_D, HERE / "ax_eval_arm.py", "--name", name, "--ckpt", self.root / "ckpt" / f"{name}.pt", "--tokens", self.tokens_path, "--out", out, "--tag", self.tag], token=f"ZZAXEVAL {name} OK")
        if "_AXON_" in name:                                           # AXON checkpoints are 1.29 GB scratch; final.pt (decoder tensors) stays and can re-compose them
            try:
                os.remove(self.root / "ckpt" / f"{name}.pt")
            except OSError:
                pass

    def run_rung(self, rung, cand):
        order = [("AXOFF", 0), ("AXON", 0)] + [(a, s) for s in self.seeds[1:] for a in ("AXOFF", "AXON")]
        for a, s in order:
            self.ready[f"{rung}_{a}_s{s}"] = threading.Event()
        errs = []

        def cpu_thread():
            try:
                for a, s in order:
                    if self.fail:
                        return
                    self.cpu_arm(rung, cand, a, s)
            except Exception:
                errs.append(("cpu", traceback.format_exc()))
                self.fail = ("cpu", errs[-1][1])

        def gpu_thread():
            try:
                for a, s in order:
                    n = f"{rung}_{a}_s{s}"
                    while not self.ready[n].wait(20):
                        if self.fail:
                            raise RuntimeError(f"aborting wait for {n}: {self.fail[0]} failed")
                    self.eval_arm(n)
            except Exception:
                errs.append(("gpu", traceback.format_exc()))
                self.fail = ("gpu", errs[-1][1])
        th = [threading.Thread(target=cpu_thread), threading.Thread(target=gpu_thread)]
        [t.start() for t in th]
        [t.join() for t in th]
        if errs:
            raise RuntimeError("; ".join(f"{w}: {m[-1800:]}" for w, m in errs))
        txt = self.sh(f"analyze_{rung}", [C.PY_T, HERE / "ax_analyze.py", "--root", self.root, "--rung", rung, "--tokens", self.tokens_path, "--seeds", ",".join(map(str, self.seeds))], env={"PYTHONIOENCODING": "utf-8"})
        line = next(l for l in txt.splitlines() if l.startswith("ZZAXVERDICT"))
        v = line.split(" ", 2)[2].split("  (")[0].strip()
        self.st["verdicts"][rung] = v
        self.status(stage=f"analysis {rung}: {v}")
        self.log(line)
        return v

    def run(self):
        for nm, fn in (("pins", self.pins), ("gates_real", self.gates_real), ("g8_mini", self.g8_mini), ("a0_banked", self.a0_banked)):
            self.status(stage=nm)
            self.stage(nm, fn)
        self.status(stage="select")
        sel = self.stage("select", self.select)
        final = None
        for rung, cand in (("X1", sel["chosen_X1"]), ("X2", sel["other_X2"])):
            if self.smoke and rung == "X2":
                break
            self.status(stage=f"rung {rung} ({cand})")
            v = self.run_rung(rung, cand)
            final = (rung, cand, v)
            if v == "CANDIDATE":
                break
            if v == "VOID":
                raise RuntimeError(f"{rung}: VOID -- a gate is red; see analysis/{rung}.json gates_red")
            self.log(f"{rung}: {v} -> next recipe rung if any (Rule Zero)")
        self.write_final(final, sel)

    def write_final(self, final, sel):
        rung, cand, v = final
        L = [f"# YAX FINAL ({self.tag}) {C.now()}", "", f"Verdicts: {json.dumps(self.st['verdicts'])}  | selection: {json.dumps(sel)}", f"Chain ended at {rung} (recipe {cand}): {v}", "",
             "Tier: NAVSIM v1.1 pseudo-simulation, open loop, NOT closed loop. Estimator: paired log-cluster bootstrap 10,000, seed 20260927.", ""]
        if v == "CANDIDATE":
            L += [f"CANDIDATE at {rung}: send ckpt/{rung}_AXOFF_s0.pt (and _s1) to the sim session; closed-loop ADOPT rule is PREREG section 6 (no stall: min speed > 1.0 m/s before the stop line on a green arrow; EPDMS >= 0.748)."]
        elif self.smoke:
            L += ["SMOKE run: mechanical check of the pipeline on a tiny subset; the verdict above is NOT a result."]
        else:
            L += ["Both recipe rungs are NOT PROVEN / REFUTED. Named levers, each with what unblocks it:",
                  "(i) ax-noise/dropout augmentation (keep ax, break its loop gain): CPU-only, needs a PI/MM go on a new PREREG;",
                  "(ii) P7 from scratch without the ax slot: GPU-days, PI decision;", "(iii) larger cache (4,000 frames is the capacity limit): ~1.1 s/frame GPU encoder pass (about 1.2 h per 4,000), needs the GPU lock."]
        (self.root / "FINAL.md").write_text("\n".join(L) + "\n", encoding="utf-8")
        self.status(stage="FINAL", state="DONE", final={"rung": rung, "verdict": v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    md = HERE / "markers"
    md.mkdir(exist_ok=True)
    p = Pipe(a.root, a.smoke)
    tag = f"yax_{p.tag}"
    (md / f"{tag}.STARTED").write_text(json.dumps({"pid": os.getpid(), "root": str(p.root), "t": C.now(), "argv": sys.argv}), encoding="utf-8")
    for m in ("DONE", "FAILED"):
        if (md / f"{tag}.{m}").exists():
            os.remove(md / f"{tag}.{m}")
    try:
        p.run()
        (md / f"{tag}.DONE").write_text(json.dumps({"state": "DONE", "status_file": str(p.root / "status.json"), "verdicts": p.st["verdicts"], "finished": C.now()}, indent=1), encoding="utf-8")   # LAST ACTION
        return 0
    except Exception:
        why = traceback.format_exc()[-3500:]
        p.log("FAILED: " + why)
        p.status(state="FAILED", why=why)
        (md / f"{tag}.FAILED").write_text(json.dumps({"state": "FAILED", "why": why, "status_file": str(p.root / "status.json"), "finished": C.now()}, indent=1), encoding="utf-8")
        return 1


if __name__ == "__main__":
    sys.exit(main())
