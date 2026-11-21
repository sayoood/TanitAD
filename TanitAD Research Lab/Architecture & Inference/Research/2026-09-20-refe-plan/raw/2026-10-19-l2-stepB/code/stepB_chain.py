#!/usr/bin/env python3
"""L2 STEP B CHAIN: one detached, CPU-only chain that runs the SEALED Step B (`PREREG_L2_STEPB.md`, Revision 1) from "the carve-B bundle exists" to "the verdict line".  It is a script, never a model turn.

    (a) WAIT    until the sealed Step B text + its .sha256 exist and hold (the seal guard), until L2-FRESH's chain printed `CHAIN_DONE_NO_READ` and its gates hold, and until L2-FRESH's own seal
                (`PREREG_L2_FRESH.sha256`) exists.  A FAIL marker of the L2-FRESH chain, a dead chain without a marker, or a timeout STOPS this chain.
    (b) GATE    the DATA CLAUSE (the bundle's token list must equal the sealed carve, sha256 `0629811f...`; mismatch REFUSES), then G-REACH-STEPB on each REAL head (s0, s1) -> PASS stamps.  Any failure
                prints ZZSTEPB_STOP BEFORE any search.
    (c) SEARCH  arms L2-s0, L2-s1, P2HEAD x 3 search seeds (resumable: shards, merged files and stamps already valid are skipped).
    (d) LABEL   the outputs (label_stepB.py: the planted identity at the labeller is enforced there, exit 3 = STOP).
    (e) ANALYSE the registered analysis (`analyse_stepB.py`): the four families (STRATEGIC PAUSED), O1 / TRADE / NOT CLEARED / O4, worst of the two head seeds, the stamps, the NAMED next arm.
    (f) print   `ZZSTEPB_DONE <combined outcome>`.

CPU ONLY: CUDA_VISIBLE_DEVICES=-1, threads capped at 4 (label step: 1), NO lock, no pod.  Every marker is at the START of its line (`^ZZSTEPB_`); the children's output goes to run/logs/*.log, never into this log,
so a grep anchored on `^ZZSTEPB_` cannot match a line that merely MENTIONS a marker (the self-matching-filter trap).

    python stepB_chain.py                       # the real chain (defaults below), launched detached (WMI, `< NUL >> log 2>&1`)
Exit codes: 0 done; 3 gate / wait failure; 4 timeout; 5 seal or data-clause refusal; 6 search; 7 label; 8 analysis; 9 crash (a marker is printed).
NOTHING here reads a Step B outcome until the registered analysis has produced it; the chain's tests run on PLANTED inputs only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import stepB  # noqa: E402  (puts L1's code on sys.path; check_stamp, check_data_clause, StepBConfig)

# ------------------------------------------------------------------------------------------------------------------------------ the real inputs (all overridable for tests)
PKG = "C:/Users/Admin/par18/l2stepb"
FRESH = "C:/Users/Admin/par18/l2fresh"
B_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/B_token_disjoint"
HEADS_DIR = "D:/Projects/TanitAD-artifacts/l2_fresh_stepA/heads"
CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
P2_CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"          # MEASURED 2026-10-19 (sha256 of CKPT); also named in PREREG_L2_FRESH.md section 8.3
REFE_DIR = "C:/Users/Admin/par18/l1run/tree_hybrid/refe"
PLAN = "C:/Users/Admin/p2heldout/plan/heldout_tokens.json"
PY_CPU = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
PY_LABEL = "C:/Users/Admin/navsim-crun/venv/Scripts/python.exe"
LABEL_PYTHONPATH = "D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896;C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/refe"
LABEL_MAPS = "D:/Archive/devbox-C/navsim/data/maps"
SEALED_MARK, DRAFT_MARK = "Status: SEALED", "Status: DRAFT"
PINS_BEGIN, PINS_END = "<!-- PINS-BEGIN -->", "<!-- PINS-END -->"
OPERATIVE = ("l2_scorer.py", "stepB.py", "g_reach_stepb.py", "label_stepB.py", "analyse_stepB.py", "stepB_chain.py")
REQUIRED_PINS = ("carve_sha256", "head_s0_sha256", "head_s1_sha256", "ckpt_sha256", "config_sha256", "config_o4_sha256")
ARMS = (("L2s0", "L2", "s0"), ("L2s1", "L2", "s1"), ("P2HEAD", "P2HEAD", None))        # (analysis name, stepB.py --arm, head seed)

DONE_RE = re.compile(r"^CHAIN_DONE_NO_READ(?:\s|$)", re.M)
FAIL_RE = re.compile(r"^(?:STEP\d+_[A-Z_]*FAIL\b|G2B_GATE_NOT_PASSED|EXPORT_FAIL|MERGE_FAIL|SCORES_FAIL\w*|NO_SAMPLE|NO_N\b|EMPTY_SAMPLE_REFUSED|ZZ[A-Z0-9_]*FAIL)", re.M)
BUNDLE_OK_RE = re.compile(r"^\s*ZZBUNDLE_OK 400(?:\s|$)", re.M)


class StopChain(Exception):
    def __init__(self, code: int, reason: str, detail: str = ""):
        super().__init__(f"{reason} {detail}".strip())
        self.code, self.reason, self.detail = code, reason, detail


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_text(path: str):
    """tolerant read of a log (cmd redirects are OEM / may carry NULs); None when the file does not exist yet"""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except FileNotFoundError:
        return None
    return raw.replace(b"\x00", b"").decode("utf-8", "replace")


class Cfg:
    """every path and parameter of the chain; `Cfg()` is the real chain, tests pass planted directories"""

    def __init__(self, pkg=PKG, fresh=FRESH, b_dir=B_DIR, b_log=None, b_pid=None, heads_dir=HEADS_DIR, ckpt=CKPT, refe_dir=REFE_DIR, plan=PLAN, py_cpu=PY_CPU, py_label=PY_LABEL, code_dir=HERE,
                 poll_s=60.0, max_wait_s=36 * 3600.0, heartbeat_s=1800.0, min_ram_gb=8.0, ram_max_wait_s=6 * 3600.0, threads=4, n_expected=400):
        self.pkg, self.fresh, self.b_dir, self.heads_dir, self.ckpt, self.refe_dir, self.plan = pkg, fresh, b_dir, heads_dir, ckpt, refe_dir, plan
        self.py_cpu, self.py_label, self.code_dir = py_cpu, py_label, code_dir
        self.prereg, self.prereg_sha = f"{pkg}/PREREG_L2_STEPB.md", f"{pkg}/PREREG_L2_STEPB.sha256"
        self.fresh_prereg, self.fresh_sha = f"{fresh}/PREREG_L2_FRESH.md", f"{fresh}/PREREG_L2_FRESH.sha256"
        self.b_log, self.b_pid = b_log or f"{fresh}/chain_B.log", b_pid
        self.config, self.config_o4 = f"{pkg}/raw/stepB_config.json", f"{pkg}/raw/stepB_config_O4.json"
        self.run = f"{pkg}/run"
        self.poll_s, self.max_wait_s, self.heartbeat_s, self.min_ram_gb, self.ram_max_wait_s = poll_s, max_wait_s, heartbeat_s, min_ram_gb, ram_max_wait_s
        self.threads, self.n_expected = int(threads), int(n_expected)

    @property
    def bundle(self):
        return f"{self.b_dir}/bundle/bundle.npz"

    @property
    def ctx(self):
        return f"{self.b_dir}/bundle/ctx.npy"

    @property
    def ctx_json(self):
        return f"{self.b_dir}/bundle/ctx.json"


# ------------------------------------------------------------------------------------------------------------------------------ child processes (CPU only, capped, never a shell string)
def child_env(label: bool = False) -> dict:
    env = dict(os.environ)
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    if label:                                                                         # the env of L1's step 5 (READY.md section 2.4)
        env.update({"PYTHONPATH": LABEL_PYTHONPATH, "NUPLAN_MAPS_ROOT": LABEL_MAPS, "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0", "REFE_SIM_HZ": "10", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    else:
        env.pop("PYTHONPATH", None)
        env.update({"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return env


def real_runner(argv: list, env: dict, log_path: str, cwd: str | None = None, on_pid=None) -> int:
    """run one child to completion, its stdout+stderr appended to `log_path`; below-normal priority on Windows (the box is shared); returns the exit code"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    flags = (0x00004000 | 0x08000000) if os.name == "nt" else 0                         # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
    with open(log_path, "ab") as lf:
        lf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(argv)}\n".encode("utf-8", "replace"))
        lf.flush()
        p = subprocess.Popen(argv, env=env, cwd=cwd, stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT, creationflags=flags)
        if on_pid:
            on_pid(p.pid)
        return p.wait()


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True, text=True, timeout=30).stdout
        return f'"{int(pid)}"' in out
    except Exception:
        return True                                                                   # cannot tell: assume alive, the timeout still ends the wait


def avail_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


# ------------------------------------------------------------------------------------------------------------------------------ the pins (a machine-readable block inside the SEALED text)
def parse_pins(text: str) -> dict:
    """{key: value} from the PINS block (`key value` per line); `code:<file>` and `l1:<file>` keys carry the LF sha256 of a code file"""
    if PINS_BEGIN not in text or PINS_END not in text:
        raise ValueError("no PINS block in the sealed text")
    body = text.split(PINS_BEGIN, 1)[1].split(PINS_END, 1)[0]
    pins = {}
    for ln in body.splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("```"):
            continue
        k, _, v = ln.partition(" ")
        pins[k.strip()] = v.strip()
    return pins


def collect_pins(cfg: Cfg, fresh_prereg_sha256: str | None = None, ckpt_sha256: str = P2_CKPT_SHA256) -> dict:
    """the pins as they are NOW on disk (used by the sealer to WRITE the block and by the tests to plant a consistent one)"""
    pins = {"carve_sha256": stepB.SEALED_CARVE_SHA256, "head_s0_sha256": sha256_file(f"{cfg.heads_dir}/final_head_s0.npz"), "head_s1_sha256": sha256_file(f"{cfg.heads_dir}/final_head_s1.npz"),
            "ckpt_sha256": ckpt_sha256, "config_sha256": stepB.sha256_lf(cfg.config), "config_o4_sha256": stepB.sha256_lf(cfg.config_o4), "fresh_prereg_sha256": fresh_prereg_sha256}
    for fn in sorted(os.listdir(cfg.code_dir)):
        if fn.endswith(".py") and fn != "make_stage.py":                              # make_stage.py only copies files for landing and is edited after the seal; it never runs in the chain
            pins[f"code:{fn}"] = stepB.sha256_lf(os.path.join(cfg.code_dir, fn))
    for fn, sh in stepB.L1_PINS.items():
        pins[f"l1:{fn}"] = sh
    return pins


def render_pins(pins: dict) -> str:
    return PINS_BEGIN + "\n```\n" + "\n".join(f"{k} {v}" for k, v in pins.items() if v is not None) + "\n```\n" + PINS_END


class Chain:
    def __init__(self, cfg: Cfg, runner=real_runner, clock=time, out=None, ram_fn=None, alive_fn=None):
        self.cfg, self.runner, self.clock = cfg, runner, clock
        self._out = out or (lambda s: print(s, flush=True))
        self.ram_fn, self.alive_fn = ram_fn or avail_gb, alive_fn or pid_alive
        self.logs = f"{cfg.run}/logs"
        self.pins: dict = {}
        self._stage = "init"

    # ---------------------------------------------------------------------------------------------------------- output
    def emit(self, marker: str, *parts) -> None:
        line = " ".join([marker, *[str(p) for p in parts]])
        self._out(line.replace("\r", " ").replace("\n", " | "))      # one marker = one line: a multi-line detail must not leave a line that does not start with ZZSTEPB_

    def stop(self, code: int, reason: str, detail: str = ""):
        self.emit("ZZSTEPB_STOP", reason, detail)
        raise StopChain(code, reason, detail)

    def state(self, stage: str, **kw) -> None:
        self._stage = stage
        os.makedirs(self.cfg.run, exist_ok=True)
        with open(f"{self.cfg.run}/chain_state.json", "w", encoding="utf-8", newline="\n") as f:
            json.dump({"pid": os.getpid(), "stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **kw}, f, indent=1)

    # ---------------------------------------------------------------------------------------------------------- (a) the seals
    def check_seal(self) -> dict:
        """the SEAL GUARD: refuses (exit 5) unless PREREG_L2_STEPB.md and its .sha256 exist, agree, say SEALED, and every pin holds against the files and constants the chain will actually use"""
        c = self.cfg
        if not os.path.isfile(c.prereg) or not os.path.isfile(c.prereg_sha):
            self.stop(5, "SEAL_MISSING", f"{c.prereg} / {c.prereg_sha}: no Step B without the sealed text")
        text = open(c.prereg, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
        rec = open(c.prereg_sha, encoding="utf-8").read().split()
        got = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if not rec or rec[0] != got:
            self.stop(5, "SEAL_MISMATCH", f"the sealed text's sha256 {got[:12]} != the recorded {(rec[0] if rec else '')[:12]}: it was edited after sealing")
        if SEALED_MARK not in text or DRAFT_MARK in text:
            self.stop(5, "SEAL_NOT_SEALED", f"the text must carry {SEALED_MARK!r} and not {DRAFT_MARK!r}")
        try:
            pins = parse_pins(text)
        except ValueError as e:
            self.stop(5, "SEAL_NO_PINS", str(e))
        missing = [k for k in REQUIRED_PINS if not pins.get(k)] + [f"code:{fn}" for fn in OPERATIVE if f"code:{fn}" not in pins]
        if missing:
            self.stop(5, "SEAL_PINS_MISSING", str(missing))
        bad = []
        for k, v in pins.items():
            if k.startswith("code:"):
                p = os.path.join(c.code_dir, k[5:])
                if not os.path.isfile(p) or stepB.sha256_lf(p) != v:
                    bad.append(k)
        if bad:
            self.stop(5, "SEAL_CODE_PIN_MISMATCH", f"{bad}: a pinned code file changed after the seal; a change goes into a sealed amendment")
        if pins["carve_sha256"] != stepB.SEALED_CARVE_SHA256:
            self.stop(5, "SEAL_CARVE_PIN_VS_CODE", "the sealed carve sha256 differs from stepB.SEALED_CARVE_SHA256")
        l1 = {k[3:]: v for k, v in pins.items() if k.startswith("l1:")}
        if l1 != dict(stepB.L1_PINS):
            self.stop(5, "SEAL_L1_PINS", "the sealed L1 pins differ from stepB.L1_PINS")
        try:
            stepB.verify_l1_code()
        except Exception as e:
            self.stop(5, "SEAL_L1_CODE", str(e)[:300])
        for key, path, o4 in (("config_sha256", c.config, False), ("config_o4_sha256", c.config_o4, True)):
            if not os.path.isfile(path) or stepB.sha256_lf(path) != pins[key]:
                self.stop(5, "SEAL_CONFIG_PIN_MISMATCH", key)
            cfg = stepB.StepBConfig.from_json(path)
            stepB.validate_config(cfg)
            if o4 and cfg != stepB.c8_from_grid(m_pop=256, n_elite=32, widen=2.0):
                self.stop(5, "SEAL_O4_CONFIG", "the O4 config is not c8_from_grid(m_pop=256, n_elite=32, widen=2.0)")
        for s in ("s0", "s1"):
            p = f"{c.heads_dir}/final_head_{s}.npz"
            if not os.path.isfile(p) or sha256_file(p) != pins[f"head_{s}_sha256"]:
                self.stop(5, "SEAL_HEAD_PIN_MISMATCH", s)
        fh = f"{c.heads_dir}/final_heads.json"
        if os.path.isfile(fh):
            man = json.load(open(fh, encoding="utf-8")).get("heads", {})
            for s in ("s0", "s1"):
                if man.get(s, {}).get("sha256") != pins[f"head_{s}_sha256"]:
                    self.stop(5, "SEAL_HEAD_MANIFEST", f"final_heads.json disagrees with the sealed pin for {s}")
        self.pins = pins
        self.emit("ZZSTEPB_SEAL_OK", got)
        return pins

    def fresh_seal_ready(self) -> bool:
        """L2-FRESH's amendment: its .sha256 exists, equals the sha256 of its text, and its text names the same carve and heads as this seal (the two amendments are about one carve)"""
        c = self.cfg
        if not (os.path.isfile(c.fresh_sha) and os.path.isfile(c.fresh_prereg)):
            return False
        rec = open(c.fresh_sha, encoding="utf-8").read().split()
        text = open(c.fresh_prereg, "rb").read().replace(b"\r\n", b"\n")
        if not rec or rec[0] != hashlib.sha256(text).hexdigest():
            self.stop(5, "FRESH_SEAL_MISMATCH", "PREREG_L2_FRESH.sha256 does not match PREREG_L2_FRESH.md")
        t = text.decode("utf-8", "replace")
        for k in ("carve_sha256", "head_s0_sha256", "head_s1_sha256"):
            if self.pins[k] not in t:
                self.stop(5, "FRESH_SEAL_PIN_DISAGREES", f"{k} is not named in L2-FRESH's sealed text")
        if self.pins.get("fresh_prereg_sha256") and self.pins["fresh_prereg_sha256"] != rec[0]:
            self.emit("ZZSTEPB_NOTE", "L2-FRESH's seal sha256 differs from the one recorded at this seal (it was re-sealed); carve and head pins still agree")
        return True

    # ---------------------------------------------------------------------------------------------------------- (a) the wait for carve B
    def wait_for_b(self) -> None:
        c = self.cfg
        t0 = last_hb = self.clock.time()
        self.state("waiting_for_carve_B")
        self.emit("ZZSTEPB_WAIT_B", c.b_log)
        while True:
            txt = read_text(c.b_log)
            if txt is not None:
                m = FAIL_RE.search(txt)
                if m:
                    self.stop(3, "B_CHAIN_FAILED", txt[m.start():m.start() + 120].splitlines()[0])
                if DONE_RE.search(txt):
                    if self.fresh_seal_ready():
                        self.emit("ZZSTEPB_B_DONE")
                        return
                elif c.b_pid is not None and not self.alive_fn(c.b_pid):
                    txt2 = read_text(c.b_log) or ""                                       # re-read: it may have finished between the read and the liveness probe
                    if not DONE_RE.search(txt2) and not FAIL_RE.search(txt2):
                        self.stop(3, "B_CHAIN_GONE", f"pid {c.b_pid} is not running and the log has neither CHAIN_DONE_NO_READ nor a FAIL marker")
                    continue
            now = self.clock.time()
            if now - t0 > c.max_wait_s:
                self.stop(4, "WAIT_TIMEOUT", f"{c.max_wait_s / 3600:.1f} h without CHAIN_DONE_NO_READ (and L2-FRESH's seal)")
            if now - last_hb >= c.heartbeat_s:
                last_hb = now
                self.emit("ZZSTEPB_WAITING", f"{(now - t0) / 60:.0f} min")
            self.clock.sleep(c.poll_s)

    def check_b_gates(self) -> None:
        """the L2-FRESH gates (their HANDOFF): ZZBUNDLE_OK 400, the labels G1 identity (+ its control), the score gates G2a / G2c / G3 and the cross-device pick agreement, C-FEAT.  Reads gate JSONs only, never a label array."""
        c, W = self.cfg, self.cfg.b_dir
        fails = []
        t3 = read_text(f"{W}/step3.log") or ""
        if not BUNDLE_OK_RE.search(t3):
            fails.append("ZZBUNDLE_OK 400 (G2b, bit for bit) not in step3.log")

        def js(name):
            try:
                return json.load(open(f"{W}/{name}", encoding="utf-8"))
            except Exception as e:
                fails.append(f"{name}: unreadable ({type(e).__name__})")
                return None

        g1 = js("labels.npz.g1.json")
        if g1 is not None and not (g1.get("passed") is True and g1.get("control_ok") is True):
            fails.append("labels G1 (identity + control)")
        g = js("scores.npz.gates.json")
        if g is not None:
            for k, key in (("G2a", "passed"), ("G3", "passed"), ("G2c", "premise_true")):
                v = g.get(k)
                if not v or not all(x.get(key) is True for x in v):
                    fails.append(f"scores gate {k}")
            x = (g.get("G2b_cross_device") or [{}])[0].get("pick_agreement")
            if not isinstance(x, (int, float)) or x < 0.99:
                fails.append(f"G2b cross-device pick_agreement {x} < 0.99 (the GPU re-score contingency is a new decision, not automated)")
        cf = js("fresh_ep_scores.json")
        if cf is not None and cf.get("pass") is not True:
            fails.append("C-FEAT (fresh_ep_scores.json pass != true)")
        if fails:
            self.stop(3, "B_GATES_FAILED", "; ".join(fails))
        self.emit("ZZSTEPB_B_GATES_OK")

    # ---------------------------------------------------------------------------------------------------------- (b) the data clause + G-REACH-STEPB
    def check_bundle(self) -> dict:
        c = self.cfg
        for p in (c.bundle, c.ctx, c.ctx_json):
            if not os.path.isfile(p):
                self.stop(5, "BUNDLE_MISSING", p)
        b = np.load(c.bundle, allow_pickle=False)
        try:
            clause = stepB.check_data_clause({"token": b["token"]}, self.pins["carve_sha256"])
        except SystemExit as e:
            self.stop(5, "DATA_CLAUSE", str(e))
        meta = json.load(open(c.ctx_json, encoding="utf-8"))
        if meta.get("tokens_sha256") != self.pins["carve_sha256"] or int(meta.get("n", -1)) != c.n_expected or list(meta.get("shape", [-1]))[0] != c.n_expected:
            self.stop(5, "DATA_CLAUSE_CTX", "ctx.json's token-list sha256 / n / shape is not the sealed carve")
        if len(b["token"]) != c.n_expected:
            self.stop(5, "DATA_CLAUSE_N", f"n = {len(b['token'])} != {c.n_expected}")
        if sha256_file(c.ctx) != meta.get("ctx_sha256"):
            self.stop(5, "CTX_SHA", "ctx.npy's sha256 differs from ctx.json (a torn or changed ctx)")
        info = {"bundle_sha256": sha256_file(c.bundle), "ctx_sha256": meta["ctx_sha256"], "n": int(len(b["token"]))}
        self.emit("ZZSTEPB_DATA_CLAUSE_OK", self.pins["carve_sha256"][:12], f"n={info['n']}", f"bundle={info['bundle_sha256'][:12]}")
        return info

    def head_path(self, s: str) -> str:
        return f"{self.cfg.heads_dir}/final_head_{s}.npz"

    def stamp_path(self, s: str) -> str:
        return f"{self.cfg.run}/gate_{s}/G-REACH-STEPB_PASS_{self.pins[f'head_{s}_sha256'][:12]}.json"

    def stamp_ok(self, arm: str, s: str | None, info: dict) -> bool:
        p = self.stamp_path(s or "s0")
        try:
            stepB.check_stamp(p, arm=arm, head_sha256=self.pins[f"head_{s}_sha256"] if s else None, ckpt_sha256=self.pins["ckpt_sha256"], config_sha256=self.pins["config_sha256"], bundle_sha256=info["bundle_sha256"])
            return True
        except RuntimeError:
            return False

    def run_child(self, argv: list, log_name: str, label: bool = False) -> int:
        env = child_env(label)
        return self.runner(argv, env, f"{self.logs}/{log_name}.log", cwd=self.cfg.code_dir, on_pid=lambda pid: self.state(self._stage, child_pid=pid))

    def common_argv(self, info: dict, head_s: str | None, arm: str) -> list:
        c, P = self.cfg, self.pins
        a = ["--arm", arm]
        if head_s:
            a += ["--head", self.head_path(head_s), "--head-sha256", P[f"head_{head_s}_sha256"]]
        return a + ["--bundle", c.bundle, "--ctx", c.ctx, "--ckpt", c.ckpt, "--ckpt-sha256", P["ckpt_sha256"], "--refe-dir", c.refe_dir, "--config", c.config, "--config-sha256", P["config_sha256"]]

    def mint_gates(self, info: dict) -> None:
        c = self.cfg
        for s in ("s0", "s1"):
            if self.stamp_ok("L2", s, info):
                self.emit("ZZSTEPB_REACH_PASS", s, "(existing stamp, still binds this head, checkpoint, config, bundle and code)")
                continue
            self.state(f"gate_{s}")
            argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "g_reach_stepb.py")] + self.common_argv(info, s, "L2") + ["--out-dir", f"{c.run}/gate_{s}", "--threads", str(c.threads)]
            rc = self.run_child(argv, f"gate_{s}")
            if rc != 0 or not self.stamp_ok("L2", s, info):
                self.stop(3, "REACH_GATE_FAILED", f"head {s}: rc={rc}; no PASS stamp that binds this head: NO SEARCH IS RUN")
            self.emit("ZZSTEPB_REACH_PASS", s, self.stamp_path(s))

    # ---------------------------------------------------------------------------------------------------------- (c) the searches
    def search_file(self, name: str) -> str:
        return f"{self.cfg.run}/search_{name}.npz"

    def search_valid(self, name: str, arm: str, s: str | None, info: dict) -> bool:
        p = self.search_file(name)
        if not os.path.isfile(p):
            return False
        try:
            z = np.load(p, allow_pickle=False)
            m = json.loads(str(z["meta"]))
            ok = (m.get("arm") == arm and m.get("head_sha256") == (self.pins[f"head_{s}_sha256"] if s else None) and m.get("ckpt_sha256") == self.pins["ckpt_sha256"]
                  and m.get("config_sha256") == self.pins["config_sha256"] and m.get("bundle_sha256") == info["bundle_sha256"]
                  and (m.get("data_clause") or {}).get("carve_sha256") == self.pins["carve_sha256"] and int(m.get("n_tokens", -1)) == info["n"] and len(z["token"]) == info["n"]
                  and np.array_equal(np.asarray(z["bundle_index"]), np.arange(info["n"])))
            return bool(ok)
        except Exception:
            return False

    def run_searches(self, info: dict) -> dict:
        c = self.cfg
        files = {}
        for name, arm, s in ARMS:
            if self.search_valid(name, arm, s, info):
                self.emit("ZZSTEPB_SEARCH_DONE", name, "(existing merged file, still binds this head, checkpoint, config, bundle and carve)")
                files[name] = self.search_file(name)
                continue
            self.state(f"search_{name}")
            stamp = self.stamp_path(s or "s0")
            d = f"{c.run}/search_{name}"
            argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "stepB.py"), "search"] + self.common_argv(info, s, arm) + ["--gate", stamp, "--out-dir", d, "--threads", str(c.threads)]
            rc = self.run_child(argv, f"search_{name}")
            if rc != 0:
                self.stop(6, "SEARCH_FAILED", f"{name}: rc={rc} (the shards written so far are kept: rerunning the chain resumes)")
            rc = self.run_child([c.py_cpu, "-u", os.path.join(c.code_dir, "stepB.py"), "merge", "--out-dir", d, "--arm", arm, "--out", self.search_file(name), "--n", str(info["n"])], f"search_{name}")
            if rc != 0 or not self.search_valid(name, arm, s, info):
                self.stop(6, "MERGE_FAILED", f"{name}: rc={rc}, or the merged file does not bind this head / checkpoint / config / bundle / carve")
            files[name] = self.search_file(name)
            self.emit("ZZSTEPB_SEARCH_DONE", name)
        return files

    # ---------------------------------------------------------------------------------------------------------- (d) the labels
    def labels_file(self, name: str) -> str:
        return f"{self.cfg.run}/labels_{name}.npz"

    def labels_valid(self, name: str, search_path: str) -> bool:
        p = self.labels_file(name)
        if not os.path.isfile(p):
            return False
        try:
            z = np.load(p, allow_pickle=False)                                           # provenance fields only: never the label arrays
            ident = json.loads(str(z["identity"]))
            return bool(str(z["search_sha256"]) == sha256_file(search_path) and ident.get("passed") and ident.get("control_ok"))
        except Exception:
            return False

    def wait_ram(self) -> None:
        c = self.cfg
        t0 = self.clock.time()
        while True:
            av = self.ram_fn()
            if av >= c.min_ram_gb:
                return
            if self.clock.time() - t0 > c.ram_max_wait_s:
                self.stop(7, "RAM_TIMEOUT", f"available {av:.1f} GiB < {c.min_ram_gb} GiB for {c.ram_max_wait_s / 3600:.1f} h (the labeller is RAM-fragile)")
            self.clock.sleep(20)

    def run_labels(self, info: dict, files: dict) -> dict:
        c = self.cfg
        labs = {}
        for name, arm, s in ARMS:
            if self.labels_valid(name, files[name]):
                self.emit("ZZSTEPB_LABEL_DONE", name, "(existing labels, bound to this search file)")
                labs[name] = self.labels_file(name)
                continue
            self.state(f"label_{name}")
            self.wait_ram()
            argv = [c.py_label, "-u", os.path.join(c.code_dir, "label_stepB.py"), "--bundle", c.bundle, "--search", files[name], "--plan", c.plan, "--out", self.labels_file(name)]
            rc = self.run_child(argv, f"label_{name}", label=True)
            if rc == 3:
                self.stop(7, "LABEL_IDENTITY_FAILED", f"{name}: the planted identity or its control failed at the labeller; no labels were written")
            if rc != 0 or not self.labels_valid(name, files[name]):
                self.stop(7, "LABEL_FAILED", f"{name}: rc={rc}, or the labels are not bound to this search file")
            labs[name] = self.labels_file(name)
            self.emit("ZZSTEPB_LABEL_DONE", name)
        return labs

    # ---------------------------------------------------------------------------------------------------------- (e)(f) the registered analysis and the verdict line
    def run_analysis(self, info: dict, files: dict, labs: dict) -> dict:
        c = self.cfg
        if sha256_file(c.bundle) != info["bundle_sha256"]:
            self.stop(5, "BUNDLE_CHANGED", "the bundle changed during the run: no analysis")
        self.state("analysis")
        out = f"{c.run}/RESULT_STEPB.json"
        argv = [c.py_cpu, "-u", os.path.join(c.code_dir, "analyse_stepB.py"), "--bundle", c.bundle, "--out", out]
        for name, _, _ in ARMS:
            argv += ["--search", f"{name}={files[name]}"]
        for name, _, _ in ARMS:
            argv += ["--labels", f"{name}={labs[name]}"]
        rc = self.run_child(argv, "analysis")
        if rc != 0 or not os.path.isfile(out):
            self.stop(8, "ANALYSIS_FAILED", f"rc={rc}")
        res = json.load(open(out, encoding="utf-8"))
        v = res.get("verdict")
        if not v or v.get("combined") not in ("O1_PASS", "TRADE", "NOT_CLEARED", "O4_INERT") or not v.get("next_arm"):
            self.stop(8, "NO_VERDICT", "the analysis result carries no verdict with a named next arm")
        self.emit("ZZSTEPB_ANALYSIS_DONE", out)
        return v

    def emit_verdict(self, v: dict) -> None:
        self.emit("ZZSTEPB_DONE", v["combined"])
        self.emit("ZZSTEPB_PER_HEAD", "  ".join(f"{k}={d['outcome']} dPDMS={d['dPDMS_points']:+.3f}pts CI[{d['ci'][0]:+.3f},{d['ci'][1]:+.3f}] seed_range={d['seed_range_points']:.3f} moved={d['moved_share']:.3f}" for k, d in v["per_head"].items()))
        self.emit("ZZSTEPB_RULE", v["rule"])
        self.emit("ZZSTEPB_NEXT", v["next_arm"])
        self.emit("ZZSTEPB_LICENSING", v["licensing"])
        self.emit("ZZSTEPB_STAMPS", v["stamps"])
        self.emit("ZZSTEPB_STRATEGIC", v["STRATEGIC"])

    # ---------------------------------------------------------------------------------------------------------- the chain
    def run(self) -> int:
        c = self.cfg
        os.makedirs(self.logs, exist_ok=True)
        self.emit("ZZSTEPB_START", f"pid={os.getpid()}", f"cuda={os.environ.get('CUDA_VISIBLE_DEVICES')!r}", f"threads={c.threads}", f"run={c.run}")
        try:
            self.check_seal()                                  # refuses at once: no sealed text, no chain
            self.wait_for_b()
            self.check_b_gates()
            self.check_seal()                                  # again, after a wait of hours
            info = self.check_bundle()
            self.mint_gates(info)                              # both PASS stamps before ANY search
            files = self.run_searches(info)
            labs = self.run_labels(info, files)
            v = self.run_analysis(info, files, labs)
            self.state("done", verdict=v["combined"])
            self.emit_verdict(v)
            return 0
        except StopChain as e:
            self.state("stopped", reason=e.reason, detail=e.detail[:500])
            return e.code
        except (Exception, SystemExit) as e:                    # a bug or a child-library refusal must still end in a marker, never in a silent traceback
            self.emit("ZZSTEPB_STOP", "CRASH", f"{type(e).__name__}: {str(e)[:300]}")
            self.state("crashed", reason=type(e).__name__, detail=str(e)[:500])
            return 9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pkg", default=PKG)
    ap.add_argument("--fresh", default=FRESH)
    ap.add_argument("--b-dir", default=B_DIR)
    ap.add_argument("--b-log", default=None)
    ap.add_argument("--b-pid", type=int, default=None, help="the L2-FRESH chain's cmd PID: if it is gone and its log has neither CHAIN_DONE_NO_READ nor a FAIL marker, STOP")
    ap.add_argument("--max-wait-hours", type=float, default=36.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    a = ap.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # the log is read by cmd redirects (cp1252 by default)
    except Exception:
        pass
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    cfg = Cfg(pkg=a.pkg, fresh=a.fresh, b_dir=a.b_dir, b_log=a.b_log, b_pid=a.b_pid, max_wait_s=a.max_wait_hours * 3600.0, poll_s=a.poll_s)
    return Chain(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
