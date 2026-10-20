#!/usr/bin/env python3
"""HNSCREEN-FP-COST at CHAIN level: the P6-HN screen chain with the REAL launch gate (anchor / run / verify / end-verify, real HMAC-signed tokens, real anchor and END records), a REAL sealed cache, stub trainer and stage tools.
(MM_RULING_FPCOST.md; the gate-level arms are refe/test_hn_ctx_anchor.py.)  Sibling-suite test (it needs eval/ for the landed GPU lock, like test_hnscreen_chain.py, which it reuses: World, Sc, make_cfg, run_chain).

Run:  PYTHONIOENCODING=utf-8 python refe/test_hnscreen_ctx_chain.py   ->   ZZTEST_PASS <n>   (arms, then the mutation matrix on hnscreen_devbox.py)      --arms-only   arms only

WHAT IS REAL AND WHAT IS STUBBED.  REAL: the chain (`hnscreen_devbox.Chain`, from source), `launch_gate.py` run via a shim process (`anchor`, `verify`, `end-verify` unchanged; `run` is the real `cmd_run` -> the real `finalize` and the real
`profile_refusals`, with `run_checks`' GPU/pod checks replaced by synthetic PASS evidence -- the cache-touching lines of `run_checks` (the data manifest and the refusals) are kept, so the reads are the real reads), the cache
(built by `ctx_cache.finalize`), the reader's argument parsing and cache-binding gate (the shim imports the landed eval/hn_screen_read.py and stops it at the first score file).  STUBBED: the trainer, S0/S1 builders, I1-I6/G15, S4 scorers.
Every gate process appends {sub, reads, files} to a log: a READ is one `gate_token._sha256_and_size` call (one file fully hashed), so reads / n_files = FULL PASSES over the cache.

THE ARMS  C_* : the anchored chain end to end (the default mode; stage order, the calls each stage makes, ZERO reads at the intermediate calls, 1 + 1 at the ends, the END_VERIFY_OK record, the reader gate admits);
 the content-mode chain is today's chain (stage order, gate calls, 6 full reads per seed); R1 mid-chain same-size same-mtime edit -> ENDV fails -> VOID -> no outputs read -> a resume cannot launder it; R2 a size change between two
 passes -> the next intermediate call refuses and the trainer is not started; R5 the END record deleted -> the reader refuses the chain's own read command; a refused anchor trains nothing; resume keeps the chain; a bad ctx_mode is a
 config error.  Every mutation edits hnscreen_devbox.py SOURCE (each snippet occurs exactly once) and must turn the named arms RED.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_PKG = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import test_hnscreen_chain as TC  # noqa: E402  -- the chain harness this file builds on

PY = TC.PY
SRC_PATH = TC.SRC_PATH
ARMS_ONLY = "--arms-only" in sys.argv
ONLY = set(sys.argv[sys.argv.index("--only") + 1].split(",")) if "--only" in sys.argv else None

GATE_SHIM = r'''
import json, os, sys, time
REFE = os.environ["SHIM_REFE"]
sys.path.insert(0, REFE)
import gate_token as GT
import launch_gate as L
import gate_checks as C
L.PROFILES["refe_hnscreen"] = {**L.PROFILES["refe_hnscreen"], "fingerprints_expected": json.loads(os.environ["SHIM_FX"])}
reads = []
_real = GT._sha256_and_size


def counting(path, *a, **k):
    reads.append(os.path.abspath(str(path)))
    return _real(path, *a, **k)


GT._sha256_and_size = counting


def fake_run_checks(prof, argv, pkg, out_dir, *, only=None, arm=None, commit=None, preflight_json=None, **kw):
    """launch_gate.run_checks with the pod/GPU checks replaced by synthetic PASS evidence; the lines that TOUCH THE CACHE (the data manifest, the profile refusals) are run exactly as run_checks runs them"""
    out_dir.mkdir(parents=True, exist_ok=True)
    tree = GT.tree_manifest(pkg)
    binding = GT.binding_of(prof["name"], commit, GT.tree_digest(tree), argv)
    data = GT.data_manifest(argv, prof["data_flags"], L.gate_inputs_of(None, None, None, None, preflight_json), content_flags=L.content_flags_of(prof), ctx_anchor=prof.get("ctx_anchor"))
    refusals = L.profile_refusals(prof, argv)
    status = {}
    for c in prof["checks"] + tuple(prof.get("na_ok", ())):
        ev = C.evidence(c, "FAIL" if refusals else "PASS", ["profile refusal: " + r for r in refusals], {}, arm=arm)
        L.write_evidence(out_dir, ev, binding, data)
        status[c] = ev["status"]
    return status


L.run_checks = fake_run_checks
sub = sys.argv[1]
args = sys.argv[1:] + (["--commit", "a" * 40] if sub == "run" else [])
rc = L.main(args)
with open(os.environ["SHIM_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps({"sub": sub, "reads": len(reads), "files": len(set(reads)), "rc": rc, "args": [a for a in sys.argv[2:] if a.startswith("--ctx-mode")], "t": time.time()}) + "\n")
sys.exit(rc)
'''

READER_SHIM = r'''
import os, sys
sys.path.insert(0, os.environ["STUB_DIR"])
sys.path.insert(0, os.path.join(os.environ["SHIM_REAL_PKG"], "eval"))
sys.path.insert(0, os.environ["SHIM_REFE"])
import stublib
import hn_screen_read as HR


class Past(Exception):
    pass


def stop(*a, **k):
    raise Past()


HR.RV.load_scores = stop                                     # the landed reader's gate has run by the time it asks for the first score file
try:
    rc = HR.main(sys.argv[1:])
except Past:
    rc = stublib.main("read", sys.argv[1:])
sys.exit(rc)
'''

TAMPER = r'''
import os, sys
mode, cache, only_seed, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
if only_seed != "any" and only_seed != seed:
    sys.exit(0)
p = sorted(os.scandir(os.path.join(cache, "units")), key=lambda e: e.name)[0].path
for e in sorted(os.scandir(os.path.join(cache, "units")), key=lambda e: e.name):
    if e.name.endswith(".ctx.u16"):
        p = e.path
        break
st = os.stat(p)
b = bytearray(open(p, "rb").read())
if mode == "flip":                                           # same size, same mtime_ns
    b[len(b) // 2] ^= 1
elif mode == "grow":                                         # a size change, mtime_ns restored
    b += b"\0"
open(p, "wb").write(bytes(b))
os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
print("TAMPERED", mode, p)
'''


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception as e:                                            # noqa: BLE001 -- an exception IS a failed check
        ok = False
        if os.environ.get("HNCC_DEBUG"):
            import traceback
            traceback.print_exc()
            print("  [check raised]", name, type(e).__name__, str(e)[:300])
    if not ok:
        bad.append(name)
    return ok


class Real:
    """the real fixtures of the whole run: a sealed cache (pristine), the shims, the key, the registered pins, the real argv template"""

    def __init__(self):
        import gate_token as GT
        import gate_checks as C
        import ctx_cache as CC
        import test_ctx_cache as TCC
        import torch  # noqa: F401
        self.GT, self.C = GT, C
        self.w = TC.World()
        base = self.w.base
        units = TCC.mk_units()
        model = TCC.Fake().eval()
        fp = CC.trunk_fingerprint(model.state_dict())
        self.pristine = os.path.join(base, "pristine_cache")
        CC.build_units(model, units, TCC.load_batch, TCC.make_forward(model), self.pristine, trunk_fp=fp, log=lambda *_: None)
        CC.finalize(self.pristine, units, trunk_fp=fp, calib_sha256="calib-1")
        self.n_files = GT.content_fingerprint(self.pristine)["n_files"]
        self.cache_bytes = GT.content_fingerprint(self.pristine)["bytes"]
        self.pin = GT.fp_key(GT.content_fingerprint(self.pristine))
        self.fx = {"--hn-bank": "file:" + hashlib.sha256(b"BANK").hexdigest(), "--ctx-cache": self.pin}      # the stub S1 writes the bank "BANK"
        self.keyf = os.path.join(base, "gate.key").replace("\\", "/")
        self.shim = os.path.join(base, "gate_shim.py")
        open(self.shim, "w", encoding="utf-8").write(GATE_SHIM)
        self.tamper = os.path.join(base, "tamper.py")
        open(self.tamper, "w", encoding="utf-8").write(TAMPER)
        rd = os.path.join(self.w.fake, "eval", "hn_screen_read.py")           # the chain's reader: the landed one behind a one-file stop (see the module doc)
        open(rd, "w", encoding="utf-8").write(READER_SHIM)
        self.dummy = os.path.join(base, "dummy_input.bin").replace("\\", "/")
        open(self.dummy, "wb").write(b"an input the pod would hold; only its existence matters here")
        import launch_gate as L
        c1 = list(json.load(open(os.path.join(REAL_PKG, "eval", "rls1b_arms.json"), encoding="utf-8"))["arms"]["C1"]["argv"])
        for f_ in L.PROFILES["refe_hnscreen"]["data_flags"]:
            if f_ in c1 and f_ not in L.HNSCREEN_FLAGS:
                c1 = C._set(c1, f_, self.dummy)
        c1 = C._set(C._set(C._set(c1, "--seed", "{seed}"), "--out", "{out}"), "--dump-scores", "{scores}")
        self.template = c1 + ["--hn-bank", "{bank_train}", "--ctx-cache", "{cache_dir}"]

    def close(self):
        self.w.close()


def scenario(R, M, name, *, seeds=(0,), mode="default", tamper=None, spec=None, stop_after=None, pre=None, cfg_extra=None):
    """one chain run: a fresh state dir, a fresh COPY of the sealed cache, the shim gate.  tamper = (mode, seed|'any') runs after the trainer of that pass.  -> (sc, rc, chain, shim log rows)"""
    sc = TC.Sc(R.w, name)
    cache = f"{sc.d}/cache"
    shutil.copytree(R.pristine, cache)
    if pre:
        pre(cache)
    log = f"{sc.d}/shim.jsonl"
    os.environ.update({"SHIM_REFE": HERE, "SHIM_FX": json.dumps(R.fx), "SHIM_LOG": log, "SHIM_REAL_PKG": REAL_PKG})
    over = {"python": sys.executable, "gate.cmd": [sys.executable, R.shim], "gate.key_file": R.keyf, "trainer.seeds": list(seeds), "trainer.argv_template": R.template, "commands.s2_cache": [PY, sc.w.stub, "step", "s2_cache"]}      # NO gate.token_check_names override (HNSCREEN-TOKEN-NAMES): the REAL mint writes the profile's checks, and the chain's DEFAULT names must find G14 / G15 in them
    over["gate.ctx_mode"] = None if mode == "default" else mode
    doc = TC.make_cfg(sc, **over)
    if tamper:
        doc["commands"]["post_pass"].append({"name": "tamper", "argv": [PY, R.tamper, tamper[0], cache, str(tamper[1]), "{seed}"]})
    for k, v in (cfg_extra or {}).items():
        doc[k] = v
    rc, ch, clk = TC.run_chain(M, sc, spec={**TC.PARTIAL2, **(spec or {})}, cfg_doc=doc, stop_after=stop_after)
    rows = [json.loads(x) for x in open(log, encoding="utf-8")] if os.path.exists(log) else []
    return sc, rc, ch, rows, cache


def passes(R, rows, sub=None):
    return sum(r["reads"] for r in rows if sub is None or r["sub"] == sub) / R.n_files


def arms(R, M, only=None):
    bad = []
    n = [0]

    def chk(name, fn):
        if only is not None and name not in only:
            return
        n[0] += 1
        check(bad, name, fn)
    GT = R.GT

    def key():
        return GT.load_key(R.keyf, create=False)

    def subs(rows):
        return [r["sub"] for r in rows]

    def c_end_to_end():
        sc, rc, ch, rows, cache = scenario(R, M, "C1", seeds=(0,))                # NO gate.ctx_mode in the config: the DEFAULT must be anchor-bound
        tok = sorted(__import__("glob").glob(f"{sc.root}/passes/s0/lg/PASS_*.json"))
        d = json.load(open(tok[-1], encoding="utf-8"))["binding"]["data"]["--ctx-cache"] if tok else {}
        end = json.load(open(f"{sc.root}/ctx_chain/end_verify.json", encoding="utf-8"))
        rd = [r for r in sc.J() if r["label"] == "read"]
        return (rc == 0 and ch.stage_order_run == ["S0", "S1", "S2", "ANCHOR", "T0", "P0", "ENDV", "S4", "VERDICT", "NEXT"] and subs(rows) == ["anchor", "run", "verify", "verify", "end-verify"]
                and [r["reads"] for r in rows] == [R.n_files, 0, 0, 0, R.n_files] and passes(R, rows) == 2.0 and d.get("kind") == "dir-anchor" and d.get("mode") == "anchor-bound"
                and end["status"] == "END_VERIFY_OK" and end["seeds"] == [0] and os.path.isfile(f"{sc.root}/ctx_chain/anchor.json") and os.path.isfile(f"{sc.root}/ctx_chain/chain.json")
                and len(rd) == 1 and "--chain-root" in rd[0]["argv"] and R.keyf in rd[0]["argv"] and GT.ctx_chain_refusals(sc.root, [0], key()) == []
                and all(r["args"] == ["--ctx-mode"] for r in rows if r["sub"] in ("run", "verify")))
    chk("C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it", c_end_to_end)

    def c_content():
        sc, rc, ch, rows, cache = scenario(R, M, "C2", seeds=(0,), mode="content")
        tok = sorted(__import__("glob").glob(f"{sc.root}/passes/s0/lg/PASS_*.json"))
        d = json.load(open(tok[-1], encoding="utf-8"))["binding"]["data"]["--ctx-cache"]
        today = [x for x in TC.expected_labels(seeds=(0,)) if not x.startswith("gate:")]
        return (rc == 0 and ch.stage_order_run == ["S0", "S1", "S2", "T0", "P0", "S4", "VERDICT", "NEXT"] and subs(rows) == ["run", "verify", "verify"] and [r["reads"] for r in rows] == [4 * R.n_files, R.n_files, R.n_files]
                and all(r["args"] == [] for r in rows) and d["kind"] == "dir-content" and not os.path.exists(f"{sc.root}/ctx_chain") and sc.labels() == today
                and GT.ctx_chain_refusals(sc.root, [0], key()) == [])
    chk("C_R7_forcing_content_mode_is_todays_chain_same_stages_same_gate_calls_no_anchor_6_full_reads_per_seed_and_the_reader_admits_it", c_content)

    def c_r1():
        sc, rc, ch, rows, cache = scenario(R, M, "C3", seeds=(0,), tamper=("flip", 0))
        end = json.load(open(f"{sc.root}/ctx_chain/end_verify.json", encoding="utf-8"))
        outcome = json.load(open(f"{sc.root}/outcome.json", encoding="utf-8"))
        ok1 = (rc == 2 and subs(rows) == ["anchor", "run", "verify", "verify", "end-verify"] and end["status"] == "VOID" and os.path.isfile(f"{sc.root}/ctx_chain/VOID.json") and outcome["row"] == 0
               and "read" not in sc.labels() and "S4" not in ch.stage_order_run and "VOID" in GT.ctx_chain_refusals(sc.root, [0], key())[0])
        # the residual is real: nothing before the end verify saw it; and a hand-restore cannot launder the voided chain on a relaunch
        p = sorted(__import__("glob").glob(f"{cache}/units/*.ctx.u16"))[0]
        st = os.stat(p)
        b = bytearray(open(p, "rb").read())
        b[len(b) // 2] ^= 1
        open(p, "wb").write(bytes(b))
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))                              # the cache is byte-identical to the anchor again
        rc2, ch2, _ = TC.run_chain(M, sc, spec={**TC.PARTIAL2}, cfg_doc=json.load(open(f"{sc.d}/config.json", encoding="utf-8")))
        rows2 = [json.loads(x) for x in open(f"{sc.d}/shim.jsonl", encoding="utf-8")]
        return ok1 and rc2 == 2 and subs(rows2) == subs(rows) and "read" not in sc.labels() and os.path.isfile(f"{sc.root}/ctx_chain/VOID.json")
    chk("C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it", c_r1)

    def c_r2():
        sc, rc, ch, rows, cache = scenario(R, M, "C4", seeds=(0, 1), tamper=("grow", 0))
        lab = sc.labels()
        return (rc == 5 and "trainer:train:0" in lab and "trainer:train:1" not in lab and "read" not in lab and subs(rows)[-1] == "verify" and rows[-1]["rc"] == 5 and "end-verify" not in subs(rows))
    chk("C_R2_a_size_change_after_pass_0_refuses_the_start_verify_of_pass_1_and_the_trainer_is_not_started", c_r2)

    def c_r5():
        sc, rc, ch, rows, cache = scenario(R, M, "C5", seeds=(0,))
        argv = ch.read_argv([0])
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        env.pop("HF_TOKEN", None)
        ok_run = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, encoding="utf-8", errors="replace")
        os.remove(f"{sc.root}/ctx_chain/end_verify.json")                                # the end verify deleted
        gone = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, encoding="utf-8", errors="replace")
        return (rc == 0 and ok_run.returncode == 0 and gone.returncode == 6 and "ZZHNSCREEN_REFUSED" in gone.stdout and "no END_VERIFY_OK record" in gone.stdout)
    chk("C_R5_the_chains_own_read_command_is_refused_by_the_landed_reader_once_the_end_record_is_deleted", c_r5)

    def c_refused_anchor():
        def flip(cache):
            p = sorted(__import__("glob").glob(f"{cache}/units/*.ctx.u16"))[0]
            st = os.stat(p)
            b = bytearray(open(p, "rb").read())
            b[0] ^= 1
            open(p, "wb").write(bytes(b))
            os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        sc, rc, ch, rows, cache = scenario(R, M, "C6", seeds=(0,), pre=flip)
        lab = sc.labels()
        return rc == 5 and subs(rows) == ["anchor"] and rows[0]["rc"] == 5 and "trainer:pre:0" not in lab and "trainer:train:0" not in lab and not os.path.exists(f"{sc.root}/ctx_chain/anchor.json")
    chk("C_a_tampered_cache_is_refused_at_the_anchor_and_nothing_is_trained", c_refused_anchor)

    def c_resume():
        sc, rc, ch, rows, cache = scenario(R, M, "C7", seeds=(0,), stop_after="T0")
        info1 = json.load(open(f"{sc.root}/ctx_chain/chain.json", encoding="utf-8"))
        a1 = open(f"{sc.root}/ctx_chain/anchor.json", "rb").read()
        sha_cfg = json.load(open(f"{sc.d}/config.json", encoding="utf-8"))
        rc2, ch2, _ = TC.run_chain(M, sc, spec={**TC.PARTIAL2}, cfg_doc=sha_cfg)
        rows2 = [json.loads(x) for x in open(f"{sc.d}/shim.jsonl", encoding="utf-8")]
        info2 = json.load(open(f"{sc.root}/ctx_chain/chain.json", encoding="utf-8"))
        return (rc == 0 and rc2 == 0 and subs(rows)[:2] == ["anchor", "run"] and subs(rows2).count("anchor") == 1 and info1 == info2 and open(f"{sc.root}/ctx_chain/anchor.json", "rb").read() == a1
                and subs(rows2) == ["anchor", "run", "verify", "verify", "end-verify"] and ch2.stage_order_run == ["P0", "ENDV", "S4", "VERDICT", "NEXT"])
    chk("C_a_resumed_chain_keeps_its_chain_and_its_anchor_and_ends_with_one_end_verify", c_resume)

    def c_bad_mode():
        sc, rc, ch, rows, cache = scenario(R, M, "C8", seeds=(0,), mode="bogus")
        return rc == 4 and rows == [] and sc.labels() == []
    chk("C_an_unknown_ctx_mode_is_a_config_error_before_anything_runs", c_bad_mode)

    def c_two_seeds_reads():
        sc, rc, ch, rows, cache = scenario(R, M, "C9", seeds=(0, 1))
        return (rc == 0 and subs(rows) == ["anchor", "run", "verify", "run", "verify", "verify", "verify", "end-verify"] and passes(R, rows) == 2.0)
    chk("C_two_seeds_still_cost_exactly_two_full_reads_the_anchor_and_the_end_verify", c_two_seeds_reads)
    return bad, n[0]


def read_src():
    return open(SRC_PATH, encoding="utf-8").read()


MUTATIONS = [
    ("stage_list: ENDV is not a stage (the end verify is skipped)",
     [("+ [f\"P{s}\" for s in seeds] + ([\"ENDV\"] if self.anchored else []) + [\"S4\"", "+ [f\"P{s}\" for s in seeds] + [\"S4\"")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it",
      "C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it"]),
    ("stage_ENDV: a failed end verify does not void the chain",
     [("        if rc != 0 or END_OK_MARK not in text or status != \"END_VERIFY_OK\":\n", "        if False:\n")],
     ["C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it"]),
    ("stage_ENDV: a VOID chain is re-verified on a relaunch (a hand-restored cache launders it)",
     [("        if os.path.exists(void):\n", "        if False:\n")],
     ["C_R1_a_same_size_same_mtime_edit_mid_chain_voids_the_chain_no_output_is_read_and_a_relaunch_cannot_launder_it"]),
    ("ctx_args: the intermediate gate calls get no anchor flags (every call re-reads in full)",
     [("        if not self.anchored:\n            return []\n        ci = self.chain_info()\n        return [\"--ctx-mode\"", "        return []\n        ci = self.chain_info()\n        return [\"--ctx-mode\"")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it", "C_two_seeds_still_cost_exactly_two_full_reads_the_anchor_and_the_end_verify"]),
    ("read_argv: the reader is not given --chain-root",
     [(" + [\"--chain-root\", self.root] + self.gate_key_args()", "")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it"]),
    ("stage_list: ANCHOR is not a stage (the chain has no start anchor)",
     [("(\"ANCHOR\"] if self.anchored else [])".replace("(\"", "([\""), "[]")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it"]),
    ("preflight: any gate.ctx_mode is accepted",
     [("        if self.cfg.get(\"gate.ctx_mode\") not in CTX_MODES:\n", "        if False:\n")],
     ["C_an_unknown_ctx_mode_is_a_config_error_before_anything_runs"]),
    ("DEFAULTS: the chain's default gate.ctx_mode is content",
     [("\"ctx_mode\": \"anchor-bound\"},", "\"ctx_mode\": \"content\"},")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it"]),
    ("DEFAULTS: gate.token_check_names is the pre-fix G14 / G15 pair (no real token has those checks, so G14 = G15 = False)",
     [("TOKEN_CHECK_NAMES = {\"G14\": \"G-PREFLIGHT\", \"G15\": \"G-PREFLIGHT\"}\n", "TOKEN_CHECK_NAMES = {\"G14\": \"G14\", \"G15\": \"G15\"}\n")],
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it"]),
    ("stage_ANCHOR: a refused anchor does not stop the chain",
     [("        if rc != 0 or ANCHOR_OK not in text or not os.path.isfile(out):\n", "        if False:\n")],
     ["C_a_tampered_cache_is_refused_at_the_anchor_and_nothing_is_trained"]),
    ("verify_pass_token: the start verify of a pass is not run with the anchor flags (a stale-token-only check)",
     [("        cmd += self.ctx_args()                                                              # HNSCREEN-FP-COST: [] in content mode\n        r = self.runner.run(cmd, f\"{self.dirs['logs']}/token_verify_s{seed}.log\")",
       "        r = self.runner.run(cmd, f\"{self.dirs['logs']}/token_verify_s{seed}.log\")")],
     # ONLY the cost arm can see this mutant, by design: without the anchor flags the start verify runs in CONTENT mode, and a content verify refuses a size change too, so R2's outcome (rc 5, the trainer
     # not started) is the same in both modes.  The mutant is a COST regression (the intermediate call re-reads the cache in full); the arm that counts the full reads is the one that must go RED.
     # (An earlier version of this table also demanded R2 and failed the matrix with "expected RED and stayed green" -- an over-specified expectation, not an undetected mutant.)
     ["C_the_anchored_chain_end_to_end_the_default_mode_zero_reads_at_the_intermediate_calls_one_at_each_end_and_the_reader_admits_it"]),
]


def apply_edits(src, edits, name):
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise AssertionError(f"mutation {name!r}: snippet occurs {n} times (need exactly 1): {old[:90]!r}")
        src = src.replace(old, new)
    return src


def main() -> int:
    R = Real()
    try:
        M = TC.load_module(read_src(), "real")
        bad, n = arms(R, M, ONLY)
        if bad:
            print("\n".join("PROBLEM the unmutated chain FAILS: " + b for b in bad))
            print(f"ZZTEST_FAIL {len(bad)} of {n} arm(s) failed on the real chain")
            return 1
        if ARMS_ONLY or ONLY is not None:
            print(f"ZZTEST_PASS {n}  ({n} chain arms; the mutation matrix was not run)")
            return 0
        print(f"{n} chain arms green on the real chain.  Mutation matrix (each mutant is hnscreen_devbox.py with ONE source edit; it runs the arms it must turn RED):")
        problems = []
        for k, (name, edits, expect) in enumerate(MUTATIONS, 1):
            try:
                Mm = TC.load_module(apply_edits(read_src(), edits, name), f"m{k}")
                red, _ = arms(R, Mm, set(expect))
                red = list(red)
            except AssertionError as e:
                problems.append(str(e))
                continue
            except Exception as e:                                                # noqa: BLE001 -- a mutant that cannot even run its arms is RED, and said so
                red = [f"arms raised {type(e).__name__}: {str(e)[:80]}"]
            miss = [c for c in expect if c not in red]
            if not red:
                problems.append(f"mutation {name!r} stayed GREEN: no arm can see it")
            elif miss and not any(r.startswith("arms raised") for r in red):
                problems.append(f"mutation {name!r}: expected RED and stayed green: {miss} (RED: {red[:6]})")
            print(f"{name[:100]:100s} RED {len(red):2d}  e.g. {', '.join(red[:1])[:110]}", flush=True)
        if problems:
            print("\n".join("PROBLEM " + p for p in problems))
            print(f"ZZTEST_FAIL {len(problems)} problem(s)")
            return 1
        print(f"ZZTEST_PASS {n + len(MUTATIONS)}  ({n} chain arms + {len(MUTATIONS)} RED mutations of hnscreen_devbox.py)")
        return 0
    finally:
        R.close()


if __name__ == "__main__":
    sys.exit(main())
