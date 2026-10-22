#!/usr/bin/env python3
"""Build a SELECTION-TIER BANK for any REFe checkpoint on S0's tokens, in S0's exact format (`<unit>.sets.npz` + `.fan.npz` + `.report.json` + `MANIFEST.sha256.json`), so that `ep_admission.py` can
read it against P2's bank.  A parameterised wrapper around the LANDED S0 machinery (SR-TTC S0, `raw/2026-10-18-sel-rank/code/`): nothing of the dump, the gates, the staging or the resume logic is
re-implemented.  The landed scripts are imported / run BY PATH after a sha256 check; only four module constants of the landed driver are overridden (SRC, CKPT, CKPT_SHA256, the lock OWNER).

    all sub-commands run under  C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I  (numpy, torch, psutil); `-I` drops the script directory from sys.path, this file adds its own.

    preflight   --ckpt C [--ckpt-sha256 X]                       CPU, seconds: the checkpoint is a REFe checkpoint of P2's architecture (score_head 256 -> 6, same tensor names / shapes), sha256 recorded
    prepare     --root R --ckpt C [--ckpt-sha256 X]               lay out R (run/plan, work/{bank,sets}), copy S0's registered plan (sha256-verified against S0's manifest), write R/candidate.json
    sets        --root R [--skip-fetch]                          fetch the 15,808 fit sets + P2's PDM labels from the (idle) pod and build R/work/sets/<log>.npz   (the LANDED srttc_s0_podsets.py)
    verify-sets --root R                                         the rebuilt set stores equal S0's labels6 / ckpt_step on every key (catches a changed pod store BEFORE hours of GPU)
    dump        --root R [--device cuda|cpu] [--no-stage]         the LANDED S0 driver, resumable, on the candidate checkpoint (GPU behind the shared dev-box lock; this wrapper never takes a lock itself)
    launch      --root R                                         print the detached (WMI) PowerShell body for `dump`, like S0's launch_s0.ps1
    status      --root R                                         markers, counts, last STATUS lines
    manifest    --root R                                         MANIFEST.sha256.json of the bank (sha256 of every unit file) + the manifest's own sha256, which the registration must NAME
    verify      --root R [--p2-bank B]                           the finished bank is on P2's sets with P2's labels (C-KEYS / C-LABELS), finite, identity controls == 0
    smoke       --ckpt C [--n-tokens 3] [--device cpu]            CPU smoke: the landed dump script, FAN path, on a few phase-L tokens, vs S0's banked fan logits (tolerance 5e-3, the landed CPU-vs-GPU one)

NO GPU is used by any sub-command except `dump --device cuda`, and nothing here is run on a GPU by the package that ships it.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True                  # never write __pycache__ next to the landed (read-only) S0 code
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

S0_ROOT = "C:/Users/Admin/srttc_s0"             # READ ONLY: S0's plan, bank, code
S0_CODE = f"{S0_ROOT}/code"
P2_BANK = f"{S0_ROOT}/work/bank"
P2_CKPT = "D:/Projects/TanitAD/data/refe_runs_eval/p2/model_final.pt"
P2_CKPT_SHA256 = "ea31db2296210b6a35dbf32ad95f1813edfa259ab330002b3ca7564aac727e51"
DRIVERL_PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
TREE = os.environ.get("P2HO_TREE", "C:/Users/Admin/p2heldout/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan")
BANK_ON = "D:/Projects/TanitAD-artifacts"       # R belongs on D: (C: is nearly full; the S0 driver stops below 5 GiB free on the work drive)
N_TOKENS, N_SETS, N_LOGS = 8922, 15808, 117     # S0's registered sample
# sha256 of the LANDED S0 code this wrapper runs (identical on the dev box and at the tip 807a29c, compared 2026-10-19)
# P6-READ-GAPS G4 (2026-10-22): srttc_s0_dump.py has TWO accepted forms. The landed one refuses every command-mode checkpoint; the G4 one (raw/2026-10-18-sel-rank/code) adds `--commands`
# (the P6 training command table) and is otherwise the landed script (a goal-mode run is unchanged). Both are CRLF files; the pins are raw sha256.
S0_DUMP_LANDED_SHA256 = "1b41a9fbc21426f949dd4be2362dda467a7d04c630790df9a703cb2b184d2ac3"
S0_DUMP_G4_SHA256 = "28e5e56bf2f2b15825a74b81549ea2f078bd7e86bd95c24047d0f1bbfea4441f"
S0_PINS = {
    "srttc_s0_driver.py": "117dc228f13b6260b3d95f9525c99be3137100845df755f00b3a12d6f8a70f06",
    "srttc_s0_dump.py": (S0_DUMP_LANDED_SHA256, S0_DUMP_G4_SHA256),        # goal-mode (P2): either; a command-mode candidate needs the G4 one (apply_command_source)
    "srttc_s0_stage.py": "cb2a42120768c40903a061d06307368cb15c949400a162431911c8c0ffeb287a",
    "srttc_s0_podsets.py": "87498ec421f8128364314996ad18c416b197ca6d77b78c39fc475a9c1272df6b",
    "srttc_s0_dgdump.py": "b0fd031294e4ca249fdf4230bafe29b831c127ce67b678a604ed33a2619dba29",
    "srttc_s0_audit.py": "a64da38c8653bb412225ef9c6fc0a694bb05d3a34081ca356d0207977cb84645",
    "srttc_s0_plan.py": "4d30c676709765aa96ce84cc79b618456c1ac52860d697b073dd124af196c292",
    "podcopy.py": "d0a2f355b5cff6b4bafd524030207498a2a3c51fb12ea24bcf0c2d2119847db1",
    "landed/c3_run_p2full.py": "3604949a51c64cd4e352104d1564a0445187acc0fffe80f384b1e3d9bfd39df0",
}
PLAN_FILES = ("master_plan.json", "ranked_fit_logs.tsv", "selection.json")      # copied from S0's run/plan; podsets_provenance.json is rebuilt by `sets`
RANKED_LOGS_SHA256 = "a61275037a3d2b06903dc7e1ad0f6f99fb740f668a6ec7488fdef6fe901ce9b1"
SMOKE_LOGIT_TOL = 5e-3                          # the landed DG-DUMP CPU-vs-GPU tolerance (srttc_s0_dgdump.py; CPU-vs-GPU noise measured 1.74e-3 in its first smoke)
SMOKE_NEG_MIN = 5e-2                            # the negative control (another token's banked logits) must differ by more than this
MIN_RAM_GB_CPU = 4.0
SELF_FILES = ("build_candidate_bank.py", "ep_admission.py")


class BuildError(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jdump(o, p):
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(o, f, indent=1, sort_keys=True, default=str)


def free_gb(path):
    d = path
    while d and not os.path.exists(d):
        d = os.path.dirname(d)
    return shutil.disk_usage(d or ".").free / 2 ** 30


def ram_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def check_s0_pins(s0_code=S0_CODE):
    """Refuse unless every landed S0 script this wrapper runs hashes to its pin (a changed driver would change what a bank means)."""
    bad, got = [], {}
    for rel, want in S0_PINS.items():
        ok = (want,) if isinstance(want, str) else tuple(want)
        p = os.path.join(s0_code, rel)
        if not os.path.exists(p):
            bad.append(f"{rel}: missing under {s0_code}")
        elif sha256_file(p) not in ok:
            bad.append(f"{rel}: sha256 {sha256_file(p)[:16]}... != pinned {', '.join(w[:16] + '...' for w in ok)}")
        else:
            got[rel] = sha256_file(p)
    if bad:
        raise BuildError("the landed S0 code differs from the pinned copy: " + "; ".join(bad))
    return got


def ckpt_nav_mode(path) -> str:
    """the checkpoint nav mode as the planner loader reads it (meta.nav_mode, default 'goal'; refe/ckpt_io.py config_for_checkpoint)"""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)          # the loader's own call (ckpt_io.py:122): our checkpoint, sha256-checked by `prepare`
    m = ck.get("meta") if isinstance(ck, dict) and isinstance(ck.get("meta"), dict) else {}
    return str(m.get("nav_mode", "goal"))


def check_command_source(path, sha256, plan_tokens) -> dict:
    """G4: the P6 command table (HF logs/commands.json: {"<log>/<token>": one-hot [LEFT, STRAIGHT, RIGHT, UNKNOWN]}) must cover EVERY token of the S0 plan; refuses otherwise"""
    if not os.path.exists(path):
        raise BuildError(f"command source {path} does not exist")
    sha = sha256_file(path)
    if sha256 and sha != sha256.lower():
        raise BuildError(f"command source sha256 {sha} != the given {sha256}")
    tab = json.load(open(path, encoding="utf-8"))
    missing = [f"{lg}/{tk}" for tk, lg in plan_tokens.items() if f"{lg}/{tk}" not in tab]
    if missing:
        raise BuildError(f"the command source lacks {len(missing)} of the {len(plan_tokens)} plan tokens (e.g. {missing[:3]}): a command-mode checkpoint is never run on a default command")
    bad = [k for k in (f"{lg}/{tk}" for tk, lg in plan_tokens.items()) if sorted(int(x) for x in tab[k]) != [0, 0, 0, 1]]
    if bad:
        raise BuildError(f"the command source has non-one-hot rows for plan tokens (e.g. {bad[:3]})")
    return {"path": os.path.abspath(path).replace(chr(92), "/"), "sha256": sha, "rows": len(tab), "plan_tokens": len(plan_tokens), "plan_tokens_covered": len(plan_tokens)}


def apply_command_source(cand: dict, nav_mode: str, environ=None) -> dict:
    """G4: decide, BEFORE any GPU lock, how the dump gets its commands. command-mode + source: export SRTTC_COMMANDS_JSON / _SHA256 (the landed driver builds the dump argv and
    cannot forward a flag; its subprocess env is a copy of os.environ) and require the G4 dump; command-mode with NO source: REFUSE (as the landed dump did);
    goal-mode: the source is ignored and nothing is exported (a P2 bank is the landed run)."""
    environ = os.environ if environ is None else environ
    src = cand.get("commands")
    if nav_mode == "command":
        if not src:
            raise BuildError("a command-mode checkpoint needs a command source: `prepare --commands <P6 commands.json>` (the dump REFUSES a command-mode planner without one)")
        if sha256_file(src["path"]) != src["sha256"]:
            raise BuildError("the command source changed since `prepare` (sha256)")
        if cand["s0_code_pins"].get("srttc_s0_dump.py") != S0_DUMP_G4_SHA256:
            raise BuildError("a command-mode checkpoint needs the G4 srttc_s0_dump.py (--commands); the landed one refuses it: point `prepare --s0-code` at a copy of the S0 code dir with the G4 dump")
        environ["SRTTC_COMMANDS_JSON"], environ["SRTTC_COMMANDS_SHA256"] = src["path"], src["sha256"]
        return {"nav_mode": nav_mode, "commands_exported": True, "sha256": src["sha256"]}
    for k in ("SRTTC_COMMANDS_JSON", "SRTTC_COMMANDS_SHA256"):
        environ.pop(k, None)
    return {"nav_mode": nav_mode, "commands_exported": False, "note": "goal-mode checkpoint: a supplied command source is ignored" if src else None}


def read_candidate(root):
    p = os.path.join(root, "candidate.json")
    if not os.path.exists(p):
        raise BuildError(f"{p} missing: run `prepare` first")
    return json.load(open(p, encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# preflight: is this a REFe checkpoint of P2's architecture?
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_state_dict(path):
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    return ck, sd


def cmd_preflight(a) -> int:
    t0 = time.time()
    if not os.path.exists(a.ckpt):
        raise BuildError(f"{a.ckpt} does not exist")
    sha = sha256_file(a.ckpt)
    if a.ckpt_sha256 and sha != a.ckpt_sha256.lower():
        raise BuildError(f"checkpoint sha256 {sha} != the given {a.ckpt_sha256}")
    ck, sd = load_state_dict(a.ckpt)
    ref_ck, ref = load_state_dict(a.reference_ckpt)
    rep = {"ckpt": a.ckpt, "sha256": sha, "bytes": os.path.getsize(a.ckpt), "format": ck.get("format") if isinstance(ck, dict) else None, "n_tensors": len(sd),
           "meta_keys": sorted((ck.get("meta") or {}).keys())[:40] if isinstance(ck, dict) and isinstance(ck.get("meta"), dict) else None}
    sw, sb = sd.get("score_head.weight"), sd.get("score_head.bias")
    rep["score_head"] = {"weight": list(sw.shape) if sw is not None else None, "bias": list(sb.shape) if sb is not None else None}
    miss = sorted(set(ref) - set(sd))
    extra = sorted(set(sd) - set(ref))
    shape_diff = sorted(k for k in set(ref) & set(sd) if tuple(ref[k].shape) != tuple(sd[k].shape))
    rep["vs_P2"] = {"reference": a.reference_ckpt, "reference_format": ref_ck.get("format"), "missing_tensors": miss[:30], "n_missing": len(miss), "extra_tensors": extra[:30], "n_extra": len(extra),
                    "shape_differs": shape_diff[:30], "n_shape_differs": len(shape_diff)}
    m_c = ck.get("meta") if isinstance(ck, dict) and isinstance(ck.get("meta"), dict) else {}
    m_r = ref_ck.get("meta") if isinstance(ref_ck, dict) and isinstance(ref_ck.get("meta"), dict) else {}
    rep["meta_differs_from_P2"] = {k: [m_c.get(k), m_r.get(k)] for k in sorted(set(m_c) | set(m_r)) if k in ("backbone", "cameras", "n_cameras", "per_sample_calib", "frozen_sha256", "weights_dir") and m_c.get(k) != m_r.get(k)}
    rep["meta_candidate"] = {k: m_c.get(k) for k in ("backbone", "cameras", "n_cameras", "per_sample_calib", "frozen_sha256", "weights_dir", "step", "epoch", "finished_at") if k in m_c}
    ok_head = bool(sw is not None and tuple(sw.shape) == (6, 256) and sb is not None and tuple(sb.shape) == (6,))
    same = bool(not miss and not extra and not shape_diff)
    rep["nav_mode"] = m_c.get("nav_mode", "goal")
    rep["needs_command_source"] = bool(rep["nav_mode"] == "command")
    rep["checks"] = {"score_head_is_256_to_6": ok_head, "same_tensor_names_and_shapes_as_P2": same, "format_equals_P2": bool(rep["format"] == ref_ck.get("format"))}
    rep["PASS"] = bool(ok_head)
    rep["verdict"] = ("PASS: same architecture as P2 (the S0 dump runs unchanged)" + ("; BUT meta differs from P2's in " + ", ".join(rep["meta_differs_from_P2"]) + " (read it: a different frozen backbone / camera set changes what the planner loads)" if rep["meta_differs_from_P2"] else "") if (ok_head and same) else
                      "PASS WITH DIFFERENCES: the score head is 256 -> 6 but tensor names / shapes differ from P2's; the landed dump may refuse or mis-hook (read vs_P2, run `smoke` on this checkpoint first)" if ok_head else
                      "FAIL: the score head is not 256 -> 6; the S0 gate (feat_dim_ok) would refuse every unit")
    rep["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(rep, indent=1))
    if a.out:
        jdump(rep, a.out)
    return 0 if rep["PASS"] else 2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# prepare / sets / verify-sets
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_prepare(a) -> int:
    root = os.path.abspath(a.root).replace("\\", "/")
    if os.path.abspath(root).replace("\\", "/").startswith(S0_ROOT):
        raise BuildError("R must not be inside S0's directory (read only)")
    if os.path.splitdrive(root)[0].upper() == "C:":
        raise BuildError(f"R={root} is on C:, which is nearly full; use a directory on D: (e.g. {BANK_ON}/p6_bank)")
    if free_gb(root) < 12.0:
        raise BuildError(f"{root} has {free_gb(root):.1f} GiB free (< 12 GiB: bank ~0.6 GiB + the set stores ~1 GiB + staging headroom)")
    pins = check_s0_pins(a.s0_code)
    sha = sha256_file(a.ckpt)
    if a.ckpt_sha256 and sha != a.ckpt_sha256.lower():
        raise BuildError(f"checkpoint sha256 {sha} != the given {a.ckpt_sha256}")
    man = json.load(open(f"{a.s0_root}/run/MANIFEST.sha256.json", encoding="utf-8"))["files"]
    for d in ("run/plan", "run/logs", "run/state", "run/batches", "work/bank", "work/sets"):
        os.makedirs(f"{root}/{d}", exist_ok=True)
    copied = {}
    for n in PLAN_FILES:
        src, dst = f"{a.s0_root}/run/plan/{n}", f"{root}/run/plan/{n}"
        s = sha256_file(src)
        want = man.get(f"run/plan/{n}", {}).get("sha256")
        if want is not None and s != want:
            raise BuildError(f"S0's {n} (sha256 {s}) differs from S0's own manifest ({want}): refuse")
        if n == "ranked_fit_logs.tsv" and s != RANKED_LOGS_SHA256:
            raise BuildError(f"ranked_fit_logs.tsv sha256 {s} != the registered {RANKED_LOGS_SHA256}")
        shutil.copyfile(src, dst)
        copied[n] = {"sha256": s, "in_s0_manifest": want is not None}
    mp = json.load(open(f"{root}/run/plan/master_plan.json", encoding="utf-8"))
    if not (mp["n_logs"] == N_LOGS and mp["n_tokens"] == N_TOKENS and mp["n_sets"] == N_SETS):
        raise BuildError(f"master_plan.json is not the registered S0 sample ({mp['n_logs']}/{mp['n_tokens']}/{mp['n_sets']})")
    cmd_rec = None
    if getattr(a, "commands", None):
        cmd_rec = check_command_source(a.commands, getattr(a, "commands_sha256", None), mp["tokens"])
    cand = {"created": now(), "candidate_id": a.candidate_id, "commands": cmd_rec, "ckpt": a.ckpt.replace("\\", "/"), "ckpt_sha256": sha, "ckpt_bytes": os.path.getsize(a.ckpt), "root": root, "s0_root": a.s0_root, "s0_code": a.s0_code,
            "plan_files": copied, "s0_code_pins": pins, "p2_ckpt": P2_CKPT, "p2_ckpt_sha256": P2_CKPT_SHA256, "n_logs": N_LOGS, "n_tokens": N_TOKENS, "n_sets": N_SETS,
            "wrapper_sha256": {n: sha256_file(os.path.join(HERE, n)) for n in SELF_FILES if os.path.exists(os.path.join(HERE, n))}}
    jdump(cand, f"{root}/candidate.json")
    print(json.dumps({k: cand[k] for k in ("candidate_id", "ckpt", "ckpt_sha256", "root")}, indent=1))
    print("PREPARE_OK")
    return 0


def cmd_sets(a) -> int:
    cand = read_candidate(a.root)
    check_s0_pins(cand["s0_code"])
    cmd = [DRIVERL_PY, "-I", os.path.join(cand["s0_code"], "srttc_s0_podsets.py"), "--plan-dir", f"{cand['root']}/run/plan", "--work", f"{cand['root']}/work"]
    if a.skip_fetch:
        cmd.append("--skip-fetch")
    print("running:", " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", P2HO_TREE=TREE)
    return subprocess.call(cmd, env=env)


def cmd_verify_sets(a) -> int:
    """Every rebuilt per-log set store equals S0's labels6 and ckpt_step on the same (log, token, step, rank): the pod's stores did not change since S0."""
    cand = read_candidate(a.root)
    import ep_admission as A
    rows, _ = A.read_ranked(f"{cand['root']}/run/plan/ranked_fit_logs.tsv", expected_sha256=RANKED_LOGS_SHA256)
    p2 = A.load_bank_units(a.p2_bank, rows)
    pos = {k: i for i, k in enumerate(p2["keys"])}
    n_ok = n_bad = n_missing = 0
    bad_ex = []
    for r in rows:
        p = f"{cand['root']}/work/sets/{r['log']}.npz"
        if not os.path.exists(p):
            n_missing += 1
            continue
        z = np.load(p, allow_pickle=False)
        for i in range(len(z["token"])):
            k = (r["log"], str(z["token"][i]), int(z["step"][i]), int(z["rank"][i]))
            j = pos.get(k)
            if j is None or not np.array_equal(z["labels6"][i], p2["lab"][j]) or int(z["ckpt_step"][i]) != int(p2["ckpt_step"][j]):
                n_bad += 1
                if len(bad_ex) < 3:
                    bad_ex.append(list(k))
            else:
                n_ok += 1
        z.close()
    rep = {"sets_equal_to_S0": n_ok, "sets_differ_or_unknown": n_bad, "logs_without_store": n_missing, "examples": bad_ex, "PASS": bool(n_ok == N_SETS and n_bad == 0 and n_missing == 0)}
    jdump(rep, f"{cand['root']}/run/verify_sets.json")
    print(json.dumps(rep, indent=1))
    return 0 if rep["PASS"] else 2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# dump: the landed S0 driver on the candidate checkpoint
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def load_s0_driver(cand: dict):
    """Import the LANDED srttc_s0_driver by path (sha256-pinned) and override exactly four constants: SRC (the run root), CKPT, CKPT_SHA256 and the GPU-lock OWNER.  Nothing else of it is touched."""
    check_s0_pins(cand["s0_code"])
    if os.path.abspath(cand["root"]).replace("\\", "/").rstrip("/").lower() == S0_ROOT.lower() or os.path.abspath(cand["root"]).replace("\\", "/").lower().startswith(S0_ROOT.lower() + "/"):
        raise BuildError("the run root is S0's own directory: the landed driver deletes the S0_DONE marker of its run root on start; refused")
    d = cand["s0_code"]
    path = os.path.join(d, "srttc_s0_driver.py")
    spec = importlib.util.spec_from_file_location("srttc_s0_driver", path)
    D = importlib.util.module_from_spec(spec)
    sys.modules["srttc_s0_driver"] = D
    spec.loader.exec_module(D)
    D.SRC = cand["root"]
    D.CKPT = cand["ckpt"]
    D.CKPT_SHA256 = cand["ckpt_sha256"]
    D.C.OWNER = f"p6bank_{cand['candidate_id']}"[:40]
    return D


def cmd_dump(a) -> int:
    cand = read_candidate(a.root)
    if sha256_file(cand["ckpt"]) != cand["ckpt_sha256"]:
        raise BuildError("the checkpoint changed since `prepare` (sha256)")
    for lg in json.load(open(f"{cand['root']}/run/plan/master_plan.json", encoding="utf-8"))["logs"]:
        if not os.path.exists(f"{cand['root']}/work/sets/{lg['log']}.npz"):
            raise BuildError(f"fit-set store of {lg['log']} missing: run `sets` (and `verify-sets`) first")
    vs = f"{cand['root']}/run/verify_sets.json"
    if not (os.path.exists(vs) and json.load(open(vs)).get("PASS")):
        raise BuildError("run `verify-sets` first (the rebuilt set stores must equal S0's labels before GPU hours are spent)")
    cmdinfo = apply_command_source(cand, ckpt_nav_mode(cand["ckpt"]))
    print(f"[{now()}] command source: {cmdinfo}", flush=True)
    D = load_s0_driver(cand)
    argv = ["--tag", "main", "--device", a.device, "--min-free-ram-mb", str(a.min_free_ram_mb)]
    if a.no_stage:
        argv.append("--no-stage")
    print(f"[{now()}] dump: SRC={D.SRC} CKPT={D.CKPT} sha256={D.CKPT_SHA256[:16]} OWNER={D.C.OWNER} argv={argv}", flush=True)
    if a.dry_run:
        print("DRY RUN: the landed driver was imported with the overrides above and would run now (no GPU lock taken, nothing dumped).")
        return 0
    return D.main(argv)


def cmd_launch(a) -> int:
    cand = read_candidate(a.root)
    me = os.path.join(HERE, "build_candidate_bank.py").replace("/", "\\")
    root = cand["root"].replace("/", "\\")
    py = DRIVERL_PY.replace("/", "\\")
    body = f'''# Launch (or RESUME) the candidate bank dump DETACHED (same pattern as S0's launch_s0.ps1: WMI Win32_Process.Create, CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW, stdin from NUL).
# Script execution is disabled on this box and -ExecutionPolicy Bypass is never used: PASTE this body into a PowerShell prompt.  The same command RESUMES (banked units are skipped).
$py = "{py}"
$me = "{me}"
$run = "{root}\\run"
$cmdline = 'cmd.exe /c ""' + $py + '" -I "' + $me + '" dump --root "{root}" --device cuda --min-free-ram-mb 4500 < NUL >> "' + $run + '\\driver_main.log" 2>&1"'
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{{ CreateFlags = [uint32]0x08000200 }}
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{{ CommandLine = $cmdline; CurrentDirectory = "{HERE.replace("/", chr(92))}"; ProcessStartupInformation = $si }}
if ($r.ReturnValue -ne 0) {{ throw "Win32_Process.Create failed: $($r.ReturnValue)" }}
"launched cmd.exe pid $($r.ProcessId)"
'''
    print(body)
    return 0


def cmd_status(a) -> int:
    cand = read_candidate(a.root)
    run = f"{cand['root']}/run"
    out = {"candidate": cand["candidate_id"], "markers": [m for m in ("S0_DONE", "S0_FAILED", "FINALIZED.txt") if os.path.exists(f"{run}/{m}")]}
    sp = f"{run}/STATUS.json"
    if os.path.exists(sp):
        st = json.load(open(sp, encoding="utf-8"))
        out["status"] = {k: st.get(k) for k in ("stage", "state", "progress", "guard", "result", "heartbeat", "n_tokens_planned", "n_sets_planned")}
    out["units_banked"] = len([f for f in os.listdir(f"{cand['root']}/work/bank") if f.endswith(".sets.npz")]) if os.path.isdir(f"{cand['root']}/work/bank") else 0
    for m in ("S0_FAILED",):
        if os.path.exists(f"{run}/{m}"):
            out[m] = open(f"{run}/{m}", encoding="utf-8").read()[:400]
    print(json.dumps(out, indent=1, default=str))
    return 0


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# manifest / verify
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def cmd_manifest(a) -> int:
    cand = read_candidate(a.root)
    run, bank = f"{cand['root']}/run", f"{cand['root']}/work/bank"
    if not os.path.exists(f"{run}/S0_DONE"):
        raise BuildError("no S0_DONE marker: the dump has not finished")
    summ = json.load(open(f"{run}/S0_SUMMARY.json", encoding="utf-8"))
    if summ.get("PARTIAL") or summ.get("n_tokens_banked") != N_TOKENS or summ.get("n_sets_banked") != N_SETS:
        raise BuildError(f"the bank is partial or not S0's sample: {summ.get('n_tokens_banked')}/{N_TOKENS} tokens, {summ.get('n_sets_banked')}/{N_SETS} sets")
    files = {}
    for n in sorted(os.listdir(bank)):
        if n.endswith((".sets.npz", ".fan.npz", ".report.json")):
            p = f"{bank}/{n}"
            files[f"work/bank/{n}"] = {"path": p, "sha256": sha256_file(p), "bytes": os.path.getsize(p)}
    doc = {"written": now(), "what": "candidate bank outputs: sha256 + bytes of every unit file as written", "candidate_id": cand["candidate_id"], "ckpt": cand["ckpt"], "ckpt_sha256": cand["ckpt_sha256"],
           "n_files": len(files), "files": files}
    jdump(doc, f"{bank}/MANIFEST.sha256.json")
    msha = sha256_file(f"{bank}/MANIFEST.sha256.json")
    print(json.dumps({"n_files": len(files), "manifest": f"{bank}/MANIFEST.sha256.json", "manifest_sha256": msha, "ckpt_sha256": cand["ckpt_sha256"]}, indent=1))
    print("The registration (EP-ADMISSION prereg) must NAME both sha256 values above BEFORE `ep_admission.py run`.")
    return 0


def cmd_verify(a) -> int:
    cand = read_candidate(a.root)
    import ep_admission as A
    rows, _ = A.read_ranked(f"{cand['root']}/run/plan/ranked_fit_logs.tsv", expected_sha256=RANKED_LOGS_SHA256)
    p2 = A.load_bank_units(a.p2_bank, rows)
    c = A.load_bank_units(f"{cand['root']}/work/bank", rows)
    al = A.align_candidate(p2, c)                                     # raises on a key / label / ckpt_step mismatch
    ctrl = {"fanpath_logit_max": 0.0, "n_sets": 0, "setpath_logit_differs_from_fan_gt_1e-3": 0}
    for n in sorted(os.listdir(f"{cand['root']}/work/bank")):
        if n.endswith(".sets.npz"):
            z = np.load(f"{cand['root']}/work/bank/{n}", allow_pickle=False)
            ctrl["fanpath_logit_max"] = max(ctrl["fanpath_logit_max"], float(z["ctrl_fanpath_logit_maxabs"].max()))
            ctrl["n_sets"] += int(len(z["token"]))
            ctrl["setpath_logit_differs_from_fan_gt_1e-3"] += int((z["ctrl_logit_maxabs"] > 1e-3).sum())
            z.close()
    rep = {"alignment": al["checks"], "n_sets": al["n_sets"], "identity_controls": ctrl, "bank_ckpt": c["ckpt"], "candidate_ckpt": cand["ckpt"],
           "ckpt_matches": bool(str(c["ckpt"]).replace("\\", "/") == cand["ckpt"]), "manifest_sha256": c["manifest_sha256"], "same_logits_as_p2_on_any_set": int((al["z6"] == p2["z6"]).all(axis=(1, 2)).sum()),
           "PASS": bool(al["n_sets"] == N_SETS and ctrl["fanpath_logit_max"] <= 1e-4 and ctrl["n_sets"] == N_SETS and ctrl["setpath_logit_differs_from_fan_gt_1e-3"] >= 0.9 * N_SETS
                        and str(c["ckpt"]).replace("\\", "/") == cand["ckpt"])}
    rep["note"] = "same_logits_as_p2_on_any_set > 0 would mean the candidate's scorer reproduced P2's logits bit for bit on a set: the dump was run on the wrong checkpoint"
    rep["PASS"] = bool(rep["PASS"] and rep["same_logits_as_p2_on_any_set"] == 0)
    jdump(rep, f"{cand['root']}/run/verify_report.json")
    print(json.dumps(rep, indent=1))
    return 0 if rep["PASS"] else 2


# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# smoke: CPU, the landed dump script, FAN path, a few phase-L tokens vs S0's banked fan logits
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _pick_margin(z6):
    """top-1 minus top-2 of the frozen aggregate for one set's [64, 6] logits (how close a pick flip would be)."""
    p = 1.0 / (1.0 + np.exp(-np.asarray(z6, np.float64)))
    agg = np.sort(p[:, 0] * p[:, 1] * (5 * p[:, 2] + 5 * p[:, 3] + 2 * p[:, 4]) / 12.0)[::-1]
    return float(agg[0] - agg[1])


def cmd_smoke(a) -> int:
    import psutil
    pins = check_s0_pins(a.s0_code)
    av = ram_gb()
    if av < MIN_RAM_GB_CPU:
        raise BuildError(f"available RAM {av:.2f} GB < {MIN_RAM_GB_CPU} GB: the CPU forward is heavy, step stopped")
    work = os.path.abspath(a.workdir)
    os.makedirs(work, exist_ok=True)
    unit = a.unit
    bt = json.load(open(f"{a.s0_root}/run/batches/{unit}/batch_tokens.json", encoding="utf-8"))
    lg = bt["token_log"][bt["tokens"][0]]
    toks = [t for t in bt["tokens"] if bt["token_log"][t] == lg][:a.n_tokens]
    plan = {"tokens": toks, "token_log": {t: lg for t in toks}, "token_root": {t: bt["token_root"][t] for t in toks}, "db": {lg: bt["db"][lg]}}
    pp = f"{work}/plan.json"
    jdump(plan, pp)
    out = f"{work}/dump.npz"
    for f in (out, os.path.splitext(out)[0] + ".report.json"):
        if os.path.exists(f):
            os.remove(f)
    sys.path.insert(0, os.path.join(TREE, "eval"))
    import eval_checkpoint as EC
    env = EC.env_driverl()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "P2HO_TREE": TREE})
    cmd = [DRIVERL_PY, "-I", os.path.join(a.s0_code, "srttc_s0_dump.py"), "--ckpt", a.ckpt, "--plan", pp, "--out", out, "--device", a.device, "--lean", "--arm", "p6prep_smoke"]
    print(f"[{now()}] smoke: {len(toks)} tokens of {lg}, device {a.device}, RAM available {av:.2f} GB", flush=True)
    t0 = time.time()
    logf = open(f"{work}/dump.log", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, cwd=work)
    peak = [0]

    def sample():
        try:
            p = psutil.Process(proc.pid)
            while proc.poll() is None:
                try:
                    rss = p.memory_info().rss + sum(c.memory_info().rss for c in p.children(recursive=True))
                    peak[0] = max(peak[0], rss)
                except psutil.Error:
                    pass
                time.sleep(1.0)
        except psutil.Error:
            pass
    th = threading.Thread(target=sample, daemon=True)
    th.start()
    rc = proc.wait()
    th.join(timeout=5)
    logf.close()
    rep = {"what": "CPU smoke of the landed S0 dump script (fan path) against S0's banked fan logits", "ckpt": a.ckpt, "device": a.device, "unit": unit, "log": lg, "tokens": toks, "dump_rc": rc,
           "seconds": round(time.time() - t0, 1), "peak_rss_gb": round(peak[0] / 2 ** 30, 2), "tolerance_logits": SMOKE_LOGIT_TOL, "s0_code_pins": pins}
    if rc != 0 or not os.path.exists(out):
        rep["PASS"] = False
        rep["tail"] = open(f"{work}/dump.log", encoding="utf-8", errors="replace").read()[-1500:]
        jdump(rep, f"{work}/smoke_report.json")
        print(json.dumps(rep, indent=1))
        return 2
    mine = np.load(out, allow_pickle=False)
    ref = np.load(f"{a.s0_root}/work/bank/{unit}.fan.npz", allow_pickle=False)
    rt = [str(x) for x in ref["token"]]
    rows = []
    for i, t in enumerate([str(x) for x in mine["token"]]):
        j = rt.index(t)
        d = float(np.abs(mine["logits"][i].astype(np.float64) - ref["logits"][j].astype(np.float64)).max())
        margin = _pick_margin(ref["logits"][j])
        # negative control: THIS token's CPU logits against ANOTHER banked token's logits must differ clearly (the comparison can fail)
        other = (j + 1) % len(rt)
        neg = float(np.abs(mine["logits"][i].astype(np.float64) - ref["logits"][other].astype(np.float64)).max())
        rows.append({"token": t, "max_abs_logit_diff_cpu_vs_s0_gpu": d, "pass": bool(d <= SMOKE_LOGIT_TOL), "pick_cpu": int(mine["pick"][i]), "pick_s0": int(ref["pick"][j]), "pick_equal": bool(int(mine["pick"][i]) == int(ref["pick"][j])), "s0_pick_margin_top1_minus_top2_aggregate": margin,
                     "fingerprint_equal": bool(str(mine["fingerprint"][i]) == str(ref["fingerprint"][j])), "negative_control_max_abs_diff_vs_other_token": neg, "negative_control_reads_FAIL": bool(neg > SMOKE_NEG_MIN)})
    rep["tokens_compared"] = rows
    rep["max_abs_logit_diff"] = max(r["max_abs_logit_diff_cpu_vs_s0_gpu"] for r in rows)
    rep["PASS"] = bool(all(r["pass"] and r["negative_control_reads_FAIL"] for r in rows) and len(rows) == len(toks))
    rep["NOT_REPRODUCED"] = ("the SET path (score_trajectories on the stored fit sets' traj3): the stored trajectories live only on the pod (`/workspace/data/refe_navtrain/onpolicy/sets`) and S0's work/sets was deleted after the bank was "
                             "banked; the pod is off limits to this step.  The fan path above is the same forward, the same weights and the same hooks; the set path differs only in the trajectories it is fed.")
    jdump(rep, f"{work}/smoke_report.json")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("s0_code_pins",)}, indent=1))
    return 0 if rep["PASS"] else 2


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-sha256", default=None)
    p.add_argument("--reference-ckpt", default=P2_CKPT)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_preflight)
    p = sub.add_parser("prepare")
    p.add_argument("--root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ckpt-sha256", default=None)
    p.add_argument("--candidate-id", default="p6")
    p.add_argument("--commands", default=None, help="G4: the P6 command table (HF logs/commands.json), required for a command-mode checkpoint; must cover every S0 plan token")
    p.add_argument("--commands-sha256", default=None)
    p.add_argument("--s0-root", default=S0_ROOT)
    p.add_argument("--s0-code", default=S0_CODE)
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("sets")
    p.add_argument("--root", required=True)
    p.add_argument("--skip-fetch", action="store_true")
    p.set_defaults(fn=cmd_sets)
    p = sub.add_parser("verify-sets")
    p.add_argument("--root", required=True)
    p.add_argument("--p2-bank", default=P2_BANK)
    p.set_defaults(fn=cmd_verify_sets)
    p = sub.add_parser("dump")
    p.add_argument("--root", required=True)
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--min-free-ram-mb", type=int, default=4500)
    p.add_argument("--no-stage", action="store_true", help="phase L only (817 locally available tokens); ends S0_FAILED rc5 by design because coverage is incomplete")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_dump)
    p = sub.add_parser("launch")
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_launch)
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("manifest")
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("verify")
    p.add_argument("--root", required=True)
    p.add_argument("--p2-bank", default=P2_BANK)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("smoke")
    p.add_argument("--ckpt", default=P2_CKPT)
    p.add_argument("--n-tokens", type=int, default=3)
    p.add_argument("--device", default="cpu", choices=("cpu",))
    p.add_argument("--unit", default="L1b95fda46f")
    p.add_argument("--workdir", required=True)
    p.add_argument("--s0-root", default=S0_ROOT)
    p.add_argument("--s0-code", default=S0_CODE)
    p.set_defaults(fn=cmd_smoke)
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except BuildError as e:
        print(f"REFUSED: {e}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
