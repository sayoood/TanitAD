"""PREREG_V8F_STRAT1 study checker (BUILD_SPEC section 8; the pattern of ``obd_check.py``). It maps the seven launch-gate checks onto the trainer's own selfcheck:

  G-HYG    V8FMiniConfig refuses bad values / unknown keys                      (tests: test_v8f_strat1.py::test_hygiene_*)           -> ``hyg``  (runs the refusals here)
  G-DVB    trainer ``declared_vs_built`` incl. ``bad_ctx`` + FLAG_CHECKS          -> ``gate`` asserts every argv lever against ``selfcheck.levers``
  G-LIVE   ``live`` / ``ctx_head_probe`` / ``nav_source_probe`` / inherited ``acc_probe`` ``otband_probe`` ``dagger_probe``             -> ``gate``
  G-CLOCK  label time: ``ctx_label_arrays`` alignment asserts (trainer) + ``build_navtext_nav.py`` G-CLOCK-NAV (its report)                -> ``gate`` reads the nav report sha
  G-EVAL   a checkpoint loads through each evaluator's own build path, strictly; built levers == the selfcheck's; a cfg that drops ctx_heads is REFUSED -> ``geval``
  G-CKPT   save / load round trip bit-equal                                       (tests + the post check)                              -> ``post``
  G-SUITE  pytest on a clean export of the branch tip, 0 regressions vs cdbff971   (run by the MM in the pod environment)                 -> NOT run here

Subcommands: ``argv TAG --root R --out O`` (render a frozen argv), ``gate TAG --selfcheck F`` (problems), ``post TAG --init I --final F`` (moved identity), ``geval --ckpt C --selfcheck F`` and ``token``.
NOT built here (UNVERIFIED, the MM's launch loop): the pilot timing / chain runner / pod preflight of ``obd_check.py``. This file only DECIDES; it launches nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "stack"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

#: the control's code of record (PREREG: the OBD-CTL recipe): the frozen box-2 argvs the STRAT1 argvs extend (read-only input; box-2 paths are mapped to --root)
BASE_ARGVS = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/obd/box2/frozen/argvs_obd.json"
BASE_SHA256 = None            # pinned by the MM when the frozen file is copied into the chain's folder (UNVERIFIED here: read in place)
STRAT_LABELS_SHA256 = "8055a321df6db1725ea17abde2999d48bcb68d43f1a480a69ed42dd0cf184eff"
NINE = ("--nav-source", "--nav-text-npz", "--nav-text-sha256", "--ctx-heads", "--w-ctx", "--ctx-labels", "--ctx-labels-sha256", "--ctx-lr-mult", "--ctx-detach")
KINDS = ("C0", "N1", "S1")
SEEDS = {"s0": {"base": "OBD-CTL_s0"}, "s1": {"base": "OBD-CTL_s1"}, "rep": {"base": "OBD-CTL_s0", "seed": "2"}}      # rep = init 0, bank s0, dagger-seed 0, seed 2 (OBD-NULL_rep's form)
RUN_STEPS = 3000
MOVED = {"adapter": ("adapter", "pos", "time"), "S": ("read_s", "head_s"), "T": ("read_t", "trunk_t", "h_goal", "h_lat", "h_lon", "h_cons"), "O": ("read_o", "head_o"), "WM": ("wm.",)}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def flag_value(toks, name):
    return toks[toks.index(name) + 1] if name in toks else None


def render(kind, seed, root, out, nav_npz="", ctx_npz="", base_argvs=BASE_ARGVS, cache_dir=None):
    """The frozen argv of an arm: the OBD-CTL argv of that seed, box-2 paths mapped under ``root`` (the pod layout: ``root/runs/v8f_mini_full/X`` -> ``root/X``, the tracks next to ``root``, the cache at ``cache_dir``), plus the nine
    STRAT1 flags (ALL named, the control included)."""
    assert kind in KINDS and seed in SEEDS
    base = json.load(open(base_argvs, encoding="utf-8"))[SEEDS[seed]["base"]]["argv"]
    r = root.rstrip("/")
    BOX2 = "D:" + chr(92) + "tanitad"
    t = [x.replace(BOX2, r).replace(chr(92), "/") if isinstance(x, str) and x.startswith(BOX2) else x for x in base]
    t = [x.replace(r + "/runs/v8f_mini_full/", r + "/") if isinstance(x, str) else x for x in t]
    t = [r + "/native_tracks.pt" if x == r + "/cache_full/native_tracks.pt" else x for x in t]
    if cache_dir:
        t = [cache_dir if x == r + "/cache_full" else x for x in t]
    t[t.index("--out") + 1] = out
    if "seed" in SEEDS[seed]:
        t[t.index("--seed") + 1] = SEEDS[seed]["seed"]
    nav = ["--nav-source", "text", "--nav-text-npz", nav_npz, "--nav-text-sha256", sha_file(nav_npz) if os.path.exists(nav_npz) else "<sha of navtext_nav.npz>"] if kind in ("N1", "S1") else         ["--nav-source", "v9", "--nav-text-npz", "", "--nav-text-sha256", ""]
    ctx = ["--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", ctx_npz, "--ctx-labels-sha256", STRAT_LABELS_SHA256, "--ctx-lr-mult", "10", "--ctx-detach", "off"] if kind == "S1" else         ["--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "", "--ctx-lr-mult", "1", "--ctx-detach", "off"]
    return t + nav + ctx


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def gate_problems(kind, toks, sc, steps=RUN_STEPS):
    """The G-DVB + G-LIVE read of a selfcheck.json against its argv. [] = pass."""
    bad = []
    miss = [f for f in NINE if f not in toks]
    if miss:
        bad.append(f"the argv does not name {miss}: a flag whose default disables it must be named")
    if sc.get("declared_vs_built") != "PASS":
        bad.append("declared_vs_built is not PASS")
    lv, live = sc.get("levers", {}), sc.get("live", {})
    if not live.get("pass"):
        bad.append("G-LIVE (live) did not pass")
    if sc.get("steps_total") != steps or flag_value(toks, "--steps") != str(steps):
        bad.append(f"steps_total {sc.get('steps_total')} / argv --steps {flag_value(toks, '--steps')} != {steps}")
    nsp = sc.get("nav_source_probe") or {}
    if not nsp.get("pass"):
        bad.append("nav_source_probe missing or not pass")
    want_src = "text" if kind in ("N1", "S1") else "v9"
    if nsp.get("source") != want_src or flag_value(toks, "--nav-source") != want_src:
        bad.append(f"nav source: arm {kind} needs {want_src}, probe read {nsp.get('source')}, argv {flag_value(toks, '--nav-source')}")
    if (nsp.get("n_text_ne_v9", 0) > 0) != (want_src == "text"):
        bad.append(f"n_text_ne_v9 {nsp.get('n_text_ne_v9')} is inconsistent with source {want_src}")
    on = kind == "S1"
    if lv.get("ctx_heads", "off") != ("on" if on else "off") or flag_value(toks, "--ctx-heads") != ("on" if on else "off"):
        bad.append(f"ctx_heads: arm {kind} needs {'on' if on else 'off'}, built {lv.get('ctx_heads', 'off')}")
    if on:
        cp = sc.get("ctx_head_probe") or {}
        if not cp.get("pass"):
            bad.append("ctx_head_probe missing or not pass")
        if cp.get("control_head_grad_w0") != 0.0 or cp.get("control_allmasked_head_grad") != 0.0 or cp.get("control_adapter_grad_detach_on") != 0.0:
            bad.append("a ctx exact-0.0 control is not exactly 0.0")
        if not (live.get("groups_grad") or {}).get("CTX", 0.0) > 0.0:
            bad.append("G-LIVE: the CTX group has no gradient")
        if float(lv.get("w_ctx", 0)) != float(flag_value(toks, "--w-ctx")) or lv.get("ctx_dim") != 41 or not lv.get("ctx_vision_only_source"):
            bad.append("built ctx levers (w_ctx / ctx_dim 41 / vision-only source) differ from the argv")
        if (sc.get("ctx_labels") or {}).get("sha256") != STRAT_LABELS_SHA256:
            bad.append("the loaded ctx label file is not sha 8055a321")
    elif "ctx_head_probe" in sc or "ctx_head." in json.dumps(sc.get("levers", {})):
        bad.append("a ctx-off arm carries a ctx probe / lever")
    for k in ("acc_probe", "dagger_probe", "otband_probe"):                 # the inherited probes (OBD-CTL's gate read them)
        if not isinstance(sc.get(k), dict) or sc[k].get("skipped"):
            bad.append(f"inherited probe {k} is missing or skipped")
    if not lv.get("detach_upper"):
        bad.append("--detach-upper is not built (the OBD-CTL recipe)")
    return bad


def moved_identity(init_sd, fin_sd, kind):
    """The post check: every non-ctx group moved from the init; ``ctx_head.*`` exists and moved iff S1; no ctx key in C0 / N1."""
    import torch
    bad = []
    for g, pre in MOVED.items():
        ks = [k for k in fin_sd if k.startswith(pre) and k in init_sd]
        if ks and all(torch.equal(init_sd[k], fin_sd[k]) for k in ks):
            bad.append(f"group {g} did not move")
    ck = [k for k in fin_sd if k.startswith("ctx_head.")]
    if kind == "S1":
        if not ck:
            bad.append("S1 checkpoint has no ctx_head keys")
        else:                                                           # the head's seeded init (CTX_INIT_SEED, forked RNG) is the fresh start state: it must have moved
            from tanitad.models import v8f_mini as M
            d = int(fin_sd["ctx_head.0.weight"].shape[1])
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(M.CTX_INIT_SEED)
                fresh = M._mlp(d, d, M.CTX_DIM).state_dict()
            if all(torch.equal(fresh[k[len("ctx_head."):]], fin_sd[k]) for k in ck):
                bad.append("ctx_head did not move from its seeded init")
    elif ck:
        bad.append(f"ctx_head keys {ck[:2]} in a {kind} checkpoint")
    return bad


def geval(ckpt_path, selfcheck_levers=None):
    """G-EVAL: load through the evaluators' own build path (``V8FMiniConfig(**ck['cfg'])`` -> ``V8FMini`` -> STRICT ``load_state_dict``); built levers equal the selfcheck's; the deliberate-regression arm (a cfg that drops
    ``ctx_heads`` from a ctx checkpoint) must be REFUSED. Returns (problems, record)."""
    import torch
    from tanitad.models import v8f_mini as M
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    bad, rec = [], {}

    def build(cfgd):
        m = M.V8FMini(M.V8FMiniConfig(**cfgd), M.P10Prior(torch.zeros(8), torch.ones(8), torch.zeros(2 * M.K30)))
        m.load_state_dict(ck["model"])
        return m
    try:
        m = build(ck["cfg"])
    except Exception as e:                                              # noqa: BLE001
        return [f"strict load failed: {type(e).__name__}: {str(e)[:200]}"], rec
    lv = m.built_levers()
    rec["built_levers"] = {k: lv[k] for k in lv if k.startswith(("ctx", "w_ctx"))}
    if selfcheck_levers is not None:
        for k in [k for k in set(lv) | set(selfcheck_levers) if k.startswith(("ctx", "w_ctx")) or k in ("navonly", "wm", "detach_upper")]:
            if lv.get(k) != selfcheck_levers.get(k):
                bad.append(f"built lever {k}: evaluator {lv.get(k)!r} vs selfcheck {selfcheck_levers.get(k)!r}")
    if ck["cfg"].get("ctx_heads") == "on":
        try:
            build({**ck["cfg"], "ctx_heads": "off", "w_ctx": 0.0})
            bad.append("regression arm: a cfg that drops ctx_heads was NOT refused (the evaluator would silently skip the head)")
        except Exception:                                               # noqa: BLE001
            rec["regression_arm_refused"] = True
    return bad, rec


def hyg():
    from tanitad.models import v8f_mini as M
    n = 0
    for kw in (dict(ctx_heads="maybe"), dict(ctx_detach="x"), dict(w_ctx=-1.0), dict(w_ctx=float("nan")), dict(nav_source="osm")):
        try:
            M.V8FMiniConfig(**kw)
        except ValueError:
            n += 1
    try:
        M.V8FMiniConfig(ctx_headz="on")
    except TypeError:
        n += 1
    return [] if n == 6 else [f"only {n} of 6 bad configs were refused"]


# ---- PREREG_V8F_STRAT1 launch gate: token, tree manifest, data bindings, G-SUITE result, mint -------------------------------------------------------------------------------------------------
TOKEN_SCHEMA = "strat1-gate-token-v1"
WRITTEN_BY = "strat1_check.py mint"
DATA_FLAGS = ("--v9-train", "--lead-train", "--speed-train", "--join-train", "--p10", "--tracks", "--nav-text-npz", "--ctx-labels", "--init-ckpt")
REQUIRED_CHECKS = ("G-HYG", "G-DVB", "G-LIVE", "G-CLOCK", "G-EVAL", "G-CKPT", "G-SUITE")
TREE_DIRS = (("stack", "tanitad"), ("stack", "scripts"), ("stack", "tests"), ("taniteval",))
TREE_EXT = (".py", ".txt", ".json", ".sh")
TRAINER_REL = os.path.join("stack", "scripts", "v8f_mini_train.py")
EVAL_GRID_REL = ("TanitAD Research Lab", "Architecture & Inference", "Research", "2026-10-17-v8f-mini", "raw", "eval_grid_windows.json")


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hash_entry(p):
    """size, sha256 and md5 of one input file (None when absent)."""
    if not p or not os.path.isfile(p):
        return None
    return {"size": os.path.getsize(p), "sha256": sha_file(p), "md5": md5_file(p)}


def tree_manifest(root):
    """sha256 of every code / test file the run executes (``stack/tanitad``, ``stack/scripts``, ``stack/tests``, ``taniteval``; no caches), keyed by the path relative to ``root``."""
    out = {}
    for parts in TREE_DIRS:
        base = os.path.join(root, *parts)
        for dp, dn, fn in os.walk(base):
            dn[:] = sorted(d for d in dn if d != "__pycache__")
            for f in sorted(fn):
                if f.endswith(TREE_EXT):
                    full = os.path.join(dp, f)
                    out[os.path.relpath(full, root).replace(os.sep, "/")] = sha_file(full)
    return out


def tree_manifest_sha(root):
    return hashlib.sha256(json.dumps(tree_manifest(root), sort_keys=True).encode("utf-8")).hexdigest()


def data_binding(toks):
    """Every data input the argv names: its size, sha256 and md5 (recomputed now). Returns (dict, problems). A named path that does not exist is a problem; an empty value is not named."""
    d, bad = {}, []
    for fl in DATA_FLAGS:
        v = flag_value(toks, fl)
        if not v:
            continue
        e = hash_entry(v)
        if e is None:
            bad.append(f"{fl} {v!r} is not a file")
        else:
            d[fl] = {"path": v, **e}
    bank = flag_value(toks, "--dagger-bank")
    if bank:
        if not os.path.isdir(bank):
            bad.append(f"--dagger-bank {bank!r} is not a directory")
        else:
            for f in sorted(os.listdir(bank)):
                if f.endswith((".npz", ".json")):
                    d["bank/" + f] = {"path": os.path.join(bank, f), **hash_entry(os.path.join(bank, f))}
            bj = os.path.join(bank, "bank.json")
            if not os.path.isfile(bj):
                bad.append("the dagger bank has no bank.json")
            elif flag_value(toks, "--dagger-bank-sha256") != sha_file(bj):
                bad.append("--dagger-bank-sha256 differs from the sha256 of bank.json")
    cache = flag_value(toks, "--cache-dir")
    if cache:
        for split in ("train", "eval"):
            m = os.path.join(cache, split, "manifest.jsonl")
            e = hash_entry(m)
            if e is None:
                bad.append(f"the cache has no {split}/manifest.jsonl ({m})")
            else:
                d[f"cache/{split}/manifest.jsonl"] = {"path": m, **e}
    nav = flag_value(toks, "--nav-text-npz")
    if nav and flag_value(toks, "--nav-text-sha256") != d.get("--nav-text-npz", {}).get("sha256"):
        bad.append("--nav-text-sha256 differs from the sha256 of the nav npz")
    cl = flag_value(toks, "--ctx-labels")
    if cl and flag_value(toks, "--ctx-labels-sha256") != d.get("--ctx-labels", {}).get("sha256"):
        bad.append("--ctx-labels-sha256 differs from the sha256 of the ctx label file")
    return d, bad


def suite_problems(path, commit):
    """The G-SUITE result file written by ``suite``: 0 regressions of the new tree against the base, for this commit. Returns (sha256 of the file or None, problems)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None, ["no G-SUITE result file"]
    bad = []
    if r.get("schema") != "strat1-gsuite-result-v1" or r.get("written_by") != "strat1_check.py suite":
        bad.append("the G-SUITE result file was not written by the checker")
    if r.get("n_regressions") != 0 or r.get("regressions"):
        bad.append(f"G-SUITE has {r.get('n_regressions')} regressions")
    if not r.get("new", {}).get("n_tests") or not r.get("base", {}).get("n_tests"):
        bad.append("the G-SUITE result has an empty side")
    if r.get("new_commit") != commit:
        bad.append(f"the G-SUITE result is for commit {r.get('new_commit')}, not {commit}")
    return sha_file(path), bad


def nav_report_problems(kind, nav_npz, report_path):
    """G-CLOCK-NAV: the nav builder's report must read PASS on train and eval, its regression arm must FAIL, and it must describe this npz."""
    if kind not in ("N1", "S1"):
        return []
    try:
        r = json.load(open(report_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no nav builder report"]
    bad = []
    for sp in ("train", "eval"):
        if not r.get(sp, {}).get("G-CLOCK-NAV_pass"):
            bad.append(f"G-CLOCK-NAV {sp} is not pass")
        if r.get(sp, {}).get("G-CLOCK-NAV_regression_arm_plus_travelled", {}).get("pass") is not False:
            bad.append(f"G-CLOCK-NAV {sp} regression arm did not fail as required")
    if r.get("navtext_nav_sha256") != sha_file(nav_npz):
        bad.append("the nav report describes a different npz")
    return bad


def binding(kind, toks, root, commit, suite_path, trainer=None, nav_report=""):
    """Everything the token binds, recomputed NOW. Returns (dict, problems)."""
    d, bad = data_binding(toks)
    tr = trainer or os.path.join(root, TRAINER_REL)
    grid = os.path.join(root, *EVAL_GRID_REL)                                 # the trainer's DAGGER_EVAL_GRID (K-B3 refuses a launch without it): a data input of the tree, bound when present
    if os.path.isfile(grid):
        d["eval_grid_windows.json"] = {"path": grid, **hash_entry(grid)}
    ssha, sbad = suite_problems(suite_path, commit)
    bad = list(bad) + sbad
    if nav_report:
        bad += nav_report_problems(kind, flag_value(toks, "--nav-text-npz"), nav_report)
    b = {"kind": kind, "commit": commit, "argv_sha256": argv_sha(toks), "data": d, "tree_manifest_sha256": tree_manifest_sha(root),
         "trainer_sha256": sha_file(tr) if os.path.isfile(tr) else None, "gsuite_result_sha256": ssha,
         "checker_sha256": sha_file(os.path.abspath(__file__)),
         "nav_report_sha256": sha_file(nav_report) if nav_report and os.path.isfile(nav_report) else None}
    if b["trainer_sha256"] is None:
        bad.append("trainer not found")
    return b, bad


def body_sha(tok):
    return hashlib.sha256(json.dumps({k: v for k, v in tok.items() if k != "body_sha256"}, sort_keys=True).encode("utf-8")).hexdigest()


def write_token(path, b, checks, selfcheck_sha):
    """The ONLY writer of a token (called by ``mint`` after every required check read PASS). ``checks`` maps each required check to a record whose ``pass`` is True."""
    miss = [c for c in REQUIRED_CHECKS if not (checks.get(c) or {}).get("pass")]
    if miss:
        raise ValueError(f"refusing to write a token: checks not PASS: {miss}")
    tok = {"schema": TOKEN_SCHEMA, "written_by": WRITTEN_BY, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "selfcheck_sha256": selfcheck_sha, "checks": checks, **b}
    tok["body_sha256"] = body_sha(tok)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(tok, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return tok


def token_problems(token_path, kind, toks, root, commit, suite_path, selfcheck_path="", trainer=None, nav_report=""):
    """[] iff a token written by ``mint`` still binds this arm kind, commit, argv, every data file, the tree, the trainer, the checker and the G-SUITE result. Each differing key is named."""
    try:
        tok = json.load(open(token_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no gate token"]
    p = []
    if tok.get("schema") != TOKEN_SCHEMA or tok.get("written_by") != WRITTEN_BY:
        p.append("token not written by the checker")
    if tok.get("body_sha256") != body_sha(tok):
        p.append("token body was edited (body_sha256)")
    miss = [c for c in REQUIRED_CHECKS if not (tok.get("checks", {}).get(c) or {}).get("pass")]
    if miss:
        p.append(f"token checks not PASS: {miss}")
    want, bad = binding(kind, toks, root, commit, suite_path, trainer, nav_report)
    p += list(bad)
    for k, v in want.items():
        if k == "data":
            for name in sorted(set(v) | set(tok.get("data", {}))):
                a, b2 = v.get(name), tok.get("data", {}).get(name)
                if a != b2:
                    p.append(f"token data {name} differs from the current file")
        elif tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    if selfcheck_path and (not os.path.isfile(selfcheck_path) or sha_file(selfcheck_path) != tok.get("selfcheck_sha256")):
        p.append("selfcheck.json differs from the token's")
    return p


# ---- G-SUITE comparison ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def junit_results(d):
    """{test id: 'pass'|'fail'|'error'|'skip'} over every shard xml of a junit directory."""
    import glob
    import xml.etree.ElementTree as ET
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.xml"))):
        for tc in ET.parse(f).getroot().iter("testcase"):
            tid = (tc.get("classname") or "") + "::" + (tc.get("name") or "")
            st = "pass"
            for ch in tc:
                if ch.tag in ("failure", "error", "skipped"):
                    st = {"failure": "fail", "error": "error", "skipped": "skip"}[ch.tag]
                    break
            out[tid] = st
    return out


def suite_compare(base_dir, new_dir, new_commit, base_commit):
    b, n = junit_results(base_dir), junit_results(new_dir)
    bad = ("fail", "error")
    reg = sorted(t for t, s in n.items() if s in bad and b.get(t) == "pass")
    newonly = sorted(t for t in n if t not in b)
    return {"schema": "strat1-gsuite-result-v1", "written_by": "strat1_check.py suite", "base_commit": base_commit, "new_commit": new_commit,
            "base": {"n_tests": len(b), "n_pass": sum(s == "pass" for s in b.values()), "n_fail": sum(s == "fail" for s in b.values()), "n_error": sum(s == "error" for s in b.values()), "n_skip": sum(s == "skip" for s in b.values())},
            "new": {"n_tests": len(n), "n_pass": sum(s == "pass" for s in n.values()), "n_fail": sum(s == "fail" for s in n.values()), "n_error": sum(s == "error" for s in n.values()), "n_skip": sum(s == "skip" for s in n.values())},
            "n_regressions": len(reg), "regressions": reg,
            "new_test_failures": sorted(t for t in newonly if n[t] in bad),
            "fail_on_both": sorted(t for t, s in n.items() if s in bad and b.get(t) in bad),
            "missing_in_new": sorted(t for t in b if t not in n), "n_new_only": len(newonly)}


# ---- mint: the checker runs the trainer's own selfcheck and a short resume run, reads every check, and only then writes the token ----------------------------------------------------------
def _run(py, trainer, toks, cwd, env_extra, log):
    import subprocess
    env = dict(os.environ)
    env.update(env_extra)
    with open(log, "ab") as lf:
        return subprocess.run([py, trainer] + list(toks), cwd=cwd, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def _with(toks, **kv):
    t = list(toks)
    for k, v in kv.items():
        f = "--" + k.replace("_", "-")
        if f in t:
            t[t.index(f) + 1] = str(v)
        else:
            t += [f, str(v)]
    return t


def arm_tag(kind, seed, steps):
    return f"{kind}_{seed}" if steps == RUN_STEPS else f"pilot{steps}_{kind}_{seed}"


def mint(tag, kind, seed, a):
    """Mint ``<runs>/<tag>/gate/gate_token.json``. Returns the list of problems (empty = token written)."""
    rd = os.path.join(a.runs, tag)
    gd = os.path.join(rd, "gate")
    out = os.path.join(rd, "train")
    os.makedirs(gd, exist_ok=True)
    frozen = os.path.join(a.inputs, "frozen", "argvs_obd.json")             # the frozen OBD-CTL argvs, shipped to the pod with the other inputs (the dev-box path is not there)
    toks = render(kind, seed, a.inputs, out, os.path.join(a.inputs, "strat1", "navtext_nav.npz"), os.path.join(a.inputs, "strat1", "strat_labels_table.npz"), base_argvs=frozen, cache_dir=a.cache_dir)
    if a.steps != RUN_STEPS:                                                  # a timing pilot: the SAME argv with --steps moved (nothing else), its own token
        toks = _with(toks, steps=a.steps)
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    nav_report = os.path.join(a.inputs, "strat1", "navtext_nav_report.json")
    b, bad = binding(kind, toks, a.root, a.commit, a.suite, nav_report=nav_report if kind in ("N1", "S1") else "")
    if bad:
        return ["binding: " + x for x in bad]
    trainer = os.path.join(a.root, TRAINER_REL)
    env = {"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"}
    checks = {"G-SUITE": {"pass": True, "result_sha256": b["gsuite_result_sha256"]}, "frozen_argvs_sha256": {"pass": True, "sha256": sha_file(frozen)}}
    h = hyg()
    checks["G-HYG"] = {"pass": not h, "problems": h}
    if kind in ("N1", "S1"):
        checks["G-CLOCK"] = {"pass": True, "nav_report_sha256": b["nav_report_sha256"], "note": "G-CLOCK-NAV read from the builder report (train + eval PASS, regression arm FAILS); ctx label alignment asserts run inside the trainer's selfcheck"}
    else:
        checks["G-CLOCK"] = {"pass": True, "note": "control: no new label clock; the inherited label clock is the OBD-CTL recipe's"}
    # selfcheck-only on the exact argv (--out moved to the gate folder)
    gsc = os.path.join(gd, "selfcheck_run")
    shutil.rmtree(gsc, ignore_errors=True)
    glog = os.path.join(gd, "gate.log")
    rc = _run(a.py, trainer, _with(toks, out=gsc) + ["--selfcheck-only"], a.root, env, glog)
    scp = os.path.join(gsc, "selfcheck.json")
    if rc != 0 or not os.path.isfile(scp):
        return [f"the trainer's --selfcheck-only exited rc={rc} and {'wrote' if os.path.isfile(scp) else 'did not write'} selfcheck.json (log {glog})"]
    shutil.copy2(scp, os.path.join(gd, "selfcheck.json"))
    sc = json.load(open(scp, encoding="utf-8"))
    gp = gate_problems(kind, toks, sc, a.steps)
    dvb = sc.get("declared_vs_built") == "PASS" and not gp
    checks["G-DVB"] = {"pass": dvb, "declared_vs_built": sc.get("declared_vs_built"), "problems": gp}
    checks["G-LIVE"] = {"pass": bool((sc.get("live") or {}).get("pass")) and not gp, "problems": gp, "groups_grad": (sc.get("live") or {}).get("groups_grad")}
    # a short resume run on the same argv: G-CKPT (round trip, resume) and G-EVAL (the evaluators' build path)
    short = os.path.join(gd, "short")
    shutil.rmtree(short, ignore_errors=True)
    st = _with(toks, out=short, steps=4, ckpt_every=2, log_every=1)
    slog = os.path.join(gd, "short.log")
    import torch
    rc1 = _run(a.py, trainer, st, a.root, {**env, "V8F_TRAIN_CRASH_AT": "3"}, slog)
    ck = os.path.join(short, "ckpt_last.pt")
    ok1 = rc1 != 0 and os.path.isfile(ck)
    step1 = torch.load(ck, map_location="cpu", weights_only=False).get("step") if ok1 else None
    rc2 = _run(a.py, trainer, st, a.root, env, slog) if ok1 else -1
    fin = os.path.join(short, "ckpt_last.pt")
    ckrec = {"crash_rc": rc1, "resume_rc": rc2, "step_at_crash_ckpt": step1}
    if ok1 and rc2 == 0:
        c1 = torch.load(fin, map_location="cpu", weights_only=False)
        ckrec["final_step"] = c1.get("step")
        sd = c1["model"]
        torch.save(sd, os.path.join(gd, "rt.pt"))
        rt = torch.load(os.path.join(gd, "rt.pt"), map_location="cpu", weights_only=False)
        eq = set(rt) == set(sd) and all(torch.equal(rt[k], sd[k]) for k in sd)
        os.remove(os.path.join(gd, "rt.pt"))
        ckrec["roundtrip_bit_equal"] = bool(eq)
        init_sd = torch.load(flag_value(toks, "--init-ckpt"), map_location="cpu", weights_only=False)["model"]
        mv = moved_identity(init_sd, sd, kind)
        ckrec["moved_identity_problems"] = mv
        ge, rec = geval(fin, sc.get("levers"))
        checks["G-EVAL"] = {"pass": not ge, "problems": ge, **rec, "scope": "strict load through the evaluators' build path, built levers == selfcheck levers, regression arm (cfg drops ctx_heads) refused; the 8-window forward_infer parity is NOT run here"}
        checks["G-CKPT"] = {"pass": bool(eq) and not mv and ckrec.get("final_step") is not None and step1 is not None and int(ckrec["final_step"]) > int(step1), **ckrec}
    else:
        checks["G-EVAL"] = {"pass": False, "problems": ["no checkpoint to load"]}
        checks["G-CKPT"] = {"pass": False, **ckrec, "problems": ["the crash run did not leave a checkpoint, or the resume run failed (logs in the gate folder)"]}
    shutil.rmtree(short, ignore_errors=True)                                  # own temporary data
    shutil.rmtree(gsc, ignore_errors=True)
    failed = [c for c in REQUIRED_CHECKS if not (checks.get(c) or {}).get("pass")]
    json.dump(checks, open(os.path.join(gd, "checks.json"), "w", encoding="utf-8"), indent=1)
    if failed:
        return [f"check {c} did not pass: {json.dumps(checks.get(c))[:600]}" for c in failed]
    write_token(os.path.join(gd, "gate_token.json"), b, checks, sha_file(os.path.join(gd, "selfcheck.json")))
    return []


def run_arm(tag, kind, a):
    """The runner: REFUSE unless the token binds exactly the argv on disk, the commit, every data file and the tree; then start the trainer under the same argv. Returns the process return code (2 = refused)."""
    import subprocess
    rd = os.path.join(a.runs, tag)
    try:
        toks = json.load(open(os.path.join(rd, "argv.json"), encoding="utf-8"))
    except (OSError, ValueError):
        print(f"REFUSED {tag}: no argv.json")
        return 2
    nav_report = os.path.join(a.inputs, "strat1", "navtext_nav_report.json") if kind in ("N1", "S1") else ""
    trainer = a.trainer or os.path.join(a.root, TRAINER_REL)
    bad = token_problems(os.path.join(rd, "gate", "gate_token.json"), kind, toks, a.root, a.commit, a.suite, os.path.join(rd, "gate", "selfcheck.json"), trainer, nav_report)
    if bad:
        print(f"REFUSED {tag}: " + "; ".join(bad))
        return 2
    print(f"TOKEN-OK {tag}")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"})
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    with open(os.path.join(rd, "logs", "train.log"), "ab") as lf:
        rc = subprocess.run([a.py, trainer] + list(toks), cwd=a.root, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("argv"); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=tuple(SEEDS)); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--nav-npz", default=""); p.add_argument("--ctx-npz", default="")
    p = sub.add_parser("gate"); p.add_argument("kind", choices=KINDS); p.add_argument("--selfcheck", required=True); p.add_argument("--argv-json", required=True)
    p = sub.add_parser("post"); p.add_argument("kind", choices=KINDS); p.add_argument("--init", required=True); p.add_argument("--final", required=True)
    p = sub.add_parser("geval"); p.add_argument("--ckpt", required=True); p.add_argument("--selfcheck", default="")
    sub.add_parser("hyg")
    for nm in ("mint", "run"):
        p = sub.add_parser(nm); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=tuple(SEEDS))
        p.add_argument("--root", required=True, help="the exported tree (holds stack/ and taniteval/)"); p.add_argument("--inputs", required=True); p.add_argument("--cache-dir", required=True)
        p.add_argument("--runs", required=True); p.add_argument("--commit", required=True); p.add_argument("--suite", required=True, help="the G-SUITE result file written by `suite`")
        p.add_argument("--steps", type=int, default=RUN_STEPS); p.add_argument("--py", default=sys.executable); p.add_argument("--gpu", default="0"); p.add_argument("--trainer", default="")
    p = sub.add_parser("suite"); p.add_argument("--base-junit", required=True); p.add_argument("--new-junit", required=True); p.add_argument("--base-commit", required=True); p.add_argument("--new-commit", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "argv":
        print(json.dumps(render(a.kind, a.seed, a.root, a.out, a.nav_npz, a.ctx_npz)))
        return 0
    if a.cmd == "gate":
        bad = gate_problems(a.kind, json.load(open(a.argv_json, encoding="utf-8")), json.load(open(a.selfcheck, encoding="utf-8")))
    elif a.cmd == "post":
        import torch
        bad = moved_identity(torch.load(a.init, map_location="cpu", weights_only=False)["model"], torch.load(a.final, map_location="cpu", weights_only=False)["model"], a.kind)
    elif a.cmd == "suite":
        r = suite_compare(a.base_junit, a.new_junit, a.new_commit, a.base_commit)
        json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
        print(json.dumps({k: r[k] for k in ("base", "new", "n_regressions", "n_new_only")}))
        return 0 if r["n_regressions"] == 0 else 1
    elif a.cmd == "mint":
        tag = arm_tag(a.kind, a.seed, a.steps)
        bad = mint(tag, a.kind, a.seed, a)
        print("TOKEN-WRITTEN " + tag if not bad else "REFUSED: " + "; ".join(bad))
        return 0 if not bad else 1
    elif a.cmd == "run":
        return run_arm(arm_tag(a.kind, a.seed, a.steps), a.kind, a)
    elif a.cmd == "geval":
        bad, rec = geval(a.ckpt, json.load(open(a.selfcheck, encoding="utf-8"))["levers"] if a.selfcheck else None)
        print(json.dumps(rec))
    else:
        bad = hyg()
    print("PASS" if not bad else "REFUSED: " + "; ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
