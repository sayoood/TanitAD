"""PREREG_V8F_CKPT120 study checker (the pattern of ``strat1_check.py``; this file imports it and adds nothing to it). It maps the launch-gate checks onto the trainer's own selfcheck for the seven CKPT120 arms:

  H1_s0 H1_s1 H1_rep   N1's argv + the far-point levers          (--s-far on, no nav_s)
  H2_s0 H2_s1          H1 + --nav-strat on  (real nav_s)
  R1_s0                H1 + --s-far-label-perm clip   (deliberate regression of the far head; must fail the read)
  R2_s0                H2 + --nav-strat-variant donor (deliberate regression of nav_s; must fail the read)

  G-HYG    V8FMiniConfig refuses every bad CKPT120 value / unknown key            -> ``hyg``
  G-DVB    trainer ``declared_vs_built`` (bad_sfar, nav_strat) + every argv lever == ``selfcheck.levers``  -> ``gate_problems``
  G-LIVE   ``ckpt120_probe`` (every key with the value it must read: non-zero where the lever is on, EXACTLY 0.0 for each control) + the inherited probes  -> ``gate_problems``
  G-CLOCK  the nav_s builder report (G-CLOCK-NAV-S: countdown error <= raw error, the ``+ travelled`` arm FAILS) for the nav_s arms; the label clock of RC-A120 is K-LABEL, bound by the table's report sha
  G-EVAL   strict load through the evaluator's build path, built levers == selfcheck levers, a cfg that drops the far head / e_nav_s is REFUSED  -> ``geval``
  G-CKPT   save / load round trip bit-equal, resume, far head and e_nav_s MOVED from their inits   -> ``moved_identity``
  G-SUITE  pytest on a clean export of the commit, 0 regressions vs the base commit  -> ``suite`` (junit compare) + ``suite_problems``

The token ``ckpt120-gate-token-v1`` binds the commit, the argv sha256, EVERY data sha256 (the RC-A120 table, the nav_s table, the nav and the other STRAT1 inputs), the tree manifest, the trainer, this checker and the G-SUITE result.
``run`` REFUSES (rc 2) unless the token still binds exactly what is on disk. The data sha256s are PARAMETERS: nothing here pins the RC-A120 / nav_s tables (the final tables are produced on the release of record).
This file only DECIDES and launches the trainer under a valid token; the pilot timing / chain runner / pod preflight is the MM's launch loop (UNVERIFIED here).

Subcommands: ``argv KIND SEED``, ``gate KIND --selfcheck F --argv-json J``, ``post KIND --init I --final F``, ``geval --ckpt C``, ``hyg``, ``suite``, ``mint KIND SEED``, ``run KIND SEED``.
"""
from __future__ import annotations

import argparse
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

import strat1_check as S1  # noqa: E402

TOKEN_SCHEMA = "ckpt120-gate-token-v1"
WRITTEN_BY = "ckpt120_check.py mint"
SUITE_SCHEMA = "ckpt120-gsuite-result-v1"
SUITE_WRITER = "ckpt120_check.py suite"
RUN_STEPS = S1.RUN_STEPS
W_CKPT_FAR = "0.3333"
S_FAR_LR_MULT = "10"
FAR_ARC_M = 120.0
#: the 11 CKPT120 flags; EVERY one is named in EVERY arm (a flag whose default disables it must be named, or the arm did not test it)
CKPT_FLAGS = ("--s-far", "--w-ckpt-far", "--s-far-lr-mult", "--rc120-npz", "--rc120-sha256", "--s-far-label-perm", "--s-adapter-detach", "--nav-strat", "--nav-strat-npz", "--nav-strat-sha256", "--nav-strat-variant")
ARMS = {"H1": ("s0", "s1", "rep"), "H2": ("s0", "s1"), "R1": ("s0",), "R2": ("s0",)}      # the seven arms of PREREG_V8F_CKPT120 section 5 (H1_rep = H1_s0's flags, --seed 2)
KINDS = tuple(ARMS)
#: per kind: (s_far_label_perm, nav_strat, nav_strat_variant)
SPEC = {"H1": ("off", "off", "real"), "H2": ("off", "on", "real"), "R1": ("clip", "off", "real"), "R2": ("off", "on", "donor")}
DATA_EXTRA = ("--rc120-npz", "--nav-strat-npz")
REQUIRED_CHECKS = S1.REQUIRED_CHECKS
TRAINER_REL = S1.TRAINER_REL
NAV_S_REPORT = "navtext_nav_s_report.json"


def arm_tag(kind, seed, steps=RUN_STEPS):
    return f"{kind}_{seed}" if steps == RUN_STEPS else f"pilot{steps}_{kind}_{seed}"


def check_arm(kind, seed):
    if kind not in ARMS or seed not in ARMS[kind]:
        raise ValueError(f"{kind}_{seed} is not an arm of PREREG_V8F_CKPT120 (arms: {[f'{k}_{s}' for k, v in ARMS.items() for s in v]})")


def _sha(path):
    return S1.sha_file(path) if path and os.path.exists(path) else None


def render(kind, seed, root, out, nav_npz="", rc120_npz="", nav_s_npz="", base_argvs=S1.BASE_ARGVS, cache_dir=None):
    """The frozen argv of an arm: N1's argv of that seed (strat1_check.render, the nine STRAT1 flags named) plus ALL eleven CKPT120 flags. The sha256 of a table that does not exist yet is a placeholder (the mint refuses it)."""
    check_arm(kind, seed)
    perm, nav, var = SPEC[kind]
    t = S1.render("N1", seed, root, out, nav_npz, "", base_argvs=base_argvs, cache_dir=cache_dir)
    rc = _sha(rc120_npz) or "<sha of the RC-A120 table>"
    ns_npz, ns_sha = (nav_s_npz, _sha(nav_s_npz) or "<sha of navtext_nav_s.npz>") if nav == "on" else ("", "")
    return t + ["--s-far", "on", "--w-ckpt-far", W_CKPT_FAR, "--s-far-lr-mult", S_FAR_LR_MULT, "--rc120-npz", rc120_npz, "--rc120-sha256", rc, "--s-far-label-perm", perm, "--s-adapter-detach", "off",
                "--nav-strat", nav, "--nav-strat-npz", ns_npz, "--nav-strat-sha256", ns_sha, "--nav-strat-variant", var]


def argv_sha(toks):
    return S1.argv_sha(toks)


def _eq(a, b):
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return False


def gate_problems(kind, toks, sc, steps=RUN_STEPS):
    """The G-DVB + G-LIVE read of a selfcheck.json against its argv. [] = pass. Every ``ckpt120_probe`` key is read with the value it must have."""
    perm, nav, var = SPEC[kind]
    bad = list(S1.gate_problems("N1", toks, sc, steps))                 # the inherited reads: nine STRAT1 flags, declared_vs_built, live, steps, nav source text, ctx off, inherited probes, detach-upper
    miss = [f for f in CKPT_FLAGS if f not in toks]
    if miss:
        bad.append(f"the argv does not name {miss}: a flag whose default disables it must be named")
    want = {"--s-far": "on", "--s-far-lr-mult": S_FAR_LR_MULT, "--s-far-label-perm": perm, "--s-adapter-detach": "off", "--nav-strat": nav, "--nav-strat-variant": var}
    for f, v in want.items():
        if S1.flag_value(toks, f) != v:
            bad.append(f"argv {f} is {S1.flag_value(toks, f)!r}, arm {kind} needs {v!r}")
    if not _eq(S1.flag_value(toks, "--w-ckpt-far"), W_CKPT_FAR):
        bad.append(f"argv --w-ckpt-far is {S1.flag_value(toks, '--w-ckpt-far')!r}, the card's value is {W_CKPT_FAR}")
    if not S1.flag_value(toks, "--rc120-npz") or not S1.flag_value(toks, "--rc120-sha256"):
        bad.append("--rc120-npz / --rc120-sha256 are not both named")
    if (nav == "on") != bool(S1.flag_value(toks, "--nav-strat-npz") and S1.flag_value(toks, "--nav-strat-sha256")) or (nav == "off" and (S1.flag_value(toks, "--nav-strat-npz") or S1.flag_value(toks, "--nav-strat-sha256"))):
        bad.append(f"--nav-strat-npz / --nav-strat-sha256 do not match --nav-strat {nav}")
    lv = sc.get("levers", {})
    if lv.get("s_far") != "on" or not _eq(lv.get("s_far_arc_m"), FAR_ARC_M):
        bad.append(f"built lever s_far {lv.get('s_far')!r} / arc {lv.get('s_far_arc_m')!r}: the far head is not built at {FAR_ARC_M} m")
    if not _eq(lv.get("w_ckpt_far"), S1.flag_value(toks, "--w-ckpt-far")):
        bad.append(f"built w_ckpt_far {lv.get('w_ckpt_far')!r} differs from the argv {S1.flag_value(toks, '--w-ckpt-far')!r}")
    for k, v in (("s_far_label_perm", perm), ("nav_strat", nav), ("nav_strat_variant", var), ("s_adapter_detach", "off")):
        if lv.get(k, "off" if k != "nav_strat_variant" else "real") != v:
            bad.append(f"built lever {k} {lv.get(k)!r} differs from the declared {v!r}")
    rc, ns = sc.get("rc120") or {}, sc.get("nav_strat") or {}
    if rc.get("sha256") != S1.flag_value(toks, "--rc120-sha256") or rc.get("perm") != perm or not rc.get("n_ok_after_perm", 0) > 0:
        bad.append(f"rc120 record {sorted(rc)} does not match the argv (sha256 / perm {perm} / survivors)")
    if nav == "on":
        if ns.get("sha256") != S1.flag_value(toks, "--nav-strat-sha256") or ns.get("variant") != var:
            bad.append("nav_strat record does not match the argv (sha256 / variant)")
        if (var == "donor") != (ns.get("n_rows_differ_from_real", 0) > 0):
            bad.append(f"nav_strat variant {var}: n_rows_differ_from_real {ns.get('n_rows_differ_from_real')} is inconsistent")
    elif ns:
        bad.append("a nav_strat-off arm carries a nav_strat record")
    p = sc.get("ckpt120_probe")
    if not isinstance(p, dict) or p.get("skipped") or not p.get("pass"):
        bad.append("ckpt120_probe is missing, skipped or not pass")
        return bad
    nz = ("head_grad", "term_ckpt_far", "adapter_grad_s_detach_off")
    zero = ("control_head_grad_w0", "control_allmasked_term", "control_allmasked_head_grad", "control_adapter_grad_s_detach_on")
    for k in nz:
        if not (isinstance(p.get(k), (int, float)) and p[k] > 0.0):
            bad.append(f"ckpt120_probe {k} must read > 0, read {p.get(k)!r}")
    for k in zero:
        if p.get(k) != 0.0:
            bad.append(f"ckpt120_probe control {k} must read EXACTLY 0.0, read {p.get(k)!r}")
    if p.get("t_iso_far") is not True:
        bad.append("ckpt120_probe t_iso_far is not True (T-ISO-FAR)")
    if not _eq(p.get("w_ckpt_far"), S1.flag_value(toks, "--w-ckpt-far")):
        bad.append("ckpt120_probe w_ckpt_far differs from the argv")
    td, te = p.get("total_diff"), p.get("total_diff_expected")
    if not (isinstance(td, (int, float)) and isinstance(te, (int, float)) and te > 0.0 and abs(td - te) <= 1e-3 * te):
        bad.append(f"ckpt120_probe: the total moved by {td!r}, w x term is {te!r}")
    if nav == "on":
        if not (isinstance(p.get("nav_s_grad"), (int, float)) and p["nav_s_grad"] > 0.0):
            bad.append(f"ckpt120_probe nav_s_grad must read > 0, read {p.get('nav_s_grad')!r}")
        for k in ("nav_s_moves_s", "nav_s_t_o_identical", "nav_s_in_keys_read"):
            if p.get(k) is not True:
                bad.append(f"ckpt120_probe {k} must be True, read {p.get(k)!r}")
    elif any(k.startswith("nav_s") for k in p):
        bad.append("a nav_strat-off arm carries nav_s probe keys")
    if not (sc.get("live", {}).get("groups_grad") or {}).get("SFAR", 1.0) > 0.0:     # the in-trainer group name is not part of this contract: read it only when it exists
        bad.append("G-LIVE: the SFAR group has no gradient")
    return bad


def moved_identity(init_sd, fin_sd, kind):
    """The post check: the STRAT1 groups moved; ``s_far_out.*`` exists and left its zero init; ``e_nav_s.*`` exists iff the arm has nav_s and left its seeded init; no ctx key."""
    import torch
    bad = list(S1.moved_identity(init_sd, fin_sd, "N1"))
    fk = [k for k in fin_sd if k.startswith("s_far_out.")]
    if not fk:
        bad.append("checkpoint has no s_far_out keys")
    elif all(not bool(fin_sd[k].abs().sum() > 0) for k in fk):
        bad.append("s_far_out is still at its zero init")
    nk = [k for k in fin_sd if k.startswith("e_nav_s.")]
    if SPEC[kind][1] == "on":
        if not nk:
            bad.append(f"{kind} checkpoint has no e_nav_s keys")
        else:
            from tanitad.models import v8f_mini as M
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(M.NAV_S_INIT_SEED)
                fresh = torch.nn.Linear(5, int(fin_sd["e_nav_s.weight"].shape[0])).state_dict()
            if all(torch.equal(fresh[k[len("e_nav_s."):]], fin_sd[k]) for k in nk):
                bad.append("e_nav_s did not move from its seeded init")
    elif nk:
        bad.append(f"e_nav_s keys {nk[:2]} in a {kind} checkpoint")
    return bad


CKPT_LEVER_KEYS = ("s_far", "w_ckpt_far", "s_far_arc_m", "nav_strat", "nav_strat_variant", "s_far_label_perm", "s_adapter_detach")


def geval(ckpt_path, selfcheck_levers=None):
    """G-EVAL: strict load through the evaluators' own build path; built CKPT120 levers == the selfcheck's; the deliberate-regression arms (a cfg that drops ``s_far``; for a nav_s arm a cfg that drops ``nav_strat``) must be REFUSED.
    Returns (problems, record)."""
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
    rec["built_levers"] = {k: lv[k] for k in CKPT_LEVER_KEYS if k in lv}
    if selfcheck_levers is not None:
        for k in CKPT_LEVER_KEYS + ("navonly", "wm", "detach_upper"):
            if lv.get(k) != selfcheck_levers.get(k):
                bad.append(f"built lever {k}: evaluator {lv.get(k)!r} vs selfcheck {selfcheck_levers.get(k)!r}")
    for name, drop in (("s_far", {"s_far": "off", "w_ckpt_far": 0.0, "s_far_label_perm": "off", "s_adapter_detach": "off"}), ("nav_strat", {"nav_strat": "off", "nav_strat_variant": "real"})):
        if ck["cfg"].get(name) == "on":
            try:
                build({**ck["cfg"], **drop})
                bad.append(f"regression arm: a cfg that drops {name} was NOT refused (the evaluator would silently skip the head)")
            except Exception:                                           # noqa: BLE001
                rec[f"regression_arm_drop_{name}_refused"] = True
    return bad, rec


def hyg():
    from tanitad.models import v8f_mini as M
    n = 0
    bads = (dict(s_far="maybe"), dict(w_ckpt_far=-1.0), dict(w_ckpt_far=float("nan")), dict(nav_strat="x"), dict(nav_strat_variant="x"), dict(s_far_label_perm="x"), dict(s_adapter_detach="x"))
    for kw in bads:
        try:
            M.V8FMiniConfig(**kw)
        except ValueError:
            n += 1
    try:
        M.V8FMiniConfig(s_farz="on")
    except TypeError:
        n += 1
    return [] if n == len(bads) + 1 else [f"only {n} of {len(bads) + 1} bad configs were refused"]


# ---- the token ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def argv_kind_problems(kind, toks):
    """The argv on disk must be the argv of THIS arm kind (the lever values the kind fixes), so a token cannot start a different arm than it was minted for."""
    perm, nav, var = SPEC[kind]
    got = {f: S1.flag_value(toks, f) for f in ("--s-far-label-perm", "--nav-strat", "--nav-strat-variant")}
    want = {"--s-far-label-perm": perm, "--nav-strat": nav, "--nav-strat-variant": var}
    return [f"argv {f} is {got[f]!r}, arm {kind} needs {want[f]!r}" for f in want if got[f] != want[f]]


def data_binding(toks):
    """STRAT1's data binding (every input the argv names: size, sha256, md5) plus the RC-A120 table and the nav_s table, each re-hashed against the sha256 the argv declares. Returns (dict, problems)."""
    d, bad = S1.data_binding(toks)
    for fl, shf in (("--rc120-npz", "--rc120-sha256"), ("--nav-strat-npz", "--nav-strat-sha256")):
        v = S1.flag_value(toks, fl)
        if not v:
            continue
        e = S1.hash_entry(v)
        if e is None:
            bad.append(f"{fl} {v!r} is not a file")
            continue
        d[fl] = {"path": v, **e}
        if S1.flag_value(toks, shf) != e["sha256"]:
            bad.append(f"{shf} differs from the sha256 of the file named by {fl}")
    return d, bad


def suite_problems(path, commit):
    """The G-SUITE result file written by ``suite``: 0 regressions of the new tree against the base, for this commit. Returns (sha256 of the file or None, problems)."""
    try:
        r = json.load(open(path, encoding="utf-8"))
    except (OSError, ValueError):
        return None, ["no G-SUITE result file"]
    bad = []
    if r.get("schema") != SUITE_SCHEMA or r.get("written_by") != SUITE_WRITER:
        bad.append("the G-SUITE result file was not written by the checker")
    if r.get("n_regressions") != 0 or r.get("regressions"):
        bad.append(f"G-SUITE has {r.get('n_regressions')} regressions")
    if not r.get("new", {}).get("n_tests") or not r.get("base", {}).get("n_tests"):
        bad.append("the G-SUITE result has an empty side")
    if r.get("new_commit") != commit:
        bad.append(f"the G-SUITE result is for commit {r.get('new_commit')}, not {commit}")
    return S1.sha_file(path), bad


def suite_compare(base_dir, new_dir, new_commit, base_commit):
    r = S1.suite_compare(base_dir, new_dir, new_commit, base_commit)
    r["schema"], r["written_by"] = SUITE_SCHEMA, SUITE_WRITER
    return r


def nav_s_report_problems(kind, nav_s_npz, report_path):
    """G-CLOCK-NAV-S, read from the REAL builder report (``build_navtext_nav.py --nav-s``): per split ``G-CLOCK-NAV_pass`` must be true, the ``G-CLOCK-NAV_regression_arm_plus_travelled`` arm must FAIL, ``rule`` must name the nav_s variant (``d_ann = infinity``), and ``navtext_nav_sha256`` must describe this npz."""
    if SPEC[kind][1] != "on":
        return []
    if not nav_s_npz or not os.path.isfile(nav_s_npz):
        return [f"arm kind {kind} needs a nav_s table, the argv names {nav_s_npz!r}"]
    try:
        r = json.load(open(report_path, encoding="utf-8"))
    except (OSError, ValueError):
        return ["no nav_s builder report"]
    bad = []
    if "d_ann = infinity" not in str(r.get("rule", "")):
        bad.append("the nav_s report's rule does not name the nav_s variant (d_ann = infinity)")
    for sp in ("train", "eval"):
        if r.get(sp, {}).get("G-CLOCK-NAV_pass") is not True:
            bad.append(f"G-CLOCK-NAV {sp} is not pass")
        if r.get(sp, {}).get("G-CLOCK-NAV_regression_arm_plus_travelled", {}).get("pass") is not False:
            bad.append(f"G-CLOCK-NAV {sp} regression arm did not fail as required")
    if r.get("navtext_nav_sha256") != S1.sha_file(nav_s_npz):
        bad.append("the nav_s report describes a different npz")
    return bad


def binding(kind, toks, root, commit, suite_path, trainer=None, nav_s_report=""):
    """Everything the token binds, recomputed NOW. Returns (dict, problems)."""
    d, bad = data_binding(toks)
    tr = trainer or os.path.join(root, TRAINER_REL)
    grid = os.path.join(root, *S1.EVAL_GRID_REL)
    if os.path.isfile(grid):
        d["eval_grid_windows.json"] = {"path": grid, **S1.hash_entry(grid)}
    ssha, sbad = suite_problems(suite_path, commit)
    bad = list(bad) + sbad
    if nav_s_report:
        bad += nav_s_report_problems(kind, S1.flag_value(toks, "--nav-strat-npz"), nav_s_report)
    b = {"kind": kind, "commit": commit, "argv_sha256": argv_sha(toks), "data": d, "tree_manifest_sha256": S1.tree_manifest_sha(root),
         "trainer_sha256": S1.sha_file(tr) if os.path.isfile(tr) else None, "gsuite_result_sha256": ssha, "checker_sha256": S1.sha_file(os.path.abspath(__file__)),
         "nav_s_report_sha256": S1.sha_file(nav_s_report) if nav_s_report and os.path.isfile(nav_s_report) else None}
    if b["trainer_sha256"] is None:
        bad.append("trainer not found")
    return b, bad


def body_sha(tok):
    return S1.body_sha(tok)


def write_token(path, b, checks, selfcheck_sha):
    """The ONLY writer of a token (called by ``mint`` after every required check read PASS)."""
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


def token_problems(token_path, kind, toks, root, commit, suite_path, selfcheck_path="", trainer=None, nav_s_report=""):
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
    p += argv_kind_problems(kind, toks)
    want, bad = binding(kind, toks, root, commit, suite_path, trainer, nav_s_report)
    p += list(bad)
    for k, v in want.items():
        if k == "data":
            for name in sorted(set(v) | set(tok.get("data", {}))):
                if v.get(name) != tok.get("data", {}).get(name):
                    p.append(f"token data {name} differs from the current file")
        elif tok.get(k) != v:
            p.append(f"token {k} differs from the current {k}")
    if selfcheck_path and (not os.path.isfile(selfcheck_path) or S1.sha_file(selfcheck_path) != tok.get("selfcheck_sha256")):
        p.append("selfcheck.json differs from the token's")
    return p


# ---- mint and run ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def _inputs(a, kind, seed, out):
    frozen = os.path.join(a.inputs, "frozen", "argvs_obd.json")
    toks = render(kind, seed, a.inputs, out, os.path.join(a.inputs, "strat1", "navtext_nav.npz"), os.path.join(a.inputs, "ckpt120", "rc120_table.npz"),
                  os.path.join(a.inputs, "ckpt120", "navtext_nav_s.npz"), base_argvs=frozen, cache_dir=a.cache_dir)
    if a.steps != RUN_STEPS:
        toks = S1._with(toks, steps=a.steps)
    return toks, frozen, os.path.join(a.inputs, "ckpt120", NAV_S_REPORT)


def mint(tag, kind, seed, a):
    """Mint ``<runs>/<tag>/gate/gate_token.json``. Returns the list of problems (empty = token written)."""
    rd = os.path.join(a.runs, tag)
    gd = os.path.join(rd, "gate")
    out = os.path.join(rd, "train")
    os.makedirs(gd, exist_ok=True)
    toks, frozen, nsr = _inputs(a, kind, seed, out)
    json.dump(toks, open(os.path.join(rd, "argv.json"), "w", encoding="utf-8"))
    b, bad = binding(kind, toks, a.root, a.commit, a.suite, nav_s_report=nsr if SPEC[kind][1] == "on" else "")
    if bad:
        return ["binding: " + x for x in bad]
    trainer = os.path.join(a.root, TRAINER_REL)
    env = {"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"}
    checks = {"G-SUITE": {"pass": True, "result_sha256": b["gsuite_result_sha256"]}, "frozen_argvs_sha256": {"pass": True, "sha256": S1.sha_file(frozen)}}
    h = hyg() + S1.hyg()
    checks["G-HYG"] = {"pass": not h, "problems": h}
    checks["G-CLOCK"] = {"pass": True, "nav_s_report_sha256": b["nav_s_report_sha256"], "note": "G-CLOCK-NAV-S read from the builder report for nav_s arms; the RC-A120 label clock is K-LABEL (the table's own report, bound by its sha256); the inherited label clocks are N1's"}
    gsc = os.path.join(gd, "selfcheck_run")
    shutil.rmtree(gsc, ignore_errors=True)
    glog = os.path.join(gd, "gate.log")
    rc = S1._run(a.py, trainer, S1._with(toks, out=gsc) + ["--selfcheck-only"], a.root, env, glog)
    scp = os.path.join(gsc, "selfcheck.json")
    if rc != 0 or not os.path.isfile(scp):
        return [f"the trainer's --selfcheck-only exited rc={rc} and {'wrote' if os.path.isfile(scp) else 'did not write'} selfcheck.json (log {glog})"]
    shutil.copy2(scp, os.path.join(gd, "selfcheck.json"))
    sc = json.load(open(scp, encoding="utf-8"))
    gp = gate_problems(kind, toks, sc, a.steps)
    checks["G-DVB"] = {"pass": sc.get("declared_vs_built") == "PASS" and not gp, "declared_vs_built": sc.get("declared_vs_built"), "problems": gp}
    checks["G-LIVE"] = {"pass": bool((sc.get("live") or {}).get("pass")) and not gp, "problems": gp, "groups_grad": (sc.get("live") or {}).get("groups_grad"), "ckpt120_probe": sc.get("ckpt120_probe")}
    short = os.path.join(gd, "short")
    shutil.rmtree(short, ignore_errors=True)
    st = S1._with(toks, out=short, steps=4, ckpt_every=2, log_every=1)
    slog = os.path.join(gd, "short.log")
    import torch
    rc1 = S1._run(a.py, trainer, st, a.root, {**env, "V8F_TRAIN_CRASH_AT": "3"}, slog)
    ck = os.path.join(short, "ckpt_last.pt")
    ok1 = rc1 != 0 and os.path.isfile(ck)
    step1 = torch.load(ck, map_location="cpu", weights_only=False).get("step") if ok1 else None
    rc2 = S1._run(a.py, trainer, st, a.root, env, slog) if ok1 else -1
    ckrec = {"crash_rc": rc1, "resume_rc": rc2, "step_at_crash_ckpt": step1}
    if ok1 and rc2 == 0:
        c1 = torch.load(ck, map_location="cpu", weights_only=False)
        ckrec["final_step"] = c1.get("step")
        sd = c1["model"]
        torch.save(sd, os.path.join(gd, "rt.pt"))
        rt = torch.load(os.path.join(gd, "rt.pt"), map_location="cpu", weights_only=False)
        eq = set(rt) == set(sd) and all(torch.equal(rt[k], sd[k]) for k in sd)
        os.remove(os.path.join(gd, "rt.pt"))
        ckrec["roundtrip_bit_equal"] = bool(eq)
        init_sd = torch.load(S1.flag_value(toks, "--init-ckpt"), map_location="cpu", weights_only=False)["model"]
        mv = moved_identity(init_sd, sd, kind)
        ckrec["moved_identity_problems"] = mv
        ge, rec = geval(ck, sc.get("levers"))
        checks["G-EVAL"] = {"pass": not ge, "problems": ge, **rec, "scope": "strict load through the evaluator build path, built CKPT120 levers == selfcheck levers, regression arms (cfg drops s_far / nav_strat) refused; the window-batch parity of the far wrappers is their own test"}
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
    write_token(os.path.join(gd, "gate_token.json"), b, checks, S1.sha_file(os.path.join(gd, "selfcheck.json")))
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
    nsr = os.path.join(a.inputs, "ckpt120", NAV_S_REPORT) if SPEC[kind][1] == "on" else ""
    trainer = a.trainer or os.path.join(a.root, TRAINER_REL)
    bad = token_problems(os.path.join(rd, "gate", "gate_token.json"), kind, toks, a.root, a.commit, a.suite, os.path.join(rd, "gate", "selfcheck.json"), trainer, nsr)
    if bad:
        print(f"REFUSED {tag}: " + "; ".join(bad))
        return 2
    print(f"TOKEN-OK {tag}")
    env = dict(os.environ)
    env.update({"PYTHONPATH": os.path.join(a.root, "stack") + os.pathsep + a.root, "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_DISABLE_XET": "1"})
    os.makedirs(os.path.join(rd, "logs"), exist_ok=True)
    with open(os.path.join(rd, "logs", "train.log"), "ab") as lf:
        return subprocess.run([a.py, trainer] + list(toks), cwd=a.root, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    seeds = sorted({s for v in ARMS.values() for s in v})
    p = sub.add_parser("argv"); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=seeds); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--nav-npz", default=""); p.add_argument("--rc120-npz", default=""); p.add_argument("--nav-s-npz", default="")
    p = sub.add_parser("gate"); p.add_argument("kind", choices=KINDS); p.add_argument("--selfcheck", required=True); p.add_argument("--argv-json", required=True)
    p = sub.add_parser("post"); p.add_argument("kind", choices=KINDS); p.add_argument("--init", required=True); p.add_argument("--final", required=True)
    p = sub.add_parser("geval"); p.add_argument("--ckpt", required=True); p.add_argument("--selfcheck", default="")
    sub.add_parser("hyg")
    for nm in ("mint", "run"):
        p = sub.add_parser(nm); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=seeds)
        p.add_argument("--root", required=True); p.add_argument("--inputs", required=True); p.add_argument("--cache-dir", required=True); p.add_argument("--runs", required=True)
        p.add_argument("--commit", required=True); p.add_argument("--suite", required=True); p.add_argument("--steps", type=int, default=RUN_STEPS)
        p.add_argument("--py", default=sys.executable); p.add_argument("--gpu", default="0"); p.add_argument("--trainer", default="")
    p = sub.add_parser("suite"); p.add_argument("--base-junit", required=True); p.add_argument("--new-junit", required=True); p.add_argument("--base-commit", required=True); p.add_argument("--new-commit", required=True); p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd in ("argv", "mint", "run"):
        try:
            check_arm(a.kind, a.seed)
        except ValueError as e:
            print("REFUSED: " + str(e))
            return 1
    if a.cmd == "argv":
        print(json.dumps(render(a.kind, a.seed, a.root, a.out, a.nav_npz, a.rc120_npz, a.nav_s_npz)))
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
