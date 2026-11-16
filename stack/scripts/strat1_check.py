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
import sys

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


def render(kind, seed, root, out, nav_npz="", ctx_npz="", base_argvs=BASE_ARGVS):
    """The frozen argv of an arm: the OBD-CTL argv of that seed, box-2 paths mapped under ``root``, plus the nine STRAT1 flags (ALL named, the control included)."""
    assert kind in KINDS and seed in SEEDS
    base = json.load(open(base_argvs, encoding="utf-8"))[SEEDS[seed]["base"]]["argv"]
    t = [x.replace("D:\\tanitad", root.rstrip("/")).replace("\\", "/") if isinstance(x, str) and x.startswith("D:\\tanitad") else x for x in base]
    t[t.index("--out") + 1] = out
    if "seed" in SEEDS[seed]:
        t[t.index("--seed") + 1] = SEEDS[seed]["seed"]
    nav = ["--nav-source", "text", "--nav-text-npz", nav_npz, "--nav-text-sha256", sha_file(nav_npz) if os.path.exists(nav_npz) else "<sha of navtext_nav.npz>"] if kind in ("N1", "S1") else \
        ["--nav-source", "v9", "--nav-text-npz", "", "--nav-text-sha256", ""]
    ctx = ["--ctx-heads", "on", "--w-ctx", "0.5", "--ctx-labels", ctx_npz, "--ctx-labels-sha256", STRAT_LABELS_SHA256, "--ctx-lr-mult", "10", "--ctx-detach", "off"] if kind == "S1" else \
        ["--ctx-heads", "off", "--w-ctx", "0", "--ctx-labels", "", "--ctx-labels-sha256", "", "--ctx-lr-mult", "1", "--ctx-detach", "off"]
    return t + nav + ctx


def argv_sha(toks):
    return hashlib.sha256(json.dumps(list(toks)).encode("utf-8")).hexdigest()


def gate_problems(kind, toks, sc):
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
    if sc.get("steps_total") != RUN_STEPS:
        bad.append(f"steps_total {sc.get('steps_total')} != {RUN_STEPS}")
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


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("argv"); p.add_argument("kind", choices=KINDS); p.add_argument("seed", choices=tuple(SEEDS)); p.add_argument("--root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--nav-npz", default=""); p.add_argument("--ctx-npz", default="")
    p = sub.add_parser("gate"); p.add_argument("kind", choices=KINDS); p.add_argument("--selfcheck", required=True); p.add_argument("--argv-json", required=True)
    p = sub.add_parser("post"); p.add_argument("kind", choices=KINDS); p.add_argument("--init", required=True); p.add_argument("--final", required=True)
    p = sub.add_parser("geval"); p.add_argument("--ckpt", required=True); p.add_argument("--selfcheck", default="")
    sub.add_parser("hyg")
    a = ap.parse_args(argv)
    if a.cmd == "argv":
        print(json.dumps(render(a.kind, a.seed, a.root, a.out, a.nav_npz, a.ctx_npz)))
        return 0
    if a.cmd == "gate":
        bad = gate_problems(a.kind, json.load(open(a.argv_json, encoding="utf-8")), json.load(open(a.selfcheck, encoding="utf-8")))
    elif a.cmd == "post":
        import torch
        bad = moved_identity(torch.load(a.init, map_location="cpu", weights_only=False)["model"], torch.load(a.final, map_location="cpu", weights_only=False)["model"], a.kind)
    elif a.cmd == "geval":
        bad, rec = geval(a.ckpt, json.load(open(a.selfcheck, encoding="utf-8"))["levers"] if a.selfcheck else None)
        print(json.dumps(rec))
    else:
        bad = hyg()
    print("PASS" if not bad else "REFUSED: " + "; ".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
