#!/usr/bin/env python3
"""G-REACH-STEPB: does the L2 EP head REACH the search, on the ACTUAL caller?  (CLAUDE.md "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER", the class that cost four arms in one day.)

The chain proved here is the one `stepB.py search` runs:   argv -> stepB.build_scorer -> l2_scorer (head file, sha256, install) -> real_scorer.make_score_fn -> score_fn
                                                            -> stepB.search_token -> tts_cem.cem_search -> score_fn(population) -> logits -> aggregate -> elites.
Every check observes the LOGITS THE SEARCH RECEIVES (a spy on score_fn) and the (u, d) THE SEARCH REQUESTS (a spy on tts_cem.composite); none inspects a flag.

  (a) C-ID      a head file whose EP row is P2's own  =>  the search output is BIT FOR BIT P2's (raw / exec / theta / moved / history), every set scored is bit identical, logit diff EXACTLY 0.0
  (b) INVERTED  a planted head with the EP row negated  =>  on every set the two arms score identically, the EP logits the search sees are EXACTLY negated, every other column identical
  (d) DIFFER    the real L2 head vs P2 on one real set: EP column differs (> 0.0), the five other columns EXACTLY 0.0   (and exactly 0.0 under (a))
  (e) C8 REACH  the config's sigma0 and clips reach the search: the first population equals the one recomputed from (seed, sigma0, clips) bit for bit; with a planted scorer that rewards each extreme,
                the extreme observed (u, d) EQUAL the configured clip, never exceeding it, in both directions
  (f) END-TO-END `stepB.run_search_cli` (the launch itself, minus only the stamp it is about to earn) writes a shard whose recorded logits show the head: the pick's EP logit differs from plain P2's, the five
                other columns are EXACTLY P2's, and the recorded head provenance names the file's sha256 and an installed EP row that is not P2's
  (c) MUTATE    delete the line that installs the head (@INSTALL) -> the read-back raises; delete install AND read-back -> check (b) ALONE must go RED; the caller wiring (@WIRE) -> (b) and (d) RED;
                the clip lines (@CLIPS) -> (e) RED;  the sigma kwarg (@SIGMA) -> (e) RED;  the launch's own scorer build (@CLIFN) -> the launch refuses / (f) RED.
                A mutation that changed nothing, or touched an unexpected number of lines, is itself a failure.
  (a), (b) and (e) drive `stepB.run_search` (what the launch loops), not a private copy of the call; (a) is run on a token where the search MOVES the pick, so the identity is not trivially 'nothing moved'.

PASS writes `G-REACH-STEPB_PASS_<head12>.json`, the stamp `stepB.py search` requires: it binds head sha256, checkpoint sha256, config sha256, bundle sha256, this code's sha256 and L1's pinned code.
Any failure writes `..._FAIL_...json` with the reasons.  CPU only.

    CUDA_VISIBLE_DEVICES=-1 python g_reach_stepb.py --arm L2 --head <npz> --head-sha256 <hex> --bundle <npz> --ctx <npy> --ckpt <pt> --ckpt-sha256 <hex> --refe-dir <refe> \
        --config stepB_config.json --config-sha256 <hex> --out-dir <dir> [--token N]
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import l2_scorer  # noqa: E402
import stepB  # noqa: E402

A, T = stepB.A, stepB.T
EP = l2_scorer.EP
OTHERS = [c for c in range(6) if c != EP]


# ------------------------------------------------------------------------------------------------------------------------------ spies
class LogitSpy:
    """wraps a score_fn: records (sha1 of the set it was handed, the logits it returned) for EVERY call the search makes"""

    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def bind(self, n):
        self.fn.bind(n)

    @property
    def info(self):
        return self.fn.info

    def __call__(self, S):
        out = self.fn(S)
        h = hashlib.sha1(np.ascontiguousarray(np.asarray(S), dtype=np.float32).tobytes()).hexdigest()
        self.calls.append((h, np.array(out, dtype=np.float64, copy=True)))
        return out


@contextlib.contextmanager
def spy_composite(Tmod=None):
    """records every (u, d) tts_cem.cem_search asks `composite` for, in order"""
    Tm = Tmod or T
    orig, calls = Tm.composite, []

    def spy(raw_pick, u, d, *a, **k):
        calls.append((float(u), float(d)))
        return orig(raw_pick, u, d, *a, **k)

    Tm.composite = spy
    try:
        yield calls
    finally:
        Tm.composite = orig


def planted_score_fn(sign: float):
    """a scorer that needs NO model: EP rises with forward progress and with left offset (sign +1) or falls (sign -1); every other column constant.  It rewards the extremes of the search space."""
    def fn(S):
        S = np.asarray(S, np.float64)
        lg = np.zeros((S.shape[0], 6))
        lg[:, 0] = lg[:, 1] = 4.0
        lg[:, EP] = sign * (0.5 * S[:, -1, 0] + 0.5 * S[:, -1, 1])
        return lg
    fn.bind = lambda n: None
    fn.info = {"planted": sign}
    return fn


# ------------------------------------------------------------------------------------------------------------------------------ mutation harness ("MUTATE, do not inspect")
def mutate_module(path: str, edits: dict, name: str):
    """load `path` as a module with every line tagged `# @TAG` replaced by `edits[TAG]` (keeping its indentation).  `edits` = {tag: (replacement, expected_count)}.  A mutation that touched fewer or more lines than
    expected is an ERROR (a vacuous mutation would read RED-by-accident or GREEN-by-accident)."""
    src = open(path, encoding="utf-8").read().replace("\r\n", "\n")
    lines, counts = src.split("\n"), {t: 0 for t in edits}
    for i, ln in enumerate(lines):
        for tag, (rep, _) in edits.items():
            if ln.rstrip().endswith("# @" + tag):
                lines[i] = ln[:len(ln) - len(ln.lstrip())] + rep
                counts[tag] += 1
    for tag, (_, want) in edits.items():
        if counts[tag] != want:
            raise RuntimeError(f"mutation {tag}: edited {counts[tag]} lines, expected {want}")
    new = "\n".join(lines)
    if new == src:
        raise RuntimeError("mutation changed nothing")
    mod = types.ModuleType(name)
    mod.__file__ = path
    sys.modules[name] = mod
    exec(compile(new, path, "exec"), mod.__dict__)
    return mod


# ------------------------------------------------------------------------------------------------------------------------------ head files for the checks
def p2_rows(scorer_mod, ckpt: str, refe_dir: str):
    ns, _ = scorer_mod.load_scorer(ckpt, refe_dir, "cpu")
    return ns.score_head.weight.detach().numpy().astype(np.float32).copy(), ns.score_head.bias.detach().numpy().astype(np.float32).copy()


def write_head(path: str, W, b, kind: str) -> str:
    np.savez(path, weight=np.asarray(W, np.float32), bias=np.asarray(b, np.float32), kind=np.array(kind))
    return l2_scorer.sha256_file(path)


def search_args(base: argparse.Namespace, arm: str, head: str | None, head_sha: str | None) -> argparse.Namespace:
    """the SAME parser the launch uses, so the CLI's own wiring is part of what is proven"""
    argv = ["--arm", arm, "--bundle", base.bundle, "--ctx", base.ctx, "--ckpt", base.ckpt, "--ckpt-sha256", base.ckpt_sha256, "--refe-dir", base.refe_dir, "--config", base.config,
            "--config-sha256", base.config_sha256, "--gate", "unused", "--out-dir", "unused"]
    if arm == "L2":
        argv += ["--head", head, "--head-sha256", head_sha]
    a = stepB.build_search_parser().parse_args(argv)
    stepB.validate_search_args(a)
    return a


# ------------------------------------------------------------------------------------------------------------------------------ the checks (each takes `mods`: build_scorer + run_search (+ run_search_cli), possibly MUTATED)
CMP_KEYS = ("exec", "scorer_in", "theta", "moved", "hist_best_agg", "hist_mu", "pred_logit_out", "pred_logit_pick", "arm_argmax", "pick_exec")


def _same_out(o1: dict, o2: dict) -> bool:
    return all(np.array_equal(o1[k], o2[k]) for k in CMP_KEYS)


def check_d(mods, a_arm, a_p2, bundle, n0, expect) -> dict:
    fa, fp = mods.build_scorer(a_arm), mods.build_scorer(a_p2)
    rec = stepB.preflight_reach(fa, fp, bundle, n0, expect)
    return {"name": "d_logit_diff_" + expect, **rec}


def check_a(mods, a_id, a_p2, bundle, n0, cfg, seed) -> dict:
    s_id, s_p2 = LogitSpy(mods.build_scorer(a_id)), LogitSpy(mods.build_scorer(a_p2))
    o_id = mods.run_search(s_id, bundle, [n0], cfg, seeds=[seed])
    o_p2 = mods.run_search(s_p2, bundle, [n0], cfg, seeds=[seed])
    same_calls = len(s_id.calls) == len(s_p2.calls) and all(x[0] == y[0] and np.array_equal(x[1], y[1]) for x, y in zip(s_id.calls, s_p2.calls))
    maxd = max((float(np.abs(x[1] - y[1]).max()) for x, y in zip(s_id.calls, s_p2.calls)), default=float("nan"))
    nontrivial = bool(np.asarray(o_p2["moved"]).any())
    ok = bool(_same_out(o_id, o_p2) and same_calls and maxd == 0.0 and len(s_id.calls) > 0 and nontrivial)
    return {"name": "a_cid_identity_head_equals_p2", "ok": ok, "output_bit_identical": _same_out(o_id, o_p2), "all_scored_sets_and_logits_bit_identical": same_calls, "n_scorings": len(s_id.calls),
            "max_abs_logit_diff": maxd, "search_moved_the_pick_on_this_token (non-vacuous)": nontrivial}


def check_b(mods, a_inv, a_p2, bundle, n0, cfg, seed) -> dict:
    s_inv, s_p2 = LogitSpy(mods.build_scorer(a_inv)), LogitSpy(mods.build_scorer(a_p2))
    mods.run_search(s_inv, bundle, [n0], cfg, seeds=[seed])
    mods.run_search(s_p2, bundle, [n0], cfg, seeds=[seed])
    ref = {h: lg for h, lg in s_p2.calls}
    matched = [(lg, ref[h]) for h, lg in s_inv.calls if h in ref]               # sets the two arms were handed IDENTICALLY (the first population, the base fan re-score, ...)
    flips = [bool(np.array_equal(li[:, EP], -lp[:, EP])) for li, lp in matched]
    others = [bool(np.array_equal(li[:, OTHERS], lp[:, OTHERS])) for li, lp in matched]
    nz = np.concatenate([lp[:, EP] for _, lp in matched]) if matched else np.zeros(0)
    ni = np.concatenate([li[:, EP] for li, _ in matched]) if matched else np.zeros(0)
    sign_flipped = bool(len(nz) > 0 and np.all(np.sign(ni[nz != 0]) == -np.sign(nz[nz != 0])) and np.any(nz != 0))
    ok = bool(len(matched) >= 2 and all(flips) and all(others) and sign_flipped)
    return {"name": "b_inverted_head_negates_ep_logits_the_search_sees", "ok": ok, "n_identical_sets": len(matched), "ep_exactly_negated_on_all": bool(all(flips)) if matched else False,
            "other_columns_identical_on_all": bool(all(others)) if matched else False, "sign_flipped_everywhere": sign_flipped}


def expected_first_population(cfg, seed: int) -> np.ndarray:
    """the first CEM population as `tts_cem.cem_search` must draw it from (seed, sigma0 * widen, clips): mean 0, member 0 = the mean"""
    rng = np.random.default_rng(int(seed))
    sig = np.asarray(cfg.sigma0, np.float64) * cfg.widen
    th = 0.0 + sig * rng.standard_normal((cfg.m_pop, 2))
    th[0] = 0.0
    th[:, 0] = np.clip(th[:, 0], cfg.u_lo, cfg.u_hi)
    th[:, 1] = np.clip(th[:, 1], cfg.d_lo, cfg.d_hi)
    return th


def check_e(mods, bundle, n0, cfg) -> dict:
    first_ok, in_bounds, hit = [], [], {"u_hi": False, "u_lo": False, "d_hi": False, "d_lo": False}
    for seed in cfg.seeds:
        for sign in (1.0, -1.0):
            with spy_composite() as calls:
                mods.run_search(planted_score_fn(sign), bundle, [n0], cfg, seeds=[seed])
            th = np.asarray(calls[:cfg.m_pop * cfg.k_iters], np.float64)          # the populations (the final elite re-score calls come after)
            if sign == 1.0:
                first_ok.append(bool(np.array_equal(th[:cfg.m_pop], expected_first_population(cfg, seed))))
            u, d = th[:, 0], th[:, 1]
            in_bounds.append(bool(u.min() >= cfg.u_lo and u.max() <= cfg.u_hi and d.min() >= cfg.d_lo and d.max() <= cfg.d_hi))
            hit["u_hi"] |= bool(u.max() == cfg.u_hi)
            hit["u_lo"] |= bool(u.min() == cfg.u_lo)
            hit["d_hi"] |= bool(d.max() == cfg.d_hi)
            hit["d_lo"] |= bool(d.min() == cfg.d_lo)
    ok = bool(all(first_ok) and all(in_bounds) and all(hit.values()))
    return {"name": "e_c8_sigma0_and_clips_reach_the_search", "ok": ok, "first_population_bit_exact_all_seeds": bool(all(first_ok)), "never_exceeds_clip": bool(all(in_bounds)), "clip_attained_exactly": hit,
            "sigma0": list(cfg.sigma0), "clips_u": [cfg.u_lo, cfg.u_hi], "clips_d": [cfg.d_lo, cfg.d_hi]}


def check_f(mods, base, head: str, head_sha: str, a_p2, bundle, n0: int, out_dir: str) -> dict:
    """(f) the launch itself, end to end on one real token, then read what it WROTE"""
    argv = ["--arm", "L2", "--head", head, "--head-sha256", head_sha, "--bundle", base.bundle, "--ctx", base.ctx, "--ckpt", base.ckpt, "--ckpt-sha256", base.ckpt_sha256, "--refe-dir", base.refe_dir,
            "--config", base.config, "--config-sha256", base.config_sha256, "--gate", "unused", "--out-dir", out_dir, "--start", str(n0), "--end", str(n0 + 1), "--shard", "1"]
    a = stepB.build_search_parser().parse_args(argv)
    shard = os.path.join(out_dir, f"shard_L2_{n0:04d}.npz")
    if os.path.exists(shard):
        os.remove(shard)
    mods.run_search_cli(a, enforce_stamp=False)
    z = np.load(shard, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    fp = mods.build_scorer(a_p2)
    fp.bind(n0)
    pick = int(bundle["pick"][n0])
    lp = np.asarray(fp(np.asarray(bundle["traj_raw"][n0], np.float32).copy()), np.float64)[pick]
    rec = np.asarray(z["pred_logit_pick"][0], np.float64)
    head_rec = (meta.get("scorer_info") or {}).get("head") or {}
    ok = bool(np.array_equal(rec[OTHERS], lp[OTHERS]) and rec[EP] != lp[EP] and head_rec.get("head_sha256") == head_sha and head_rec.get("installed_ep_row_sha256") != head_rec.get("p2_ep_row_sha256")
              and (meta.get("preflight") or {}).get("ok") is True)
    return {"name": "f_end_to_end_launch_shard_shows_the_head", "ok": ok, "recorded_ep_logit_minus_p2": float(rec[EP] - lp[EP]), "other_columns_equal_p2": bool(np.array_equal(rec[OTHERS], lp[OTHERS])),
            "shard_head_sha256_matches": head_rec.get("head_sha256") == head_sha, "installed_ep_row_differs_from_p2": head_rec.get("installed_ep_row_sha256") != head_rec.get("p2_ep_row_sha256"),
            "preflight_ok": (meta.get("preflight") or {}).get("ok")}


# ------------------------------------------------------------------------------------------------------------------------------ the gate
def pick_gate_token(mods, a_p2, bundle, cfg, seed, max_tries: int = 40):
    """the first token whose pick MOVES >= 5 m and on which plain P2's search actually moves the pick (so the C-ID comparison is of non-trivial outputs, not of 'nothing happened' twice)"""
    fn, tried = mods.build_scorer(a_p2), 0
    for n in range(len(bundle["token"])):
        p = A.repair_last_heading_np(np.asarray(bundle["traj_raw"][n][int(bundle["pick"][n])], np.float64))
        if float(np.linalg.norm(np.diff(np.concatenate([np.zeros((1, 3)), p], 0)[:, :2], axis=0), axis=1).sum()) < 5.0:
            continue
        tried += 1
        if np.asarray(mods.run_search(fn, bundle, [n], cfg, seeds=[seed])["moved"]).any():
            return n, tried
        if tried >= max_tries:
            break
    raise RuntimeError(f"no token on which P2's search moves the pick in {tried} tries: C-ID would be vacuous")


def run_gate(base: argparse.Namespace, out_dir: str, scorer_mod=None, token: int | None = None, verbose: bool = True, carve_sha256: str | None = None) -> dict:
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    l1 = stepB.verify_l1_code()
    cfg_sha = stepB.sha256_lf(base.config)
    if cfg_sha != base.config_sha256:
        raise SystemExit(f"config sha256 {cfg_sha[:12]} != --config-sha256")
    cfg = stepB.StepBConfig.from_json(base.config)
    stepB.validate_config(cfg)
    ck_sha = l2_scorer.sha256_file(base.ckpt)
    if ck_sha != base.ckpt_sha256:
        raise SystemExit("checkpoint sha256 mismatch")
    bundle = A.load_npz(base.bundle)
    clause = stepB.check_data_clause(bundle, carve_sha256)                  # the DATA CLAUSE: no gate is minted for a bundle that is not the sealed carve (carve_sha256=None = the sealed carve B)
    SM = scorer_mod or stepB._real_scorer_module()
    mods = types.SimpleNamespace(build_scorer=lambda a: stepB.build_scorer(a, scorer_mod=SM), run_search=stepB.run_search,
                               run_search_cli=lambda a, **k: stepB.run_search_cli(a, scorer_mod=SM, carve_sha256=carve_sha256, **k))
    log = (lambda *x: print(*x, flush=True)) if verbose else (lambda *x: None)
    seed = cfg.seeds[0]
    a_p2 = search_args(base, "P2HEAD", None, None)
    n0, tried = (pick_gate_token(mods, a_p2, bundle, cfg, seed) if token is None else (int(token), 0))
    log(f"gate token index {n0} (tried {tried})")

    # head files: P2's own rows (identity), P2's rows with the EP row negated (inverted)
    W2, b2 = p2_rows(SM, base.ckpt, base.refe_dir)
    id_path, inv_path = os.path.join(tmp, "head_identity_p2.npz"), os.path.join(tmp, "head_inverted_ep.npz")
    id_sha = write_head(id_path, W2, b2, "gate-identity-p2")
    Winv, binv = W2.copy(), b2.copy()
    Winv[EP], binv[EP] = -W2[EP], -b2[EP]
    inv_sha = write_head(inv_path, Winv, binv, "gate-inverted-ep")
    a_id = search_args(base, "L2", id_path, id_sha)
    a_inv = search_args(base, "L2", inv_path, inv_sha)
    a_l2 = search_args(base, base.arm, base.head, base.head_sha256) if base.arm == "L2" else None
    checks, muts = [], []

    def run(name, fn):
        """a check that RAISES (e.g. the launch's own SystemExit refusal of an identity head) is a failed check, recorded, never a crash of the gate"""
        try:
            return fn()
        except (Exception, SystemExit) as e:
            return {"name": name, "ok": False, "raised": type(e).__name__, "message": str(e)[:300]}

    log("(a) C-ID ...")
    checks.append(run("a_cid_identity_head_equals_p2", lambda: check_a(mods, a_id, a_p2, bundle, n0, cfg, seed)))
    checks.append(run("d_logit_diff_identical", lambda: check_d(mods, a_id, a_p2, bundle, n0, "identical")))
    log("(b) inverted ...")
    checks.append(run("b_inverted_head_negates_ep_logits_the_search_sees", lambda: check_b(mods, a_inv, a_p2, bundle, n0, cfg, seed)))
    if a_l2 is not None:
        log("(d) the real L2 head vs P2 ...")
        checks.append(run("d_logit_diff_differs", lambda: check_d(mods, a_l2, a_p2, bundle, n0, "differs")))
    log("(e) C8 reach ...")
    checks.append(run("e_c8_sigma0_and_clips_reach_the_search", lambda: check_e(mods, bundle, n0, cfg)))
    if a_l2 is not None:
        log("(f) the launch, end to end ...")
        checks.append(run("f_end_to_end_launch_shard_shows_the_head", lambda: check_f(mods, base, base.head, base.head_sha256, a_p2, bundle, n0, os.path.join(tmp, "e2e"))))

    # ---- (c) MUTATE: every arm must read RED
    log("(c) mutations ...")
    src_l2s, src_stepB = os.path.join(HERE, "l2_scorer.py"), os.path.join(HERE, "stepB.py")

    def red(name, fn, why):
        try:
            r = fn()
            muts.append({"mutation": name, "must_be_red_by": why, "check": r["name"] if isinstance(r, dict) else None, "red": bool(not r["ok"]), "detail": {k: v for k, v in r.items() if k != "name"}})
        except (Exception, SystemExit) as e:         # a raise (or the launch's own SystemExit refusal) IS red for a mutation that must be refused loudly
            muts.append({"mutation": name, "must_be_red_by": why, "check": "raised", "red": True, "detail": {"exception": type(e).__name__, "message": str(e)[:200]}})

    m1 = mutate_module(src_l2s, {"INSTALL": ("pass", 2)}, "l2_scorer_mut1")
    m2 = mutate_module(src_l2s, {"INSTALL": ("pass", 2), "READBACK": ("pass", 1)}, "l2_scorer_mut2")
    m_wire = mutate_module(src_stepB, {"WIRE": ("head = None", 1)}, "stepB_mut_wire")
    m_clip = mutate_module(src_stepB, {"CLIPS": ("pass", 2)}, "stepB_mut_clips")
    m_sig = mutate_module(src_stepB, {"SIGMA": ("#", 1)}, "stepB_mut_sigma")
    m_cli = mutate_module(src_stepB, {"CLIFN": ('fn = build_scorer(argparse.Namespace(**{**vars(a), "arm": "P2HEAD", "head": None, "head_sha256": None}), scorer_mod=scorer_mod)', 1)}, "stepB_mut_clifn")
    mk = lambda hm=None, sm=None: types.SimpleNamespace(build_scorer=lambda a: (sm or stepB).build_scorer(a, scorer_mod=SM, head_mod=hm), run_search=(sm or stepB).run_search,   # noqa: E731
                                                       run_search_cli=lambda a, **k: (sm or stepB).run_search_cli(a, scorer_mod=SM, carve_sha256=carve_sha256, **k))
    red("M1 delete the @INSTALL lines (read-back must raise)", lambda: check_b(mk(hm=m1), a_inv, a_p2, bundle, n0, cfg, seed), "the install read-back raising")
    red("M2 delete @INSTALL and @READBACK (only the caller-level observation can see it)", lambda: check_b(mk(hm=m2), a_inv, a_p2, bundle, n0, cfg, seed), "check (b) ALONE")
    red("M2d same mutation under check (d)", lambda: check_d(mk(hm=m2), a_inv, a_p2, bundle, n0, "differs"), "check (d)")
    red("M3 caller wiring: build_scorer ignores the head (@WIRE) under (b)", lambda: check_b(mk(sm=m_wire), a_inv, a_p2, bundle, n0, cfg, seed), "check (b)")
    red("M3d same mutation under (d)", lambda: check_d(mk(sm=m_wire), a_inv, a_p2, bundle, n0, "differs"), "check (d)")
    red("M4 the C8 clip lines deleted (@CLIPS)", lambda: check_e(mk(sm=m_clip), bundle, n0, cfg), "check (e)")
    red("M5 the sigma0 kwarg deleted (@SIGMA)", lambda: check_e(mk(sm=m_sig), bundle, n0, cfg), "check (e)")
    if a_l2 is not None:
        red("M6 the launch builds a P2 scorer instead of the head scorer (@CLIFN)", lambda: check_f(mk(sm=m_cli), base, base.head, base.head_sha256, a_p2, bundle, n0, os.path.join(tmp, "e2e_mut")),
            "the launch's own preflight refusing, or check (f)")
    # the controls of the controls: the UNmutated modules must read GREEN under the same calls (a check that is red on the original proves nothing about the mutation)
    ctl = [run("b_control_unmutated", lambda: check_b(mk(), a_inv, a_p2, bundle, n0, cfg, seed)), run("e_control_unmutated", lambda: check_e(mk(), bundle, n0, cfg))]
    if a_l2 is not None:
        ctl.append(run("f_control_unmutated", lambda: check_f(mk(), base, base.head, base.head_sha256, a_p2, bundle, n0, os.path.join(tmp, "e2e_ctl"))))
    all_checks_ok = all(c["ok"] for c in checks)
    all_red = all(m["red"] for m in muts)
    verdict = "PASS" if (all_checks_ok and all_red and all(c["ok"] for c in ctl)) else "FAIL"
    stamp = {"schema": "tanitad.g_reach_stepb/1", "verdict": verdict, "arm": base.arm, "head_sha256": base.head_sha256 if base.arm == "L2" else None, "head_path": base.head, "ckpt_sha256": ck_sha,
             "config_sha256": cfg_sha, "bundle_sha256": l2_scorer.sha256_file(base.bundle), "data_clause": clause, "code_sha256": stepB.code_shas(), "l1_pins": l1, "token_index": n0, "token": str(bundle["token"][n0]),
             "search_seed": seed, "checks": checks, "mutations": muts, "controls_unmutated": ctl, "seconds": round(time.time() - t0, 1),
             "evidence": "MEASURED (ours): CPU, the real P2 scorer, one real set; mechanics only, no outcome read"}
    p = os.path.join(out_dir, f"G-REACH-STEPB_{verdict}_{(base.head_sha256 or 'p2')[:12]}.json")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(stamp, f, indent=1)
    log("ZZGATE", verdict, p)
    stamp["stamp_path"] = p
    return stamp


def main(argv=None) -> int:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "CPU only (set CUDA_VISIBLE_DEVICES=-1)"
    ap = argparse.ArgumentParser(prog="g_reach_stepb.py")
    for k in ("--arm",):
        ap.add_argument(k, required=True, choices=("L2",))
    for k in ("--head", "--head-sha256", "--bundle", "--ctx", "--ckpt", "--ckpt-sha256", "--refe-dir", "--config", "--config-sha256", "--out-dir"):
        ap.add_argument(k, required=True)
    ap.add_argument("--token", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    import torch
    torch.set_num_threads(a.threads)
    st = run_gate(a, a.out_dir, token=a.token)
    return 0 if st["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
