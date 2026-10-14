#!/usr/bin/env python3
"""Tests for RL-S1b (eval/PREREG_RL_S1.md section 10): the EPO DOSE lever, --epo-lam 0.25, and its launch-gate profile `refe_rls1b`.
Run:  PYTHONIOENCODING=utf-8 python refe/test_rl_s1b.py   ->   ZZTEST_PASS <n>   (prints the mutation table).   CPU only, no pod, no checkpoint.

WHY THERE IS NO NEW TRAINER FLAG.  The lever the attribution chose (eval/PREREG_RL_S1.md 10.1-10.2; raw/2026-10-14-rls1b-attribution/attribution.json, dose.json) is the final EPO weight lam_pg, which
scorer_finetune.py already exposes as --epo-lam (default 1.0, G-DVB kind `built`, observed through the data event's epo.lam).  Adding a second flag for the
same number would create two levers for one quantity.  So RL-S1b changes NO trainer code and NO reward code; what is proved here instead:

  P  PINNED: the reward module and every EPO / loss code segment are the bytes R0 trained with (launch commit a120418a, blob fec6174d of scorer_finetune.py),
     so R0 -> R0b differs by the argv value alone; and the parser default stays 1.0, so every refe_rls1 argv behaves exactly as before.
  D  DOSE REACHES THE LOSS, EXACTLY: on the real epo_loss, the gradient at lam_pg 0.25 is 0.25 x the gradient at 1.0 BIT FOR BIT (0.25 is a power of two,
     so the product is exact), non-zero; the ramp's mid-point (0.25 x epo_ramp(100, 200) = 0.125) likewise; and lam_pg 0.0 gives EXACTLY 0.0 (the control
     that proves the probe can read zero).
  S  END TO END through the real main() (test_rl_epo.smoke: CPU stubs for the GPU parts only): --epo-lam 0.25 trains (rc 0, preflight G1-G11 PASS), the data
     event records epo.lam 0.25 (what G-DVB compares), arm B moves, arm B differs from the 1.0 run's arm B, arm A (the control) is BIT-IDENTICAL across
     doses (the dose never reaches the control), and --epo-lam 0 leaves B == A exactly.
  G  THE GATE: refe_rls1b accepts exactly C1 / R0b / R1b (C1 = RL-S1's C1 argv, accepted by both profiles), refuses every mis-declared dose, seed, class,
     the R2 class (R2b needs a PI decision) and an omitted --epo-lam; refe_rls1 is unchanged; G-DVB passes on built epo.lam 0.25 and FAILS on built 1.0.
Every mutation below edits a SOURCE file (the snippet must occur EXACTLY once, else the test errors) and must turn >= 1 check RED.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
import sys
import tempfile
import textwrap
import types

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SF_PATH, LG_PATH, GC_PATH, RR_PATH = (os.path.join(HERE, f) for f in ("scorer_finetune.py", "launch_gate.py", "gate_checks.py", "rl_reward.py"))

# sha256 of each code segment of scorer_finetune.py AS R0 TRAINED IT (commit a120418a870f, blob fec6174de90d7f428579d9dc2ac85889170096b0); main.<name> = the closure
# inside main(), dedented.  main.fwd is deliberately NOT pinned: it changed after R0 (the default-OFF aux head's feed path, D-AUX-1 S1 wiring) -- see PREREG_RL_S1 10.3 and outcome O5.
R0_SEGMENTS = {
    "epo_ramp": "66889cf721361cc701a8efb1d90856b9fd0cbeb84581afe9e571e900e452c98c",
    "epo_policy": "9d9fa0746ed57a9ac985bf21ab290b952aabdb4329f8c5352c88e7364a84ba13",
    "epo_loss": "c4fd550f6dee16f85a95a7e0eb6414c29358635eefc036817e5386ab67ebcb89",
    "epo_reward": "dcf191d56ca6bfd3faa4a7e972f250386dccd1c7db41e52d48152ae791077573",
    "_rl_terms_record": "8298fdde47863194522a02adc7f25e401acf5d1338e54151b9a3277addb9a4b6",
    "load_rl_terms": "ef49e1b049507950c09fea50930d74a2092f5b31e56b8275db8bbb9f0f1a5d92",
    "epo_refusals": "b97a83d1db6f7a3f19885fe9350f68426f7576e64367bc82bd523efb2f502c10",
    "main.epo_term": "cf1fcdf97c513ea03b12107d73038f3cb171fa33e5e61e01af141e5997b44df9",
    "main.train_losses": "959a8cb562172bb20e334779a414339b733e76e702f81b0be35059e82a60f7ed",
    "main.losses": "07141375b5a64dc0cfbae335aeb53118f57d795b43569bdd9207dd02188e7537",
    "main.losses6": "bb6593db6f3e94aad1477629a7181fd42323bec49e4473c45334eb7348895cc0",
    "main.sel_score": "554302540d0df41262d40786c662a5f8eb35ca59a294665d1472edb716994d4e",
    "main.lane_bce": "06aad73427000b603ee067f211c0e8214d4cb75475c1e50a10fbaeaef008c9a3",
}
R0_REWARD_GIT_BLOB = "cd5c51445ad4ce38153d72ceab0aa01226aa7372"          # refe/rl_reward.py at the R0 launch commit (and today)
BASE = ("--ckpt /w/model_sft4_B_u400.pt --backbone vitl16 --targets /w/train --images /w/pix --calib /w/calib.json --onpolicy /w/onp --heldout /w/onh "
        "--heldout-logs /w/hl.txt --out /w/rl --lane-head --teacher-lane-train /w/tl --teacher-lane-heldout /w/tlh --pdm-labels-train /w/pdm "
        "--pdm-labels-heldout /w/pdmh --require-pdm --b-mode listnet --lam 0 --lr 1e-4 --max-updates 1000 --epochs 4 --workers 3")


def load(src: str, name: str, path: str):
    m = types.ModuleType(name)
    m.__file__ = path
    sys.modules[name] = m
    exec(compile(src, path, "exec"), m.__dict__)
    return m


def segments(src: str) -> dict:
    lines, out = src.split("\n"), {}
    for n in ast.parse(src).body:
        if isinstance(n, ast.FunctionDef):
            out[n.name] = "\n".join(lines[n.lineno - 1:n.end_lineno])
            if n.name == "main":
                for m in n.body:
                    if isinstance(m, ast.FunctionDef):
                        out["main." + m.name] = textwrap.dedent("\n".join(lines[m.lineno - 1:m.end_lineno]))
    return out


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check(bad, name, fn):
    try:
        ok = bool(fn())
    except Exception:                                                     # noqa: BLE001 -- an exception IS a failed check
        ok = False
    if not ok:
        bad.append(name)


# ================================================================================================ part SF: the trainer (P, D, S)
def suite_sf(sf_src: str, rr_bytes: bytes, tag: str) -> list:
    import test_rl_epo as TE                                             # the real-main() smoke harness (stubs only the GPU / checkpoint / corpus parts)
    bad: list = []
    M = load(sf_src, f"sf_s1b_{tag}", SF_PATH)
    seg = segments(sf_src)
    # ---- P: the code R0 trained with, and the default that keeps refe_rls1 unchanged
    for k, h in R0_SEGMENTS.items():
        check(bad, f"P_pinned_{k}", lambda k=k, h=h: hashlib.sha256(seg[k].encode()).hexdigest() == h)
    check(bad, "P_reward_is_R0s", lambda: git_blob(rr_bytes) == R0_REWARD_GIT_BLOB)
    check(bad, "P_parser_default_lam_1", lambda: M.build_parser().get_default("epo_lam") == 1.0)
    # ---- D: the dose identity on the REAL epo_loss (fresh forward per call; a tiny linear scorer, 7 outputs, the base's rule incl. the lane output)
    g = torch.Generator().manual_seed(7)
    x = torch.randn(3, 16, 5, generator=g)
    R = torch.rand(3, 16, generator=g)
    lin0 = nn.Linear(5, 7)
    with torch.no_grad():
        for p in lin0.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.5)
    rule = lambda p: M.agg_v1(p[..., :6]) * p[..., 6]                   # noqa: E731  (main()'s sel_score for a 7-output scorer)

    def grads(lam):
        lin = nn.Linear(5, 7)
        lin.load_state_dict(lin0.state_dict())
        with torch.no_grad():
            ref = lin0(x) + 0.3                                         # pi_SFT != pi, so the KL term is live too
        loss = M.epo_loss(lin(x), ref, R, rule, 0.1, 0.1, lam)
        loss.backward()
        return float(loss.detach()), [p.grad.detach().clone() for p in lin.parameters()]
    l1, g1 = grads(1.0)
    lq, gq = grads(0.25)
    le, ge = grads(0.25 * M.epo_ramp(100, 200))
    l0, g0 = grads(0.0)
    check(bad, "D_quarter_dose_exact", lambda: all(torch.equal(a, b * 0.25) for a, b in zip(gq, g1)) and lq == l1 * 0.25)
    check(bad, "D_quarter_dose_nonzero", lambda: all(float(a.abs().max()) > 0.0 for a in gq))
    check(bad, "D_ramp_midpoint_exact", lambda: 0.25 * M.epo_ramp(100, 200) == 0.125 and all(torch.equal(a, b * 0.125) for a, b in zip(ge, g1)))
    check(bad, "D_zero_dose_exactly_zero", lambda: l0 == 0.0 and all(float(a.abs().max()) == 0.0 for a in g0))
    # ---- S: the real main(), three doses on the same stub data
    with tempfile.TemporaryDirectory() as td:
        runs = {}
        for lam in (0.25, 1.0, 0.0):
            d = os.path.join(td, str(lam))
            os.makedirs(d)
            runs[lam] = TE.smoke(sf_src, d, epo=True, epo_lam=lam, native7=True, terms=True, tag=f"{tag}_{str(lam).replace('.', '_')}")
    q = runs[0.25]
    data_ev = [r for r in q["rows"]["sft.jsonl"] if r.get("event") == "data"]
    trains = [r for r in q["rows"]["sft.jsonl"] if r.get("event") == "train"]
    check(bad, "S_quarter_runs", lambda: q["pre"] == 0 and q["run"] == 0 and q["A"] is not None and q["B"] is not None)
    check(bad, "S_preflight_G11_pass", lambda: q["pass_json"] is not None and "G11_epo_live_and_identity" in str(q["pass_json"]))
    check(bad, "S_data_event_lam_0_25", lambda: len(data_ev) == 1 and data_ev[0]["epo"]["lam"] == 0.25)
    check(bad, "S_logged_lam_pg_schedule", lambda: trains and all(r["epo_lam_pg"] == 0.25 * min(1.0, r["update"] / 2) for r in trains))
    check(bad, "S_quarter_moves_B", lambda: not TE.scorer_equal(q["A"], q["B"]))
    check(bad, "S_dose_reaches_B", lambda: not TE.scorer_equal(q["B"], runs[1.0]["B"]))
    check(bad, "S_control_untouched_by_dose", lambda: TE.scorer_equal(q["A"], runs[1.0]["A"]) and TE.scorer_equal(q["A"], runs[0.0]["A"]))
    check(bad, "S_zero_dose_B_equals_A", lambda: runs[0.0]["run"] == 0 and TE.scorer_equal(runs[0.0]["A"], runs[0.0]["B"]))
    return bad


# ================================================================================================ part G: the gate (refe_rls1b)
def suite_g(lg_src: str, gc_src: str, tag: str) -> list:
    bad: list = []
    GC = load(gc_src, f"gc_s1b_{tag}", GC_PATH)
    L = load(lg_src, f"lg_s1b_{tag}", LG_PATH)
    P, Pb = L.PROFILES["refe_rls1"], L.PROFILES["refe_rls1b"]
    argv = lambda s: L.parse_args_string(BASE + " " + s)                 # noqa: E731
    c1 = argv("--seed 1")
    r0b = argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.25")
    r1b = argv("--seed 1 --epo --epo-terms-train /w/t --epo-lam 0.25")
    r0 = argv("--seed 0 --epo --epo-terms-train /w/t")

    def refs(prof, arm, a):
        return L.profile_refusals({**prof, "declared_arm": arm} if arm else prof, a)

    def red(prof, arm, a, frag):
        return any(frag in r for r in refs(prof, arm, a))
    check(bad, "G_clean_C1", lambda: refs(Pb, "C1", c1) == [])
    check(bad, "G_clean_R0b", lambda: refs(Pb, "R0b", r0b) == [])
    check(bad, "G_clean_R1b", lambda: refs(Pb, "R1b", r1b) == [])
    check(bad, "G_C1_same_argv_both_profiles", lambda: refs(P, "C1", c1) == [] and refs(Pb, "C1", c1) == [])
    check(bad, "G_R0b_lam_omitted", lambda: red(Pb, "R0b", r0, "--epo-lam 0.25"))
    check(bad, "G_R0b_lam_1", lambda: red(Pb, "R0b", argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 1.0"), "--epo-lam 0.25"))
    check(bad, "G_R0b_lam_0_5", lambda: red(Pb, "R0b", argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.5"), "--epo-lam 0.25"))
    check(bad, "G_R1b_seed_0", lambda: red(Pb, "R1b", argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.25"), "registered with --seed 1"))
    check(bad, "G_R0b_seed_1", lambda: red(Pb, "R0b", r1b, "registered with --seed 0"))
    check(bad, "G_C1_with_epo", lambda: red(Pb, "C1", argv("--seed 1 --epo --epo-terms-train /w/t --epo-lam 0.25"), "must NOT carry"))
    check(bad, "G_R0b_without_epo", lambda: red(Pb, "R0b", c1 if False else argv("--seed 0"), "requires --epo"))
    check(bad, "G_R2_class_needs_PI", lambda: red(Pb, None, argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.25 --epo-k --epo-l"), "PI decision"))
    check(bad, "G_other_levers_still_registered", lambda: red(Pb, "R0b", argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.25 --epo-beta 0.2"), "--epo-beta 0.1"))
    check(bad, "G_refe_rls1_unchanged", lambda: refs(P, "R0", r0) == [] and red(P, "R0", argv("--seed 0 --epo --epo-terms-train /w/t --epo-lam 0.25"), "--epo-lam 1.0")
          and refs(P, "R2", argv("--seed 0 --epo --epo-terms-train /w/t --epo-k --epo-l")) == [])

    def pf_err(profile, arm):
        try:
            L._profile_for(argparse.Namespace(profile=profile, arm_name=arm), minting=True)
        except SystemExit as e:
            return str(e)
        return ""
    check(bad, "G_cli_R2b_is_PI_decision", lambda: "PI decision" in pf_err("refe_rls1b", "R2b"))
    check(bad, "G_cli_R0_not_rls1b_arm", lambda: "is not an arm of refe_rls1b" in pf_err("refe_rls1b", "R0"))
    check(bad, "G_cli_R0b_not_rls1_arm", lambda: "is not an arm of refe_rls1" in pf_err("refe_rls1", "R0b"))
    check(bad, "G_cli_mint_needs_arm", lambda: "mints for ONE named arm" in pf_err("refe_rls1b", None))
    check(bad, "G_suite_has_s1b", lambda: [e["script"] for e in Pb["suite"]] == ["test_rl_reward.py", "test_rl_epo.py", "test_launch_gate.py", "test_rl_s1b.py"]
          and all(e["token"] in open(os.path.join(HERE, e["script"]), encoding="utf-8").read() for e in Pb["suite"]))
    check(bad, "G_same_checks_as_rls1", lambda: Pb["checks"] == P["checks"] and Pb["data_flags"] == P["data_flags"] and Pb["forbidden_flags"] == P["forbidden_flags"]
          and Pb["required_flags"] == P["required_flags"] and Pb["trainer"] == P["trainer"])
    # ---- G-DVB on the scorer: the argv's 0.25 is compared with what the trainer's data event says it BUILT
    import scorer_finetune as S
    ns = S.build_parser().parse_args(r0b)

    def data(lam):
        return {"event": "data", "b_mode": ns.b_mode, "b_compw": ns.b_compw, "epo": {**{k: getattr(ns, "epo_" + k) for k in GC.SCORER_EPO_KEYS}, "lam": lam}}
    pb_arm = {**Pb, "declared_arm": "R0b"}
    ev_ok = GC.check_dvb_scorer(r0b, {"data": data(0.25)}, pb_arm)
    ev_bad = GC.check_dvb_scorer(r0b, {"data": data(1.0)}, pb_arm)
    check(bad, "G_dvb_pass_built_0_25", lambda: ev_ok["status"] == "PASS")
    check(bad, "G_dvb_fail_built_1_0", lambda: ev_bad["status"] == "FAIL" and any("--epo-lam: declared 0.25 != built 1.0" in r for r in ev_bad["reasons"]))
    return bad


# (name, part, file, old snippet, new snippet) -- each snippet must occur EXACTLY once in its file
MUTATIONS = [
    ("sf_dose_ignored_by_the_loss", "SF", SF_PATH, "lam_pg = a.epo_lam * epo_ramp(upd, a.epo_ramp)", "lam_pg = 1.0 * epo_ramp(upd, a.epo_ramp)"),
    ("sf_epo_loss_drops_lam", "SF", SF_PATH, "return lam_pg * (-J + beta * kl).mean()", "return (lam_pg != 0.0) * (-J + beta * kl).mean()"),
    # (test_rl_epo's `epo_leaks_into_arm_A` -- ep added to lA too -- does NOT touch arm A here: one backward runs on lA + lB and ep's graph is arm B's, so it DOUBLES
    # B's dose instead; the pin catches that.  This mutation is the behavioural one: arm A gets its OWN dosed EPO term, so the control would move with the dose.)
    ("sf_dose_reaches_control", "SF", SF_PATH, "            ep = epo_term(sx0, sxB, tg, rows, upd, lam_pg)\n",
     "            ep = epo_term(sx0, sxB, tg, rows, upd, lam_pg)\n            lA = lA + epo_term(sx0, sxA, tg, rows, upd, lam_pg)\n"),
    ("sf_data_event_lam_not_built_value", "SF", SF_PATH, '"ramp": a.epo_ramp, "lam": a.epo_lam,', '"ramp": a.epo_ramp, "lam": 1.0,'),
    ("sf_parser_default_moved", "SF", SF_PATH, 'ap.add_argument("--epo-lam", type=float, default=1.0,', 'ap.add_argument("--epo-lam", type=float, default=0.25,'),
    ("rr_reward_edited", "SF", RR_PATH, "lam_d: float = 0.15", "lam_d: float = 0.15 "),
    ("lg_rls1b_lam_registered_1", "G", LG_PATH, "RLS1B_EPO_LAM = 0.25", "RLS1B_EPO_LAM = 1.0"),
    ("lg_profile_registry_ignored", "G", LG_PATH, 'reg = dict(prof.get("epo_registered", RLS1_EPO_REGISTERED))', "reg = dict(RLS1_EPO_REGISTERED)"),
    ("lg_profile_seeds_ignored", "G", LG_PATH, 'arm_seeds = prof.get("arm_seeds", RLS1_ARM_SEEDS)', "arm_seeds = RLS1_ARM_SEEDS"),
    ("lg_profile_arms_ignored", "G", LG_PATH, 'arm_class = prof.get("arm_class", RLS1_ARM_CLASS)', "arm_class = RLS1_ARM_CLASS"),
    ("lg_unregistered_class_accepted", "G", LG_PATH, 'if cls != "R?" and cls not in set(arm_class.values()):', "if False:"),
    ("lg_R2b_PI_refusal_removed", "G", LG_PATH, '        if arm_name in prof.get("arms_need_pi", {}):\n', "        if False:\n"),
    ("lg_rls1b_seeds_swapped", "G", LG_PATH, 'RLS1B_ARM_SEEDS = {"C1": 1, "R0b": 0, "R1b": 1}', 'RLS1B_ARM_SEEDS = {"C1": 1, "R0b": 1, "R1b": 0}'),
    ("lg_rls1_registry_moved", "G", LG_PATH, '"epo_lam": 1.0, "epo_lam_d": 0.15', '"epo_lam": 0.25, "epo_lam_d": 0.15'),
    ("lg_cli_arm_table_ignored", "G", LG_PATH, 'arms = {"rls1": prof.get("arm_class", RLS1_ARM_CLASS), "auxs1": AUXS1_ARM_CLASS}', 'arms = {"rls1": RLS1_ARM_CLASS, "auxs1": AUXS1_ARM_CLASS}'),
    ("lg_suite_without_s1b", "G", LG_PATH, 'RLS1B_SUITE = RLS1_SUITE + [{"script": "test_rl_s1b.py", "token": "ZZTEST_PASS"}]', "RLS1B_SUITE = RLS1_SUITE"),
    ("gc_dvb_does_not_observe_lam", "G", GC_PATH, 'SCORER_EPO_KEYS = ("temp", "beta", "ramp", "lam", ', 'SCORER_EPO_KEYS = ("temp", "beta", "ramp", '),
]


def main() -> int:
    srcs = {p: open(p, encoding="utf-8").read() for p in (SF_PATH, LG_PATH, GC_PATH)}
    rr = open(RR_PATH, "rb").read()
    n = 0
    real = suite_sf(srcs[SF_PATH], rr, "real") + suite_g(srcs[LG_PATH], srcs[GC_PATH], "real")
    assert not real, f"the unmutated sources FAIL their own checks: {real}"
    n += 1
    print(f"{'mutation':38s} {'part':4s} caught-by (failing checks)")
    for i, (name, part, path, old, new) in enumerate(MUTATIONS):
        base = rr.decode("utf-8") if path == RR_PATH else srcs[path]
        cnt = base.count(old)
        assert cnt == 1, f"mutation {name}: the snippet occurs {cnt} times in {os.path.basename(path)} (must be exactly 1) -- a vacuous mutation proves nothing"
        mut = base.replace(old, new)
        try:
            if part == "SF":
                failing = suite_sf(mut if path == SF_PATH else srcs[SF_PATH], mut.encode("utf-8") if path == RR_PATH else rr, f"m{i}")
            else:
                failing = suite_g(mut if path == LG_PATH else srcs[LG_PATH], mut if path == GC_PATH else srcs[GC_PATH], f"m{i}")
        except Exception as e:                                          # noqa: BLE001 -- the mutated source breaks the suite's own setup: RED, and said so
            failing = [f"suite setup raised {type(e).__name__}: {str(e)[:60]}"]
        assert failing, f"mutation {name} stayed GREEN: the suite cannot see it"
        print(f"{name:38s} {part:4s} {', '.join(failing[:4])}")
        n += 1
    print(f"ZZTEST_PASS {n}  (1 unmutated suite + {len(MUTATIONS)} RED mutations)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
