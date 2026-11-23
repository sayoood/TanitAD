#!/usr/bin/env python3
"""SCORER-FT launch-gate PROFILES and ROWS (DESIGN_SCORER_FT section 3; PI go 2026-11-20): `refe_scorerft_t` (T), `refe_scorerft_c` (C), `refe_scorerft_t_s1` (Ts), `refe_scorerft_c_s1` (Cs), and the G-DVB / G-LIVE rows
`rank_dvb_rows`, `rank_live_rows` (L1-L5), the routing in `check_live` and the generalised (h).

  python test_scorerft_profile.py            everything, incl. the source-mutation harness
  python test_scorerft_profile.py --fast     no mutation harness
  python test_scorerft_profile.py --check N  one named check (what the harness runs on a mutated copy of this tree)

CPU only, no scipy, stdlib + numpy + torch + the gate modules.  Prints ZZSCORERFTPROF_PASS <n>.  Asserts:
  * the four profiles exist, share the recipe, and NO existing profile moved (every pre-existing profile's whole-profile fingerprint equals its pin below: the MEASURED 2026-11-07 trap);
  * argv(T) - argv(C) = {--rank-w, --rank-heads, --rank-ep-gap, --rank-gate-topk, --out};  argv(Ts) - argv(T) = {--seed, --out};  argv(Cs) - argv(C) = {--seed, --out};  C names `--rank-w 0.0`;
  * the profiles REFUSE while the base sha256 or the weight is unfilled; with the pins filled the template argv is clean; each lever the card names is refused when it moves; REFE_RANK_MUTATE in the environment is refused;
  * the rows (pure functions of a capture) judge a CLEAN synthetic ranking term PASS and turn RED on each deliberate-regression arm of DESIGN 3.4 that is observable without the trainer (R-DROP, R-UNSAFE, R-SHUF, R-LEAK, R-HEAD,
    R-CTRL), with a numpy reference independent of the term;
  * G-LIVE's (h) keys on `traj_label_required`, and the scorerft arms are routed to the same live rows as C1.
INTEGRATION_PENDING (printed, not a failure): the checks that need the built trainer half (train.py's four flags, rank_loss.RankLoss) are re-run by `chk_trainer_integration` once `--rank-w` is in the parser."""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import gate_checks as GC  # noqa: E402

N = 0
HEX_BASE = "ab" * 32
W_FIX = 0.25                                                       # a fixture rank_w for the numeric gradient captures below (the MM's real one is the weight rule's output)
# the weight the TEMPLATE argv and the filled profile carry: the registered SCORERFT_RANK_W once the MM fills it (the profile then REQUIRES that exact --rank-w,
# so a fixture value would be refused), else the fixture. MM 2026-11-21: with W_FIX here this file went RED as soon as the pin was filled, and it is in
# SCORERFT_SUITE_EXTRA, so T's G-SUITE would have refused T (MEASURED on a scratch copy with SCORERFT_RANK_W = 0.5).
W_ARGV = float(L.SCORERFT_RANK_W) if L.SCORERFT_RANK_W is not None else W_FIX
ARMS = ("T", "C", "Ts", "Cs")
NAMES = {"T": "refe_scorerft_t", "C": "refe_scorerft_c", "Ts": "refe_scorerft_t_s1", "Cs": "refe_scorerft_c_s1"}
# whole-profile fingerprints sha256(repr(sorted(items)))[:16] of EVERY profile that existed at 36b92a2b (the 2026-11-07 trap: a launch_gate.py edit can move them, and other tests pin them BY VALUE)
FP_PINS = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
           "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p5c": "ed995961b67113bd", "refe_p5c_c0": "65b0a2e8fc15b36e",
           "refe_p5c_c0r": "e89c54c4fa046afd", "refe_field_p7": "6950f70944a0fbda", "refe_field_p7_ctrl": "3dc54a1a4de1a7d7", "refe_field_p7_ctrl_s1": "84134e3ebb2d3f93",
           "refe_field_p7_s1": "b502f0275a4862d8", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94", "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35",
           "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_s1zsm_p6": "86380fbb786daa75", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a", "refe_rep_s1zsm_p6": "27fe236a2fd68ca4",
           "refe_dose_s1z_p6": "9085f9e5cf59c016", "refe_div_p6": "e51f9374a8f29816", "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94",
           "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957", "refe_aux_s1c": "dcdf2dbd2a674f1c", "refe_vdac_p6": "d4e05641baf8a48e", "refe_p5c_c1r": "4a76a09574fbc144"}          # + refe_p5c_c1r (C1's seed-1 replicate, PI go 2026-11-20): pinned by value too


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def filled(arm: str) -> dict:
    """the profile with every MM-filled pin given a (fixture) value: what the gate sees once the real objects exist."""
    p = L.scorerft_profile(arm)
    p["init_from_sha256"] = HEX_BASE
    if L.SCORERFT_ARM_RANK[arm]:
        p["rank_w"] = W_ARGV
    return p


def targv(arm: str) -> list:
    return L.scorerft_argv(arm, rank_w=W_ARGV if L.SCORERFT_ARM_RANK[arm] else None)


def refusals(prof: dict, argv: list) -> str:
    return " || ".join(L.profile_refusals(prof, argv))


def flagset(argv):
    return {f: v for f, v in GT.flag_pairs(argv)}


def delta(a: list, b: list) -> set:
    fa, fb = flagset(a), flagset(b)
    return {f for f in set(fa) | set(fb) if fa.get(f) != fb.get(f)}


# ------------------------------------------------------------------------------------------------ profile checks
def chk_registry():
    for arm, nm in NAMES.items():
        ok(nm in L.PROFILES and L.PROFILES[nm]["name"] == nm and L.PROFILES[nm]["scorerft_rules"] is True and L.PROFILES[nm]["scorerft_arm"] == arm, nm)
        p = L.PROFILES[nm]
        ok(not p.get("p5c_rules") and not p.get("p7field_rules") and "p5c_arm" not in p, nm)
        ok(p["traj_label_required"] is True and p["required_flags"]["--steps"] == "1000" and p["required_flags"]["--epochs"] == "0" and "--traj-label-table" in p["required_flags"], nm)
        ok(p["expect_schedule"]["steps"] == 1000 and p["expect_schedule"]["epochs"] == 0 and p["expect_schedule"]["epoch_len"] == L.P7_SCENES, nm)
        ok(p["traj_label_sha256"] == L.P5C_LABEL_TABLE_SHA256 and p["traj_label_rank_policy"] == "rank0", nm)
        ok(any(e.get("script") == "test_rank_loss.py" and e["token"] == "ZZRANK_PASS" for e in p["suite"]), nm)
        ok(any(e.get("script") == "test_scorerft_profile.py" and e["token"] == "ZZSCORERFTPROF_PASS" for e in p["suite"]), nm)
        ok("REFE_RANK_MUTATE" in p["forbidden_env"], nm)
    for nm in ("refe_scorerft_t", "refe_scorerft_t_s1"):
        ok(L.PROFILES[nm]["live_terms"] == L.P6_TERMS + ("rank_L",) and L.PROFILES[nm]["required_flags"]["--rank-heads"] == "ep,gate", nm)
    for nm in ("refe_scorerft_c", "refe_scorerft_c_s1"):                                           # a control that declared its term FAILED G-LIVE (field_L ABSENT, C, 2026-11-19)
        ok("rank_L" not in L.PROFILES[nm]["live_terms"] and L.PROFILES[nm]["required_flags"]["--rank-w"] == "0.0" and "--rank-heads" not in L.PROFILES[nm]["required_flags"], nm)
    ok(L.PROFILES["refe_scorerft_t"]["required_flags"]["--seed"] == "0" and L.PROFILES["refe_scorerft_t_s1"]["required_flags"]["--seed"] == "1", "seeds")
    ok(L.SCORERFT_BASE_SHA256 is None or re.fullmatch(r"[0-9a-f]{64}", L.SCORERFT_BASE_SHA256), "pin shape")
    for k in ("rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"):
        ok(GC.DVB_KINDS.get(k) == "built", k)


def chk_existing_unchanged():
    """every profile that existed at 36b92a2b has its pinned whole-profile fingerprint; no other profile gained a scorerft key; the new ones are exactly the four."""
    for nm, fp in FP_PINS.items():
        got = hashlib.sha256(repr(sorted(L.PROFILES[nm].items())).encode()).hexdigest()[:16]
        ok(got == fp, f"{nm}: fingerprint moved {fp} -> {got}")
    ok(set(L.PROFILES) == set(FP_PINS) | set(NAMES.values()), sorted(set(L.PROFILES) ^ (set(FP_PINS) | set(NAMES.values()))))
    for nm in FP_PINS:
        ok(not any(k.startswith(("scorerft", "rank_")) or k == "traj_label_required" for k in L.PROFILES[nm]), nm)


def chk_argv_delta():
    T, C, Ts, Cs = (targv(a) for a in ARMS)
    ok(delta(T, C) == {"--rank-w", "--rank-heads", "--rank-ep-gap", "--rank-gate-topk", "--out"}, delta(T, C))
    ok(delta(Ts, T) == {"--seed", "--out"}, delta(Ts, T))
    ok(delta(Cs, C) == {"--seed", "--out"}, delta(Cs, C))
    ok(flagset(C)["--rank-w"] == ["0.0"] and "--rank-heads" not in flagset(C), "C names --rank-w 0.0 explicitly and no head")
    ok(flagset(T)["--rank-heads"] == ["ep,gate"] and float(flagset(T)["--rank-w"][0]) == W_ARGV and flagset(T)["--rank-gate-topk"] == ["8"], "T rule")
    c1 = flagset(L.p5c_argv("C1"))
    for a_, v_ in zip(ARMS, (T, C, Ts, Cs)):
        d = delta(v_, L.p5c_argv("C1"))
        ok(d == {"--init-from", "--steps", "--out", "--rank-w"} | ({"--rank-heads", "--rank-ep-gap", "--rank-gate-topk"} if L.SCORERFT_ARM_RANK[a_] else set()) | ({"--seed"} if a_ in ("Ts", "Cs") else set()), (a_, d))
        ok(flagset(v_)["--traj-label-table"] == c1["--traj-label-table"] and flagset(v_)["--score-w"] == ["0.1"] and "--scorer-sees-trunk" in flagset(v_), a_)
        ok(flagset(v_)["--steps"] == ["1000"] and flagset(v_)["--init-from"] == [L.SCORERFT_DEFAULT_INIT], a_)
    try:
        L.scorerft_argv("T")                                                                      # no template without the weight
        ok(L.SCORERFT_RANK_W is not None, "unfilled weight must raise")
    except ValueError:
        ok(L.SCORERFT_RANK_W is None)


def chk_pins_refuse():
    for arm in ARMS:
        argv = targv(arm)
        p0 = L.scorerft_profile(arm)
        p0["init_from_sha256"] = None
        ok("SCORERFT_BASE_SHA256" in refusals(p0, argv), arm)                                      # an unfilled base pin refuses
        p1 = filled(arm)
        ok(refusals(p1, argv) == "", f"{arm}: the template argv with the pins filled must be clean: {refusals(p1, argv)}")
        p2 = filled(arm)
        p2["traj_label_sha256"] = "cd" * 32
        ok("is not C1's" in refusals(p2, argv), arm)
        p3 = filled(arm)
        p3["traj_label_rank_policy"] = "all"
        ok("rank policy" in refusals(p3, argv), arm)
    pw = L.scorerft_profile("T")
    pw["init_from_sha256"] = HEX_BASE
    pw["rank_w"] = None
    ok("rank_w" in refusals(pw, targv("T")) and "finite float" in refusals(pw, targv("T")), "an unfilled weight refuses")
    pw["rank_w"] = 0.0
    ok("must be > 0" in refusals(pw, targv("T")), "T at weight 0 refuses")
    pc = filled("C")
    pc["rank_w"] = 0.1
    ok("EXACTLY 0.0" in refusals(pc, targv("C")), "C at a weight refuses")
    pr = filled("T")
    pr["rank_gate_topk"] = 16
    ok("not the card's" in refusals(pr, targv("T")), "the registered rule is the card's")
    pe = filled("T")
    pe["expect_schedule"] = dict(pe["expect_schedule"], steps=2000)
    ok("not the card's" in refusals(pe, targv("T")), "the schedule is 1,000 steps")
    pt = filled("T")
    pt["traj_label_required"] = False
    ok("traj_label_required" in refusals(pt, targv("T")), "(h) must key on the profile")


def mv(argv: list, flag: str, val: str) -> list:
    a = list(argv)
    a[a.index(flag) + 1] = val
    return a


def drop(argv: list, flag: str) -> list:
    a = list(argv)
    i = a.index(flag)
    del a[i:i + 2]
    return a


def chk_levers():
    pT, pC = filled("T"), filled("C")
    T, C = targv("T"), targv("C")
    for nm, bad in (("steps", mv(T, "--steps", "2000")), ("seed", mv(T, "--seed", "1")), ("rank-w", mv(T, "--rank-w", repr(W_ARGV + 0.5))), ("heads", mv(T, "--rank-heads", "ep")),
                    ("gap", mv(T, "--rank-ep-gap", "0.05")), ("topk", mv(T, "--rank-gate-topk", "16")), ("lr", mv(T, "--lr", "1e-4")), ("score-w", mv(T, "--score-w", "0.2")),
                    ("batch", mv(T, "--batch", "2")), ("nav", mv(T, "--nav-mode", "goal")), ("no rank-w (R-CTRL)", drop(T, "--rank-w")), ("no heads", drop(T, "--rank-heads")),
                    ("no table", drop(T, "--traj-label-table")), ("no init", drop(T, "--init-from")), ("extra flag", T + ["--field-labels", "/x"]), ("w-field", mv(T, "--w-field", "2.0"))):
        ok(refusals(pT, bad) != "", f"T must refuse: {nm}")
    for nm, bad in (("rank-w on C", mv(C, "--rank-w", "0.5")), ("no rank-w on C", drop(C, "--rank-w")), ("heads on C", C + ["--rank-heads", "ep,gate"]), ("seed on C", mv(C, "--seed", "1"))):
        ok(refusals(pC, bad) != "", f"C must refuse: {nm}")
    ok(refusals(pT, T + ["--rank-w", "0.25"]) != "", "a lever carried twice")
    ok(refusals(pT, T + ["--synthetic"]) != "", "a forbidden flag")
    old = os.environ.get("REFE_RANK_MUTATE")
    os.environ["REFE_RANK_MUTATE"] = "drop"
    try:
        ok("REFE_RANK_MUTATE" in refusals(pT, T) and "REFE_RANK_MUTATE" in refusals(pC, C), "the regression-arm environment switch is refused in a launch")
    finally:
        if old is None:
            os.environ.pop("REFE_RANK_MUTATE")
        else:
            os.environ["REFE_RANK_MUTATE"] = old
    ok(refusals(pT, T) == "" and refusals(pC, C) == "", "env restored")
    # R-CTRL: the control declares rank_L / the treatment omits it
    pc = filled("C")
    pc["live_terms"] = pc["live_terms"] + ("rank_L",)
    ok("declares rank_L live" in refusals(pc, C), "R-CTRL: C declaring rank_L")
    pt = filled("T")
    pt["live_terms"] = L.P6_TERMS
    ok("omit rank_L" in refusals(pt, T), "R-CTRL: T not declaring rank_L")


def chk_fresh_out():
    with tempfile.TemporaryDirectory(prefix="sft_out_") as td:
        (Path(td) / "ckpt_last.pt").write_bytes(b"x")
        bad = mv(targv("T"), "--out", td)
        ok("starts FRESH" in refusals(filled("T"), bad), "a seeded checkpoint is refused")
        ok("starts FRESH" not in refusals(filled("T"), mv(targv("T"), "--out", td + "/new")), "a fresh --out is accepted")


def chk_live_wiring():
    src = (HERE / "gate_checks.py").read_text(encoding="utf-8")
    ok('if profile.get("scorerft_rules"):' in src and "rank_live_rows(cap, profile, arm)" in src and "p5c_live_rows(cap, build_cap, profile)" in src, "check_live routes scorerft_rules")
    ok('profile["traj_label_required"] if "traj_label_required" in profile else profile.get("p5c_arm") in ("C1", "C1r")' in src, "(h) keys on the profile")
    ok('profile.get("scorerft_rules")))' in src, "G-CLOCK requires the bank content pin")
    lg = (HERE / "launch_gate.py").read_text(encoding="utf-8")
    ok('if prof.get("scorerft_rules"):\n        out += scorerft_refusals(prof, argv)' in lg, "profile_refusals routes scorerft_rules")
    ok('prof.get("scorerft_rules"))' in lg, "run_checks builds the BUILD capture for G-LIVE (f)")


def chk_h_generalised():
    """(h): a profile that says `traj_label_required` needs the table whatever the arm's NAME; a profile without the key behaves as before (C1 needs one, C0 / field arms must not carry one)."""
    ds = SimpleNamespace(traj_label=None, traj_label_sha256=None, traj_label_cov=None)
    cap = SimpleNamespace(args=SimpleNamespace(init_from=None), bank_objs=[ds], model=None, step0=None)
    r, _ = GC.p5c_live_rows(cap, SimpleNamespace(step0=None), {"scorerft_arm": "T", "traj_label_required": True, "init_from_sha256": HEX_BASE})
    ok(any("carries no label table" in x for x in r), r)
    ds2 = SimpleNamespace(traj_label=object(), traj_label_sha256="zz", traj_label_cov=None)
    cap2 = SimpleNamespace(args=SimpleNamespace(init_from=None), bank_objs=[ds2], model=None, step0=None)
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"p5c_arm": "C0", "init_from_sha256": HEX_BASE})
    ok(any("must not carry a label table" in x for x in r), "C0 unchanged: still must not carry a table")
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"p7field_arm": "C", "init_from_sha256": HEX_BASE})
    ok(any("must not carry a label table" in x for x in r), "the field arms unchanged")
    r, _ = GC.p5c_live_rows(cap2, SimpleNamespace(step0=None), {"scorerft_arm": "C", "traj_label_required": True, "init_from_sha256": HEX_BASE})
    ok(not any("must not carry a label table" in x for x in r), "a scorerft control carries the table and is NOT refused for it")


# ------------------------------------------------------------------------------------------------ the rows, on synthetic captures
def make_sets(seed=7, B=3, M=16):
    rng = np.random.default_rng(seed)
    y = np.zeros((B, M, 6), np.float32)
    y[..., 0] = (rng.random((B, M)) > 0.25)
    y[..., 1] = (rng.random((B, M)) > 0.2)
    y[..., 2] = rng.random((B, M))
    y[..., 3] = (rng.random((B, M)) > 0.4)
    y[..., 4] = (rng.random((B, M)) > 0.1)
    y[..., 5] = 1.0
    s = rng.normal(size=(B, M, 6)).astype(np.float32)
    return s, y


def torch_term(logits, labels, msk, batch, heads, ep_gap, gate_topk, mutate=None):
    """A torch stand-in for the TRAINER's term (DESIGN 2.2) with the env-mutation switches of 3.4: the system under test for the rows."""
    import torch
    F = torch.nn.functional
    s, y = logits.float(), labels.float()
    B, M, _ = s.shape
    safe = (y[..., 0] > 0.5) & (y[..., 1] > 0.5)
    if mutate == "unsafe_ep":
        safe_ep = torch.ones_like(safe)
    else:
        safe_ep = safe
    if mutate == "shuffle":
        y = y.roll(1, dims=1)
        safe, safe_ep = (y[..., 0] > 0.5) & (y[..., 1] > 0.5), (y[..., 0] > 0.5) & (y[..., 1] > 0.5)
    pdms = y[..., 0] * y[..., 1] * (5 * y[..., 2] + 5 * y[..., 3] + 2 * y[..., 4]) / 12
    p = torch.sigmoid(s.detach())
    agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12
    top = torch.zeros_like(safe)
    ti = torch.topk(agg, int(gate_topk), dim=1).indices
    top.scatter_(1, ti, True)
    hs = {h.strip() for h in heads.split(",") if h.strip()}
    total, pairs = s.new_zeros(()), {}
    for b in range(B):
        for h in ("ep", "gate", "ttc"):
            if h not in hs:
                continue
            if h == "ep":
                m_ = safe_ep[b][:, None] & safe_ep[b][None, :] & ((y[b, :, 2][:, None] - y[b, :, 2][None, :]) >= ep_gap)
                a = (5 / 12) * (y[b, :, 2][:, None] - y[b, :, 2][None, :])
                z = s[b, :, 2][:, None] - s[b, :, 2][None, :]
            elif h == "gate":
                m_ = safe[b][:, None] & (~safe[b])[None, :] & top[b][None, :] & (pdms[b][:, None] > pdms[b][None, :])
                a = pdms[b][:, None] - pdms[b][None, :]
                zz = F.logsigmoid(s[b, :, 0]) + F.logsigmoid(s[b, :, 3 if mutate == "head_swap" else 1])
                z = zz[:, None] - zz[None, :]
            else:
                m_ = safe[b][:, None] & safe[b][None, :] & (y[b, :, 3][:, None] == 1) & (y[b, :, 3][None, :] == 0)
                a = torch.full((M, M), 5 / 12)
                z = s[b, :, 3][:, None] - s[b, :, 3][None, :]
            pairs[h] = pairs.get(h, 0) + int(m_.sum())
            if m_.any():
                af = torch.where(m_, a, torch.zeros_like(a))
                total = total + msk[b] * (af * F.softplus(-z)).sum() / af.sum() / float(batch)
    out = total
    if mutate == "drop":
        out = total.detach()                                              # computed, never reaches the loss
    return out, {"pairs": pairs}


class FakeRank:
    mutate = None

    def __init__(self, heads="ep,gate", ep_gap=0.02, gate_topk=8):
        self.heads, self.ep_gap, self.gate_topk = heads, ep_gap, gate_topk

    def facts(self):
        return {"heads": self.heads, "ep_gap": self.ep_gap, "gate_topk": self.gate_topk}

    def term(self, score, labels, msk, batch):
        return torch_term(score, labels, msk, batch, self.heads, self.ep_gap, self.gate_topk, FakeRank.mutate)


def make_model(leaky=False):
    import torch
    nn = torch.nn

    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.backbone = nn.Module()
            self.backbone.lo = nn.Module()
            self.backbone.lo.A = nn.Parameter(torch.ones(1, 8) * 0.5)
            self.scene_proj = nn.Linear(8, 8)
            self.queries = nn.Parameter(torch.randn(16, 8) * 0.1)
            self.dec = nn.Linear(8, 8)
            self.traj_head = nn.Linear(8, 4)
            self.score_q_mlp = nn.Sequential(nn.Linear(4, 8))
            self.score_dec = nn.Linear(8, 8)
            self.score_head = nn.Linear(8, 6)

        def forward(self, img, ego, goal, calib=None, detach_scorer=True):
            vis = self.scene_proj(img * self.backbone.lo.A)
            h = self.dec(img[:, None, :] + self.queries[None])
            traj = self.traj_head(h)
            q = self.score_q_mlp((traj if (leaky or not detach_scorer) else traj.detach()))
            s = self.score_head(torch.relu(q + self.score_dec(vis)[:, None, :]))
            return traj, s
    return Fake()


def install_fake_rank_module():
    mod = types.ModuleType("rank_loss")
    mod.RankLoss = FakeRank
    sys.modules["rank_loss"] = mod


def reach_cap(heads="ep,gate", w=W_FIX, leaky=False, seed=3):
    import torch
    m = make_model(leaky)
    s_, y = make_sets(seed=seed, B=1, M=16)
    img = torch.randn(2, 8)
    cap = SimpleNamespace(model=m, first_batch=(img, torch.zeros(2, 5), torch.zeros(2, 2), None), rank_first=[{"labels": y, "msk": np.ones(1, np.float32)}],
                          args=SimpleNamespace(rank_w=w, rank_heads=heads, rank_ep_gap=0.02, rank_gate_topk=8))
    return cap


def chk_reach_and_regressions():
    """L3 on a tiny stand-in model: the CLEAN term passes; R-DROP (detached), R-HEAD (the gate reads the wrong logit) and R-LEAK (traj not detached) each turn it RED; the weight-0 control reads exactly 0.0."""
    install_fake_rank_module()
    FakeRank.mutate = None
    cap = reach_cap()
    fr = GC.rank_reach(cap, {}, None)
    ok(fr["ran"] and fr["labelled"] == 1, fr)
    ok(GC.judge_rank_reach(fr, "ep,gate") == [], GC.judge_rank_reach(fr, "ep,gate"))
    rows = fr["declared"]["score_head_rows"]
    ok(rows["nc"] > 0 and rows["dac"] > 0 and rows["ep"] > 0 and rows["ttc"] == 0.0 and rows["c"] == 0.0 and rows["ddc"] == 0.0, rows)
    ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()) and all(v == 0.0 for v in fr["zero"]["score_head_rows"].values()), "the weight-0 control reads exactly 0.0")
    ok(fr["declared"]["groups"]["traj_head"]["abs_sum"] == 0.0 and fr["declared"]["groups"]["dec"]["abs_sum"] == 0.0 and fr["declared"]["groups"]["queries"]["abs_sum"] == 0.0, "trajectory side exactly 0.0")
    ok(fr["declared"]["groups"]["backbone.lora"]["abs_sum"] > 0 and fr["declared"]["groups"]["scene_proj"]["abs_sum"] > 0, "the trunk is reached")
    for mut, frag in (("drop", "does not reach a head it trains"), ("head_swap", "row TTC")):
        FakeRank.mutate = mut
        try:
            r = GC.judge_rank_reach(GC.rank_reach(reach_cap(), {}, None), "ep,gate")
            ok(r != [] and any(frag in x for x in r), (mut, r))                                   # R-DROP / R-HEAD: RED
        finally:
            FakeRank.mutate = None
    r = GC.judge_rank_reach(GC.rank_reach(reach_cap(leaky=True), {}, None), "ep,gate")
    ok(any("trajectory side" in x for x in r), ("R-LEAK", r))                                    # R-LEAK: RED on the trajectory zero control
    cap_ttc = reach_cap(heads="ep,gate,ttc")
    r = GC.judge_rank_reach(GC.rank_reach(cap_ttc, {}, None), "ep,gate,ttc")
    ok(r == [], ("ttc named: its row must be non-zero", r))
    cap_ttc2 = reach_cap(heads="ep,gate")
    fr2 = GC.rank_reach(cap_ttc2, {}, None)
    ok(GC.judge_rank_reach(fr2, "ep,gate,ttc") != [], "a TTC head the term does not train but the arm names is RED")
    ok(GC.judge_rank_reach({"ran": False}, "ep,gate") != [] and GC.judge_rank_reach({"ran": True, "labelled": 0}, "ep,gate") != [], "an unrun / unlabelled probe is not a pass")


def audit_call(mutate, seed=11, heads="ep,gate", gap=0.02, topk=8):
    import torch
    s_, y = make_sets(seed=seed)
    msk = np.ones(s_.shape[0], np.float32)
    FakeRank.mutate = mutate
    try:
        out, st = torch_term(torch.from_numpy(s_), torch.from_numpy(y), torch.from_numpy(msk), s_.shape[0], heads, gap, topk, mutate)
    finally:
        FakeRank.mutate = None
    return {"logits": s_, "labels": y, "msk": msk, "batch": int(s_.shape[0]), "heads": heads, "ep_gap": gap, "gate_topk": topk, "value": float(out), "pairs": st["pairs"]}


def chk_audit():
    ref_ok, _ = GC.judge_rank_audit([audit_call(None)], None)
    ok(ref_ok == [], ref_ok)
    c = audit_call(None)
    ref = GC.rank_reference(c["logits"], c["labels"], c["msk"], c["batch"], c["heads"], c["ep_gap"], c["gate_topk"])
    ok(ref["pairs"]["ep"] > 0 and ref["pairs"]["gate"] > 0 and ref["L"] > 0, ref["pairs"])
    ok(abs(ref["L"] - c["value"]) <= 1e-6 * abs(c["value"]), (ref["L"], c["value"]))
    for mut in ("unsafe_ep", "shuffle"):                                                            # R-UNSAFE, R-SHUF: RED by value and by the pair counts
        r, _ = GC.judge_rank_audit([audit_call(mut)], None)
        ok(r != [], mut)
    ok(GC.judge_rank_audit([audit_call("head_swap")], None)[0] != [], "R-HEAD also moves the loss value")
    c2 = audit_call(None)
    c2["value"] *= 1.001
    ok(GC.judge_rank_audit([c2], None)[0] != [], "a 0.1 % value error is caught (tol 1e-6)")
    c3 = audit_call(None)
    c3["pairs"] = dict(c3["pairs"], ep=c3["pairs"]["ep"] + 1)
    ok(GC.judge_rank_audit([c3], None)[0] != [], "a wrong pair count is caught")
    ok(GC.judge_rank_audit([], None)[0] != [], "no captured call is not a pass")
    # analytic identities of the reference itself: a bar below everything / above everything
    s_, y = make_sets()
    msk = np.ones(3, np.float32)
    full = GC.rank_reference(s_, y, msk, 3, "ep,gate", 0.02, 16)
    wide = GC.rank_reference(s_, y, msk, 3, "ep,gate", 0.02, 10 ** 6)
    ok(full["L"] == wide["L"] and full["pair_sets"] == wide["pair_sets"], "topk >= M is the unrestricted reference")
    none = GC.rank_reference(s_, y, msk, 3, "ep", 1.5, 8)
    ok(none["L"] == 0.0 and none["pairs"]["ep"] == 0, "a gap above everything: zero pairs, exactly 0.0")
    part = GC.rank_reference(s_, y, msk, 3, "ep", 0.5, 8)
    ok(0 < part["pairs"]["ep"] < GC.rank_reference(s_, y, msk, 3, "ep", 0.02, 8)["pairs"]["ep"], "a gap in the middle: partial")
    ok(GC.rank_reference(s_, y, np.zeros(3, np.float32), 3, "ep,gate", 0.02, 8)["L"] == 0.0, "an unlabelled set (mask 0) contributes exactly 0")


def dvb_cap(w=W_FIX, accum=2, steps=2, **over):
    n = steps * accum
    a = SimpleNamespace(rank_w=w, rank_heads="ep,gate" if w else "", rank_ep_gap=0.02, rank_gate_topk=8)
    sa = SimpleNamespace(steps=steps, accum=accum)
    if w:
        calls = [{"heads": "ep,gate", "ep_gap": 0.02, "gate_topk": 8} for _ in range(n)]
        grads = [w / accum] * n
        metrics = [{"event": "rank", "heads": ["ep", "gate"], "ep_gap": 0.02, "gate_topk": 8, "aggregate": "navsim_v1", "reduce": "set-wmean/const-B", "mutate": None}] + [
            {"step": i, "traj_L1": 1.0, "rank_w": w, "rank_L": 0.5, "rank_pairs_ep": 5, "rank_pairs_gate": 3} for i in range(steps)]
    else:
        calls, grads, metrics = [], [], [{"step": i, "traj_L1": 1.0} for i in range(steps)]
    s = SimpleNamespace(args=sa, rank_calls=calls, rank_grads=grads, metrics=metrics)
    for k, v in over.items():
        setattr(s, k, v)
    return a, s


def dvb_ok(a, s):
    return all(r["ok"] for r in GC.rank_dvb_rows(a, None, s)), [r for r in GC.rank_dvb_rows(a, None, s) if not r["ok"]]


def chk_dvb_rows():
    a, s = dvb_cap()
    good, bad = dvb_ok(a, s)
    ok(good, bad)
    a, s = dvb_cap(rank_grads=[])
    ok(not dvb_ok(a, s)[0], "R-DROP: the hook never fires")
    a, s = dvb_cap(rank_grads=[0.5 * W_FIX / 2] * 4)
    ok(not dvb_ok(a, s)[0], "the multiplier wired off by a factor reads the factor")
    a, s = dvb_cap(rank_grads=[W_FIX / 2] * 3)
    ok(not dvb_ok(a, s)[0], "a term that fired on only some micro-batches")
    a, s = dvb_cap(rank_calls=[{"heads": "ep,gate", "ep_gap": 0.02, "gate_topk": 8}] * 3)
    ok(not dvb_ok(a, s)[0], "the term is called once per micro-batch")
    a, s = dvb_cap()
    s.metrics[0]["heads"] = ["ep"]
    ok(not dvb_ok(a, s)[0], "the event's heads must equal the argv's")
    a, s = dvb_cap()
    s.rank_calls[0]["gate_topk"] = 4
    ok(not dvb_ok(a, s)[0], "the kwargs the term was called with must equal the argv's")
    a, s = dvb_cap()
    s.metrics[0]["mutate"] = "drop"
    ok(not dvb_ok(a, s)[0], "a term built under REFE_RANK_MUTATE can never count")
    a, s = dvb_cap()
    s.metrics = [m for m in s.metrics if m.get("event") != "rank"]
    ok(not dvb_ok(a, s)[0], "no `rank` start event")
    a, s = dvb_cap()
    s.metrics[1]["rank_w"] = 0.5
    ok(not dvb_ok(a, s)[0], "the logged rank_w must be the argv's on every row")
    a, s = dvb_cap(w=0.0)
    ok(dvb_ok(a, s)[0], "the control: nothing fires, nothing is logged")
    a, s = dvb_cap(w=0.0)
    s.rank_grads = [0.1]
    ok(not dvb_ok(a, s)[0], "R-CTRL: a control whose hook fires")
    a, s = dvb_cap(w=0.0)
    s.metrics[0]["rank_L"] = 0.3
    ok(not dvb_ok(a, s)[0], "R-CTRL: a control that logs rank_L")
    ok(GC.rank_dvb_rows(SimpleNamespace(rank_w=0.0), None, None) == [], "no smoke capture, no rows")


def chk_live_rows_control_and_t():
    install_fake_rank_module()
    FakeRank.mutate = None
    capC = SimpleNamespace(args=SimpleNamespace(rank_w=0.0, rank_heads=""), metrics=[{"step": 0, "traj_L1": 1.0}], rank_calls=[])
    ok(GC.rank_live_rows(capC, {}, None)[0] == [], "control passes")
    capC.metrics[0]["rank_L"] = 0.1
    ok(GC.rank_live_rows(capC, {}, None)[0] != [], "R-CTRL: control logging rank_L is RED")
    cap = reach_cap()
    cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_ep": 4, "rank_pairs_gate": 2} for i in range(2)]
    cap.rank_calls = [audit_call(None)]
    cap.lab_submits = []
    r, d = GC.rank_live_rows(cap, {}, None)
    ok(len(r) == 1 and "L5" in r[0], r)                                                              # only L5 (NAVSIM-bound, pod-only) is unproved on a dev box; L1-L4 pass
    cap.rank_traj_reach = {"n": 2, "n_none": 0, "abs_sum": 0.5}
    ok(any("R-LEAK" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "R-LEAK: the score the term got reaches traj_head")
    cap.rank_traj_reach = {"n": 2, "n_none": 2, "abs_sum": 0.0}
    cap.metrics[1]["rank_pairs_gate"] = 0
    cap.metrics[2]["rank_pairs_gate"] = 0
    ok(any("L2" in x and "gate" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "a head with zero pairs is structurally dead")
    cap.metrics[1].pop("rank_L")
    ok(any("L1" in x for x in GC.rank_live_rows(cap, {}, None)[0]), "rank_L absent from a row")
    ok(GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": 0.0}) == [] and GC.judge_harness_identity({"ran": True, "tokens": 8, "max_abs_diff": 1e-6}) != []
       and GC.judge_harness_identity({"ran": False, "why": "x"}) != [] and GC.judge_harness_identity({"ran": True, "tokens": 0, "max_abs_diff": 0.0}) != [], "L5 judge")


def chk_trainer_integration():
    """Re-run once the trainer half has landed: train.py's parser carries the four flags (default OFF), the registry == the parser (stale / unknown), and the real `rank_loss.RankLoss` passes L3 / L4 on the stand-in model."""
    import train as T
    dests = {a.dest for a in T.build_parser()._actions if a.dest != "help"}
    have = {"rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"} & dests
    if not have:
        print("  INTEGRATION_PENDING: train.py has none of --rank-w/--rank-heads/--rank-ep-gap/--rank-gate-topk (the trainer half is not in this tree): the registry<->parser identity, the real term's L3/L4 and the live G-DVB/G-LIVE run are NOT proved here", flush=True)
        return
    ok(have == {"rank_w", "rank_heads", "rank_ep_gap", "rank_gate_topk"}, have)
    d = T.build_parser().parse_args([])
    ok(d.rank_w == 0.0 and d.rank_heads == "" and d.rank_ep_gap == 0.02 and d.rank_gate_topk == 8, "defaults OFF")
    ok(set(GC.DVB_KINDS) == dests, sorted(set(GC.DVB_KINDS) ^ dests))
    import importlib
    sys.modules.pop("rank_loss", None)
    RL = importlib.import_module("rank_loss")
    ok(hasattr(RL, "RankLoss"), "rank_loss.RankLoss")
    FakeRank.mutate = None
    cap = reach_cap()
    fr = GC.rank_reach(cap, {}, None)
    ok(GC.judge_rank_reach(fr, "ep,gate") == [], GC.judge_rank_reach(fr, "ep,gate"))
    import torch
    c = audit_call(None)
    out = RL.RankLoss(heads="ep,gate", ep_gap=0.02, gate_topk=8).term(torch.from_numpy(c["logits"]), torch.from_numpy(c["labels"]), torch.from_numpy(c["msk"]), c["batch"])
    out = out[0] if isinstance(out, tuple) else out
    ref = GC.rank_reference(c["logits"], c["labels"], c["msk"], c["batch"], "ep,gate", 0.02, 8)
    ok(abs(float(out) - ref["L"]) <= 1e-6 * abs(ref["L"]), (float(out), ref["L"]))


def chk_no_scipy():
    src = (HERE / "test_scorerft_profile.py").read_text(encoding="utf-8")
    ok(not re.search(r"^\s*(import|from)\s+scipy", src, re.M))
    for m in re.finditer(r"subprocess\.run\(", src):
        depth, j = 0, m.end() - 1
        for k in range(j, len(src)):
            depth += src[k] == "("
            depth -= src[k] == ")"
            if depth == 0:
                break
        call = src[j:k + 1]
        ok("text=True" not in call or "encoding=" in call, "subprocess.run with text=True and no encoding=")


def chk_census_l2():
    """SFTT launch gate G-LIVE L2 (MM 2026-11-23): a head the smoke leaves at ZERO pairs passes iff the registered, hash-pinned census proves pairs > 0 under the DECLARED rule.  Fixtures are built in a tmp dir and the pin
    is the fixture's own sha (profile['rank_census']); the imported rank_loss is a fake module whose __file__ is a fixture."""
    import json
    install_fake_rank_module()
    FakeRank.mutate = None
    pr = L.scorerft_profile("T")
    ok(pr.get("rank_census") == L.SCORERFT_RANK_CENSUS and all(len(L.SCORERFT_RANK_CENSUS[k]) == 64 for k in ("sha256", "rank_loss_sha256")), "the profile of a ranking arm carries the pin")
    ok("rank_census" not in L.scorerft_profile("C") and "rank_census" not in L.scorerft_profile("Cs"), "a control arm carries no census")
    ok(L.SCORERFT_RANK_CENSUS["sha256"].startswith("a9f13c78") and L.SCORERFT_RANK_CENSUS["rank_loss_sha256"].startswith("c0e027d2"), "the pins are the MM's")
    INIT = "/workspace/data/refe_runs/p5c_c1/model_final.pt"
    base = {"heads": ["ep", "gate"], "ep_gap": 0.02, "gate_topk": 8, "label_backend": "pdm", "NOT_A_DECISION": False, "init_from": INIT, "rule": {"status": "SELECT", "rank_w": W_FIX},
            "pairs_by_head": {"ep": 100, "gate": 5}, "sets_with_pairs_by_head": {"ep": 20, "gate": 2}, "labelled_sets": 32}
    td = Path(tempfile.mkdtemp(prefix="sftcen_"))
    try:
        rl = td / "rank_loss.py"
        rl.write_bytes(b"# fixture rank_loss\r\nx = 1\r\n")
        rl_sha = hashlib.sha256(b"# fixture rank_loss\nx = 1\n").hexdigest()
        rl2 = td / "rank_loss_other.py"
        rl2.write_bytes(b"# a different rank_loss\n")

        def case(over=None, rank_loss_file=rl, pin_over=None, write=True, smoke_gate=0, drop_msk=False):
            d = json.loads(json.dumps(base))
            d.update(over or {})
            path = td / f"census_{len(list(td.iterdir()))}.json"
            if write:
                path.write_text(json.dumps(d), encoding="utf-8")
            pin = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if write else "00" * 32, "rank_loss_sha256": rl_sha}
            pin.update(pin_over or {})
            profile = {"rank_census": pin}
            sys.modules["rank_loss"].__file__ = str(rank_loss_file)
            cap = reach_cap()
            cap.args.init_from = INIT
            cap.metrics = [{"event": "rank"}] + [{"step": i, "traj_L1": 1.0, "rank_L": 0.4, "rank_pairs_ep": 4, "rank_pairs_gate": smoke_gate} for i in range(2)]
            cap.rank_calls = [audit_call(None)]
            if drop_msk:                                       # a 4th call: the trainer keeps the mask of the first 3 calls only
                cap.rank_calls += [dict(cap.rank_calls[0]) for _ in range(3)]
                for c_ in cap.rank_calls[3:]:
                    c_.pop("msk")
            cap.lab_submits = []
            r, det = GC.rank_live_rows(cap, profile, None)
            return [x for x in r if x.startswith("L2")], [x for x in r if not x.startswith("L2")], det

        # PASS (a): zero smoke gate pairs + a valid census
        l2, rest, det = case()
        ok(l2 == [] and len(rest) == 1 and "L5" in rest[0], (l2, rest))
        ok(det["census"]["verdict"]["gate"] == {"source": "census", "ok": True} and det["census"]["verdict"]["ep"] == {"source": "smoke", "ok": True}, det["census"]["verdict"])
        ok(det["census"]["n_smoke_sets"] == 3 and abs(det["census"]["p_zero_smoke"]["gate"] - (1 - 2 / 32) ** 3) < 1e-12, det["census"])
        ok(det["census"]["facts"]["sha256"] and det["census"]["facts"]["rank_loss_sha256"] == rl_sha, "the facts record both shas")
        ok(case(drop_msk=True)[2]["census"]["n_smoke_sets"] is None and case(drop_msk=True)[2]["census"]["p_zero_smoke"]["gate"] is None, "an unrecoverable smoke set count is None, never invented")
        # PASS (b): smoke pairs > 0 and NO census file: the old path, untouched
        l2, rest, det = case(write=False, smoke_gate=3)
        ok(l2 == [] and "census" not in det and det["pairs_logged"]["gate"] == 6, (l2, det.get("census")))
        # FAIL (c)-(k): each ONE reason, the original L2 text verbatim + the census clause
        fails = {"c_no_census_file": dict(write=False), "d_sha_mismatch": dict(pin_over={"sha256": "ab" * 32}), "e_census_gate_pairs_zero": dict(over={"pairs_by_head": {"ep": 100, "gate": 0}}),
                 "e2_census_gate_sets_zero": dict(over={"sets_with_pairs_by_head": {"ep": 20, "gate": 0}}), "f_gate_topk_16": dict(over={"gate_topk": 16}), "g_init_from_differs": dict(over={"init_from": "/x/other.pt"}),
                 "h_rank_loss_sha_differs": dict(rank_loss_file=rl2), "i_not_a_decision": dict(over={"NOT_A_DECISION": True}), "j_synthetic": dict(over={"label_backend": "synthetic"}),
                 "k_heads_ep_only": dict(over={"heads": ["ep"]}), "l_ep_gap": dict(over={"ep_gap": 0.05}), "m_rank_w": dict(over={"rule": {"status": "SELECT", "rank_w": 0.5}}),
                 "n_pin_rank_loss_sha": dict(pin_over={"rank_loss_sha256": "cd" * 32}), "o_no_pin": dict(pin_over={"sha256": ""})}
        for nm, kw in fails.items():
            l2, rest, det = case(**kw)
            ok(len(l2) == 1 and l2[0].startswith("L2 head 'gate' formed ZERO pairs over the whole smoke (structurally dead: the 'built, tested, unreachable' class); the registered census does not prove it: "), (nm, l2))
            ok(det["census"]["verdict"]["gate"] == {"source": "census", "ok": False}, nm)
        # judge / facts never raise
        ok(GC.rank_census_facts({}) ["ran"] is False and GC.rank_census_facts(None)["ran"] is False and GC.rank_census_facts({"rank_census": {"path": str(td / "nope.json")}})["ran"] is False, "facts: never raise")
        (td / "bad.json").write_text("{not json", encoding="utf-8")
        ok(GC.rank_census_facts({"rank_census": {"path": str(td / "bad.json")}})["ran"] is False, "facts: a corrupt file")
        ok(GC.judge_census_head("gate", {"ran": False, "why": "x"}, SimpleNamespace(), {}) != [], "judge: an unrun census")
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_field_reach_active_sample():
    """SFTT launch gate 1b (MM 2026-11-23, Fs): `field_reach` probes the FIRST sample whose hinge loss is > 0, not idx[0].  A fake `field_loss` (hinge on a per-sample scalar 'sdf') stands in for the real one."""
    import torch
    mod = types.ModuleType("field_loss")

    def field_loss(traj, sdf, margin=0.0, interp=None):                    # sum_b relu(margin - s_b) * mean(traj_b): exactly 0 (loss AND gradient) where s_b >= margin
        s = sdf.reshape(sdf.shape[0], -1)[:, 0]
        return (torch.relu(margin - s) * traj.reshape(traj.shape[0], -1).mean(1)).sum() / traj.shape[0]
    mod.field_loss = field_loss
    old = sys.modules.get("field_loss")
    sys.modules["field_loss"] = mod
    try:
        def fcap(s_vals, valid=None):
            B = len(s_vals)
            sdf = torch.tensor(s_vals, dtype=torch.float32).reshape(B, 1, 1)
            return SimpleNamespace(model=make_model(), first_batch=(torch.randn(B, 8), torch.zeros(B, 5), torch.zeros(B, 2), None), field_first=(sdf, valid),
                                   args=SimpleNamespace(w_field=2.0, field_margin=0.5, field_interp="5hz"))
        fr = GC.field_reach(fcap([5.0, 5.0, 0.1, 0.1]), None)                      # index 0 inactive (sdf 5 >= margin), index 2 the first active
        ok(fr["active_index"] == 2 and fr["n_active"] == 2 and fr["losses"][0] == 0.0 and fr["losses"][1] == 0.0 and fr["losses"][2] > 0.0, fr.get("losses"))
        ok(fr["declared"]["loss"] != 0.0 and fr["declared"]["groups"]["traj_head"]["abs_sum"] > 0.0 and fr["declared"]["groups"]["queries"]["abs_sum"] > 0.0, "an active sample reaches the groups")
        ok(all(v["abs_sum"] == 0.0 for v in fr["zero"]["groups"].values()), "the weight-0 control reads exactly 0.0")
        rs = GC.judge_field_reach(fr, 2.0)
        ok(not any("EXACTLY zero on 'traj_head'" in x or "EXACTLY zero on 'queries'" in x or "UNPROVED" in x for x in rs), rs)
        fr0 = GC.field_reach(fcap([0.1, 5.0, 5.0, 5.0]), None)                     # the old path: index 0 active
        ok(fr0["active_index"] == 0 and fr0["n_active"] == 1 and fr0["declared"]["groups"]["traj_head"]["abs_sum"] > 0.0, "index 0 active: unchanged")
        fr_n = GC.field_reach(fcap([5.0, 5.0, 5.0, 5.0]), None)                    # no labelled sample is active
        ok(fr_n["ran"] and fr_n["labelled"] == 4 and fr_n["n_active"] == 0 and "declared" not in fr_n, fr_n)
        rn = GC.judge_field_reach(fr_n, 2.0)
        ok(len(rn) == 1 and "UNPROVED" in rn[0] and "every hinge loss is exactly 0.0" in rn[0] and "does not reach" not in rn[0], rn)
        frv = GC.field_reach(fcap([0.1, 5.0, 0.1, 0.1], valid=torch.tensor([False, True, True, True])), None)   # an unlabelled sample is skipped; first ACTIVE labelled = 2
        ok(frv["labelled"] == 3 and frv["active_index"] == 2, frv.get("active_index"))
        ok(GC.judge_field_reach({"ran": True, "labelled": 0}, 2.0) != [] and GC.judge_field_reach({"ran": False}, 2.0) != [], "unchanged: unlabelled / not run")
    finally:
        if old is None:
            sys.modules.pop("field_loss", None)
        else:
            sys.modules["field_loss"] = old


CHECKS = {"registry": chk_registry, "existing_unchanged": chk_existing_unchanged, "argv_delta": chk_argv_delta, "pins_refuse": chk_pins_refuse, "levers": chk_levers, "fresh_out": chk_fresh_out,
          "live_wiring": chk_live_wiring, "h_generalised": chk_h_generalised, "reach_and_regressions": chk_reach_and_regressions, "audit": chk_audit, "dvb_rows": chk_dvb_rows,
          "live_rows": chk_live_rows_control_and_t, "census_l2": chk_census_l2, "field_reach_active": chk_field_reach_active_sample, "trainer_integration": chk_trainer_integration, "no_scipy": chk_no_scipy}

# (file, old, new, the check that must go RED): ONE emitting line of a COPY of this tree is neutralised.
MUTATIONS = [
    ("launch_gate.py", "    if prof.get(\"scorerft_rules\"):\n        out += scorerft_refusals(prof, argv)\n", "", "levers"),                                         # the rules are not hooked into profile_refusals
    ("launch_gate.py", "    if not _p5c_hex64(prof.get(\"init_from_sha256\")):\n        out.append(f\"{name}: C1's final", "    if False:\n        out.append(f\"{name}: C1's final", "pins_refuse"),   # an unfilled base pin no longer refuses
    ("launch_gate.py", "    elif ranking and not rw > 0.0:\n", "    elif False:\n", "pins_refuse"),                                                                    # a ranking arm at weight 0 is accepted
    ("launch_gate.py", "    elif not ranking and rw != 0.0:\n", "    elif False:\n", "pins_refuse"),                                                                  # a control at a weight is accepted
    ("launch_gate.py", "    for ev in tuple(prof.get(\"forbidden_env\", ())) + SCORERFT_ENV_FORBIDDEN:\n", "    for ev in ():\n", "levers"),                           # REFE_RANK_MUTATE is accepted in a launch
    ("launch_gate.py", "    if not ranking and \"rank_L\" in terms:\n", "    if False:\n", "levers"),                                                                # a control declaring rank_L is accepted (R-CTRL)
    ("launch_gate.py", "    for f in sorted(set(got) - set(want) - {\"--resume\"}):\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arms",
     "    for f in []:\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arms", "levers"),                  # an extra lever is accepted
    ("launch_gate.py", "    a[a.index(\"--seed\") + 1] = str(SCORERFT_ARM_SEED[arm])\n", "", "argv_delta"),                                                          # the replicates no longer differ by the seed
    ("launch_gate.py", "        rk = [\"--rank-w\", \"0.0\"]\n", "        rk = []\n", "argv_delta"),                                                                   # C no longer NAMES --rank-w 0.0
    ("gate_checks.py", "    if profile.get(\"scorerft_rules\"):                                   # SCORER-FT arms", "    if False:                                   # SCORER-FT arms", "live_wiring"),   # the scorerft arms are not routed to the live rows
    ("gate_checks.py", "    needs_table = profile[\"traj_label_required\"] if \"traj_label_required\" in profile else profile.get(\"p5c_arm\") in (\"C1\", \"C1r\")", "    needs_table = profile.get(\"p5c_arm\") in (\"C1\", \"C1r\")", "h_generalised"),
    ("gate_checks.py", "        ok=bool(calls) and len(mult) == len(calls) and all(math.isclose(m_, w, rel_tol=1e-5) for m_ in mult))", "        ok=True)", "dvb_rows"),      # the hook row no longer judges the multiplier
    ("gate_checks.py", "        elif not math.isclose(v, ref[\"L\"], rel_tol=RANK_REL_TOL, abs_tol=1e-12):", "        elif False:", "audit"),                                       # the L4 value comparison is gone
    ("gate_checks.py", "        if x is not None and x[\"abs_sum\"] != 0.0:\n            out.append(f\"L3 rank-only backward reaches the trajectory side", "        if False:\n            out.append(f\"L3 rank-only backward reaches the trajectory side", "reach_and_regressions"),
    ("gate_checks.py", "        if rows.get(h, 0.0) != 0.0:\n            out.append(f\"L3 rank-only backward reaches score_head row", "        if False:\n            out.append(f\"L3 rank-only backward reaches score_head row", "reach_and_regressions"),
    ("gate_checks.py", "    c = census or {}\n    if not c.get(\"ran\"):", "    return []\n    c = census or {}\n    if not c.get(\"ran\"):", "census_l2"),                        # judge_census_head returns [] unconditionally (L2 amendment)
    ("gate_checks.py", "    out[\"active_index\"] = idx[active[0]]\n    sl = slice(idx[active[0]], idx[active[0]] + 1)", "    out[\"active_index\"] = idx[0]\n    sl = slice(idx[0], idx[0] + 1)", "field_reach_active"),   # always probe idx[0] (the Fs defect)
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="sftmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            for f in HERE.glob("p7_pod.sh"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:70]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_scorerft_profile.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
            ok(r.returncode != 0, f"MUTATION SURVIVED: {fn}: {old[:60]!r} -> check {check} still passed")
            print(f"  mutation RED as required: {fn} [{old.strip()[:56]}] -> {check}", flush=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)


def main() -> int:
    if "--check" in sys.argv:
        name = sys.argv[sys.argv.index("--check") + 1]
        CHECKS[name]()
        print(f"CHECK_OK {name} {N}")
        return 0
    for name, fn in CHECKS.items():
        fn()
        print(f"  {name}: ok ({N})", flush=True)
    if "--fast" not in sys.argv:
        run_mutations()
    print(f"ZZSCORERFTPROF_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
