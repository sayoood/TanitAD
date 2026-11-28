#!/usr/bin/env python3
"""P5-C launch-gate PROFILES (PREREG_P5C 10.1; PREREG_P5C_A1 Point 5): `refe_p5c` (C1), `refe_p5c_c0` (C0), `refe_p5c_c0r` (C0r, the conditional seed replicate), `refe_p5c_c1r` (C1r, C1's seed-1 replicate, PI go 2026-11-20).

  python test_p5c_profile.py            everything, incl. the source-mutation harness
  python test_p5c_profile.py --fast     no mutation harness
  python test_p5c_profile.py --check N  one named check (what the harness runs on a mutated copy of this tree)

CPU only, no scipy, stdlib + the gate modules.  Prints ZZP5CPROF_PASS <n>.  Asserts: the three profiles exist and share the registered recipe; argv(C1) - argv(C0) = {--traj-label-table, --out}
and argv(C0r) - argv(C0) = {--seed, --out} (card 2 / 10.1); the profiles REFUSE while a pin is unfilled (base sha256; on C1 the table sha256 and the rank policy); with every pin filled the template
argv is clean; each lever the card names (lr, steps / epochs, init-from, seed, no head, the table flag per arm, the bank, a fresh --out) is refused when it moves; B-P6 forbids the yaw table.
The mutation harness neutralises ONE emitting line of a COPY of launch_gate.py at a time and requires the check to go RED.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402

N = 0
HEX_BASE, HEX_TABLE = "ab" * 32, "cd" * 32
NAMES = {"C1": "refe_p5c", "C0": "refe_p5c_c0", "C0r": "refe_p5c_c0r", "C1r": "refe_p5c_c1r"}


def ok(cond, msg=""):
    global N
    assert cond, msg
    N += 1


def filled(arm: str, base: str = "p7") -> dict:
    """the profile with every MM-filled pin given a (fixture) value: what the gate sees once the real objects exist."""
    p = L.p5c_profile(arm, base)
    p["init_from_sha256"] = HEX_BASE
    if arm in L.P5C_TABLE_ARMS:
        p["traj_label_sha256"], p["traj_label_rank_policy"] = HEX_TABLE, "rank0"
    return p


def unfilled(arm: str) -> dict:
    """the profile with every MM-filled pin set to None, whatever the source holds now (this test must stay green after the MM fills them)."""
    p = L.p5c_profile(arm)
    p["init_from_sha256"] = None
    if arm in L.P5C_TABLE_ARMS:
        p["traj_label_sha256"] = p["traj_label_rank_policy"] = None
    return p


def refusals(prof: dict, argv: list) -> str:
    return " || ".join(L.profile_refusals(prof, argv))


def swap(argv: list, flag: str, *vals) -> list:
    a = list(argv)
    i = a.index(flag)
    j = i + 1
    while j < len(a) and not a[j].startswith("--"):
        j += 1
    return a[:i] + [flag, *vals] + a[j:]


def drop(argv: list, flag: str) -> list:
    a = list(argv)
    i = a.index(flag)
    j = i + 1
    while j < len(a) and not a[j].startswith("--"):
        j += 1
    return a[:i] + a[j:]


def chk_registry():
    for arm, nm in NAMES.items():
        ok(nm in L.PROFILES and L.PROFILES[nm]["p5c_arm"] == arm and L.PROFILES[nm]["name"] == nm and L.PROFILES[nm]["p5c_rules"] is True)
        pr = L.PROFILES[nm]
        ok("G-RESERVE" not in pr["checks"] and "--aux-drv-labels" not in pr["data_flags"] and "--init-from" in pr["data_flags"], "no head, no reserve audit, the base checkpoint is a DATA flag")
        ok("--init-from" not in pr["forbidden_flags"] and all(f in pr["forbidden_flags"] for f in L.AUX_HEAD_FLAGS), "init-from allowed; every head flag forbidden")
        ok(pr["expect_schedule"]["epochs"] == 0 and pr["expect_schedule"]["steps"] == 2000 and pr["expect_schedule"]["epoch_len"] == 92896)
        ok(pr["expect_schedule"]["n_tuples_by_bank"] == {"train_grow_cmd_3c_p7": 150603} and pr["expect_schedule"]["targets_sha256_by_bank"] == L.P7_TARGETS_SHA256)
        ok(pr["required_flags"]["--lr"] == "5e-5" and pr["required_flags"]["--steps"] == "2000" and pr["required_flags"]["--epochs"] == "0" and "--ego-yaw-table" in pr["required_flags"])
        ok(pr["required_flags"]["--seed"] == str(L.P5C_ARM_SEED[arm]))
        ok(any(e["script"] == "test_traj_label_table.py" and e["token"] == "ZZTRAJLABEL_PASS" for e in pr["suite"]))
        ok(not any(e["script"] in ("test_session_reserve.py", "test_p7_integration.py") for e in pr["suite"]), "P7's reserve suites are not P5-C's")
        ok(all(any(e["script"] == s_ for e in pr["suite"]) for s_ in ("test_p5c_live.py", "test_p5c_postgate.py", "test_p5c_chain.py")), "G-LIVE (f)/(h), the post-training gate and the chain are in P5-C's G-SUITE")
    ok("--traj-label-table" in L.PROFILES["refe_p5c"]["required_flags"] and "--traj-label-table" not in L.PROFILES["refe_p5c_c0"]["required_flags"], "C1 REQUIRES the table flag")
    ok(("--traj-label-table" in L.PROFILES["refe_p5c"]["data_flags"]) and "--traj-label-table" not in L.PROFILES["refe_p5c_c0"]["data_flags"])
    ok("--traj-label-table" in L.PROFILES["refe_p5c_c0"]["forbidden_flags"] and "--traj-label-table" in L.PROFILES["refe_p5c_c0r"]["forbidden_flags"] and "--traj-label-table" not in L.PROFILES["refe_p5c"]["forbidden_flags"])
    # C1r (C1's seed-1 replicate) is a TABLE arm exactly like C1: required + data flag, never forbidden, the pins registered; and only its seed / name / arm differ from C1's profile
    c1, c1r = L.PROFILES["refe_p5c"], L.PROFILES["refe_p5c_c1r"]
    ok("--traj-label-table" in c1r["required_flags"] and "--traj-label-table" in c1r["data_flags"] and "--traj-label-table" not in c1r["forbidden_flags"], "C1r requires the table flag")
    ok(L.P5C_TABLE_ARMS == {"C1", "C1r"} and L.P5C_ARM_SEED["C1r"] == 1 and L.P5C_ARM_SEED["C1"] == 0 and c1r["traj_label_sha256"] == c1["traj_label_sha256"] and c1r["traj_label_rank_policy"] == c1["traj_label_rank_policy"])
    ok({k for k in set(c1) | set(c1r) if c1.get(k) != c1r.get(k)} == {"name", "p5c_arm", "required_flags"} and {k for k in c1["required_flags"] if c1["required_flags"][k] != c1r["required_flags"][k]} == {"--seed"}, "C1r differs from C1 by name, arm and --seed alone")
    # every suite script the profiles name exists beside this file
    for nm in NAMES.values():
        for e in L.PROFILES[nm]["suite"]:
            ok((HERE / e["script"]).is_file(), e["script"])
    # the registered pins are UNFILLED in the source (the MM fills them): None here, never a typed value
    ok(L.P5C_BASE_SHA256 in (None,) or re.fullmatch(r"[0-9a-f]{64}", str(L.P5C_BASE_SHA256)), "a pin is None (unfilled) or a real sha256: never a typed placeholder")
    ok(L.P5C_LABEL_TABLE_SHA256 in (None,) or re.fullmatch(r"[0-9a-f]{64}", str(L.P5C_LABEL_TABLE_SHA256)))
    ok(L.P5C_RANK_POLICY in (None, "all", "rank0"))
    ok(re.fullmatch(r"[0-9a-f]{64}", L.P5C_P6_BASE_SHA256) and L.P5C_P6_BASE_SHA256.startswith("968a469d"))


def chk_argv_delta():
    a1, a0, a0r, a1r = (L.p5c_argv(x) for x in ("C1", "C0", "C0r", "C1r"))
    f = lambda a: {k: tuple(v) for k, v in GT.flag_pairs(a)}          # noqa: E731
    d10 = {k for k in set(f(a1)) | set(f(a0)) if f(a1).get(k) != f(a0).get(k)}
    ok(d10 == {"--traj-label-table", "--out"}, d10)
    ok(set(f(a1)) - set(f(a0)) == {"--traj-label-table"} and set(f(a0)) - set(f(a1)) == set())
    d0r = {k for k in set(f(a0r)) | set(f(a0)) if f(a0r).get(k) != f(a0).get(k)}
    ok(d0r == {"--seed", "--out"} and f(a0r)["--seed"] == ("1",) and f(a0)["--seed"] == ("0",), d0r)
    ok(f(a1)["--seed"] == ("0",) == f(a0)["--seed"], "C1 and C0 share the seed (common random numbers)")
    d1r = {k for k in set(f(a1r)) | set(f(a1)) if f(a1r).get(k) != f(a1).get(k)}
    ok(d1r == {"--seed", "--out"} and f(a1r)["--seed"] == ("1",) and f(a1)["--seed"] == ("0",), d1r)          # argv(C1r) - argv(C1) = {--seed, --out}
    ok(f(a1r)["--traj-label-table"] == f(a1)["--traj-label-table"] and f(a1r)["--out"][0].endswith("/p5c_c1r"), "C1r carries C1's table and its own --out")
    ok(len({f(a)["--out"] for a in (a1, a0, a0r, a1r)}) == 4, "four distinct out dirs")
    # the recipe tokens (card 3): lr 5e-5, 2,000 steps, epochs 0, batch 4 x accum 64, wd 0.01, bf16 + tf32, no head, label chunks 2, score-w 0.1, scorer sees trunk, nav command, yaw wrap off
    r = f(a1)
    ok(r["--lr"] == ("5e-5",) and r["--steps"] == ("2000",) and r["--epochs"] == ("0",) and r["--batch"] == ("4",) and r["--accum"] == ("64",) and r["--weight-decay"] == ("0.01",))
    ok(r["--amp"] == ("bf16",) and "--tf32" in r and r["--score-w"] == ("0.1",) and "--scorer-sees-trunk" in r and r["--nav-mode"] == ("command",) and r["--yaw-wrap"] == ("off",) and r["--labeller"] == ("online",))
    ok(not any(k in r for k in L.AUX_HEAD_FLAGS), "no head")
    ok("--ego-yaw-table" in r and "--init-from" in r, "B-P7: yaw table and base")
    ok("--ego-yaw-table" not in dict(GT.flag_pairs(L.p5c_argv("C1", "p6"))), "B-P6: no yaw table")
    # P7's own ARGS levers (p7_pod.sh:346) that P5-C keeps: the template does not drift from them
    pod = (HERE / "p7_pod.sh").read_text(encoding="utf-8")
    m = re.search(r'^ARGS="(.*)"$', pod, re.M)
    ok(m is not None, "p7_pod.sh's ARGS line")
    p7 = {k: tuple(v) for k, v in GT.flag_pairs(m.group(1).split())}
    shell_defaults = {"$YAW_WRAP": re.search(r'^YAW_WRAP="\$\{P7_YAW_WRAP-([^}]*)\}"', pod, re.M), "$LC": re.search(r'^LC="\$\{P7_LABEL_CHUNKS:-([^}]*)\}"', pod, re.M)}
    ok(all(v is not None for v in shell_defaults.values()), "p7_pod.sh's YAW_WRAP / LC defaults")
    p7 = {k: tuple(shell_defaults[x].group(1) if x in shell_defaults else x for x in v) for k, v in p7.items()}          # P7's ARGS names $YAW_WRAP and $LC: resolve them to the script's own defaults
    for k in ("--backbone", "--scorer-sees-trunk", "--yaw-wrap", "--nav-mode", "--labeller", "--label-chunks", "--epoch-unit", "--batch", "--accum", "--weight-decay", "--amp", "--score-w", "--w-field",
              "--tf32", "--compile", "--workers", "--log-every", "--ckpt-every-min"):
        ok(p7.get(k) == r.get(k), (k, p7.get(k), r.get(k)))


def chk_pins_refuse():
    for arm, nm in NAMES.items():
        rs = refusals(unfilled(arm), L.p5c_argv(arm))
        ok("init_from_sha256" in rs and "not REGISTERED" in rs, (nm, rs))
        if arm in L.P5C_TABLE_ARMS:
            ok("traj_label_sha256" in rs and "traj_label_rank_policy" in rs, rs)
        else:
            ok("traj_label_sha256" not in rs, rs)
        ok(refusals(filled(arm), L.p5c_argv(arm)) == "", (nm, refusals(filled(arm), L.p5c_argv(arm))))
    # malformed / typed pins are not registrations
    for bad in ("", "AB" * 32, "ab" * 31, None, 123):
        p = filled("C1")
        p["init_from_sha256"] = bad
        ok("init_from_sha256" in refusals(p, L.p5c_argv("C1")), bad)
    for bad in ("rank1", "ALL", None, ""):
        p = filled("C1")
        p["traj_label_rank_policy"] = bad
        ok("traj_label_rank_policy" in refusals(p, L.p5c_argv("C1")), bad)
    p = filled("C1")
    p["traj_label_sha256"] = "zz"
    ok("traj_label_sha256" in refusals(p, L.p5c_argv("C1")))
    p = filled("C0")
    p["traj_label_sha256"] = HEX_TABLE
    ok("NO table" in refusals(p, L.p5c_argv("C0")), "a table pin on a control profile")
    for pol in ("all", "rank0"):
        p = filled("C1")
        p["traj_label_rank_policy"] = pol
        ok(refusals(p, L.p5c_argv("C1")) == "")


def chk_levers():
    """every card lever, moved one at a time, is refused by NAME (and the control arm reads clean)."""
    for arm in ("C1", "C0", "C0r", "C1r"):
        pr, a = filled(arm), L.p5c_argv(arm)
        ok(refusals(pr, a) == "")
        for what, mut, needle in (
                ("lr 2e-4 (P7's)", swap(a, "--lr", "2e-4"), "--lr"),
                ("epochs 25", swap(a, "--epochs", "25"), "--epochs"),
                ("steps 1999", swap(a, "--steps", "1999"), "--steps"),
                ("no --init-from", drop(a, "--init-from"), "--init-from"),
                ("the head", a + ["--aux-drv", "--w-aux-drv", "0.1"], "--aux-drv"),
                ("a head label store", a + ["--aux-drv-labels", "/x"], "--aux-drv-labels"),
                ("--grow", a + ["--grow"], "--grow"),
                ("batch 8", swap(a, "--batch", "8"), "--batch"),
                ("wrong seed", swap(a, "--seed", "7"), "--seed"),
                ("no --seed", drop(a, "--seed"), "--seed"),
                ("yaw wrap on", swap(a, "--yaw-wrap", "on"), "--yaw-wrap"),
                ("an unlisted lever", a + ["--ckpt-every-steps", "100"], "--ckpt-every-steps"),
                ("a lever twice", a + ["--score-w", "0.1"], "--score-w"),
                ("no yaw table", drop(a, "--ego-yaw-table"), "--ego-yaw-table"),
                ("another bank", swap(a, "--targets", "/workspace/data/refe_navtrain/train_grow_cmd_3c"), "bank"),
                ("workers 8", swap(a, "--workers", "8"), "--workers"),
                ("log-every 5", swap(a, "--log-every", "5"), "--log-every"),
                ("score-w 0.2", swap(a, "--score-w", "0.2"), "--score-w"),
                ("weight decay 0", swap(a, "--weight-decay", "0.0"), "--weight-decay")):
            rs = refusals(pr, mut)
            ok(rs != "" and needle in rs, (arm, what, rs))
        ok(refusals(pr, a + ["--resume"]) == "" or "--resume" not in refusals(pr, a), "--resume is allowed (the supervisor's relaunch)")
        ok(refusals(pr, drop(a, "--resume")) == "", "and optional")
    # the table flag, per arm: C1 needs it; C0 / C0r forbid it; each arm refuses the other's argv
    ok("--traj-label-table" in refusals(filled("C1"), drop(L.p5c_argv("C1"), "--traj-label-table")), "C1 without the table reads as C0: refused")
    ok("FORBIDS" in refusals(filled("C0"), L.p5c_argv("C1")) and "FORBIDS" in refusals(filled("C0r"), L.p5c_argv("C1")))
    ok(refusals(filled("C0"), L.p5c_argv("C0r")) != "" and refusals(filled("C0r"), L.p5c_argv("C0")) != "", "C0 and C0r differ by the seed: each refuses the other's argv")
    ok(refusals(filled("C1"), L.p5c_argv("C0")) != "", "C1 refuses C0's argv")
    # C1r (the seed-1 replicate of C1): its real argv PASSES; each deliberate regression is REFUSED, by name
    r1, a1r = filled("C1r"), L.p5c_argv("C1r")
    ok(refusals(r1, a1r) == "" and L.p5c_refusals(r1, a1r) == [], refusals(r1, a1r))
    rs = refusals(r1, swap(a1r, "--seed", "0"))
    ok("--seed" in rs, ("C1r with --seed 0", rs))
    rs = refusals(r1, drop(a1r, "--traj-label-table"))
    ok("--traj-label-table" in rs, ("C1r with no table", rs))
    rs = refusals(r1, a1r + ["--ckpt-every-steps", "100"])
    ok("--ckpt-every-steps" in rs, ("C1r with one extra lever", rs))
    rs = refusals(r1, L.p5c_argv("C1"))
    ok("--seed" in rs, ("C1's argv under the C1r profile", rs))
    rs = refusals(filled("C1"), a1r)
    ok("--seed" in rs, ("C1r's argv under C1's profile", rs))
    ok("FORBIDS" in refusals(filled("C0r"), a1r) and "FORBIDS" in refusals(filled("C0"), a1r), "C1r's table argv under a control profile")


def chk_fresh_out():
    td = Path(tempfile.mkdtemp(prefix="p5c_out_"))
    try:
        for arm in ("C1", "C0", "C1r"):
            pr = filled(arm)
            a = swap(L.p5c_argv(arm), "--out", str(td / arm))
            ok(refusals(pr, a) == "", "no out dir yet")
            (td / arm).mkdir()
            ok(refusals(pr, a) == "", "an empty out dir is fresh")
            (td / arm / "ckpt_last.pt").write_bytes(b"x")
            rs = refusals(pr, a)
            ok("ckpt_last.pt" in rs and "FRESH" in rs, rs)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def chk_b6():
    p = filled("C1", "p6")
    ok(p["name"] == "refe_p5c_b6" and p["p5c_base"] == "p6")
    p = L.p5c_profile("C1", "p6")
    ok(p["init_from_sha256"] == L.P5C_P6_BASE_SHA256 and "--ego-yaw-table" in p["forbidden_flags"] and "--ego-yaw-table" not in p["required_flags"] and "--ego-yaw-table" not in p["data_flags"])
    p["traj_label_sha256"], p["traj_label_rank_policy"] = HEX_TABLE, "all"
    ok(refusals(p, L.p5c_argv("C1", "p6")) == "", refusals(p, L.p5c_argv("C1", "p6")))
    ok("--ego-yaw-table" in refusals(p, L.p5c_argv("C1", "p7")), "B-P6 refuses a yaw table")
    ok("p5c_b6" not in " ".join(L.PROFILES) and "refe_p5c_b6" not in L.PROFILES, "B-P6 is not registered until the P7 verdict says so")


def chk_untouched():
    """refe_p6 / refe_p7 and every older profile are what they were: no p5c key leaks in; refe_p7 still asks for its head."""
    for nm, pr in L.PROFILES.items():
        if nm in NAMES.values():
            continue
        ok(not pr.get("p5c_rules") and "p5c_arm" not in pr, nm)
    ok(L.PROFILES["refe_p7"]["required_flags"]["--w-aux-drv"] == "0.1" and L.PROFILES["refe_p7"]["required_flags"]["--epochs"] == "25" and "--init-from" in L.PROFILES["refe_p7"]["forbidden_flags"])
    ok(L.PROFILES["refe_p6"]["required_flags"]["--lr"] == "2e-4")


def chk_no_scipy():
    src = (HERE / "test_p5c_profile.py").read_text(encoding="utf-8")
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


CHECKS = {"registry": chk_registry, "argv_delta": chk_argv_delta, "pins_refuse": chk_pins_refuse, "levers": chk_levers, "fresh_out": chk_fresh_out, "b6": chk_b6, "untouched": chk_untouched,
          "no_scipy": chk_no_scipy}

# (file, old, new, the check that must go RED): ONE emitting line of a COPY of this tree is neutralised.
MUTATIONS = [
    ("launch_gate.py", "    if prof.get(\"p5c_rules\"):\n        out += p5c_refusals(prof, argv)\n", "", "pins_refuse"),                                    # the P5-C rules are not hooked into profile_refusals
    ("launch_gate.py", "    if not _p5c_hex64(prof.get(\"init_from_sha256\")):\n        out.append(f\"{name}: the base checkpoint's sha256 is not REGISTERED (init_from_sha256 = {prof.get('init_from_sha256')!r}; P5C_BASE_SHA256): the MM fills it from the sealed P7 final model_final.pt; an unpinned base is never launched from\")", "    if False:\n        out.append(f\"{name}: the base checkpoint's sha256 is not REGISTERED (init_from_sha256 = {prof.get('init_from_sha256')!r}; P5C_BASE_SHA256): the MM fills it from the sealed P7 final model_final.pt; an unpinned base is never launched from\")", "pins_refuse"),                              # an unfilled base pin no longer refuses
    ("launch_gate.py", "        if not _p5c_hex64(prof.get(\"traj_label_sha256\")):\n", "        if False:\n", "pins_refuse"),                      # an unfilled table pin no longer refuses
    ("launch_gate.py", "        if prof.get(\"traj_label_rank_policy\") not in (\"all\", \"rank0\"):\n", "        if False:\n", "pins_refuse"),               # an unfilled rank-policy pin no longer refuses
    ("launch_gate.py", "    if arm not in P5C_TABLE_ARMS:\n        forb += (\"--traj-label-table\",)\n", "", "levers"),                                           # the controls accept the table flag
    ("launch_gate.py", " + AUX_HEAD_FLAGS\n    if arm not in P5C_TABLE_ARMS:", "\n    if arm not in P5C_TABLE_ARMS:", "registry"),                                                     # the head is no longer forbidden
    ("launch_gate.py", "    if arm in P5C_TABLE_ARMS:\n        req[\"--traj-label-table\"] = None\n", "", "registry"),                                     # C1 may drop the table and read as C0
    ("launch_gate.py", "P5C_ARM_SEED = {\"C1\": 0, \"C0\": 0, \"C0r\": 1, \"C1r\": 1}", "P5C_ARM_SEED = {\"C1\": 0, \"C0\": 0, \"C0r\": 0, \"C1r\": 1}", "argv_delta"),
    ("launch_gate.py", "P5C_ARM_SEED = {\"C1\": 0, \"C0\": 0, \"C0r\": 1, \"C1r\": 1}", "P5C_ARM_SEED = {\"C1\": 0, \"C0\": 0, \"C0r\": 1, \"C1r\": 0}", "argv_delta"),          # C1r no longer differs by the seed (the C1r seed check)
    ("launch_gate.py", "P5C_TABLE_ARMS = frozenset({\"C1\", \"C1r\"})", "P5C_TABLE_ARMS = frozenset({\"C1\"})", "registry"),          # C1r is no longer a table arm             # C0r no longer differs by the seed
    ("launch_gate.py", "\"--lr\": \"5e-5\", \"--init-from\": None", "\"--lr\": \"2e-4\", \"--init-from\": None", "registry"),                           # the profile registers P7's lr, not the card's
    ("launch_gate.py", "    if ow and (Path(str(ow)) / \"ckpt_last.pt\").exists():\n        out.append(f\"{name}: --out {ow} already holds ckpt_last.pt: a P5-C arm starts FRESH (fresh AdamW; a seeded checkpoint would resume an old optimiser; card 10.1 `resume`)\")", "    if False:\n        out.append(f\"{name}: --out {ow} already holds ckpt_last.pt: a P5-C arm starts FRESH (fresh AdamW; a seeded checkpoint would resume an old optimiser; card 10.1 `resume`)\")", "fresh_out"),                                   # a seeded checkpoint is accepted
    ("launch_gate.py", "    for f in sorted(set(got) - set(want) - {\"--resume\"}):\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arm differs from its pair by the card's lever alone)\")", "    for f in []:\n        out.append(f\"{name}: the argv carries {f} {' '.join(got[f][0])}, which arm {arm}'s argv does not (the arm differs from its pair by the card's lever alone)\")", "levers"),                                  # an extra, unlisted lever is accepted
    ("launch_gate.py", "        elif f in want and f not in P5C_PATH_FLAGS and f != \"--resume\":\n            w, g = want[f], vs[0]\n            same = len(w) == len(g) and all((float(a) == float(b)) if re.fullmatch(r\"-?[0-9.eE+-]+\", a) and re.fullmatch(r\"-?[0-9.eE+-]+\", b) else a == b for a, b in zip(w, g))\n            if not same:", "        elif False:\n            w, g = want[f], vs[0]\n            same = len(w) == len(g) and all((float(a) == float(b)) if re.fullmatch(r\"-?[0-9.eE+-]+\", a) and re.fullmatch(r\"-?[0-9.eE+-]+\", b) else a == b for a, b in zip(w, g))\n            if not same:", "levers"),                       # a moved non-path lever is accepted
    ("launch_gate.py", "    forb = tuple(f for f in p6[\"forbidden_flags\"] if f != \"--init-from\")", "    forb = tuple(f for f in p6[\"forbidden_flags\"])", "registry"),    # --init-from stays forbidden (the profile could never launch)
    ("launch_gate.py", "    if base == \"p6\":\n        forb += (\"--ego-yaw-table\",)\n", "", "b6"),                                                       # B-P6 accepts the yaw table
]


def run_mutations() -> None:
    for fn, old, new, check in MUTATIONS:
        td = Path(tempfile.mkdtemp(prefix="p5cmut_"))
        try:
            for f in HERE.glob("*.py"):
                shutil.copyfile(f, td / f.name)
            for f in HERE.glob("p7_pod.sh"):
                shutil.copyfile(f, td / f.name)
            src = (td / fn).read_text(encoding="utf-8")
            ok(src.count(old) == 1, f"mutation target not found exactly once in {fn}: {old[:70]!r}")
            (td / fn).write_text(src.replace(old, new), encoding="utf-8", newline="\n")
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            r = subprocess.run([sys.executable, str(td / "test_p5c_profile.py"), "--check", check], cwd=str(td), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
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
    print(f"ZZP5CPROF_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
