#!/usr/bin/env python3
"""DATA-SCALE (PREREG_DATASCALE.md 4.3, 5 M6, 6, 9): tests of `--select cv5m`, its per-log OOF table and bootstrap, the six `refe_ds_*` launch-gate profiles, the CTRL-MARGIN
mutations, the learning-curve driver (CTRL-PARENT) and CTRL-INDEX.  SYNTHETIC data only, CPU only.  Every number printed is synthetic.

  python test_safehead_ds.py           everything: + every deliberate-regression arm of the gate, + mint / verify / the trainer's own token refusal end to end
  python test_safehead_ds.py --fast    G-SUITE's form: cv5 bit-identical, the rule on constructed tables, CTRL-MARGIN, the trainer's cv5m, the learning curve, CTRL-INDEX, refusals, clean gates

Prints ZZSHDS_PASS <n> (or ZZSHDS_FAIL).  It reuses test_safehead.py's synthetic world and helpers.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FAST = "--fast" in sys.argv
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]] + (["--fast"] if FAST else [])

import numpy as np  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import safehead_cvm as CVM  # noqa: E402
import safehead_data as D  # noqa: E402
import safehead_gate as SHG  # noqa: E402
import safehead_lc as LC  # noqa: E402
import safehead_train as T  # noqa: E402
import test_safehead as TS  # noqa: E402

N = 0
CV5_PROBE = "be809fc682974c8f4ced3f23b12f68596caf473fe8b5b118e1a5ae91338e229c"      # MEASURED on the TIP 01810a19 trainer (before this patch): EP-HEAD `--select cv5`, see test_cv5_unchanged
FAMS = {"ds_sh": {"rows": "0,1,3,5"}, "ds_ep": {"rows": "2"}}
EPS = 1e-9


def ok(name):
    global N
    N += 1
    print(f"ok {N:02d} {name}", flush=True)


def mk(tmp, name, **kw):
    return TS.make_world(tmp / name, **kw)


def argv_ds(w, out, fam, arm="s0", extra=()):
    a = TS.argv_for(w, out, seed=SHG.ARMS[arm]["seed"], perm=SHG.ARMS[arm]["perm"])
    if fam == "ds_ep":
        a = TS._swap(a, "--rows", "2")
    return a + ["--select", "cv5m"] + list(extra)


def prof_ds(w, fam, arm="s0"):
    p = copy.deepcopy(L.PROFILES[SHG.DS_FAMILIES[fam]["name"][arm]])
    p["init_from_sha256"] = D.sha256_file(w["ckpt"])
    p["suite"] = [{"raw": [sys.executable, "-c", "print('ZZTEST_PASS')"], "token": "ZZTEST_PASS"}]
    return p


def mint_env(tmp, w, fam, argv, tag):
    """A REAL minted PASS token for `argv` (the way test_e2e does): returns ({TANITAD_GATE_KEY_FILE, TOKEN_ENV}, token path)."""
    prof = prof_ds(w, fam)
    name = SHG.DS_FAMILIES[fam]["name"]["s0"]
    prof["name"] = name
    keyf = tmp / f"keys_{tag}" / "gate.key"
    key = GT.load_key(keyf, create=True)
    gate_out = tmp / f"gate_{tag}"
    L.run_checks(prof, argv, HERE.parent, gate_out, commit="ab" * 20, key=key)
    verdict, path, tok = L.finalize(prof, argv, HERE.parent, gate_out, key, commit="ab" * 20)
    assert verdict == "PASS", tok["reasons"]
    return {"TANITAD_GATE_KEY_FILE": str(keyf), T.TOKEN_ENV: str(path)}, path


def smoke(w, out, fam, steps, save, arm_env=None, extra=(), perm=None):
    a = argv_ds(w, out, fam, "red" if perm else "s0", extra=["--smoke-steps", str(steps), *extra])
    a = TS._swap(a, "--save-steps", save)
    if perm:
        a = TS._swap(a, "--permute-labels-seed", str(perm))
    r = TS.trainer(a, {T.ARM_ENV: arm_env} if arm_env else None)
    assert r.returncode == 0, (r.stdout[-300:], r.stderr[-800:])
    return out


def load_sel(out):
    return json.loads((Path(out) / "cv_select.json").read_text(encoding="utf-8"))


# ============================================================================================ constructed tables (the rule itself, no trainer)
STEPS = (0, 100, 200)


def table(deltas: dict, n_logs=24, base=90.0, seed=0):
    """A per-log OOF table.  deltas {step: {metric: [per-log deltas] | float}}; the base value varies by log so the estimator sees real between-log spread."""
    rng = np.random.default_rng(seed)
    b = {m: base + rng.normal(0, 2.0, n_logs) for m in CVM.METRICS}
    tab = {}
    for i in range(n_logs):
        lg = f"log_{i:03d}"
        tab[lg] = {"n": 10 + (i % 7), "per_step": {}}
        for s in STEPS:
            d = deltas.get(s, {})
            tab[lg]["per_step"][str(s)] = {m: float(b[m][i] + (0.0 if s == 0 else (np.asarray(d.get(m, 0.0))[i] if np.ndim(d.get(m, 0.0)) else d.get(m, 0.0)))) for m in CVM.METRICS}
    return tab


def _alt(mean, amp, n=24):
    """per-log deltas alternating mean+amp / mean-amp: a point gain `mean` whose CI straddles 0 when amp >> mean."""
    return np.array([mean + amp if i % 2 == 0 else mean - amp for i in range(n)])


def test_rule_on_tables():
    """CTRL-MARGIN (the rule): a +noise step WINS the no-margin argmax and cv5m refuses it; a clear gain is chosen; ties go earliest; EP alone separated worse blocks the step."""
    # (1) +noise: step 100's point gain is +0.1 with CI straddling 0; step 200 is slightly negative.  The parent rule picks 100; cv5m picks 0.
    noise = table({100: {"pdms": _alt(0.1, 1.5)}, 200: {"pdms": _alt(-0.3, 1.5)}})
    d = CVM.decide(noise, STEPS)
    g = d["per_step"]["100"]["gain"]
    assert g["point"] > 0 and g["lo"] < 0 < g["hi"], g
    assert d["no_margin_argmax_step_from_table"] == 100 and d["chosen_step"] == 0 and d["chosen_is_untrained"] and d["eligible_steps"] == [], d
    # (2) a clear gain with a clean guard: chosen
    clear = table({100: {"pdms": _alt(1.0, 0.1)}, 200: {"pdms": _alt(0.4, 0.1)}})
    d2 = CVM.decide(clear, STEPS)
    assert d2["chosen_step"] == 100 and d2["eligible_steps"] == [100, 200] and d2["per_step"]["100"]["gain"]["lo"] > 0, d2["eligible_steps"]
    # (3) ties -> the earliest: identical columns at 100 and 200
    tie = table({100: {"pdms": _alt(1.0, 0.1)}, 200: {"pdms": _alt(1.0, 0.1)}})
    tab = tie
    for lg in tab:
        tab[lg]["per_step"]["200"] = dict(tab[lg]["per_step"]["100"])
    assert CVM.decide(tab, STEPS)["chosen_step"] == 100
    # (4) EP ALONE separated worse: gain clearly positive, EP delta CI-hi < 0, every other sub-score clean
    ep_bad = table({100: {"pdms": _alt(1.0, 0.1), "ep": _alt(-1.0, 0.1)}})
    d4 = CVM.decide(ep_bad, STEPS)
    assert d4["per_step"]["100"]["gain"]["lo"] > 0 and d4["per_step"]["100"]["guard_violations"] == ["ep"] and not d4["per_step"]["100"]["eligible"] and d4["chosen_step"] == 0, d4["per_step"]["100"]
    # (5) the other four sub-scores are each guarded: each alone blocks the step
    for m in ("nc", "dac", "ttc", "c"):
        t = table({100: {"pdms": _alt(1.0, 0.1), m: _alt(-1.0, 0.1)}})
        dd = CVM.decide(t, STEPS)
        assert dd["per_step"]["100"]["guard_violations"] == [m] and dd["chosen_step"] == 0, (m, dd["per_step"]["100"]["guard_violations"])
    # (6) a sub-score that is worse but NOT separated (CI straddles 0) does not block
    soft = table({100: {"pdms": _alt(1.0, 0.1), "ep": _alt(-0.05, 1.5)}})
    assert CVM.decide(soft, STEPS)["chosen_step"] == 100
    # (7) all steps worse: step 0
    assert CVM.decide(table({100: {"pdms": _alt(-1.0, 0.1)}, 200: {"pdms": _alt(-2.0, 0.1)}}), STEPS)["chosen_step"] == 0
    # (8) the largest gain wins among the eligible, not the latest
    big = table({100: {"pdms": _alt(1.5, 0.1)}, 200: {"pdms": _alt(0.5, 0.1)}})
    assert CVM.decide(big, STEPS)["chosen_step"] == 100
    ok("cv5m rule on constructed tables: a +noise step wins the no-margin argmax and cv5m chooses 0; a clear gain is chosen; ties earliest; the largest gain wins; each of NC, DAC, TTC, EP, C "
       "alone (CI-hi < 0) blocks a step; a worse-but-not-separated sub-score does not; all-worse -> step 0")


def test_ctrl_margin_mutations():
    """CTRL-MARGIN: the mutation that drops the CI condition CHANGES the chosen step on the +noise fixture; the mutation that drops EP from the guard CHANGES it on the EP-only fixture."""
    noise = table({100: {"pdms": _alt(0.1, 1.5)}, 200: {"pdms": _alt(-0.3, 1.5)}})
    assert CVM.decide(noise, STEPS, CVM.rule_for_arm(None))["chosen_step"] == 0
    assert CVM.decide(noise, STEPS, CVM.rule_for_arm("ds_no_margin"))["chosen_step"] == 100, "ds_no_margin did not change the +noise fixture's choice"
    ep_bad = table({100: {"pdms": _alt(1.0, 0.1), "ep": _alt(-1.0, 0.1)}})
    assert CVM.decide(ep_bad, STEPS, CVM.rule_for_arm(None))["chosen_step"] == 0
    assert CVM.decide(ep_bad, STEPS, CVM.rule_for_arm("ds_guard_no_ep"))["chosen_step"] == 100, "ds_guard_no_ep did not change the EP-only fixture's choice"
    # the mutation of the OTHER clause does not mask this one
    assert CVM.decide(ep_bad, STEPS, CVM.rule_for_arm("ds_no_margin"))["chosen_step"] == 0 and CVM.decide(noise, STEPS, CVM.rule_for_arm("ds_guard_no_ep"))["chosen_step"] == 0
    assert CVM.rule_for_arm(None) == CVM.REGISTERED_RULE and CVM.rule_for_arm("ds_no_margin") != CVM.REGISTERED_RULE and CVM.rule_for_arm("ds_guard_no_ep") != CVM.REGISTERED_RULE
    assert CVM.rule_for_arm("ds_guard_no_ep")["guard"] == ["nc", "dac", "ttc", "c"]
    ok("CTRL-MARGIN: ds_no_margin turns the +noise fixture's choice 0 -> 100; ds_guard_no_ep turns the EP-only fixture's choice 0 -> 100; neither mutation masks the other; the built rule differs from the registered one")


def test_estimator_is_boot():
    """The CI is p5c_read.Boot's: same draws, same percentile.  The per-log table expansion reproduces Boot on the raw tokens, and no estimator is re-implemented in safehead_cvm."""
    import p5c_read as P
    assert CVM.P5C is P and CVM.P5C.Boot is P.Boot and (CVM.SEED, CVM.NBOOT) == (20260927, 10000) == (P.SEED, P.NBOOT)
    src = (HERE / "safehead_cvm.py").read_text(encoding="utf-8")
    assert "np.percentile" not in src and "default_rng" not in src and "bincount" not in src, "safehead_cvm re-implements a bootstrap estimator"
    rng = np.random.default_rng(3)
    n_logs, per = 17, 9
    logs = np.repeat([f"l{i:02d}" for i in range(n_logs)], per)
    n = len(logs)
    picks = {0: np.zeros(n, np.int64), 50: np.ones(n, np.int64)}
    vals = {m: rng.random((n, 64)) for m in CVM.METRICS}
    tab = CVM.build_table(logs, picks, vals)
    dec = CVM.decide(tab, [0, 50])
    ar = np.arange(n)
    raw = {m: vals[m][ar, 1] * 100 - vals[m][ar, 0] * 100 for m in CVM.METRICS}
    B = P.Boot(logs, n_boot=10000, seed=20260927)
    for m, key in (("pdms", "gain"), ("ep", "ep")):
        ref = B.ci(raw[m])
        got = dec["per_step"]["50"]["gain"] if m == "pdms" else dec["per_step"]["50"]["delta"]["ep"]
        assert all(abs(got[k] - ref[k]) <= EPS for k in ("point", "lo", "hi")), (m, got, ref)
    ok("estimator: the CI equals p5c_read.Boot on the RAW per-token values (10,000 draws, seed 20260927, 95 % percentile) to 1e-9 for the gain and a sub-score; safehead_cvm holds no estimator of its own")


def test_table_and_recheck(tmp, fam):
    """The trainer's cv5m on a synthetic cache: the per-log table, the pooled numbers, the choice == the rule recomputed FROM THE TABLE (CTRL-CV(e)), the parent's no-margin argmax."""
    w = mk(tmp, f"tab_{fam}", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    out = smoke(w, tmp / f"tab_{fam}_out", fam, 40, "0,20,40")
    s = load_sel(out)
    rows = FAMS[fam]["rows"]
    assert s["select"] == "cv5m" and s["n_logs"] == 9 and s["n_tokens"] == 9 * 24 and sorted(s["per_log"]) == sorted(sum(s["folds"].values(), [])) and len(s["per_log"]) == 9
    assert all(v["n"] == 24 and set(v["per_step"]) == {"0", "20", "40"} and set(v["per_step"]["0"]) == set(CVM.METRICS) for v in s["per_log"].values())
    for st, p in s["per_step"].items():                                                   # the table's pooled numbers are the trainer's token-level OOF numbers
        for m, key in (("pdms", "oof_pdms"), ("nc", "oof_nc"), ("dac", "oof_dac"), ("ttc", "oof_ttc"), ("ep", "oof_ep"), ("c", "oof_c")):
            tol = EPS if m == "pdms" else 1e-4                                            # the parent's sub-score means are float32 means (kept bit-identical); the table is float64
            assert abs(CVM.pooled(s["per_log"], int(st), m) - p[key]) <= tol, (st, m)
    rc = CVM.recheck(s)
    assert rc["match"] and rc["ci_bit_equal"], rc
    assert s["margin_rule"] == CVM.REGISTERED_RULE
    ps = s["per_step"]
    best = max(v["oof_pdms"] for v in ps.values())
    assert s["no_margin_argmax_step"] == min(int(k) for k, v in ps.items() if v["oof_pdms"] == best) == s["no_margin_argmax_step_from_table"]
    assert s["margin"]["0"]["gain"] == {"point": 0.0, "lo": 0.0, "hi": 0.0} and not s["margin"]["0"]["eligible"]
    el = [int(k) for k, v in s["margin"].items() if v["eligible"]]
    assert s["eligible_steps"] == el and (s["chosen_step"] in el if el else s["chosen_step"] == 0)
    for k, v in s["margin"].items():
        assert v["eligible"] == (int(k) > 0 and v["gain"]["lo"] > 0 and all(v["delta"][m]["hi"] >= 0 for m in CVM.GUARD)), (k, v)
    assert json.loads((out / "run.json").read_text())["chosen_step"] == s["chosen_step"] and not (out / "heldout_select.json").exists()
    assert D.load_head(out / "head_step00040.npz")["meta"]["select"] == "cv5m" and D.load_head(out / "head_step00040.npz")["meta"]["rows"] == [int(x) for x in rows.split(",")]
    # the written table is the evidence: perturb ONE log's value and the recomputation no longer reproduces the recorded CIs
    bad = json.loads(json.dumps(s))
    bad["per_log"][sorted(bad["per_log"])[0]]["per_step"]["40"]["pdms"] += 5.0
    assert not CVM.recheck(bad)["ci_bit_equal"]
    ok(f"[{fam}] cv5m (synthetic): per-log table for 9 logs x 3 steps x 6 metrics; its pooled numbers equal the token-level OOF numbers (pdms <= 1e-9, sub-scores <= 1e-4: the parent's float32 means); the choice ({s['chosen_step']}) == the rule "
       f"recomputed from the table, CIs bit-equal; no-margin argmax {s['no_margin_argmax_step']} reported; eligibility == (CI-lo > 0 and guard); a perturbed table fails the recheck")


def test_noise_world_cv5_vs_cv5m(tmp):
    """The card's fixture, through the TRAINER: labels permuted (a head fitted on noise).  Under `cv5` a +noise step WINS (its OOF PDMS exceeds step 0's); `cv5m` chooses 0.
    The deliberate-regression arm ds_no_margin changes that choice; the guard arm changes it on a trainer-level EP fixture."""
    w = mk(tmp, "nz", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    o5 = tmp / "nz_cv5"
    a = argv_ds(w, o5, "ds_ep", "red", extra=["--smoke-steps", "30"])
    a = TS._swap(a, "--save-steps", "0,15,30")
    a = TS._swap(TS._swap(a, "--select", "cv5"), "--permute-labels-seed", "7")
    r = TS.trainer(a)
    assert r.returncode == 0, r.stderr[-500:]
    s5 = load_sel(o5)
    om = smoke(w, tmp / "nz_cv5m", "ds_ep", 30, "0,15,30", perm=7)
    sm = load_sel(om)
    assert s5["chosen_step"] == 15 and "margin" not in s5 and "per_log" not in s5, ("the +noise step must win under cv5", s5["chosen_step"])     # the parent rule is fooled
    g = sm["margin"]["15"]["gain"]
    assert sm["no_margin_argmax_step"] == 15 and g["point"] > 0 and g["lo"] <= 0 and sm["chosen_step"] == 0 and sm["chosen_is_untrained"] and sm["eligible_steps"] == [], (g, sm["chosen_step"])
    assert abs(s5["per_step"]["15"]["oof_pdms"] - sm["per_step"]["15"]["oof_pdms"]) <= EPS, "cv5 and cv5m disagree on the same OOF numbers"
    oa = smoke(w, tmp / "nz_nomargin", "ds_ep", 30, "0,15,30", perm=7, arm_env="ds_no_margin")
    assert load_sel(oa)["chosen_step"] == 15, "ds_no_margin did not change the trainer's choice on the +noise fixture"
    ok(f"+noise (trainer, synthetic, labels permuted): cv5 CHOOSES step 15 (the parent rule is fooled); cv5m chooses 0 (point gain {g['point']:+.3f}, CI-lo {g['lo']:+.3f} <= 0); "
       "the same OOF numbers under both; ds_no_margin turns the choice back to 15 (CTRL-MARGIN, trainer-level)")


def test_cv5_unchanged(tmp):
    """CTRL-SH-UNCHANGED for the EP-HEAD rule: `--select cv5` is BIT-IDENTICAL to the tip trainer (heads, metrics.jsonl, cv_select.json bytes, head meta): probe pinned."""
    w = mk(tmp, "u5", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    out = tmp / "u5_out"
    a = TS.argv_for(w, out, extra=["--smoke-steps", "40"])
    a = TS._swap(TS._swap(a, "--rows", "2"), "--save-steps", "0,20,40") + ["--select", "cv5"]
    r = TS.trainer(a)
    assert r.returncode == 0, r.stderr[-500:]
    res = {"heads": {s: D.head_arrays_sha256(D.load_head(out / f"head_step{s:05d}.npz")) for s in (0, 20, 40)}, "metrics": hashlib.sha256((out / "metrics.jsonl").read_bytes()).hexdigest(),
           "cv": hashlib.sha256((out / "cv_select.json").read_bytes()).hexdigest(), "meta": D.load_head(out / "head_step00040.npz")["meta"].get("select")}
    got = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    assert got == CV5_PROBE, f"`--select cv5` moved: {got} != {CV5_PROBE}"
    s = json.loads((out / "cv_select.json").read_text())
    assert "per_log" not in s and "margin" not in s and "select" not in s
    ok("CTRL-SH-UNCHANGED (cv5): EP-HEAD `--select cv5` is bit-identical to the tip trainer: heads, metrics.jsonl, cv_select.json bytes and head meta equal the pinned probe; no cv5m key leaks in")


# ============================================================================================ the learning curve
def test_nested_sets():
    logs = [f"2021.01.01.00.00.{i:02d}_veh_{i}" for i in range(40)]
    parent = logs[3:11]
    sets = LC.nested_sets(logs, parent, [8, 16, 24, 40])
    assert sets[8] == sorted(parent) and sets[40] == sorted(logs)
    assert set(sets[8]) <= set(sets[16]) <= set(sets[24]) <= set(sets[40]) and [len(sets[n]) for n in (8, 16, 24, 40)] == [8, 16, 24, 40]
    others = sorted(set(logs) - set(parent), key=lambda lg: hashlib.sha256(lg.encode("utf-8")).hexdigest())
    assert sets[16] == sorted(parent + others[:8]) and sets[24] == sorted(parent + others[:16])
    assert LC.nested_sets(list(reversed(logs)), list(reversed(parent)), [8, 16]) == {8: sets[8], 16: sets[16]}, "the sets depend on arrival order"
    for bad in ([7], [41]):
        TS.expect_raises(lambda b=bad: LC.nested_sets(logs, parent, b), SystemExit, "outside")
    TS.expect_raises(lambda: LC.nested_sets(logs, parent + ["nope"], [9]), SystemExit, "not in the cache")
    ok("learning-curve sets: nested, the smallest is EXACTLY the parent's logs, each larger set adds the others in sha256(log name) order, independent of arrival order; out-of-range sizes and unknown parents are refused")


def _subset_index(w, logs, dst: Path):
    rows = list(csv.DictReader(open(w["sealed"], newline="", encoding="utf-8")))
    keep = [r for r in rows if r["log"] in set(logs)]
    with open(dst, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        wr.writeheader()
        wr.writerows(keep)
    return dst


LC_STATE: dict = {}                                                                     # the learning-curve test's argv + minted token, reused by test_lc_token_gate


def _lc(argv, env_extra=None):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8")
    env.pop(T.TOKEN_ENV, None)
    env.pop(T.ARM_ENV, None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(HERE / "safehead_lc.py"), *argv], cwd=str(HERE), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def test_learning_curve(tmp):
    """M6 on a synthetic cache + CTRL-PARENT: at n = |parent| the no-margin argmax and the OOF PDMS equal a banked `cv5` cv_select.json produced by the TRAINER on the parent's logs only;
    a banked table that is off by 1e-6 FAILS the control (exit 1).  AMENDMENT 1: the run is TOKEN-GATED -- it runs on a REAL minted PASS token for the lc's own trainer argv (s0 recipe:
    2,000 steps, the registered save steps), exactly as test_e2e does for the trainer."""
    w = mk(tmp, "lc", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    logs = sorted({r["log"] for r in csv.DictReader(open(w["sealed"], newline="", encoding="utf-8"))})
    parent = [logs[0], logs[2], logs[4], logs[6], logs[8]]                              # 5 logs incl. held-out ones (the trainer needs held-out tokens)
    assert any("ho" in lg for lg in parent)
    pl = tmp / "parent_logs.txt"
    pl.write_text("\n".join(parent) + "\n", encoding="utf-8")
    pidx = _subset_index(w, parent, tmp / "parent_index.csv")
    # the banked parent: the TRAINER's `cv5` over ONLY the parent's logs
    wp = dict(w, sealed=pidx)
    bank = tmp / "bank"
    a = TS.argv_for(wp, bank, extra=["--smoke-steps", "2000"])                          # the profile pins 2,000 steps + the save steps; --smoke-steps runs tokenless
    a = TS._swap(a, "--rows", "2") + ["--select", "cv5"]
    r = TS.trainer(a)
    assert r.returncode == 0, r.stderr[-600:]
    banked = bank / "cv_select.json"
    assert load_sel(bank)["n_logs"] == 5
    out = tmp / "lc_out"
    rest = argv_ds(w, out, "ds_ep")                                                    # the lc's TRAINER argv == what the token binds (s0 recipe, rows 2, cv5m)
    argv = ["--parent-logs", str(pl), "--sizes", "5,N", "--parent-cv-select", str(banked)] + rest
    env, tokpath = mint_env(tmp, w, "ds_ep", rest, "lc")
    LC_STATE.update(argv=argv, env=env, rest=rest, w=w, out=out, tokpath=tokpath)
    r = _lc(argv, env)
    assert r.returncode == 0 and "ZZSHLC_DONE" in r.stdout, (r.stdout[-600:], r.stderr[-800:])
    lc = json.loads((out / "lc.json").read_text(encoding="utf-8"))
    assert lc["token"]["profile"] == SHG.DS_FAMILIES["ds_ep"]["name"]["s0"] and lc["token"]["token_sha256"] == D.sha256_file(tokpath), lc["token"]   # the token is recorded
    assert sorted(lc["sets"]) == ["5", "9"] and [lc["sets"][k]["n_logs"] for k in ("5", "9")] == [5, 9]
    cp = lc["ctrl_parent"]
    assert cp["ok"] and cp["max_abs_diff"] <= EPS and cp["our_no_margin_argmax_step"] == cp["banked_chosen_step"], cp
    assert lc["sets"]["5"]["per_step"]["2000"]["oof_pdms"] == load_sel(bank)["per_step"]["2000"]["oof_pdms"]
    assert lc["summary"]["sizes"] == [5, 9] and len(lc["summary"]["best_step_gain_point"]) == 2
    for k in ("5", "9"):                                                         # every set carries the OOF gain + CI at every saved step, and its decision recomputes from its own table
        S = lc["sets"][k]
        assert all(set(S["margin"][s]["gain"]) == {"point", "lo", "hi"} for s in ("0", "250", "500", "1000", "2000"))
        assert CVM.recheck({"per_step": S["per_step"], "per_log": S["per_log"], "margin_rule": CVM.REGISTERED_RULE, "margin": S["margin"], "chosen_step": S["chosen_step"]})["match"]
    # the 9-log set equals what the trainer's own cv5m writes on the full cache (same recipe, same folds)
    full = smoke(w, tmp / "lc_full", "ds_ep", 2000, SHG.REGISTERED["--save-steps"])
    assert load_sel(full)["per_step"] == lc["sets"]["9"]["per_step"] and load_sel(full)["margin"] == lc["sets"]["9"]["margin"]
    # CTRL-PARENT must be able to FAIL
    bad = json.loads(banked.read_text())
    bad["per_step"]["2000"]["oof_pdms"] += 1e-6
    (tmp / "bank_bad.json").write_text(json.dumps(bad), encoding="utf-8")
    shutil.rmtree(out)                                                                  # the token binds the argv INCLUDING --out, so the second run reuses the SAME (emptied) out
    argv2 = list(argv)
    argv2[argv2.index("--parent-cv-select") + 1] = str(tmp / "bank_bad.json")           # --parent-cv-select is the lc's own flag: it is not in `rest`, the token still verifies
    r2 = _lc(argv2, env)
    assert r2.returncode == 1 and "ZZSHLC_FAIL CTRL-PARENT" in r2.stdout and json.loads((out / "lc.json").read_text())["ctrl_parent"]["ok"] is False, (r2.stdout[-400:], r2.stderr[-400:])
    shutil.rmtree(out)
    out.mkdir()                                                                         # an empty --out again, for test_lc_token_gate
    ok("learning curve (synthetic, 5/9 logs, on a REAL minted PASS token, 2,000 steps): sets nested; each set's OOF gain + CI at every step recomputes from its table; the n = |parent| set == the banked cv5 cv_select.json (argmax, OOF PDMS "
       "<= 1e-9, CTRL-PARENT PASS); the N-log set == the trainer's own cv5m; a banked table off by 1e-6 FAILS CTRL-PARENT (exit 1)")


# ============================================================================================ CTRL-INDEX
def test_lc_token_gate(tmp):
    """AMENDMENT 1: safehead_lc.py TRAINS heads, so it refuses without a launch-gate PASS token, exactly as the trainer does.  No token, a token minted for a DIFFERENT argv (--lr / --seed
    swapped) and an active regression arm are each refused BEFORE any fit (no lc.json)."""
    argv, env, out = LC_STATE["argv"], LC_STATE["env"], LC_STATE["out"]
    assert out.is_dir() and not any(out.iterdir()), "test_learning_curve must run first and leave --out empty"
    r = _lc(argv)                                                                       # no token at all
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr) and "no launch-gate token" in (r.stdout + r.stderr), (r.stdout[-300:], r.stderr[-400:])
    assert not (out / "lc.json").exists()
    r = _lc(TS._swap(argv, "--lr", "0.001"), env)                                       # a token for the s0 argv, a run with another lr
    assert r.returncode != 0 and "does not verify" in (r.stdout + r.stderr), (r.stdout[-300:], r.stderr[-400:])
    r = _lc(TS._swap(argv, "--seed", "7"), env)
    assert r.returncode != 0 and "REFUSED" in (r.stdout + r.stderr), (r.stdout[-300:], r.stderr[-400:])
    r = _lc(argv, dict(env, **{T.ARM_ENV: "ds_no_margin"}))                             # a regression arm never trains, token or not
    assert r.returncode != 0 and "regression arm is active" in (r.stdout + r.stderr), (r.stdout[-300:], r.stderr[-400:])
    assert not (out / "lc.json").exists() and not any(out.iterdir())
    ok("learning-curve token gate (AMENDMENT 1): with NO token, with a token minted for a different argv (--lr / --seed swapped) and under an active regression arm safehead_lc.py exits "
       "non-zero (REFUSED / does not verify) before any fit and writes no lc.json")


def test_ctrl_index(tmp):
    def write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["token", "log", "split", "file", "sha256"])
            wr.writerows(rows)
        return path
    par = [(f"p{i:04d}", f"logP{i // 4}", "train" if i < 8 else "heldout", f"logP{i // 4}.npz", f"{(i // 4) + 1:064x}") for i in range(12)]
    new = [(f"n{i:04d}", f"logN{i // 4}", "train", f"logN{i // 4}.npz", f"{100 + (i // 4):064x}") for i in range(12)]
    P_, I_ = write(tmp / "ci_parent.csv", par), write(tmp / "ci_185.csv", par + new)
    r = CVM.ctrl_index(I_, P_)
    assert r["ok"] and r["n_parent"] == 12 and r["n_new"] == 12 and r["n_logs"] == 6, r
    bads = {
        "a parent token's sha moved": (par[:5] + [(par[5][0], par[5][1], par[5][2], par[5][3], "f" * 64)] + par[6:] + new, "differ"),
        "a parent token dropped": (par[:5] + par[6:] + new, "missing"),
        "a parent token re-labelled to another log": (par[:5] + [(par[5][0], "logZ", par[5][2], par[5][3], par[5][4])] + par[6:] + new, "differ"),
        "a parent token's split moved": (par[:5] + [(par[5][0], par[5][1], "heldout", par[5][3], par[5][4])] + par[6:] + new, "differ"),
        "no new rows": (par, "adds no row"),
        "a log appears in the parent AND the new rows": (par + new[:11] + [("n9999", "logP1", "train", "logP1b.npz", "e" * 64)], "appear in the parent"),
        "a log carries two splits": (par + new[:10] + [("n9998", "logN0", "heldout", "logN0.npz", "100".zfill(64))], "more than one split"),
    }
    for k, (rows, needle) in bads.items():
        res = CVM.ctrl_index(write(tmp / "ci_bad.csv", rows), P_)
        assert not res["ok"] and any(needle in p for p in res["problems"]), (k, res)
    dup = write(tmp / "ci_dup.csv", par + new + [new[0]])
    TS.expect_raises(lambda: CVM.ctrl_index(dup, P_), D.ContractError, "duplicate tokens")
    nosha = tmp / "ci_nosha.csv"
    nosha.write_text("token,log,split,file\n" + "\n".join(",".join(r[:4]) for r in par + new) + "\n", encoding="utf-8")
    assert not CVM.ctrl_index(nosha, P_)["ok"]
    ok("CTRL-INDEX: the parent's 12 tokens bit-identical by sha + 12 new rows over 6 logs -> OK; 7 broken indices (a moved sha, a dropped / re-labelled / re-split parent token, no new rows, a log in "
       "both halves, a log in two splits) and a duplicate token and a missing sha256 column are each refused")


# ============================================================================================ profiles, refusals, gate
def test_profiles_registered():
    names = set(SHG.DS_PROFILE_NAMES)
    assert len(names) == 6 and names <= set(L.PROFILES) and not names & (set(SHG.PROFILE_NAME.values()) | set(SHG.NEXT_PROFILE_NAMES))
    assert len(L.PROFILES) == 55
    for n in names:
        p = L.PROFILES[n]
        assert p["runner"] == "safehead" and p["safehead_rules"] and set(p["checks"]) == set(SHG.CHECKS) and p["suite"][-1]["token"] == "ZZSHDS_PASS" and p["suite"][-2]["token"] == "ZZSHNEXT_PASS"
        assert p["required_flags"]["--seed"] in ("0", "1") and p["registered"]["--select"] == "cv5m" and p["registered"]["--steps"] == "2000" and p["registered"]["--save-steps"] == "0,250,500,1000,2000"
        assert p["registered"]["--rows"] == ("2" if "_ep_" in n else "0,1,3,5") and p["live_terms"] == (("qfl_ep",) if "_ep_" in n else SHG.LIVE_TERMS)
        assert ("--permute-labels-seed" in p["required_flags"]) == n.endswith("_red") and (("--permute-labels-seed" in p["forbidden_flags"]) == (not n.endswith("_red")))
        assert p["init_from_sha256"] == SHG.C1_SHA256 and p["card"].endswith("PREREG_DATASCALE.md") and "--mlp-hidden" not in p["registered"]
    assert L.PROFILES["refe_ds_ep_red"]["required_flags"]["--permute-labels-seed"] == str(SHG.RED_PERM_SEED)
    # NO existing profile fingerprint moves: the three SAFE-HEAD, the six L2-NEXT (pinned from the tip), and every older profile (the pre-SAFE-HEAD pin file)
    fp = lambda n: hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()[:16]        # noqa: E731
    pin = json.loads(TS.PRE_FP_FILE.read_text(encoding="utf-8"))
    moved = [n for n, f in pin.items() if fp(n) != f]
    assert not moved, f"pre-existing profiles moved: {moved}"
    nx = json.loads((HERE / "test_safehead_ds_nextfp.json").read_text(encoding="utf-8")) if (HERE / "test_safehead_ds_nextfp.json").is_file() else None
    assert nx is not None and set(nx) == set(SHG.NEXT_PROFILE_NAMES) | set(SHG.PROFILE_NAME.values()), "test_safehead_ds_nextfp.json (the 9 SAFE-HEAD / L2-NEXT fingerprints) is missing"
    moved = [n for n, f in nx.items() if fp(n) != f]
    assert not moved, f"SAFE-HEAD / L2-NEXT profiles moved: {moved}"
    ok(f"profiles: six NEW names (refe_ds_sh_*, refe_ds_ep_*), 55 in all; the 40 older, the 3 SAFE-HEAD and the 6 L2-NEXT profiles are fingerprint-identical (none moved)")


def test_refusals(tmp):
    w = mk(tmp, "ref", per_log=24)
    for fam in FAMS:
        for arm in SHG.ARMS:
            name = SHG.DS_FAMILIES[fam]["name"][arm]
            prof = L.PROFILES[name]
            a = argv_ds(w, tmp / f"ref_{name}", fam, arm)
            assert L.profile_refusals(prof, a) == [], (name, L.profile_refusals(prof, a))
            assert SHG.profile_name_for_argv(a) == name and SHG.family_of_argv(a) == fam, (SHG.profile_name_for_argv(a), name)
    ps, pe = L.PROFILES["refe_ds_sh_s0"], L.PROFILES["refe_ds_ep_s0"]
    bs, be = argv_ds(w, tmp / "ref_s", "ds_sh"), argv_ds(w, tmp / "ref_e", "ds_ep")
    bad = {"sh select cv5": TS._swap(bs, "--select", "cv5"), "sh select heldout": TS._swap(bs, "--select", "heldout"), "sh rows": TS._swap(bs, "--rows", "0,1,3"), "sh mlp": bs + ["--mlp-hidden", "64"],
           "sh seed": TS._swap(bs, "--seed", "2"), "sh smoke": bs + ["--smoke-steps", "3"], "sh perm on s0": bs + ["--permute-labels-seed", "5"], "sh lr": TS._swap(bs, "--lr", "0.001"),
           "sh steps": TS._swap(bs, "--steps", "4000"), "ep rows": TS._swap(be, "--rows", "0,1,3,5"), "ep select cv5": TS._swap(be, "--select", "cv5"), "ep gamma": TS._swap(be, "--gamma", "1.0")}
    for k, a in bad.items():
        assert L.profile_refusals(ps if k.startswith("sh") else pe, a), f"{k} was not refused"
    assert L.profile_refusals(pe, bs) and L.profile_refusals(ps, be), "the two levers' argvs cross-satisfy"
    plain = TS.argv_for(w, tmp / "ref_plain")
    assert L.profile_refusals(ps, plain) and L.profile_refusals(L.PROFILES["refe_safehead_s0"], bs)
    ep_old = TS._swap(be, "--select", "cv5")
    assert L.profile_refusals(L.PROFILES["refe_ephead_s0"], ep_old) == [] and SHG.profile_name_for_argv(ep_old) == "refe_ephead_s0", "the parent's cv5 profile no longer accepts its own argv"
    assert L.profile_refusals(L.PROFILES["refe_ephead_s0"], be), "the parent's EP-HEAD profile accepts a cv5m argv"
    orig = T.build_parser

    def bp():
        ap = orig()
        ap.add_argument("--new-lever", type=float, default=0.0)
        return ap
    T.build_parser = bp
    try:
        assert any("NO G-DVB entry" in r for r in L.profile_refusals(ps, bs)), "a new flag without a G-DVB entry was not refused"
    finally:
        T.build_parser = orig
    ok("refusals: the six template argvs are clean and route to their own profile; 12 moved / crossed levers are refused; the parent's cv5 profile still takes its own argv and refuses cv5m; "
       "a new flag without a G-DVB entry is refused")


def test_gate_clean(tmp, fam):
    w = mk(tmp, f"gc_{fam}", per_log=24)
    prof = prof_ds(w, fam)
    st = TS.gate(prof, argv_ds(w, tmp / f"gc_{fam}_launch", fam), only=[c for c in SHG.CHECKS if c != "G-SUITE"] + ["G-CLOCK"])
    assert all(st[c] == "PASS" for c in SHG.CHECKS if c != "G-SUITE") and st["G-CLOCK"] == "NA", st
    real = L.PROFILES[SHG.DS_FAMILIES[fam]["name"]["s0"]]
    st = TS.gate(real, argv_ds(w, tmp / f"gc_{fam}_real", fam), only=["G-DVB"])
    assert st["G-DVB"] == "FAIL", "the REAL profile accepted a checkpoint that is not C1"
    ok(f"[{fam}] gate: G-HYG G-DVB G-LIVE G-EVAL G-CKPT PASS on the synthetic cache (G-DVB also checks the built cv5m rule), G-CLOCK NA; the real profile's G-DVB refuses a non-C1 checkpoint")


def test_gate_mutations(tmp, fam):
    """CTRL-MARGIN at the gate: each cv5m mutation arm makes G-DVB RED (it reads the rule the trainer BUILT), on both levers; and a cv5m rule leaking into a cv5 / heldout arm is RED too."""
    w = mk(tmp, f"gm_{fam}", per_log=24)
    prof = prof_ds(w, fam)
    a = argv_ds(w, tmp / f"gm_{fam}_launch", fam)
    for arm in ("ds_no_margin", "ds_guard_no_ep"):
        st = TS.gate(prof, a, only=["G-DVB"], arm=arm)
        assert st["G-DVB"] == "FAIL", (fam, arm, st)
    st = TS.gate(prof, a, only=["G-DVB"])
    assert st["G-DVB"] == "PASS", st
    ok(f"[{fam}] mutations: ds_no_margin -> G-DVB RED, ds_guard_no_ep -> G-DVB RED; the clean arm PASSes")


def test_gate_all_arms(tmp, fam):
    w = mk(tmp, f"ga_{fam}", per_log=24)
    prof = prof_ds(w, fam)
    arms = {k: v for k, v in TS.ARM_CHECK.items()}
    arms.update({"sh_cv_token_folds": "G-DVB", "ds_no_margin": "G-DVB", "ds_guard_no_ep": "G-DVB"})
    for arm, chk in arms.items():
        st = TS.gate(prof, argv_ds(w, tmp / f"ga_{fam}_{arm}", fam), only=[chk], arm=arm)
        assert st[chk] == "FAIL", (fam, arm, st)
    ok(f"[{fam}] every deliberate-regression arm turns its check RED ({len(arms)} arms)")


def test_e2e(tmp, fam):
    w = mk(tmp, f"e2e_{fam}", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    prof = prof_ds(w, fam)
    name = SHG.DS_FAMILIES[fam]["name"]["s0"]
    prof["name"] = name
    keyf = tmp / f"keys_{fam}" / "gate.key"
    key = GT.load_key(keyf, create=True)
    out, run_out = tmp / f"e2e_{fam}_gate", tmp / f"e2e_{fam}_run"
    argv = argv_ds(w, run_out, fam)
    L.run_checks(prof, argv, HERE.parent, out, commit="ab" * 20, key=key)
    verdict, path, tok = L.finalize(prof, argv, HERE.parent, out, key, commit="ab" * 20)
    assert verdict == "PASS", tok["reasons"]
    okv, why, _ = GT.verify_token(path, argv, HERE.parent, key=key, profile=name, data_flags=SHG.DATA_FLAGS)
    assert okv, why
    okv, why, _ = GT.verify_token(path, TS._swap(argv, "--select", "cv5"), HERE.parent, key=key, profile=name, data_flags=SHG.DATA_FLAGS)
    assert not okv
    env = {"TANITAD_GATE_KEY_FILE": str(keyf), T.TOKEN_ENV: str(path)}
    r = TS.trainer(TS._swap(argv, "--lr", "0.001"), env)
    assert r.returncode != 0 and "does not verify" in (r.stdout + r.stderr), r.stderr[-300:]
    r = TS.trainer(argv, env)
    assert r.returncode == 0 and "ZZSH_TRAIN_DONE" in r.stdout and "cv5m chosen_step" in r.stdout, r.stderr[-600:]
    rj = json.loads((run_out / "run.json").read_text())
    s = load_sel(run_out)
    assert rj["token"]["profile"] == name and (run_out / "head_step02000.npz").is_file() and rj["chosen_step"] == s["chosen_step"] and CVM.recheck(s)["match"]
    ok(f"[{fam}] end to end: the gate mints PASS, verify MATCHes the argv and REFUSES a cv5 swap, the trainer refuses a mismatched token and trains (2,000 steps, cv5m) on a matching one")


def main() -> int:
    tmp = TS.mkdtemp()
    try:
        test_rule_on_tables()
        test_ctrl_margin_mutations()
        test_estimator_is_boot()
        test_nested_sets()
        test_ctrl_index(tmp)
        test_profiles_registered()
        test_cv5_unchanged(tmp)
        for fam in FAMS:
            test_table_and_recheck(tmp, fam)
        test_noise_world_cv5_vs_cv5m(tmp)
        test_learning_curve(tmp)
        test_lc_token_gate(tmp)
        test_refusals(tmp)
        for fam in FAMS:
            test_gate_clean(tmp, fam)
            test_gate_mutations(tmp, fam)
        if not FAST:
            for fam in FAMS:
                test_gate_all_arms(tmp, fam)
                test_e2e(tmp, fam)
    except Exception as e:                                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ZZSHDS_FAIL {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"ZZSHDS_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
