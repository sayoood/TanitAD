#!/usr/bin/env python3
"""L2-NEXT: tests of SH-MLP (PREREG_SHMLP.md) and EP-HEAD (PREREG_EPHEAD.md): the two trainer flags added to safehead_train.py (--mlp-hidden, --select cv5, EP as a trained row),
their G-DVB entries, their six launch-gate profiles and the reachability proofs.  SYNTHETIC data only, CPU only.  Every number printed is synthetic.

  python test_safehead_next.py           everything: + every deliberate-regression arm of the gate, + mint / verify / the trainer's own token refusal end to end
  python test_safehead_next.py --fast    G-SUITE's form: SAFE-HEAD unchanged (bit-identical), the MLP, CTRL-REACH + mutations on both arms, the CV rule, CTRL-RED, refusals, clean gates

Prints ZZSHNEXT_PASS <n> (or ZZSHNEXT_FAIL).  It reuses test_safehead.py's synthetic world and helpers.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

FAST = "--fast" in sys.argv
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.argv = [sys.argv[0]] + (["--fast"] if FAST else [])

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import safehead_data as D  # noqa: E402
import safehead_gate as SHG  # noqa: E402
import safehead_train as T  # noqa: E402
import test_safehead as TS  # noqa: E402

N = 0
IDPROBE = "d33a6ecaae8f2878e5e0333467ef4f3089a2e7af56dd966d1309c480c388ab9c"     # MEASURED on the LANDED tree e1eb6bf3 (the SAFE-HEAD trainer before this patch), see test_safehead_unchanged
NEXT = {"shmlp": {"extra": ["--mlp-hidden", "64"], "rows": "0,1,3,5"}, "ephead": {"extra": ["--rows", "2", "--select", "cv5"], "rows": "2"}}


def ok(name):
    global N
    N += 1
    print(f"ok {N:02d} {name}", flush=True)


def mk(tmp, name, **kw):
    return TS.make_world(tmp / name, **kw)


def argv_next(w, out, fam, arm="s0", extra=()):
    a = TS.argv_for(w, out, seed=SHG.ARMS[arm]["seed"], perm=SHG.ARMS[arm]["perm"])
    if fam == "ephead":
        a = TS._swap(a, "--rows", "2")
        a += ["--select", "cv5"]
    else:
        a += ["--mlp-hidden", "64"]
    return a + list(extra)


def prof_next(w, fam, arm="s0"):
    p = copy.deepcopy(L.PROFILES[SHG.FAMILIES[fam]["name"][arm]])
    p["init_from_sha256"] = D.sha256_file(w["ckpt"])
    p["suite"] = [{"raw": [sys.executable, "-c", "print('ZZTEST_PASS')"], "token": "ZZTEST_PASS"}]
    return p


def smoke(w, out, fam, steps, save, arm_env=None, extra=(), perm=None):
    a = argv_next(w, out, fam, "red" if perm else "s0", extra=["--smoke-steps", str(steps), *extra])
    a = TS._swap(a, "--save-steps", save)
    r = TS.trainer(a, {T.ARM_ENV: arm_env} if arm_env else None)
    assert r.returncode == 0, (r.stdout[-300:], r.stderr[-800:])
    return out


def metrics(out):
    return [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]


# ============================================================================================ tests
def test_safehead_unchanged(tmp):
    """The trainer refactor changes NOTHING for SAFE-HEAD: heads, metrics and the held-out rule of two 25-step runs are bit-identical to the landed tree's (probe pinned)."""
    w = mk(tmp, "idw", per_log=24, seed=3)
    res = {}
    for nm, extra in (("s0", []), ("red", ["--permute-labels-seed", "20261127"])):
        out = tmp / f"id_{nm}"
        a = TS.argv_for(w, out, extra=["--smoke-steps", "25"] + extra)
        a = TS._swap(a, "--save-steps", "0,10,25")
        r = TS.trainer(a)
        assert r.returncode == 0, r.stderr[-500:]
        hs = {s: D.head_arrays_sha256(D.load_head(out / f"head_step{s:05d}.npz")) for s in (0, 10, 25)}
        sel = json.loads((out / "heldout_select.json").read_text())
        res[nm] = {"heads": hs, "metrics_sha": hashlib.sha256((out / "metrics.jsonl").read_bytes()).hexdigest(), "sel_per_step": sel["per_step"], "chosen": sel["chosen_step"]}
        assert D.load_head(out / "head_step00025.npz")["meta"]["schema"] == D.SCHEMA_HEAD and "mlp_hidden" not in D.load_head(out / "head_step00025.npz")["meta"]
    got = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    assert got == IDPROBE, f"SAFE-HEAD output moved: {got} != {IDPROBE}"
    ok("SAFE-HEAD is bit-identical to the landed trainer: heads (W, b), metrics.jsonl and the held-out table of s0 and red equal the pinned probe; heads stay schema /1")


def test_mlp_arithmetic(tmp):
    rng = np.random.default_rng(0)
    M = T.HeadModel([0, 1, 3, 5], 64, rng.normal(0, .05, (6, 256)).astype(np.float32), rng.normal(0, 1, 6).astype(np.float32), 3e-4, None, 0)
    assert M.n_params == 66052 == 4 * (256 * 64 + 64 + 64 + 1), M.n_params
    assert M.last_layer_init_abs == 0.0
    W0, b0 = M.W0, M.b0
    S = torch.from_numpy(rng.normal(0, 1, (5, 64, 256)).astype(np.float32))
    base = D.direct_logits(S.numpy(), W0, b0)
    with torch.no_grad():
        z = M.logits(S).numpy()
    assert np.abs(z - base).max() <= 1e-5, "step 0 of the MLP head is not C1's head (the trainer's torch forward vs numpy: rounding only)"
    with torch.no_grad():
        assert all(float(M.layers[r][1].weight.abs().sum() + M.layers[r][1].bias.abs().sum()) == 0.0 for r in M.rows)
    st = M.state()
    h = T.head_dict(st, W0, b0, [0, 1, 3, 5], 64)
    assert np.array_equal(D.head_logits(base, S.numpy(), h), base) and np.array_equal(D.head_forward(S.numpy(), h), base)
    for r in (0, 1, 3, 5):                                                                   # perturb the last layer: the read must follow the trainer's forward
        M.layers[r][1].weight.data += 0.05
        M.layers[r][1].bias.data += 0.1
    with torch.no_grad():
        z = M.logits(S).numpy()
    h = T.head_dict(M.state(), W0, b0, [0, 1, 3, 5], 64)
    assert np.abs(D.head_forward(S.numpy(), h) - z).max() <= 1e-4
    assert np.abs(D.head_logits(base, S.numpy(), h) - base - (z - base)).max() <= 1e-4
    assert np.abs(z[..., [2, 4]] - base[..., [2, 4]]).max() <= 1e-5, "an untrained row (EP/C) moved"
    assert np.array_equal(D.head_logits(base, S.numpy(), h)[..., [2, 4]], base[..., [2, 4]]), "the READ moved an untrained row (EP/C)"
    p = tmp / "mlp_head.npz"
    D.save_head(p, W=h["W"], b=h["b"], W0=W0, b0=b0, meta={"rows": [0, 1, 3, 5], "mlp_hidden": 64}, mlp=h["mlp"])
    h2 = D.load_head(p)
    assert h2["meta"]["schema"] == D.SCHEMA_HEAD_MLP and D.head_arrays_sha256(h2) == D.head_arrays_sha256(h)
    h3 = dict(h)
    h3["mlp"] = {**h["mlp"], "mlp_b2": h["mlp"]["mlp_b2"] + 1.0}
    assert D.head_arrays_sha256(h3) != D.head_arrays_sha256(h), "the head hash is blind to the MLP"
    bad = tmp / "mlp_bad.npz"
    D.save_head(bad, W=h["W"] + 1e-3, b=h["b"], W0=W0, b0=b0, meta={"rows": [0, 1, 3, 5], "mlp_hidden": 64}, mlp=h["mlp"])
    TS.expect_raises(lambda: D.load_head(bad), D.ContractError, "LINEAR part must be C1's")
    ok("SH-MLP arithmetic: 66,052 parameters = 4 x (256*64+64+64+1); last layer exactly 0; step 0 = C1's logits (the read's form bit for bit, the trainer's torch forward within 1e-5); read == trainer forward (<= 1e-4); EP/C untouched; "
       "schema /2 round trip; the head hash sees the MLP; a head with a moved linear part is refused")


def test_cv_folds():
    logs = [f"log_{i:02d}" for i in (7, 3, 11, 1, 9, 0, 5)]
    f = D.cv_folds(logs, 5)
    assert f == {"log_00": 0, "log_01": 1, "log_03": 2, "log_05": 3, "log_07": 4, "log_09": 0, "log_11": 1}, f
    assert D.cv_folds(logs[::-1], 5) == f, "the folds depend on the order the logs arrive in"
    ok("cv folds: sorted log name, index mod 5, independent of arrival order")


def test_reach(tmp, fam):
    """CTRL-REACH on the new arm: non-zero gradient on every trained row, EXACTLY 0.0 on every other row, the other rows bit-identical, the trained rows moved; then the
    MUTATIONS (a trained row dropped from the optimiser; a frozen row entering the loss) turn G-LIVE RED."""
    w = mk(tmp, f"reach_{fam}", per_log=24)
    out = smoke(w, tmp / f"reach_{fam}_out", fam, 5, "0,5")
    M = metrics(out)
    rows = [int(x) for x in NEXT[fam]["rows"].split(",")]
    frozen = [r for r in range(6) if r not in rows]
    assert len(M) == 5 and all(m["grad_abs"][f"row{g}"] > 0 for m in M for g in rows), [m["grad_abs"] for m in M]
    assert all(v == 0.0 for m in M for v in m["frozen_grad_abs"].values()) and set(M[0]["frozen_grad_abs"]) == {f"row{r}" for r in frozen}
    h0, h5 = D.load_head(out / "head_step00000.npz"), D.load_head(out / "head_step00005.npz")
    assert np.array_equal(h0["W"], w["W0"]) and np.array_equal(h0["b"], w["b0"])
    assert all(SHG._row_moved(h0, h5, g, rows) > 0 for g in rows)
    assert np.array_equal(h5["W"][frozen], w["W0"][frozen]) and np.array_equal(h5["b"][frozen], w["b0"][frozen])
    if fam == "shmlp":
        assert np.array_equal(h5["W"], w["W0"]) and h5["meta"]["schema"] == D.SCHEMA_HEAD_MLP, "SH-MLP's linear part must stay C1's"
        assert [m["grad_abs"]["row0"] for m in M][0] > 0 and all(m["qfl_nc"] == m["qfl_nc"] for m in M)
    prof = prof_next(w, fam)
    a = argv_next(w, tmp / f"reach_{fam}_launch", fam)
    for arm, chk in (("sh_reach_row_dropped", "G-LIVE"), ("sh_reach_ep_in_loss", "G-LIVE")):
        st = TS.gate(prof, a, only=[chk], arm=arm)
        assert st[chk] == "FAIL", (fam, arm, st)
    ok(f"[{fam}] CTRL-REACH: non-zero gradient on rows {rows}; EXACTLY 0.0 on rows {frozen}; those rows bit-identical to C1; trained rows moved; a row dropped from the optimiser "
       f"and a frozen row in the loss turn G-LIVE RED")


def test_mutations(tmp, fam):
    w = mk(tmp, f"mut_{fam}", per_log=24)
    prof = prof_next(w, fam)
    a = argv_next(w, tmp / f"mut_{fam}_launch", fam)
    arms = {"shmlp": (("sh_mlp_off", "G-DVB"), ("sh_mlp_nonzero_init", "G-DVB"), ("sh_mlp_nonzero_init", "G-EVAL")),
            "ephead": (("sh_cv_token_folds", "G-DVB"),)}[fam]
    for arm, chk in arms:
        st = TS.gate(prof, a, only=[chk], arm=arm)
        assert st[chk] == "FAIL", (fam, arm, st)
    ok(f"[{fam}] mutations: " + ", ".join(f"{x}->{y} RED" for x, y in arms))


def test_cv_rule(tmp):
    """EP-HEAD's selection: the report's chosen step is the argmax of the out-of-fold PDMS (ties earliest); every log is in exactly one fold; step 0 reproduces C1's picks
    out of fold; the red arm (permuted EP labels) ends BELOW step 0 and chooses step 0."""
    w = mk(tmp, "cv", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    o_s, o_r = tmp / "cv_s0", tmp / "cv_red"
    smoke(w, o_s, "ephead", 400, "0,100,200,400")
    smoke(w, o_r, "ephead", 400, "0,100,200,400", perm=SHG.RED_PERM_SEED)
    s, r = json.loads((o_s / "cv_select.json").read_text()), json.loads((o_r / "cv_select.json").read_text())
    ps = s["per_step"]
    best = max(v["oof_pdms"] for v in ps.values())
    assert s["chosen_step"] == min(int(k) for k, v in ps.items() if v["oof_pdms"] == best), s["chosen_step"]
    assert s["n_logs"] == 9 and sorted(sum(s["folds"].values(), [])) == sorted(set(sum(s["folds"].values(), []))) and len(sum(s["folds"].values(), [])) == 9, s["folds"]
    assert all(len(v) >= 1 for v in s["folds"].values()) and s["n_tokens"] == 9 * 24
    assert ps["0"]["switch_rate_vs_c1"] == 0.0 and ps["0"]["oof_delta_vs_step0"] == 0.0
    assert all(abs(ps[k]["oof_pdms"] - ps["0"]["oof_pdms"] - ps[k]["oof_delta_vs_step0"]) < 1e-9 for k in ps)
    pr = r["per_step"]
    assert pr["400"]["oof_pdms"] < pr["0"]["oof_pdms"] and r["chosen_step"] == 0, (pr["0"]["oof_pdms"], pr["400"]["oof_pdms"], r["chosen_step"])
    assert json.loads((o_s / "run.json").read_text())["chosen_step"] == s["chosen_step"]
    assert not (o_s / "heldout_select.json").exists(), "cv5 must not also write the SAFE-HEAD held-out table"
    h = D.load_head(o_s / "head_step00400.npz")
    assert h["meta"]["rows"] == [2] and np.array_equal(np.delete(h["W"], 2, 0), np.delete(w["W0"], 2, 0)) and not np.array_equal(h["W"][2], w["W0"][2])
    # the rule itself on a constructed tie: earliest wins
    ties = {0: 91.0, 100: 91.5, 200: 91.5, 400: 90.0}
    assert min(k for k, v in ties.items() if v == max(ties.values())) == 100
    ok(f"EP-HEAD cv5 (synthetic): chosen step {s['chosen_step']} = argmax of the out-of-fold PDMS ({ps['0']['oof_pdms']:.3f} -> {best:.3f}), ties earliest; 9 logs each in one fold; "
       f"step 0 is C1 out of fold; RED ends below step 0 ({pr['0']['oof_pdms']:.3f} -> {pr['400']['oof_pdms']:.3f}) and chooses step 0; only EP's row moved")


def test_mlp_rule_and_red(tmp):
    w = mk(tmp, "mr", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    o_s, o_r = tmp / "mr_s0", tmp / "mr_red"
    smoke(w, o_s, "shmlp", 600, "0,250,500,600")
    smoke(w, o_r, "shmlp", 600, "0,250,500,600", perm=SHG.RED_PERM_SEED)
    s, rd = json.loads((o_s / "heldout_select.json").read_text()), json.loads((o_r / "heldout_select.json").read_text())
    p0, pr = s["per_step"]["0"], rd["per_step"]["600"]
    pc = s["per_step"][str(s["chosen_step"])]
    assert all(s["per_step"][k]["safety_miss_rate"] >= pc["safety_miss_rate"] for k in s["per_step"]), "the rule did not take the minimum"
    for k in ("auc_nc", "auc_dac", "auc_ttc"):
        assert pr[k] < p0[k], (k, p0[k], pr[k])
    assert pr["safety_miss_rate"] > p0["safety_miss_rate"], (p0["safety_miss_rate"], pr["safety_miss_rate"])
    assert rd["per_step"]["0"] == s["per_step"]["0"] and not (o_s / "cv_select.json").exists()
    ok(f"SH-MLP held-out rule (safety labels only) chooses step {s['chosen_step']} (miss {p0['safety_miss_rate']:.3f} -> {pc['safety_miss_rate']:.3f}); CTRL-RED: AUC nc/dac/ttc "
       f"fall and the miss rate rises ({pr['safety_miss_rate']:.3f}); both runs' step 0 identical")


def test_refusals(tmp):
    w = mk(tmp, "ref", per_log=24)
    for fam in NEXT:
        for arm in SHG.ARMS:
            name = SHG.FAMILIES[fam]["name"][arm]
            prof = L.PROFILES[name]
            a = argv_next(w, tmp / f"ref_{name}", fam, arm)
            assert L.profile_refusals(prof, a) == [], (name, L.profile_refusals(prof, a))
            assert SHG.profile_name_for_argv(a) == name, (SHG.profile_name_for_argv(a), name)
    pm, pe = L.PROFILES["refe_shmlp_s0"], L.PROFILES["refe_ephead_s0"]
    bm, be = argv_next(w, tmp / "ref_m", "shmlp"), argv_next(w, tmp / "ref_e", "ephead")
    bad = {"shmlp width": TS._swap(bm, "--mlp-hidden", "32"), "shmlp rows": TS._swap(bm, "--rows", "0,1,3"), "shmlp + select": bm + ["--select", "cv5"],
           "shmlp seed": TS._swap(bm, "--seed", "2"), "shmlp smoke": bm + ["--smoke-steps", "3"], "ephead rows": TS._swap(be, "--rows", "0,1,3,5"),
           "ephead select": TS._swap(be, "--select", "heldout"), "ephead + mlp": be + ["--mlp-hidden", "64"], "ephead lr": TS._swap(be, "--lr", "0.001"),
           "ephead perm on s0": be + ["--permute-labels-seed", "5"]}
    for k, a in bad.items():
        prof = pm if k.startswith("shmlp") else pe
        assert L.profile_refusals(prof, a), f"{k} was not refused"
    base_sh = TS.argv_for(w, tmp / "ref_sh")                                         # a SAFE-HEAD argv does not satisfy the new profiles, and vice versa
    assert L.profile_refusals(pm, base_sh) and L.profile_refusals(pe, base_sh) and L.profile_refusals(L.PROFILES["refe_safehead_s0"], bm)
    orig = T.build_parser

    def bp():
        ap = orig()
        ap.add_argument("--new-lever", type=float, default=0.0)
        return ap
    T.build_parser = bp
    try:
        assert any("NO G-DVB entry" in r for r in L.profile_refusals(pm, bm)), "a new flag without a G-DVB entry was not refused"
    finally:
        T.build_parser = orig
    # the real profiles pin C1's sha256 and the trainer's token
    assert all(L.PROFILES[n]["init_from_sha256"] == SHG.C1_SHA256 for n in SHG.NEXT_PROFILE_NAMES)
    ok("refusals: the six template argvs are clean and route to their own profile; 10 moved / extra / crossed levers are refused; a new flag without a G-DVB entry is refused")


def test_profiles_registered():
    import safehead_gate
    names = set(SHG.NEXT_PROFILE_NAMES)
    assert len(names) == 6 and names <= set(L.PROFILES) and not names & set(SHG.PROFILE_NAME.values())
    for n in names:
        p = L.PROFILES[n]
        assert p["runner"] == "safehead" and p["safehead_rules"] and set(p["checks"]) == set(SHG.CHECKS) and p["suite"][-1]["token"] == "ZZSHNEXT_PASS"
        assert p["required_flags"]["--seed"] in ("0", "1") and p["registered"]["--steps"] == "2000" and p["registered"]["--save-steps"] == "0,250,500,1000,2000"
        assert ("--permute-labels-seed" in p["required_flags"]) == n.endswith("_red") and (("--permute-labels-seed" in p["forbidden_flags"]) == (not n.endswith("_red")))
    assert L.PROFILES["refe_shmlp_s0"]["registered"]["--mlp-hidden"] == "64" and L.PROFILES["refe_ephead_s0"]["registered"]["--select"] == "cv5"
    assert L.PROFILES["refe_ephead_red"]["required_flags"]["--permute-labels-seed"] == str(SHG.RED_PERM_SEED)
    fps = {n: hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()[:16] for n in SHG.PROFILE_NAME.values()}
    assert fps == SH_FP, f"a SAFE-HEAD profile moved: {fps}"
    pin = json.loads(TS.PRE_FP_FILE.read_text(encoding="utf-8"))
    moved = [n for n, fp in pin.items() if hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()[:16] != fp]
    assert not moved, f"pre-existing profiles moved: {moved}"
    ok("profiles: six NEW names registered (refe_shmlp_*, refe_ephead_*); the 3 SAFE-HEAD and all 40 older profiles are fingerprint-identical")


SH_FP = {"refe_safehead_s0": "2e3c21ade449ac67", "refe_safehead_s1": "a1ae4d012c40cd53", "refe_safehead_red": "465160c03b54ab12"}     # MEASURED on the landed tree e1eb6bf3


def test_gate_clean(tmp, fam):
    w = mk(tmp, f"gc_{fam}", per_log=24)
    prof = prof_next(w, fam)
    st = TS.gate(prof, argv_next(w, tmp / f"gc_{fam}_launch", fam), only=[c for c in SHG.CHECKS if c != "G-SUITE"] + ["G-CLOCK"])
    assert all(st[c] == "PASS" for c in SHG.CHECKS if c != "G-SUITE") and st["G-CLOCK"] == "NA", st
    real = L.PROFILES[SHG.FAMILIES[fam]["name"]["s0"]]
    st = TS.gate(real, argv_next(w, tmp / f"gc_{fam}_real", fam), only=["G-DVB"])
    assert st["G-DVB"] == "FAIL", "the REAL profile accepted a checkpoint that is not C1"
    ok(f"[{fam}] gate: G-HYG G-DVB G-LIVE G-EVAL G-CKPT PASS on the synthetic cache, G-CLOCK NA; the real profile's G-DVB refuses a non-C1 checkpoint")


def test_gate_all_arms(tmp, fam):
    w = mk(tmp, f"ga_{fam}", per_log=24)
    prof = prof_next(w, fam)
    arms = dict(TS.ARM_CHECK)
    if fam == "shmlp":
        arms.update({"sh_mlp_off": "G-DVB", "sh_mlp_nonzero_init": "G-EVAL"})
    else:
        arms.update({"sh_cv_token_folds": "G-DVB"})
    for arm, chk in arms.items():
        st = TS.gate(prof, argv_next(w, tmp / f"ga_{fam}_{arm}", fam), only=[chk], arm=arm)
        assert st[chk] == "FAIL", (fam, arm, st)
    ok(f"[{fam}] every deliberate-regression arm turns its check RED ({len(arms)} arms)")


def test_e2e(tmp, fam):
    w = mk(tmp, f"e2e_{fam}", n_tr_logs=6, n_ho_logs=3, per_log=24, seed=5)
    prof = prof_next(w, fam)
    name = SHG.FAMILIES[fam]["name"]["s0"]
    prof["name"] = name
    keyf = tmp / f"keys_{fam}" / "gate.key"
    key = GT.load_key(keyf, create=True)
    out, run_out = tmp / f"e2e_{fam}_gate", tmp / f"e2e_{fam}_run"
    argv = argv_next(w, run_out, fam)
    L.run_checks(prof, argv, HERE.parent, out, commit="ab" * 20, key=key)
    verdict, path, tok = L.finalize(prof, argv, HERE.parent, out, key, commit="ab" * 20)
    assert verdict == "PASS", tok["reasons"]
    okv, why, _ = GT.verify_token(path, argv, HERE.parent, key=key, profile=name, data_flags=SHG.DATA_FLAGS)
    assert okv, why
    okv, why, _ = GT.verify_token(path, TS._swap(argv, "--lr", "0.001"), HERE.parent, key=key, profile=name, data_flags=SHG.DATA_FLAGS)
    assert not okv
    env = {"TANITAD_GATE_KEY_FILE": str(keyf), T.TOKEN_ENV: str(path)}
    r = TS.trainer(TS._swap(argv, "--lr", "0.001"), env)
    assert r.returncode != 0 and "does not verify" in (r.stdout + r.stderr), r.stderr[-300:]
    r = TS.trainer(argv, env)
    assert r.returncode == 0 and "ZZSH_TRAIN_DONE" in r.stdout, r.stderr[-600:]
    rj = json.loads((run_out / "run.json").read_text())
    assert rj["token"]["profile"] == name and (run_out / "head_step02000.npz").is_file()
    assert D.load_head(run_out / "head_step02000.npz")["meta"]["smoke"] is False
    ok(f"[{fam}] end to end: the gate mints PASS, verify MATCHes the argv and REFUSES a moved lever, the trainer refuses a mismatched token and trains (2,000 steps) on a matching one")


def main() -> int:
    tmp = TS.mkdtemp()
    try:
        test_safehead_unchanged(tmp)
        test_mlp_arithmetic(tmp)
        test_cv_folds()
        test_profiles_registered()
        for fam in NEXT:
            test_reach(tmp, fam)
            test_mutations(tmp, fam)
        test_cv_rule(tmp)
        test_mlp_rule_and_red(tmp)
        test_refusals(tmp)
        for fam in NEXT:
            test_gate_clean(tmp, fam)
        if not FAST:
            for fam in NEXT:
                test_gate_all_arms(tmp, fam)
                test_e2e(tmp, fam)
    except Exception as e:                                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ZZSHNEXT_FAIL {type(e).__name__}: {str(e)[:300]}")
        return 1
    print(f"ZZSHNEXT_PASS {N}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
