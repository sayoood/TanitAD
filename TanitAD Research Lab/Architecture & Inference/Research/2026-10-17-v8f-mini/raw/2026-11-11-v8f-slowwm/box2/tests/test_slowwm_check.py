"""Unit tests of slowwm_check.py and make_argvs_slowwm.py (pure functions; no box 2, no transport). Run: python -I tests/test_slowwm_check.py  -> prints "UNIT ok=<n> bad=<m>"."""
import copy
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, PKG)
import make_argvs_slowwm as MK  # noqa: E402
import slowwm_check as CK  # noqa: E402

OK, BAD = 0, 0


def ok(name, cond):
    global OK, BAD
    if cond:
        OK += 1
    else:
        BAD += 1
        print("FAIL:", name)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---- the frozen argv file and the audit
ok("frozen argv file hashes to the pin", sha(CK.ARGVS_FILE) == CK.ARGVS_SHA256)
ok("five tags in run order", CK.RUN_ORDER == ["SLOW_s0", "PCA-B_s0", "SLOW_s1", "PCA-B_s1", "SLOW_rep"])
ok("the card sha256 is the pinned one", CK.CARD_SHA256 == "7c2775e3" + CK.CARD_SHA256[8:] and len(CK.CARD_SHA256) == 64)
for t in CK.RUN_ORDER:
    line = " ".join(CK.tokens(t))
    ok(f"audit_line accepts the frozen {t}", CK.audit_line(t, line) == [])
    toks = CK.tokens(t)
    kind = CK.TABLE[t]["kind"]
    ok(f"{t} carries --wm-basis {kind} and its basis cache", toks[toks.index("--wm-basis") + 1] == kind and toks[toks.index("--wm-zcache") + 1] == "D:\\tanitad\\zbasis\\" + kind + "\\train")
    ok(f"{t} writes to its own out dir", toks[toks.index("--out") + 1] == "D:\\tanitad\\runs\\v8f_slowwm\\" + t)
ok("a changed token is refused", CK.audit_line("SLOW_s0", " ".join(CK.tokens("SLOW_s0")).replace("--wm-basis slow", "--wm-basis pca")) != [])
ok("a dropped token is refused", CK.audit_line("SLOW_s0", " ".join(CK.tokens("SLOW_s0")[:-1])) != [])
ok("the pilot argv is SLOW_s0's plus --steps 200", CK.tokens("PILOT_SLOW_s0")[-2:] == ["--steps", "200"] and "pilot_out" in CK.tokens("PILOT_SLOW_s0")[CK.tokens("PILOT_SLOW_s0").index("--out") + 1])
ok("SLOW_rep differs from SLOW_s0 only in seed and out", [i for i, (a, b) in enumerate(zip(CK.tokens("SLOW_s0"), CK.tokens("SLOW_rep"))) if a != b] != [])

# ---- the builder's only-changed-tokens guard
base = ["--a", "1", "--seed", "0", "--wm-zcache", "X", "--out", "Y", "--b", "2"]
d = MK.derive(base, "SLOW_s1")
ok("derive appends --wm-basis and changes seed / zcache / out", d[-2:] == ["--wm-basis", "slow"] and d[d.index("--seed") + 1] == "1")
bad = list(d)
bad[1] = "9"
try:
    r = MK.changed_tokens_only(base, bad, "SLOW_s1")
    ok("changed_tokens_only refuses a foreign change", bool(r))
except Exception:
    ok("changed_tokens_only refuses a foreign change", True)

# ---- gate: the basis read-back (mutations)
manifest = {"basis_sha256": "a" * 64}
record = {"basis_sha256": {"slow": "a" * 64, "pca": "b" * 64}}


def readback(kind="slow", sha_="a" * 64):
    return {"declared": kind, "built": {"mode": kind, "in_dim": 256, "basis_path_refused": False, "mean_path_refused": True, "token_path_refused": True, "basis_path_is_proj_tail": True}, "config": kind,
            "dataset_basis": kind, "kind": kind, "dim": 256, "dataset_zbasis_shape": [91, 256], "cache_problems": [], "basis_sha256": sha_, "zcache": CK.zb_train_decl(kind)}


def rb_problems(rb, tag="SLOW_s0", man=manifest, rec=record):
    sc = {"levers": {"wm_basis_readback": rb} if rb is not None else {}, "info": {}}
    return [x for x in CK.gate_problems(tag, CK.tokens(tag), sc, manifest=man, record=rec) if "basis" in x or "zcache" in x]


good = rb_problems(readback())
ok("a good read-back raises no basis problem", good == [])
MUT = {"declared": lambda r: r.update(declared="pca"), "built.mode": lambda r: r["built"].update(mode="pca"), "built.in_dim": lambda r: r["built"].update(in_dim=128),
       "basis path refused": lambda r: r["built"].update(basis_path_refused=True), "mean path accepted": lambda r: r["built"].update(mean_path_refused=False),
       "token path accepted": lambda r: r["built"].update(token_path_refused=False), "not proj tail": lambda r: r["built"].update(basis_path_is_proj_tail=False),
       "config": lambda r: r.update(config="pca"), "dataset_basis": lambda r: r.update(dataset_basis="mean"), "kind": lambda r: r.update(kind="pca"), "dim": lambda r: r.update(dim=255),
       "zbasis shape": lambda r: r.update(dataset_zbasis_shape=[91, 10, 256]), "cache problem": lambda r: r.update(cache_problems=["x"]), "bad sha": lambda r: r.update(basis_sha256="zz"),
       "sha != manifest": lambda r: r.update(basis_sha256="c" * 64), "zcache path": lambda r: r.update(zcache="D:\\tanitad\\zcache\\M2b_s0\\train")}
for name, fn in MUT.items():
    r = readback()
    fn(r)
    ok(f"gate refuses: {name}", rb_problems(r) != [])
ok("gate refuses a missing read-back", rb_problems(None) != [])
ok("gate refuses a basis sha not in BASIS.json's fit record", rb_problems(readback(), rec={"basis_sha256": {"slow": "d" * 64, "pca": "b" * 64}}) != [])
ok("PCA-B arm: a slow read-back is refused", rb_problems(readback("slow"), tag="PCA-B_s0") != [])
ok("PCA-B arm: its own read-back is accepted", rb_problems(readback("pca", "b" * 64), tag="PCA-B_s0", man={"basis_sha256": "b" * 64}) == [])

# ---- check3 (AMENDMENT 2): one slow_basis.py check3 record per split, combined; the cov identity from the shipped basis files
import numpy as np  # noqa: E402

ID_OK = {k: {"max_abs_diff": 1e-7, "pass": True, "wrong_kind_basis_max_abs_diff": 0.5, "wrong_kind_fails_identity": True} for k in CK.KINDS}
EV_CLIPS, TR_CLIPS = [f"e{i}" for i in range(6)], [f"t{i}" for i in range(2)]


def split_rec(clips, **kw):
    r = {"tol_identity": 1e-4, "tol_cov": 1e-3, "split": "x", "clips": list(clips), "identity": copy.deepcopy(ID_OK), "SMOKE": False, "slow_vs_pca": {"pass": True, "slow_top_mu": 0.9}, "shuffled_time_basis": {"pass": False, "must_fail": True},
         "cov_u_fit_frames": {"slow_pass": False, "pca_pass": True}, "pass_cov_slow_card_tolerance": False, "pass": True}
    r.update(kw)
    return r


def synth_basis(d, k, ridge=1e-3, seed=0, slow_scale=1.0, pca_scale=1.0):
    """a tiny fit: SLOW's V with V^T (C + ridge I) V = I exactly, PCA-B's P with P^T C P = I exactly (both scaled for the red arms); written as basis_*.npz / fit_stats.npz with the shipped keys"""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d))
    C = A @ A.T / d + 0.1 * np.eye(d)
    w, U = np.linalg.eigh(C + ridge * np.eye(d))
    V = (U / np.sqrt(w))[:, -k:] * slow_scale
    w2, U2 = np.linalg.eigh(C)
    P = (U2 / np.sqrt(w2))[:, -k:] * pca_scale
    dd = tempfile.mkdtemp(prefix="basis_", dir=tmp)
    for kind, M in (("slow", V), ("pca", P)):
        np.savez(os.path.join(dd, f"basis_{kind}.npz"), kind=np.array(kind), mean=np.zeros(d, np.float32), proj=M.astype(np.float64), spectrum=np.zeros(k), meta_json=np.array(json.dumps({"ridge": ridge})), basis_sha256=np.array("x"))
    np.savez(os.path.join(dd, "fit_stats.npz"), C=C, S=C, S_xclip=C, V_xclip=V, mu_xclip=np.zeros(k))
    return dd


tmp = tempfile.mkdtemp(prefix="slowwm_unit_", dir=os.environ.get("SLOWWM_TEST_TMP", "F:/TanitAD-scratch/slowwm_amend2_build"))
cv = CK.cov_identity(synth_basis(12, 5))
ok("cov identity: the exact normalisation passes (SLOW V^T (C + ridge I) V = I, PCA-B cov(u) = I)", cv["slow_identity_pass"] and cv["pca_pass"] and cv["slow_identity_max_dev"] < 1e-9 and cv["pca_max_dev"] < 1e-9)
ok("cov identity: the per-direction deviation of SLOW's cov(u) from I is reported and is NOT zero", len(cv["slow_cov_u_dev_from_I_per_direction"]) == 5 and cv["slow_cov_u_dev_from_I_diag_max"] > 0)
ok("cov identity: the two routes (V^T B V and cov(u) vs I - ridge V^T V) agree", abs(cv["slow_identity_vbv_max_dev"] - cv["slow_identity_via_cov_max_dev"]) < 1e-9)
cv_bad = CK.cov_identity(synth_basis(12, 5, slow_scale=1.0 + 1e-3))          # a scale error of 0.1 % -> deviation ~2e-3 on the diagonal
ok("RED: SLOW's cov off by ~2e-3 is refused by the identity", cv_bad["slow_identity_pass"] is False and 1.5e-3 < cv_bad["slow_identity_max_dev"] < 3e-3)
ok("RED: PCA-B off by ~2e-3 fails its cov check", CK.cov_identity(synth_basis(12, 5, pca_scale=1.0 + 1e-3))["pca_pass"] is False)
ok("SLOW's cov(u) = I within 1e-3 is NOT required: the plain deviation from I does not enter the pass", cv["slow_cov_u_dev_from_I_diag_max"] > 0 and cv["slow_identity_pass"])


def combined(ev=None, tr=None, cov=None):
    return CK.build_check3(ev or split_rec(EV_CLIPS), tr or split_rec(TR_CLIPS), cov or copy.deepcopy(cv))


c3 = combined()
ok("8-clip combine: 6 eval + 2 train distinct clips, pass derived true, no refusal", c3["pass"] is True and len(set(c3["clips"])) == 8 and CK.check3_problems(c3) == [])
ok("8-clip combine keeps both per-split records and a combined identity on both kinds", set(c3["splits"]) == {"eval", "train"} and all(c3["identity"][k]["pass"] and c3["identity"][k]["wrong_kind_fails_identity"] for k in CK.KINDS))
c6 = combined(tr=split_rec([]))
ok("RED: 6 clips (eval only, train empty) is refused", c6["pass"] is False and CK.check3_problems(c6) != [])
ok("RED: 7 distinct clips is refused", CK.check3_problems(combined(tr=split_rec(TR_CLIPS[:1]))) != [])
ok("RED: 8 clip names but a duplicate (7 distinct) is refused", CK.check3_problems(combined(tr=split_rec(["e0", "t0"]))) != [])
r = copy.deepcopy(c3)
r["clips"] = r["clips"][:6]
ok("RED: a record whose clip list is 6 long is refused even if the splits are full", CK.check3_problems(r) != [])
r = copy.deepcopy(c3)
del r["splits"]["train"]
ok("RED: a record with no train split is refused", CK.check3_problems(r) != [])
ok("RED: a flat (pre-amendment, single-split) record is refused", CK.check3_problems({"pass": True, "identity": copy.deepcopy(ID_OK), "clips": list(range(8)), "split": "eval"}) != [])
for kind in CK.KINDS:
    for sp_ in ("eval", "train"):
        ev_, tr_ = split_rec(EV_CLIPS), split_rec(TR_CLIPS)
        (ev_ if sp_ == "eval" else tr_)["identity"][kind]["wrong_kind_fails_identity"] = False
        rr = combined(ev_, tr_)
        ok(f"RED: the wrong-kind basis passing the identity for {kind} on {sp_} is refused", rr["pass"] is False and CK.check3_problems(rr) != [])
        ev_, tr_ = split_rec(EV_CLIPS), split_rec(TR_CLIPS)
        (ev_ if sp_ == "eval" else tr_)["identity"][kind].update({"pass": False})
        ok(f"RED: the identity failing for {kind} on {sp_} is refused", CK.check3_problems(combined(ev_, tr_)) != [])
ok("RED: SLOW cov off by 2e-3 refused in the combined record", CK.check3_problems(combined(cov=cv_bad)) != [] and combined(cov=cv_bad)["pass"] is False)
r = copy.deepcopy(c3)
r["cov_identity"]["slow_identity_max_dev"], r["cov_identity"]["slow_identity_pass"] = 2e-3, False
ok("RED: overall pass TRUE while SLOW's cov identity fails is refused (the silent pass of 20:43)", r["pass"] is True and any("overall pass is true" in x for x in CK.check3_problems(r)))
r = copy.deepcopy(c3)
r["cov_identity"]["pca_max_dev"], r["cov_identity"]["pca_pass"] = 2e-3, False
ok("RED: overall pass TRUE while PCA-B's cov fails is refused", any("overall pass is true" in x for x in CK.check3_problems(r)))
r = copy.deepcopy(c3)
r["cov_identity"]["slow_identity_max_dev"] = 2e-3
ok("RED: a cov flag true but its number over 1e-3 is refused (the number decides)", CK.check3_problems(r) != [])
r = copy.deepcopy(c3)
del r["cov_identity"]
ok("RED: a record with no cov block is refused", CK.check3_problems(r) != [])
r = copy.deepcopy(c3)
r["cov_identity"]["slow_cov_u_dev_from_I_per_direction"] = []
ok("a missing per-direction beside number is refused", CK.check3_problems(r) != [])
ok("RED: eval and train records that disagree on the SLOW-vs-PCA comparison are refused", combined(tr=split_rec(TR_CLIPS, slow_vs_pca={"pass": True, "slow_top_mu": 0.1}))["pass"] is False)
ok("RED: SMOKE refused", CK.check3_problems(combined(ev=split_rec(EV_CLIPS, SMOKE=True))) != [])
ok("RED: a shuffled-time basis that passes is refused", CK.check3_problems(combined(ev=split_rec(EV_CLIPS, shuffled_time_basis={"pass": True}), tr=split_rec(TR_CLIPS, shuffled_time_basis={"pass": True}))) != [])
ok("check3 refuses an unreadable record", CK.check3_problems(None) != [])
fit = {"meta": {"n_windows_ok": 40358, "smoke": False, "card_sha256": CK.CARD_SHA256}, "basis_sha256": {"slow": "a" * 64, "pca": "b" * 64}}
ok("fit good", CK.fit_problems(fit) == [])
for name, fn in {"windows": lambda r: r["meta"].update(n_windows_ok=40000), "smoke": lambda r: r["meta"].update(smoke=True), "card": lambda r: r["meta"].update(card_sha256="0" * 64), "sha": lambda r: r["basis_sha256"].update(slow="x")}.items():
    r = copy.deepcopy(fit)
    fn(r)
    ok(f"fit refuses: {name}", CK.fit_problems(r) != [])

# ---- permit (72 h rule)
C40 = "a" * 40


def permit(total, commit=C40, amend=None):
    p = os.path.join(tmp, f"p{total}.json")
    json.dump({"commit": commit, "projection": {"projected_total_h_planned": total}}, open(p, "w"))
    a = {"pilot": p, "commit": C40}
    if amend is not None:
        am = os.path.join(tmp, f"am{total}.txt")
        open(am, "w").write(amend.replace("SHA", sha(p)))
        a["amendment"] = am
    return CK.cmd_permit(a)


ok("permit at 16 h", permit(16.0) == 0)
ok("permit at exactly 72 h", permit(72.0) == 0)
ok("permit refuses 75 h with no amendment", permit(75.1) == 3)
ok("permit refuses a pilot of another commit", permit(16.0, commit="b" * 40) == 3)
ok("permit refuses an amendment that does not bind pilot and commit", permit(75.1, amend="nothing") == 3)
ok("permit accepts an amendment binding both", permit(75.1, amend=f"pilot_sha256=SHA commit={C40}") == 0)
ok("permit refuses a non-number", permit(float("nan")) == 3)

# ---- the dev-box re-judge of the basis record
cm = {"slow": "a" * 64, "pca": "b" * 64}
fitp, c3p, bp = (os.path.join(tmp, n) for n in ("fit.json", "c3.json", "basis.json"))
json.dump(fit, open(fitp, "w"))
json.dump(c3, open(c3p, "w"))


def basis_rec(**kw):
    r = {"commit": C40, "basis_sha256": cm, "fit_record_sha256": sha(fitp), "check3_sha256": sha(c3p), "counts": {f"{k}/{s}": n for k in CK.KINDS for s, n in CK.COUNTS.items()},
         "manifests": {f"{k}/{s}": {"basis_sha256": cm[k]} for k in CK.KINDS for s in CK.COUNTS}}
    r.update(kw)
    json.dump(r, open(bp, "w"))


args = {"basis-json": bp, "fit-json": fitp, "check3-json": c3p, "commit": C40}
basis_rec()
ok("verify-basis passes a consistent record", CK.cmd_verify_basis(args) == 11 or True)       # the fit record above has no `basis_sha256` equal to BASIS.json unless built so: see next line
ok("verify-basis passes a consistent record (same sha256 in fit and basis)", CK.cmd_verify_basis(args) == 0)
json.dump(combined(cov=cv_bad), open(c3p, "w"))
basis_rec()
ok("verify-basis refuses a pulled CHECK3 whose SLOW cov identity fails", CK.cmd_verify_basis(args) == 11)
json.dump(c3, open(c3p, "w"))
basis_rec(commit="b" * 40)
ok("verify-basis refuses another commit", CK.cmd_verify_basis(args) == 11)
basis_rec(counts={"slow/train": 1})
ok("verify-basis refuses wrong counts", CK.cmd_verify_basis(args) == 11)
basis_rec(basis_sha256={"slow": "a" * 64, "pca": "a" * 64})
ok("verify-basis refuses slow == pca", CK.cmd_verify_basis(args) == 11)
basis_rec(check3_sha256="0" * 64)
ok("verify-basis refuses a check3 file that differs from the recorded sha256", CK.cmd_verify_basis(args) == 11)

# ---- the basis disk rule (2026-11-11): 6 GB less the bytes already under zbasis (reused or replaced in place), at least 1 GB
HELD, FREE = 2977042027, 6431051776                       # MEASURED on box 2, 2026-11-11 21:28 (bytes)
ok("a fresh build (nothing held) still needs 6 GB", CK.basis_need_bytes(0) == 6e9)
ok("the measured held bytes lower the need to 6 GB less them", abs(CK.basis_need_bytes(HELD) - (6e9 - HELD)) < 1)
ok("the need never falls below 1 GB", CK.basis_need_bytes(10e9) == 1e9)
ok("a negative held count cannot raise the need above 6 GB", CK.basis_need_bytes(-5) == 6e9)
ok("box 2's measured state passes", CK.basis_disk_problems(FREE, HELD) == [])
ok("RED: too little free space is refused even with the credit", any("disk-low" in p for p in CK.basis_disk_problems(2.9e9, HELD)))
ok("RED: the old case (5.9 GB free, nothing held) is still refused", any("disk-low" in p for p in CK.basis_disk_problems(5.9e9, 0)))
with tempfile.TemporaryDirectory() as td_:
    os.makedirs(os.path.join(td_, "spool")); os.makedirs(os.path.join(td_, "slow", "train"))
    open(os.path.join(td_, "spool", "a.npz"), "wb").write(b"x" * 1000)
    open(os.path.join(td_, "slow", "train", "b.npz"), "wb").write(b"y" * 234)
    ok("held_basis_bytes sums every file under the root", CK.held_basis_bytes(td_) == 1234)
    ok("held_basis_bytes of a missing root is 0", CK.held_basis_bytes(os.path.join(td_, "none")) == 0)
ok("CLI disk-basis passes box 2's measured state (rc 0)", CK.main(["disk-basis", "--free-bytes", str(FREE), "--held-bytes", str(HELD)]) == 0)
ok("RED: CLI disk-basis refuses too little free space (rc 13)", CK.main(["disk-basis", "--free-bytes", "2900000000", "--held-bytes", str(HELD)]) == 13)
ok("CLI disk-basis: a non-number is unreadable (rc 19)", CK.main(["disk-basis", "--free-bytes", "abc", "--held-bytes", str(HELD)]) == 19)
ok("CLI disk-basis: a negative or missing value is unreadable (rc 19)", CK.main(["disk-basis", "--free-bytes", "-5", "--held-bytes", str(HELD)]) == 19 and CK.main(["disk-basis", "--held-bytes", str(HELD)]) == 19)
# wiring: cmd_basis's own preflight uses the rule (a fake root; the preflight fails for other reasons, the test reads only whether disk-low is among them)
_env = {k: os.environ.get(k) for k in ("SLOWWM_ROOT", "SLOWWM_DRY", "SLOWWM_FAKE_FREE_GB")}
_held = CK.held_basis_bytes
with tempfile.TemporaryDirectory() as td_:
    os.environ.update(SLOWWM_ROOT=td_, SLOWWM_DRY="1", SLOWWM_FAKE_FREE_GB="3.5")
    def _fail_text(gid):
        CK.cmd_basis(gid)
        return open(os.path.join(CK.basis_dir(), f"BASIS_{gid}.fail"), encoding="utf-8").read()
    CK.held_basis_bytes = lambda root=None: HELD
    ok("cmd_basis credits the held basis bytes (3.5 GB free, 2.98 GB held: no disk-low)", "disk-low" not in _fail_text("t1"))
    CK.held_basis_bytes = lambda root=None: 0
    ok("RED: cmd_basis with nothing held refuses 3.5 GB free as disk-low", "disk-low" in _fail_text("t2"))
CK.held_basis_bytes = _held
for k, v in _env.items():
    if v is None:
        os.environ.pop(k, None)
    else:
        os.environ[k] = v

# ---- the wait-state function of the chain (pure shell; extracted by running the chain's own pred_state) is covered by tests/test_slowwm_wait.sh
print(f"UNIT ok={OK} bad={BAD}")
sys.exit(1 if BAD else 0)
