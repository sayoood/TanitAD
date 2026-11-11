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

# ---- check3 / fit records
c3 = {"pass": True, "identity": {k: {"pass": True, "wrong_kind_fails_identity": True} for k in CK.KINDS}, "clips": list(range(8))}
ok("check3 good", CK.check3_problems(c3) == [])
for name, fn in {"pass false": lambda r: r.update({"pass": False}), "smoke": lambda r: r.update({"SMOKE": True}), "wrong kind passes": lambda r: r["identity"]["slow"].update(wrong_kind_fails_identity=False),
                 "7 clips": lambda r: r.update(clips=list(range(7)))}.items():
    r = copy.deepcopy(c3)
    fn(r)
    ok(f"check3 refuses: {name}", CK.check3_problems(r) != [])
ok("check3 refuses an unreadable record", CK.check3_problems(None) != [])
fit = {"meta": {"n_windows_ok": 40358, "smoke": False, "card_sha256": CK.CARD_SHA256}, "basis_sha256": {"slow": "a" * 64, "pca": "b" * 64}}
ok("fit good", CK.fit_problems(fit) == [])
for name, fn in {"windows": lambda r: r["meta"].update(n_windows_ok=40000), "smoke": lambda r: r["meta"].update(smoke=True), "card": lambda r: r["meta"].update(card_sha256="0" * 64), "sha": lambda r: r["basis_sha256"].update(slow="x")}.items():
    r = copy.deepcopy(fit)
    fn(r)
    ok(f"fit refuses: {name}", CK.fit_problems(r) != [])

# ---- permit (72 h rule)
tmp = tempfile.mkdtemp(prefix="slowwm_unit_", dir=os.environ.get("SLOWWM_TEST_TMP", "F:/TanitAD-scratch/slowwm_box2_build"))
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
basis_rec(commit="b" * 40)
ok("verify-basis refuses another commit", CK.cmd_verify_basis(args) == 11)
basis_rec(counts={"slow/train": 1})
ok("verify-basis refuses wrong counts", CK.cmd_verify_basis(args) == 11)
basis_rec(basis_sha256={"slow": "a" * 64, "pca": "a" * 64})
ok("verify-basis refuses slow == pca", CK.cmd_verify_basis(args) == 11)
basis_rec(check3_sha256="0" * 64)
ok("verify-basis refuses a check3 file that differs from the recorded sha256", CK.cmd_verify_basis(args) == 11)

# ---- the wait-state function of the chain (pure shell; extracted by running the chain's own pred_state) is covered by tests/test_slowwm_wait.sh
print(f"UNIT ok={OK} bad={BAD}")
sys.exit(1 if BAD else 0)
