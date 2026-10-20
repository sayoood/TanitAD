"""Tests of the TOKEN-LIST-GENERIC modes of SEARCH-NAVTEST-PIPE (MM scope addition 2026-10-19): the `s0bank` mode (label-only route, sealed S0-bank amendment, P2's own fan files as the identity reference), the reference
classes, the navtest table identity, and the fan-label join.  CPU only; planted data and a fake runner; every gate has a RED arm and a `@GUARD` line for mutation_check.py.

    cd code && CUDA_VISIBLE_DEVICES=-1 python -m pytest test_navpipe_s0.py -q -p no:cacheprovider --basetemp=<scratch>
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
import navpipe_chain as CH  # noqa: E402
import navpipe_bridge as NB  # noqa: E402
import navpipe_gpu_stage as GS  # noqa: E402
import test_navpipe as T  # noqa: E402
from navpipe_core import Refusal  # noqa: E402

raises = T.raises


# ====================================================================================================== references
def s0_fan_dir(tmp, tokens, logit_shift=None, pick_shift=None, units=2):
    """P2's OWN fan files of the S0 bank, planted: `<unit>.fan.npz` {token, fingerprint, logits [n,64,6], pick, log}"""
    d = os.path.join(str(tmp), "fans")
    os.makedirs(d, exist_ok=True)
    for u in range(units):
        tk = tokens[u::units]
        picks = np.array([T.fan(t)[1] for t in tk], np.int64)
        lg = np.zeros((len(tk), 64, 6), np.float32)
        if logit_shift in tk:
            lg[tk.index(logit_shift), 3, 2] = np.nextafter(np.float32(0), np.float32(1))     # ONE ulp
        if pick_shift in tk:
            picks[tk.index(pick_shift)] = (picks[tk.index(pick_shift)] + 1) % 64
        np.savez(os.path.join(d, f"u{u}.fan.npz"), token=np.array(tk), fingerprint=np.array([hashlib.sha1(t.encode()).hexdigest() for t in tk]), logits=lg, pick=picks, log=np.array(["l"] * len(tk)))
    return d


def test_s0fan_reference_levels_and_red_arms(tmp_path):
    toks = sorted(T.planted_doc()["tokens"])[:6]
    raw, pick = T.planted_bundle(toks)
    lg = np.zeros((6, 64, 6), np.float32)
    ref = NC.make_reference("s0fan", s0_fan_dir(tmp_path, toks))
    r = NC.check_reference_identity(ref, toks, raw, lg, pick)
    assert r["ok"] and r["kind"] == "s0fan" and r["n_bad"] == 0
    lg2 = lg.copy()
    lg2[1, 5, 0] = np.float32(1e-7)
    assert not NC.check_reference_identity(ref, toks, raw, lg2, pick)["ok"]                                  # one logit of one proposal differs
    p2 = pick.copy()
    p2[4] = (p2[4] + 1) % 64
    assert not NC.check_reference_identity(ref, toks, raw, lg, p2)["ok"]                                      # another proposal is the pick
    r = NC.check_reference_identity(ref, toks + ["f" * 16], np.concatenate([raw, raw[:1]]), np.concatenate([lg, lg[:1]]), np.concatenate([pick, pick[:1]]))
    assert not r["ok"] and r["n_absent_from_banked"] == 1                                                     # a token the bank does not hold
    assert not NC.check_reference_identity(ref, [], raw[:0], lg[:0], pick[:0])["ok"]                          # an empty chunk is no identity
    ref2 = NC.make_reference("s0fan", s0_fan_dir(tmp_path / "x", toks, logit_shift=toks[2]))
    assert not NC.check_reference_identity(ref2, toks, raw, lg, pick)["ok"]                                   # the BANK differs by one ulp: bytes, not tolerance
    ref3 = NC.make_reference("s0fan", s0_fan_dir(tmp_path / "y", toks, pick_shift=toks[0]))
    assert not NC.check_reference_identity(ref3, toks, raw, lg, pick)["ok"]


def test_s0fan_reference_refuses_duplicates_and_an_empty_directory(tmp_path):
    toks = sorted(T.planted_doc()["tokens"])[:6]
    d = s0_fan_dir(tmp_path, toks)
    shutil.copy(os.path.join(d, "u0.fan.npz"), os.path.join(d, "u9.fan.npz"))                                  # the same tokens in two units
    raises("gate", NC.make_reference, "s0fan", d)
    e = tmp_path / "empty"
    e.mkdir()
    raises("gate", NC.make_reference, "s0fan", str(e))


def test_fan_dir_sha_follows_every_member(tmp_path):
    toks = sorted(T.planted_doc()["tokens"])[:6]
    d = s0_fan_dir(tmp_path, toks)
    a = NC.fan_dir_sha256(d)
    assert a == NC.fan_dir_sha256(d)
    z = {k: v for k, v in np.load(os.path.join(d, "u1.fan.npz")).items()}
    z["pick"] = (z["pick"] + 1) % 64
    np.savez(os.path.join(d, "u1.fan.npz"), **z)
    assert NC.fan_dir_sha256(d) != a
    os.remove(os.path.join(d, "u0.fan.npz"))
    assert NC.fan_dir_sha256(d) not in (a,)


def planted_table(tmp, tokens, variant="repaired", shift=None, logit_shift=None):
    raw, pick = T.planted_bundle(tokens)
    props = np.zeros((len(tokens), 64, 8, 3), np.float32)
    for i in range(len(tokens)):
        r = raw[i]
        if variant == "repaired":
            props[i] = NC.exec_to_poses(NC.repair_last_heading(r))
        elif variant == "raw":
            props[i] = NC.exec_to_poses(r)
        else:
            props[i] = NC.exec_to_poses(r)
            props[i, pick[i]] = NC.exec_to_poses(NC.repair_last_heading(r[pick[i]])[None])[0]
    if shift is not None:
        props[tokens.index(shift), 17, 3, 0] = np.nextafter(props[tokens.index(shift), 17, 3, 0], np.float32(1e9))      # ONE ulp on a NON-pick proposal
    lg = np.zeros((len(tokens), 64, 6), np.float32)
    if logit_shift is not None:
        lg[tokens.index(logit_shift), 40, 1] = np.float32(1e-6)
    os.makedirs(str(tmp), exist_ok=True)
    p = os.path.join(str(tmp), "table.npz")
    np.savez(p, token=np.array(tokens), proposals=props, logits=lg, pick=pick, pdms=np.full((len(tokens), 64), np.nan), sub=np.full((len(tokens), 64, 6), np.nan), valid=np.zeros((len(tokens), 64), bool))
    return p, raw, pick


@pytest.mark.parametrize("variant", ["repaired", "raw", "pick_repaired_others_raw"])
def test_table_identity_all_64_proposals_logits_and_pick(tmp_path, variant):
    toks = sorted(T.planted_doc()["tokens"])[:5]
    p, raw, pick = planted_table(tmp_path, toks, variant)
    lg = np.zeros((5, 64, 6), np.float32)
    ref = NC.make_reference("table", p)
    r = NC.check_reference_identity(ref, toks, raw, lg, pick)
    assert r["ok"] and r["variant"] == variant
    p2, _, _ = planted_table(tmp_path / "s", toks, variant, shift=toks[3])
    assert not NC.check_reference_identity(NC.make_reference("table", p2), toks, raw, lg, pick)["ok"]            # one ulp on one NON-pick proposal: invisible to the 8-pose seam identity, seen here
    p3, _, _ = planted_table(tmp_path / "l", toks, variant, logit_shift=toks[1])
    assert not NC.check_reference_identity(NC.make_reference("table", p3), toks, raw, lg, pick)["ok"]


def test_table_locks_the_first_convention_so_a_mixed_table_is_no_identity(tmp_path):
    toks = sorted(T.planted_doc()["tokens"])[:4]
    pa, raw, pick = planted_table(tmp_path, toks, "repaired")
    pb, _, _ = planted_table(tmp_path / "b", toks, "raw")
    za, zb = np.load(pa), np.load(pb)
    props = za["proposals"].copy()
    props[2:] = zb["proposals"][2:]                                                                             # tokens 0-1 follow one convention, 2-3 another
    np.savez(pa, token=za["token"], proposals=props, logits=za["logits"], pick=za["pick"])
    ref = NC.make_reference("table", pa)
    r = NC.check_reference_identity(ref, toks, raw, np.zeros((4, 64, 6), np.float32), pick)
    assert not r["ok"] and r["n_bad"] >= 1


def test_table_reference_never_reads_a_label_array(tmp_path, monkeypatch):
    toks = sorted(T.planted_doc()["tokens"])[:3]
    p, raw, pick = planted_table(tmp_path, toks)
    seen = []
    real = np.load

    class Spy:
        def __init__(self, z):
            self.z, self.files = z, z.files

        def __getitem__(self, k):
            seen.append(k)
            return self.z[k]

    monkeypatch.setattr(np, "load", lambda *a, **k: Spy(real(*a, **k)))
    NC.make_reference("table", p)
    assert set(seen) <= {"token", "proposals", "logits", "pick"} and not ({"pdms", "sub", "valid"} & set(seen))


def test_seam_ref_from_dict_equals_the_legacy_check():
    toks = sorted(T.planted_doc()["tokens"])[:5]
    banked = T.banked_for(toks)
    raw, pick = T.planted_bundle(toks)
    ref = NC.SeamRef.from_dict(banked)
    assert NC.check_reference_identity(ref, toks, raw, np.zeros((5, 64, 6), np.float32), pick)["ok"] and NC.check_pick_identity(toks, raw, pick, banked)["ok"]
    p2 = pick.copy()
    p2[1] = (p2[1] + 1) % 64
    assert not NC.check_reference_identity(ref, toks, raw, np.zeros((5, 64, 6), np.float32), p2)["ok"]


def test_refuse_navtest_tokens_and_the_join(tmp_path):
    NC.refuse_navtest_tokens(["a", "b"], ["c", "d"])
    e = raises("seal", NC.refuse_navtest_tokens, ["a", "b"], ["b", "d"])
    assert e.reason == "REFUSED_NAVTEST_TOKEN" and e.code == 5
    toks = sorted(T.planted_doc()["tokens"])[:4]
    rs = np.random.RandomState(5)
    fl = {"token": np.array(toks[::-1]), "sub": rs.rand(4, 64, 6), "pdms": rs.rand(4, 64), "valid": np.ones((4, 64), bool)}
    b = {"token": np.array(toks), "pick": np.zeros(4, np.int64)}
    j = NC.join_fan_labels(b, fl)
    for i, t in enumerate(toks):
        assert np.array_equal(j["sub"][i], fl["sub"][3 - i]) and np.array_equal(j["pdms"][i], fl["pdms"][3 - i])        # re-aligned BY TOKEN, not by position
    raises("gate", NC.join_fan_labels, {**b, "token": np.array(toks + ["z" * 16]), "pick": np.zeros(5, np.int64)}, fl)   # a bundle token without a label row
    raises("gate", NC.join_fan_labels, b, {k: v for k, v in fl.items() if k != "valid"})
    raises("gate", NC.join_fan_labels, b, {**fl, "pdms": fl["pdms"][:, :8]})


# ====================================================================================================== the S0 seal
def make_s0_seal(tmp, pins, status="Status: **SEALED 2026-10-21**", drop=(), tamper_pin=None, name="S0AMEND"):
    pins = dict(pins)
    for f in NC.OPERATIVE:
        pins[f"code:{f}"] = NC.sha_lf(os.path.join(HERE, f))
    for k in drop:
        pins.pop(k, None)
    if tamper_pin:
        pins[tamper_pin] = "9" * 64
    body = "# S0-bank amendment (planted)\n\n" + status + "\n\nText.\n\n<!-- PINS-BEGIN -->\n" + "\n".join(f"{k} {v}" for k, v in pins.items()) + "\n<!-- PINS-END -->\n"
    am = os.path.join(str(tmp), name + ".md")
    open(am, "w", encoding="utf-8", newline="\n").write(body)
    sha = hashlib.sha256(body.encode()).hexdigest()
    open(os.path.join(str(tmp), name + ".sha256"), "w").write(sha + "  " + name + ".md\n")
    return am, sha


class S0World(T.World):
    """the planted S0-bank world: the navtest World's planted children, plus the S0 inputs (plan json, P2 fan files, fan labels), a fake `label_stepB`, and a sealed S0 amendment"""

    def __init__(self, tmp, label_fail=False, fan_label_drop=None, navtest_overlap=False, free=50.0, logit_shift=None, pick_shift=None, **kw):
        super().__init__(tmp, free=free, **kw)
        import make_navtest_bundle as MB
        self.label_fail, self.gpu_argvs, self.label_calls, self.harness_calls = label_fail, [], [], []
        self.plan_label = os.path.join(self.tmp, "s0_plan.json")
        json.dump(MB.make_plan(self.doc, "D:/db", "D:/frames"), open(self.plan_label, "w"))
        self.fandir = s0_fan_dir(self.tmp, self.tokens, logit_shift=logit_shift, pick_shift=pick_shift)
        rs = np.random.RandomState(9)
        order = [t for t in self.tokens[::-1] if t != fan_label_drop]
        self.fan_labels = os.path.join(self.tmp, "fan_labels.npz")
        self.fl = {"token": np.array(order), "sub": rs.rand(len(order), 64, 6), "pdms": rs.rand(len(order), 64), "valid": np.ones((len(order), 64), bool)}
        np.savez(self.fan_labels, **self.fl)
        nav = ["e" * 16, "d" * 16] + ([self.tokens[3]] if navtest_overlap else [])
        self.navtest = os.path.join(self.tmp, "navtest_tokens.json")
        json.dump({"tokens": nav}, open(self.navtest, "w"))
        self.s0_plan = NC.plan_doc(self.doc, NC.sha256_file(self.token_file), 11)
        self.pins0 = self.s0_pins()
        self.am0, self.sha0 = make_s0_seal(self.tmp, self.pins0)

    def s0_pins(self):
        return {"mode": "s0bank", "token_file_sha256": NC.sha256_file(self.token_file), "plan_json_sha256": NC.sha256_file(self.plan_label), "chunk_plan_sha256": self.s0_plan["plan_sha256"],
                "n_tokens": "20", "n_logs": "4", "ckpt_sha256": NC.sha256_file(self.ckpt), "config_sha256": NC.sha_lf(self.config), "head_s0_sha256": NC.sha256_file(self.head),
                "reference_sha256": NC.fan_dir_sha256(self.fandir), "reference_kind": "s0fan", "fan_labels_sha256": NC.sha256_file(self.fan_labels)}

    def cfg0(self, **kw):
        base = dict(mode="s0bank", amendment=self.am0, amendment_sha256=self.sha0, cap=11, token_file=self.token_file, ckpt=self.ckpt, config=self.config, head_s0=self.head, reference_path=self.fandir,
                    fan_labels=self.fan_labels, plan_label=self.plan_label, navtest_tokens=self.navtest, label_script=os.path.join(CH.STEPB_CODE, "label_stepB.py"))
        base.update(kw)
        return CH.Cfg(os.path.join(self.tmp, "work"), **base)

    def chain0(self, cfg=None, **kw):
        cfg = cfg or self.cfg0(**kw)

        def hf(label, seam_path, tokens_file):
            self.harness_calls.append(label)
            raise AssertionError("the T1 harness must never run in the s0bank mode")
        return CH.Chain(cfg, runner=self.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda p: self.free, harness_fn=hf)

    # children
    def runner(self, argv, env, log_path, cwd=None):
        joined = " ".join(argv)
        if "label_stepB.py" in joined:
            self.calls.append("label_stepB.py")
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.label(argv)
            return 0
        if "navpipe_gpu_stage.py" in joined:
            self.gpu_argvs.append(list(argv))
        return super().runner(argv, env, log_path, cwd)

    def label(self, argv):
        so = np.load(self.arg(argv, "--search"), allow_pickle=False)
        self.label_calls.append(self.arg(argv, "--plan"))
        S = so["exec"].shape[1]
        sub = np.zeros((len(so["token"]), S, 6))
        pdms = np.zeros((len(so["token"]), S))
        for s in range(S):
            sub[:, s], pdms[:, s] = T.f_scores(NC.exec_to_poses(so["exec"][:, s]))
        ident = {"passed": not self.label_fail, "control_ok": True, "max_abs": 0.0}
        np.savez(self.arg(argv, "--out"), token=so["token"], sub=sub, pdms=pdms, identity=np.array("IDENT " + json.dumps(ident)), sub_names=np.array(NC.SUBS))


def test_s0_seal_passes_without_a_c1_licence(tmp_path):
    w = S0World(tmp_path)
    r = NC.check_s0_seal(w.am0, w.sha0)
    assert r["licence"] is None and r["pins"]["mode"] == "s0bank" and r["amendment_sha256"] == w.sha0


@pytest.mark.parametrize("case", ["no_args", "no_amend", "no_sidecar", "typed_sha_wrong", "edited_after_seal", "draft", "mode_navtest", "mode_missing", "operative_unpinned", "code_pin_wrong"]
                         + ["pin_missing:" + k for k in NC.REQUIRED_PINS_S0 if k != "mode"])
def test_s0_seal_refuses(tmp_path, case):
    w = S0World(tmp_path)
    kw = {}
    if case == "draft":
        kw["status"] = "Status: **DRAFT**"
    if case.startswith("pin_missing:"):
        kw["drop"] = (case.split(":")[1],)
    if case == "mode_missing":
        kw["drop"] = ("mode",)
    pins = dict(w.pins0)
    if case == "mode_navtest":
        pins["mode"] = "navtest"
    if case == "operative_unpinned":
        kw["drop"] = ("code:navpipe_chain.py",)
    if case == "code_pin_wrong":
        kw["tamper_pin"] = "code:navpipe_core.py"
    am, sha = make_s0_seal(tmp_path, pins, **kw)
    if case == "no_args":
        e = raises("seal", NC.check_s0_seal, None, None)
    elif case == "no_amend":
        e = raises("seal", NC.check_s0_seal, str(tmp_path / "nope.md"), sha)
    elif case == "no_sidecar":
        os.remove(os.path.splitext(am)[0] + ".sha256")
        e = raises("seal", NC.check_s0_seal, am, sha)
    elif case == "typed_sha_wrong":
        e = raises("seal", NC.check_s0_seal, am, "0" * 64)
    elif case == "edited_after_seal":
        open(am, "a").write("\nedited\n")
        e = raises("seal", NC.check_s0_seal, am, sha)
    else:
        e = raises("seal", NC.check_s0_seal, am, sha)
    assert e.code == 5
    want = {"mode_navtest": "SEAL_WRONG_MODE", "mode_missing": "SEAL_PIN_MISSING", "operative_unpinned": "SEAL_PIN_MISMATCH", "code_pin_wrong": "SEAL_PIN_MISMATCH", "draft": "SEAL_NOT_SEALED",
            "no_args": "SEAL_MISSING_ARGS", "no_amend": "SEAL_AMENDMENT_MISSING", "no_sidecar": "SEAL_SIDECAR_MISSING", "typed_sha_wrong": "SEAL_SHA_MISMATCH", "edited_after_seal": "SEAL_SHA_MISMATCH"}.get(case, "SEAL_PIN_MISSING")
    assert e.reason == want, (case, e.reason)


def test_a_c2_amendment_cannot_unlock_the_s0_bank_nor_the_reverse(tmp_path):
    w = S0World(tmp_path)
    (tmp_path / "c2").mkdir()
    am, sha, c1, pins = T.make_seal(tmp_path / "c2")
    raises("seal", NC.check_s0_seal, am, sha)                                                   # a navtest amendment (licence pins, no S0 pins) is no S0 amendment
    e = raises("seal", NC.check_c2_seal, w.am0, w.sha0, c1.path, c1.sha)                        # an S0 amendment is no navtest amendment, even with a valid C1 licence in hand
    assert e.code == 5
    p2 = dict(w.pins0)
    p2.update({"c1_prereg_sha256": T.PLANT_PREREG, "licence_ruling_sha256": NC.LICENCE_RULING_SHA256, "inference_cost_status": "UNMEASURED", "bar_dpdms_points": "0.1", "bar_safety_points": "-0.25",
               "banked_seam_sha256": "a" * 64, "banked_csv_sha256": "a" * 64, "harness_script_sha256": "a" * 64, "seam_module_sha256": "a" * 64})
    am2, sha2 = make_s0_seal(tmp_path, p2, name="BOTH")                                          # even with every navtest pin present, mode=s0bank is not a navtest seal
    e = raises("seal", NC.check_c2_seal, am2, sha2, c1.path, c1.sha)
    assert e.reason == "SEAL_WRONG_MODE"


# ====================================================================================================== the s0bank chain
def test_s0_chain_refuses_before_opening_any_data_without_the_seal(tmp_path):
    called = []
    cfg = CH.Cfg(str(tmp_path / "w"), mode="s0bank", token_file=str(tmp_path / "does_not_exist.json"), amendment=None)
    ch = CH.Chain(cfg, runner=lambda *a, **k: called.append(a) or 0, out=io.StringIO())
    assert ch.run() == 5 and not called
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP SEAL_MISSING_ARGS" in out and "DATA_OK" not in out and not os.path.exists(str(tmp_path / "w" / "chunks"))


def test_s0_chain_happy_path_uses_the_label_route_and_never_the_t1_harness(tmp_path):
    w = S0World(tmp_path)
    ch = w.chain0(procs=2)
    assert ch.run() == 0, ch.out.getvalue()
    out = ch.out.getvalue()
    for m in ("SEAL_OK", "DATA_OK", "DISK_OK", "BUNDLE_OK 0", "SEARCH_DONE 0", "SCORE_DONE 0", "BANKED 0", "CHUNK_DONE 0", "CHUNK_DONE 1", "S0BANK_MERGED", "ANALYSIS_NOT_RUN"):
        assert "ZZNAVPIPE_" + m in out, m
    assert "ZZNAVPIPE_ANALYSIS_DONE" not in out and "ADMISSION" not in out.replace("no admission record", "").replace("ANALYSIS_NOT_RUN EP-ADMISSION is read downstream", "")
    assert w.harness_calls == [] and w.calls.count("label_stepB.py") == 2 and not os.path.exists(os.path.join(w.tmp, "work", "t1"))
    assert set(w.label_calls) == {w.plan_label}                                                  # the label route reads the PINNED plan
    for av in w.gpu_argvs:                                                                        # the GPU stage gets the pinned plan and the S0 reference, not the navtest seam
        assert av[av.index("--plan") + 1] == w.plan_label and av[av.index("--reference-kind") + 1] == "s0fan" and av[av.index("--reference-path") + 1] == w.fandir
    st = NC.State(os.path.join(w.tmp, "work", "navpipe_state.json"), w.s0_plan["plan_sha256"])
    for cid in (0, 1):
        assert st.status(cid) == "DELETED" and not os.path.exists(os.path.join(w.tmp, "work", "chunks", f"c{cid:02d}", "bundle", "ctx.npy")) and NC.verify_banked(st, cid) == []
    fin = os.path.join(w.tmp, "work", "final")
    b = np.load(os.path.join(fin, "bundle_merged.npz"), allow_pickle=False)
    s = np.load(os.path.join(fin, "search_merged.npz"), allow_pickle=False)
    lab = np.load(os.path.join(fin, "scores_merged.npz"), allow_pickle=False)
    toks = [str(t) for t in b["token"]]
    assert toks == w.tokens == [str(t) for t in s["token"]] == [str(t) for t in lab["token"]]      # canonical (sorted-token) order
    ix = {str(t): i for i, t in enumerate(w.fl["token"])}
    for i, t in enumerate(toks):                                                                   # the fan labels are the PINNED INPUT, joined by token
        assert np.array_equal(b["sub"][i], w.fl["sub"][ix[t]]) and np.array_equal(b["pdms"][i], w.fl["pdms"][ix[t]])
    assert b["sub"].shape == (20, 64, 6) and b["valid"].shape == (20, 64) and "traj_raw" not in b.files
    man = json.load(open(os.path.join(fin, "S0BANK_MANIFEST.json")))
    assert man["mode"] == "s0bank" and man["n_tokens"] == 20 and all(NC.sha256_file(v["path"]) == v["sha256"] for v in man["files"].values())


@pytest.mark.parametrize("what", ["token_file", "plan_json", "ckpt", "config", "head", "reference", "fan_labels", "kind"])
def test_s0_chain_pin_mismatch_stops_before_any_child(tmp_path, what):
    w = S0World(tmp_path)
    if what == "token_file":
        d = dict(w.doc)
        d["rule"] = "edited"
        json.dump(d, open(w.token_file, "w"))
    elif what == "plan_json":
        open(w.plan_label, "a").write(" ")
    elif what == "ckpt":
        open(w.ckpt, "ab").write(b"x")
    elif what == "config":
        open(w.config, "w").write('{"a": 1}')
    elif what == "head":
        open(w.head, "ab").write(b"x")
    elif what == "reference":
        z = {k: v for k, v in np.load(os.path.join(w.fandir, "u0.fan.npz")).items()}
        z["pick"] = (z["pick"] + 1) % 64
        np.savez(os.path.join(w.fandir, "u0.fan.npz"), **z)
    elif what == "fan_labels":
        np.savez(w.fan_labels, **{**w.fl, "pdms": w.fl["pdms"] + 1e-9})
    ch = w.chain0(**({"reference_kind": "seam"} if what == "kind" else {}))
    assert ch.run() == 5
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP SEAL_PIN_MISMATCH" in out and w.calls == [] and not os.path.exists(os.path.join(w.tmp, "work", "chunks"))


def test_s0_chain_refuses_a_token_list_that_holds_a_navtest_token(tmp_path):
    w = S0World(tmp_path, navtest_overlap=True)
    ch = w.chain0()
    assert ch.run() == 5
    assert "ZZNAVPIPE_STOP REFUSED_NAVTEST_TOKEN" in ch.out.getvalue() and w.calls == []


def test_s0_chain_refuses_a_missing_reference_or_labels_or_plan(tmp_path):
    for k, why in (("reference_path", "S0_REFERENCE_MISSING"), ("fan_labels", "S0_FAN_LABELS_MISSING"), ("plan_label", "S0_PLAN_MISSING")):
        w = S0World(tmp_path / k)
        ch = w.chain0(**{k: str(tmp_path / "nowhere")})
        assert ch.run() == 5 and why in ch.out.getvalue() and w.calls == [], k


def test_s0_chain_red_identity_against_p2s_own_fan_stops_and_keeps_the_ctx(tmp_path):
    for arm in ("logit", "pick"):
        base = tmp_path / arm
        w0 = S0World(base)
        tok = NC.chunk_tokens(w0.doc, w0.s0_plan["chunks"][1])[2]
        w = S0World(base / "w", logit_shift=tok if arm == "logit" else None, pick_shift=tok if arm == "pick" else None)
        ch = w.chain0()
        assert ch.run() == 3, arm
        out = ch.out.getvalue()
        assert "ZZNAVPIPE_STOP G_IDENT_NOT_PASSED" in out and "CHUNK_DONE 0" in out and "CHUNK_DONE 1" not in out and "S0BANK_MERGED" not in out
        assert os.path.exists(os.path.join(w.tmp, "work", "chunks", "c01", "bundle", "ctx.npy"))        # the evidence is kept


def test_s0_chain_red_a_bundle_token_without_a_fan_label_row_stops(tmp_path):
    w0 = S0World(tmp_path / "a")
    drop = NC.chunk_tokens(w0.doc, w0.s0_plan["chunks"][0])[1]
    w = S0World(tmp_path / "b", fan_label_drop=drop)
    ch = w.chain0()
    assert ch.run() == 3
    assert "ZZNAVPIPE_STOP FAN_LABELS_MISSING_TOKENS" in ch.out.getvalue() and "SEARCH_DONE" not in ch.out.getvalue()
    assert os.path.exists(os.path.join(w.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))


def test_s0_chain_red_label_planted_identity_failure_stops_before_banking(tmp_path):
    w = S0World(tmp_path, label_fail=True)
    ch = w.chain0()
    assert ch.run() == 7
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP G_LABEL_NOT_PASSED" in out and "BANKED" not in out and "S0BANK_MERGED" not in out
    st = NC.State(os.path.join(w.tmp, "work", "navpipe_state.json"), w.s0_plan["plan_sha256"])
    assert st.status(0) == "SEARCHED" and os.path.exists(os.path.join(w.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))


def test_s0_chain_red_disk_guard_stops_before_any_child(tmp_path):
    w = S0World(tmp_path, free=5.0)
    ch = w.chain0()
    assert ch.run() == 3 and "ZZNAVPIPE_STOP" in ch.out.getvalue() and "DISK" in ch.out.getvalue().upper() and "navpipe_gpu_stage.py" not in w.calls


def test_s0_chain_is_resumable_and_prefetch_gives_the_same_bank(tmp_path):
    w = S0World(tmp_path / "a")
    ch = w.chain0(max_chunks=1)
    assert ch.run() == 0 and "PARTIAL" in ch.out.getvalue()
    n1 = w.calls.count("label_stepB.py")
    ch2 = w.chain0()
    assert ch2.run() == 0 and w.calls.count("label_stepB.py") == n1 + 1                          # chunk 0 is NOT redone
    w2 = S0World(tmp_path / "b")
    ch3 = w2.chain0(prefetch=1)
    assert ch3.run() == 0
    for n in ("bundle_merged.npz", "search_merged.npz", "scores_merged.npz"):
        a, b = np.load(os.path.join(w.tmp, "work", "final", n)), np.load(os.path.join(w2.tmp, "work", "final", n))
        assert all(np.array_equal(a[k], b[k]) for k in a.files if k != "meta"), n


# ====================================================================================================== navtest mode, optional table identity
def test_navtest_chain_with_the_table_identity_and_its_red_arm(tmp_path):
    w = T.World(tmp_path / "ok")
    p, _, _ = planted_table(tmp_path / "ok", w.tokens)
    ch = CH.Chain(w.cfg(table_ref=p), runner=w.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda q: w.free, harness_fn=w.harness_fn)
    assert ch.run() == 0, ch.out.getvalue()
    idn = json.load(open(os.path.join(w.tmp, "work", "chunks", "c00", "identity.json")))
    assert idn["ok"] and idn["table"]["ok"] and idn["table"]["variant"] == "repaired"
    w2 = T.World(tmp_path / "red")
    p2, _, _ = planted_table(tmp_path / "red", w2.tokens, shift=NC.chunk_tokens(w2.doc, w2.plan["chunks"][0])[1])
    ch2 = CH.Chain(w2.cfg(table_ref=p2), runner=w2.runner, out=io.StringIO(), ram_fn=lambda: 99.0, free_fn=lambda q: w2.free, harness_fn=w2.harness_fn)
    assert ch2.run() == 3 and "G_IDENT_NOT_PASSED" in ch2.out.getvalue() and "table" in ch2.out.getvalue()      # the 8-pose seam identity passes; the table sees the non-pick proposal


def test_cli_accepts_navtest_as_the_name_of_the_c2_mode_and_requires_tokens_for_s0bank(tmp_path, capsys):
    with pytest.raises(SystemExit):
        CH.main(["--mode", "s0bank", "--work", str(tmp_path / "w")])
    rc = CH.main(["--mode", "navtest", "--work", str(tmp_path / "w2"), "--dry-run"])
    assert rc == 5 and "ZZNAVPIPE_START c2" in capsys.readouterr().out
    rc = CH.main(["--mode", "s0bank", "--tokens", str(tmp_path / "t.json"), "--work", str(tmp_path / "w3"), "--dry-run"])
    assert rc == 5 and "SEAL_MISSING_ARGS" in capsys.readouterr().out


# ====================================================================================================== builder / bridge / GPU stage, reference-generic
def test_gpu_stage_and_bridge_carry_the_reference_kind(tmp_path):
    class A:
        bridge_sample, chunk_sample, plan, ckpt, banked_seam, out_dir, n_expected = "b.json", "c.json", "p.json", "k.pt", "s.npz", "o", 7
        reference_kind, reference_path = "s0fan", "fans"
    steps = GS.plan(A, py="py")
    for _, argv, _ in steps:
        assert argv[argv.index("--reference-kind") + 1] == "s0fan" and argv[argv.index("--reference-path") + 1] == "fans" and "--banked-seam" not in argv

    class B(A):
        reference_kind, reference_path = None, None
    for _, argv, _ in GS.plan(B, py="py"):
        assert argv[argv.index("--reference-kind") + 1] == "seam" and argv[argv.index("--reference-path") + 1] == "s.npz"             # the navtest default is unchanged


def test_bridge_judge_works_on_p2s_s0_fan_reference_and_every_red_arm(tmp_path):
    toks = sorted(T.planted_doc()["tokens"])[:3]
    ref = NC.make_reference("s0fan", s0_fan_dir(tmp_path, toks))

    def calls(shift=()):
        out = {}
        for t in toks:
            f, k = T.fan(t)
            lg = np.zeros((64, 6), np.float32)
            if t in shift:
                lg = lg.copy()
                lg[3, 3] = np.float32(1e-3)
            out[t] = (f, lg, k)
        return out
    ident = {"rc": 0, "stdout": "ZZBUNDLE_OK 3 1 s", "fired": 0, "calls": calls()}
    red = {"rc": 1, "stdout": "ZZBUNDLE_FAIL missing 0 bad 3", "fired": 3, "calls": calls(shift=toks)}
    assert NB.judge(ident, red, toks, ref)["pass"]
    assert not NB.judge({**ident, "calls": calls(shift=toks[:1])}, red, toks, ref)["pass"]
    assert not NB.judge(ident, {**red, "calls": calls()}, toks, ref)["pass"]
    assert not NB.judge(ident, {**red, "fired": 1}, toks, ref)["pass"]
