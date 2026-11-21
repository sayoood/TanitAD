"""Controls of dscale10k_check_rh.py (ROLLHIST gate build B3). CPU only, fake small banks / argv files / data; the REAL pod templates (fixtures/U_sK.B.tmpl.json, B_s0.B.tmpl.json), the REAL
2bda9e1a ``rollhist_pair_check.py`` and ``v8f_dagger.py`` (ref/tree) and the reps1 checker (ref/chk_reps1.py, the unchanged original of the identity test).

The heavy data checks of a union arm (C-MERGE re-runs the merge tool on the real 14,512-clip release, C-MASK / C-JOIN-EQ read the real npz files) cannot run on fakes: the Rig replaces exactly those three
functions with a pass, IN BOTH the original and the new module, so every other check (C-ARGV, C-MD5, C-STEPS, C-NCLIPS, C-CHUNK, C-TRAIN-SHA, C-DISJ, C-HOLD, C-BANK, G-DAGGER, the binding, the token) runs for real.
"""
import copy
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "ref")
for p in (HERE,):
    if p not in sys.path:
        sys.path.insert(0, p)
import dscale10k_merge as M  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


NEW = _load("dscale10k_check_rh_new", os.environ.get("DS_CHECK_RH_PATH") or os.path.join(HERE, "dscale10k_check_rh.py"))      # the mutation run points this at the mutant copy
ORIG = _load("dscale10k_check_reps1_orig", os.path.join(REF, "chk_reps1.py"))
PAIR_CHECK = os.path.join(REF, "tree", "stack", "scripts", "rollhist_pair_check.py")
COMMIT = "2bda9e1a382986cfdce3bb455822678b42a816e2"
N_CLIPS, N_WIN = 14512, 985048


def fx(name):
    o = json.load(open(os.path.join(HERE, "fixtures", name), encoding="utf-8"))
    return o["argv"] if isinstance(o, dict) else o


def md5f(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def put_json(p, o):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(o, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True)
    return p


GEN = {"causal_g1x": {"readout": "causal_g", "acc_gain": 1.0, "path_exec": True, "kappa_max": "0.2"},
       "causal_g1": {"readout": "causal_g", "acc_gain": 1.0, "path_exec": False, "kappa_max": "0.2"}}


def ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


class Rig:
    """One seed's world: tree root, union data files, dry record, UE stage-A checkpoint, a valid 4-shard bank under ``gen``, filled RH and B0 argv, evidence, selfchecks."""

    def __init__(self, tmp, seed=0, gen="causal_g1x"):
        self.tmp, self.seed, self.gen = str(tmp), seed, gen
        T = self.tmp
        self.root = os.path.join(T, "tree")
        for rel, txt in (("stack/scripts/v8f_mini_train.py", "# trainer\n"), ("stack/tanitad/x.py", "x=1\n"), ("stack/tests/t.py", "t=1\n"), ("taniteval/e.py", "e=1\n")):
            os.makedirs(os.path.dirname(os.path.join(self.root, rel)), exist_ok=True)
            open(os.path.join(self.root, rel), "w").write(txt)
        d = os.path.join(T, "data")
        os.makedirs(os.path.join(d, "cache", "train"), exist_ok=True)
        self.files = {}
        for n, b in (("v9", b"v9-union"), ("lead", b"lead"), ("speed", b"speed"), ("tracks", b"tracks"), ("p10", b"p10"), ("join", b"join-union"), ("ckpt0", b"stageA-ckpt-s0"), ("ckpt1", b"stageA-ckpt-s1"),
                      ("merge_rec", b"{}"), ("tenk", b"tenk"), ("urel", b"urel")):
            self.files[n] = os.path.join(d, n + ".bin")
            open(self.files[n], "wb").write(b)
        self.corpus_join = os.path.join(d, "corpus_join.npz")
        np.savez(self.corpus_join, a=np.arange(3))
        self.union_join = os.path.join(d, "union_join.npz")
        np.savez(self.union_join, a=np.arange(3))
        hold = [f"hold_{i}" for i in range(5)]
        open(os.path.join(d, "cache", "train", "manifest.jsonl"), "w").write("\n".join(json.dumps({"clip": c, "n_frames": 10}) for c in hold) + "\n")
        self.holdout_manifest = os.path.join(d, "holdout_manifest.jsonl")
        open(self.holdout_manifest, "w").write("\n".join(json.dumps({"clip": c, "n_frames": 10}) for c in hold) + "\n")
        self.eval_ids = os.path.join(d, "eval_ids.txt")
        open(self.eval_ids, "w").write("eval_1\neval_2\n")
        clips = [f"tr_{i:05d}" for i in range(N_CLIPS)]
        self.dry = put_json(os.path.join(T, "dry.json"), {"n_clips": N_CLIPS, "n_clips_all": N_CLIPS + 60, "n_windows": N_WIN, "n_windows_all_clips": N_WIN, "train_sha256": M.train_sha256(clips), "train_clips": sorted(clips, key=M.sha12),
                                                          "lru": 50, "sampler_chunk": 48, "S": 4791, "S_U": 4791, "steps_mult": 4, "batch": 64, "total_steps": 3000, "v9_md5": md5f(self.files["v9"]), "n_excluded": 60,
                                                          "excluded_in_train": 0, "argv_sha256": "x"})
        self.base = os.path.join(T, "B_s0.B.tmpl.json")
        json.dump({"argv": self._remap(fx("B_s0.B.tmpl.json"), only=("--p10",))}, open(self.base, "w"))      # the base's --p10 is not a data flag: both sides name the same (test) file
        self.tmpl_toks = self._remap(fx(f"U_s{seed}.B.tmpl.json"))
        self.template = put_json(os.path.join(T, f"U_s{seed}.B.tmpl.json"), {"argv": self.tmpl_toks})
        self.init = os.path.join(T, "pod", "runs", f"UE_s{seed}", "A", "ckpt_last.pt")
        os.makedirs(os.path.dirname(self.init), exist_ok=True)
        open(self.init, "wb").write(open(self.files[f"ckpt{seed}"], "rb").read())
        self.bank_dir = os.path.join(T, "bank", f"s{seed}")
        self.make_bank(gen)
        self.rh_toks = self.make_rh(gen)
        self.b0_toks = self.make_b0()
        self.ev = self.make_evidence()
        self.sc_rh = self.make_selfcheck_rh(gen)
        self.sc_b0 = self.make_selfcheck_b0()

    # ---- builders
    def _remap(self, toks, only=None):
        m = {"--cache-dir": os.path.join(self.tmp, "data", "cache"), "--v9-train": self.files["v9"], "--v9-train-md5": md5f(self.files["v9"]), "--lead-train": self.files["lead"], "--speed-train": self.files["speed"],
             "--join-train": self.files["join"], "--p10": self.files["p10"], "--tracks": self.files["tracks"]}
        if only:
            m = {k: v for k, v in m.items() if k in only}
        out, i = [], 0
        toks = list(toks)
        while i < len(toks):
            out.append(toks[i])
            if toks[i] in m:
                out.append(m[toks[i]])
                i += 2
                continue
            i += 1
        return out

    @staticmethod
    def setf(toks, flag, val):
        t = list(toks)
        t[t.index(flag) + 1] = val
        return t

    @staticmethod
    def dropf(toks, flag):
        t = list(toks)
        i = t.index(flag)
        j = i + 1
        while j < len(t) and not (t[j].startswith("--") and len(t[j]) > 2):
            j += 1
        return t[:i] + t[j:]

    def make_bank(self, gen, with_generator=True, shas_from=None):
        bd = self.bank_dir
        os.makedirs(bd, exist_ok=True)
        g = GEN[gen] if with_generator else None
        shards = []
        for k in range(4):
            f = f"shard_{k:03d}of004.npz"
            open(os.path.join(bd, f), "wb").write(f"shard{k}".encode())
            shards.append({"k": k, "file": f, "file_sha256": sha(os.path.join(bd, f)), "n_records": 10})
            put_json(os.path.join(bd, f"shard_{k:03d}of004.provenance.json"), {"kb0_tick0_identity": {"mismatch": 0}, "kb1_replay": {"mismatch_chunks": 0}, **({"generator": g} if g else {})})
        common = {"train": {"n_clips": N_CLIPS, "n_windows": N_WIN, "train_sha256": json.load(open(self.dry))["train_sha256"]}, "eval": {"overlap": 0, "n_clips": 141}, "partial": False, "ckpt_sha256": sha(self.init),
                  **({"generator": g} if g else {})}
        bj = put_json(os.path.join(bd, "bank.json"), {"n_shards": 4, "shards": shards, "common": common, "partial": False, "coverage": {"fraction": 0.9, "blocked": False}, **({"generator": g} if g else {})})
        put_json(os.path.join(bd, "PROVENANCE.json"), {"controls": {"K-B4": {"PASS": True}, "K-B5": {"PASS": True}}, "bank_sha256": sha(bj)})
        self.bank_sha = sha(bj)
        return bj

    def make_rh(self, gen, bank_sha=None):
        t = list(self.tmpl_toks)
        out = os.path.join(self.tmp, "runs", f"RH_s{self.seed}", "train")
        t = self.setf(t, "--out", out)
        t = self.setf(t, "--init-ckpt", self.init)
        t = self.setf(t, "--dagger-bank", self.bank_dir)
        t = self.setf(t, "--dagger-bank-sha256", bank_sha or self.bank_sha)
        return t + ["--dagger-bank-readout", gen, "--dagger-bank-kappa", "0.2"]

    def make_b0(self):
        t = list(self.tmpl_toks)
        t = self.setf(t, "--out", os.path.join(self.tmp, "runs", f"B0_s{self.seed}", "train"))
        t = self.setf(t, "--init-ckpt", self.init)
        t = self.setf(t, "--dagger-p", "0")
        for f in ("--dagger-bank", "--dagger-bank-sha256", "--dagger-seed"):
            t = self.dropf(t, f)
        return t

    def make_evidence(self):
        terms = NEW.STAGE_A_TERMS
        reach = {k: {t: {"finite": True, "value": 1.0, "grad": 1.0} for t in terms} for k in ("corpus", "tenk", "mixed")}
        reach["tenk"]["goal_vlm"] = {"finite": True, "value": 1.0, "grad": 0.0}
        reach["tenk"]["cons_lc"] = {"finite": True, "value": 0.0, "grad": 1.0}
        reach["control"] = {"goal_vlm_grad": 1.0}
        hy = {"arm": {"n_trainable": NEW.BASE_PARAMS, "cfg": {"a": 1}}, "base": {"n_trainable": NEW.BASE_PARAMS, "cfg": {"a": 1}}}
        return put_json(os.path.join(self.tmp, "evidence.json"), {"hyg": hy, "reach": reach, "clock": {"tenk": [0.01, 0.02], "corpus": [0.02, 0.03]}, "G-EVAL": {"pass": True}, "G-CKPT": {"pass": True}, "G-SUITE": {"pass": True}})

    def make_selfcheck_rh(self, gen):
        sc = {"declared_vs_built": "PASS", "acc_probe": {"pass": True, "head_o_grad_diff": 1.0, "w_acc_declared": 49.6969},
              "dagger_probe": {"dagger": "on", "pass": True, "loss_diff": 0.5, "grad_diff_all": 0.5, "control_repeat_loss_diff": 0.0, "control_repeat_grad_diff": 0.0, "control_p0_loss_diff": 0.0,
                               "control_p0_grad_diff": 0.0, "control_prior_grad_diff": 0.0, "control_wm_grad_diff": 0.0, "control_p0_same_object": True},
              "info": {"dagger": {"coverage": {"fraction": 0.9, "blocked": False}}}, "levers": {"dagger": {"dagger_bank_readout": gen, "dagger_bank_kappa": "0.2"}}}
        return put_json(os.path.join(self.tmp, "selfcheck_rh.json"), sc)

    def make_selfcheck_b0(self):
        sc = {"declared_vs_built": "PASS", "acc_probe": {"pass": True, "head_o_grad_diff": 1.0, "w_acc_declared": 49.6969}, "levers": {"dagger": {"dagger_bank_readout": "pinned", "dagger_bank_kappa": "none"}}}
        return put_json(os.path.join(self.tmp, "selfcheck_b0.json"), sc)

    # ---- minting
    def argv_file(self, name, toks):
        return put_json(os.path.join(self.tmp, name + ".argv.json"), {"argv": toks})

    def patch_heavy(self, mod):
        mod.c_merge_problems = lambda rec: ([], md5f(self.files["v9"]))
        mod.mask_problems_release = lambda *a, **k: []
        mod.join_eq_problems = lambda *a, **k: []
        mod.CORPUS_JOIN_MD5_PREFIX = md5f(self.corpus_join)[:8]

    def mint_ns(self, arm, argv_path, gate, pair=None, token=None, selfcheck=None, template=None, pair_check=PAIR_CHECK, stage="B"):
        return ns(arm=arm, stage=stage, argv=argv_path, root=self.root, commit=COMMIT, gate_dir=os.path.join(self.tmp, "gate", gate), py=sys.executable, gpu="0", base_argv=self.base, evidence=self.ev,
                  dry=self.dry, merge_record=self.files["merge_rec"], union_release=self.files["urel"], tenk_v9=self.files["tenk"], r_release="", corpus_join=self.corpus_join, union_join=self.union_join,
                  eval_ids=self.eval_ids, digest_json=[], ue_dry="", selfcheck=selfcheck or "", tacsrc_list="", ue_argv="", b_dry="", holdout_manifest=self.holdout_manifest, holdout=(0, 3),
                  template=template if template is not None else self.template, pair_argv=pair or "", pair_check=pair_check, readout_token=token or "")

    def mint(self, kind, rh_toks=None, b0_toks=None, token=None, mod=None, sc_rh=None, sc_b0=None, tag=None, pair_check=PAIR_CHECK, template=None):
        mod = mod or NEW
        self.patch_heavy(mod)
        s = self.seed
        rh_toks, b0_toks = rh_toks or self.rh_toks, b0_toks or self.b0_toks
        rp, bp = self.argv_file(f"RH_s{s}{tag or ''}", rh_toks), self.argv_file(f"B0_s{s}{tag or ''}", b0_toks)
        if kind == "RH":
            a = self.mint_ns(f"RH_s{s}", rp, f"RH_s{s}{tag or ''}", pair=bp, token=token if token is not None else self.gen, selfcheck=sc_rh or self.sc_rh, pair_check=pair_check, template=template)
        else:
            a = self.mint_ns(f"B0_s{s}", bp, f"B0_s{s}{tag or ''}", pair=rp, token=token, selfcheck=sc_b0 or self.sc_b0, pair_check=pair_check, template=template)
        problems = mod.mint(a)
        return problems, a.gate_dir


@pytest.fixture()
def rig(tmp_path):
    return Rig(tmp_path / "w0")


def refused_by(problems, *names):
    txt = "\n".join(problems)
    return all(f"check {n} did not pass" in txt or n in txt for n in names)


# ---------------------------------------------------------------------------------------------------------------------------- 1. matching RH and B0 mint PASS
def test_matching_rh_and_b0_mint_pass(rig):
    pr, gd = rig.mint("RH")
    assert pr == [], pr
    pb, gb = rig.mint("B0")
    assert pb == [], pb
    for g, arm, toks in ((gd, "RH_s0", rig.rh_toks), (gb, "B0_s0", rig.b0_toks)):
        tok = json.load(open(os.path.join(g, "gate_token.json")))
        assert tok["commit"] == COMMIT and tok["arm"] == arm and tok["stage"] == "B"
        assert tok["argv_sha256"] == NEW.argv_sha(toks)
        assert tok["tree_manifest_sha256"] and tok["trainer_sha256"] and tok["data"]["cache/train/manifest.jsonl"]["sha256"]
        assert tok["data"]["--init-ckpt"]["sha256"] == sha(rig.init)
        assert tok["checks"]["C-RH-PAIR"]["readout_token"] == "causal_g1x"
        assert set(NEW.required_checks(arm, "B")) <= {c for c, v in tok["checks"].items() if v["pass"]}
    rh = json.load(open(os.path.join(gd, "gate_token.json")))
    assert rh["data"]["bank/bank.json"]["sha256"] == rig.bank_sha and rh["checks"]["C-RH-BANKGEN"]["pass"] and rh["checks"]["C-BANK"]["pass"]
    assert "C-BANK" not in json.load(open(os.path.join(gb, "gate_token.json")))["checks"]
    # the token binds every file the stage-B token binds, plus the template, the pair and the pair-check
    assert {"template", "pair_argv", "pair_check"} <= set(rh["data"])
    # token_problems (what `run` calls before it starts the trainer): clean now, REFUSED once the argv or the pair changes
    extra = {"template": rig.template, "pair_argv": os.path.join(rig.tmp, "B0_s0.argv.json"), "pair_check": PAIR_CHECK, "selfcheck": rig.sc_rh, "evidence": rig.ev, "dry_record": rig.dry, "merge_record": rig.files["merge_rec"],
             "--union-release": rig.files["urel"], "tenk_v9": rig.files["tenk"]}
    assert NEW.token_problems(os.path.join(gd, "gate_token.json"), "RH_s0", "B", rig.rh_toks, rig.root, COMMIT, extra) == []
    assert NEW.token_problems(os.path.join(gd, "gate_token.json"), "RH_s0", "B", rig.setf(rig.rh_toks, "--lr", "2e-4"), rig.root, COMMIT, extra)


# ---------------------------------------------------------------------------------------------------------------------------- 2. a wrong declared token is REFUSED
def test_wrong_declared_token_refused(rig):
    pr, _ = rig.mint("RH", token="causal_g1")                       # CLI token disagrees with the argv's declaration
    assert pr and refused_by(pr, "C-ARGV"), pr
    pr, _ = rig.mint("RH", token="causal_g2x")                      # not one of the two readouts
    assert pr and refused_by(pr, "C-ARGV"), pr
    wrong = rig.setf(rig.rh_toks, "--dagger-bank-readout", "causal_g0.5")         # argv declares a readout the bank was not generated under
    pr, _ = rig.mint("RH", rh_toks=wrong, token="causal_g0.5")
    assert pr and refused_by(pr, "C-ARGV"), pr
    # the argv and the CLI both say causal_g1x but the bank was generated WITHOUT --path-exec (causal_g1): the bank's record decides
    rig.make_bank("causal_g1")
    rh = rig.make_rh("causal_g1x")
    pr, _ = rig.mint("RH", rh_toks=rh, token="causal_g1x")
    assert pr and refused_by(pr, "C-RH-BANKGEN"), pr


def test_causal_g1_refused_pair_check_needs_a_flag(tmp_path):
    r = Rig(tmp_path / "g1", gen="causal_g1")
    pr, _ = r.mint("RH", token="causal_g1")
    assert pr and "hard-codes" in "\n".join(pr) and "committed-code change" in "\n".join(pr), pr
    assert refused_by(pr, "C-RH-PAIR")
    # ... and the rest of the checks read clean: the pair check is the ONLY thing refusing (nothing is patched in process)
    assert not [x for x in pr if "check C-RH-PAIR" not in x], pr
    assert hashlib.sha256(open(PAIR_CHECK, "rb").read()).hexdigest() == "0dc30d9ef773be761a9b36a07d0fd73ef770e87d3384826230ba798db560869f"   # the committed 2bda9e1a file, untouched


# ---------------------------------------------------------------------------------------------------------------------------- 3. a bank header WITHOUT the generator record is REFUSED
def test_bank_header_without_generator_record_refused(rig):
    rig.make_bank("causal_g1x", with_generator=False)
    rh = rig.make_rh("causal_g1x")
    pr, _ = rig.mint("RH", rh_toks=rh)
    assert pr and refused_by(pr, "C-RH-BANKGEN"), pr
    # tie the verdict to the generator comparison alone: no other required check fails
    assert not [x for x in pr if "check C-RH-BANKGEN" not in x], pr
    assert "reads readout 'pinned'" in "\n".join(pr)


# ---------------------------------------------------------------------------------------------------------------------------- 4. a bank sha mismatch is REFUSED
def test_bank_sha_mismatch_refused(rig):
    rh = rig.setf(rig.rh_toks, "--dagger-bank-sha256", "0" * 64)
    pr, _ = rig.mint("RH", rh_toks=rh)
    assert pr, "an argv whose --dagger-bank-sha256 is not the sha256 of bank.json must not mint"
    txt = "\n".join(pr)
    assert "--dagger-bank-sha256" in txt or "bank.json sha256" in txt
    # and the RH-side check itself reads it (function level)
    assert NEW.rh_bank_generator_problems(rig.bank_dir, "0" * 64, "causal_g1x") and not NEW.rh_bank_generator_problems(rig.bank_dir, rig.bank_sha, "causal_g1x")
    # a bank whose bank.json bytes changed after the argv was written
    d = json.load(open(os.path.join(rig.bank_dir, "bank.json")))
    d["note"] = "edited"
    put_json(os.path.join(rig.bank_dir, "bank.json"), d)
    pr, _ = rig.mint("RH")
    assert pr


# ---------------------------------------------------------------------------------------------------------------------------- 5. B0 carrying ANY bank flag is REFUSED
@pytest.mark.parametrize("extra", [["--dagger-bank", "/x/bank"], ["--dagger-bank-sha256", "ab" * 32], ["--dagger-seed", "1"], ["--dagger-bank-readout", "causal_g1x"], ["--dagger-bank-kappa", "0.2"],
                                   ["--dagger-bank", "@BANK@", "--dagger-bank-sha256", "@BANKSHA@"]])
def test_b0_with_a_bank_flag_refused(rig, extra):
    b0 = rig.b0_toks + extra
    pr, _ = rig.mint("B0", b0_toks=b0)
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_b0_with_dagger_p_on_refused(rig):
    pr, _ = rig.mint("B0", b0_toks=rig.setf(rig.b0_toks, "--dagger-p", "0.1"))
    assert pr and refused_by(pr, "C-ARGV"), pr


def test_b0_takes_no_readout_token(rig):
    pr, _ = rig.mint("B0", token="causal_g1x")
    assert pr and refused_by(pr, "C-ARGV"), pr


# ---------------------------------------------------------------------------------------------------------------------------- 6. RH with a FIFTH changed flag is REFUSED
@pytest.mark.parametrize("how", ["lr", "steps", "added", "removed", "batch", "data_path", "kin"])
def test_rh_with_a_fifth_changed_flag_refused(rig, how):
    t = list(rig.rh_toks)
    if how == "lr":
        t = rig.setf(t, "--lr", "2e-4")
    elif how == "steps":
        t = rig.setf(t, "--steps", "3001")
    elif how == "added":
        t = t + ["--foo", "1"]
    elif how == "removed":
        t = rig.dropf(t, "--detach-upper")
    elif how == "batch":
        t = rig.setf(t, "--batch", "32")
    elif how == "data_path":
        t = rig.setf(t, "--tracks", rig.files["lead"])
    elif how == "kin":
        t = rig.setf(t, "--kin-mode", "xy")
    pr, _ = rig.mint("RH", rh_toks=t)
    assert pr and refused_by(pr, "C-ARGV"), (how, pr)


def test_the_unchanged_rh_argv_is_not_refused_by_the_fifth_flag_rule(rig):
    assert rig.mint("RH")[0] == []


# ---------------------------------------------------------------------------------------------------------------------------- 7. an init checkpoint that is NOT the arm's stage A is REFUSED
@pytest.mark.parametrize("which", ["RH", "B0"])
@pytest.mark.parametrize("bad", ["other_seed", "placeholder", "stage_b_ckpt"])
def test_init_that_is_not_the_arms_stage_a_refused(rig, which, bad):
    wrong = {"other_seed": os.path.join(rig.tmp, "pod", "runs", "UE_s1", "A", "ckpt_last.pt"), "placeholder": "@INIT@",
             "stage_b_ckpt": os.path.join(rig.tmp, "pod", "runs", "UE_s0", "B", "ckpt_last.pt")}[bad]
    os.makedirs(os.path.dirname(wrong), exist_ok=True) if not wrong.startswith("@") else None
    if not wrong.startswith("@"):
        open(wrong, "wb").write(open(rig.init, "rb").read())              # a real file with the SAME bytes: only the NAME can refuse it
    if which == "RH":
        pr, _ = rig.mint("RH", rh_toks=rig.setf(rig.rh_toks, "--init-ckpt", wrong))
    else:
        pr, _ = rig.mint("B0", b0_toks=rig.setf(rig.b0_toks, "--init-ckpt", wrong))
    assert pr and refused_by(pr, "C-ARGV"), pr
    assert "K-B2" in "\n".join(pr) or "not filled" in "\n".join(pr)


def test_bank_made_from_another_checkpoint_refused(rig):
    open(rig.init, "ab").write(b"changed")                                # the bank's recorded ckpt_sha256 is now not this init (K-B2, the existing C-BANK)
    pr, _ = rig.mint("RH")
    assert pr and refused_by(pr, "C-BANK"), pr


# ---------------------------------------------------------------------------------------------------------------------------- 8. IDENTITY: the new file's UE path equals the original's, field for field
def _strip(tok):
    return {k: v for k, v in tok.items() if k not in ("at", "checker_sha256", "body_sha256")}


def test_identity_ue_s0_stage_b_token_equals_the_original(rig):
    """A UE_s0 stage-B argv (the U_s0.B template filled with UE_s0's own checkpoint and bank) minted through the original reps1 file and through dscale10k_check_rh.py: equal token, field for field.
    ``at`` (the clock), ``checker_sha256`` and ``body_sha256`` (which hashes the previous) differ BY DESIGN: they are the file's own identity."""
    t = rig.tmpl_toks
    t = rig.setf(t, "--out", os.path.join(rig.tmp, "runs", "UE_s0", "B"))
    t = rig.setf(t, "--init-ckpt", rig.init)
    t = rig.setf(t, "--dagger-bank", rig.bank_dir)
    t = rig.setf(t, "--dagger-bank-sha256", rig.bank_sha)
    af = rig.argv_file("UE_s0.B", t)
    out = {}
    for name, mod in (("orig", ORIG), ("new", NEW)):
        rig.patch_heavy(mod)
        a = rig.mint_ns("UE_s0", af, "UE_" + name, selfcheck=rig.sc_rh, template="", pair_check="")
        pr = mod.mint(a)
        assert pr == [], (name, pr)
        out[name] = json.load(open(os.path.join(a.gate_dir, "gate_token.json")))
    assert out["orig"]["checker_sha256"] != out["new"]["checker_sha256"]
    assert _strip(out["orig"]) == _strip(out["new"])
    assert set(out["orig"]) == set(out["new"]) and set(out["orig"]["checks"]) == set(out["new"]["checks"])
    # the same through the runner's verifier: each file's token verifies under its own file only
    assert ORIG.token_problems(os.path.join(rig.tmp, "gate", "UE_orig", "gate_token.json"), "UE_s0", "B", t, rig.root, COMMIT, {"evidence": rig.ev, "dry_record": rig.dry, "selfcheck": rig.sc_rh,
                               "merge_record": rig.files["merge_rec"], "--union-release": rig.files["urel"], "tenk_v9": rig.files["tenk"]}) == []


def test_identity_argv_diff_and_required_checks_unchanged_for_every_original_arm():
    base, tmpl = fx("B_s0.B.tmpl.json"), fx("U_s0.B.tmpl.json")
    for arm in sorted(set(ORIG.ARMS)):
        for stage in ("A", "B"):
            assert ORIG.required_checks(arm, stage) == NEW.required_checks(arm, stage), (arm, stage)
            assert ORIG.argv_diff_problems(base, tmpl, arm, stage, 76656) == NEW.argv_diff_problems(base, tmpl, arm, stage, 76656), (arm, stage)
    assert set(NEW.ARMS) - set(ORIG.ARMS) == {"RH_s0", "RH_s1", "B0_s0", "B0_s1"}
    for arm in ORIG.ARMS:
        assert ORIG.ARMS[arm] == NEW.ARMS[arm]


def test_the_real_pod_templates_pass_the_existing_stage_b_diff():
    base = fx("B_s0.B.tmpl.json")
    assert NEW.argv_diff_problems(base, fx("U_s0.B.tmpl.json"), "U_s0", "B") == []
    assert NEW.argv_diff_problems(base, fx("U_s1.B.tmpl.json"), "U_s1", "B") == []


# ---------------------------------------------------------------------------------------------------------------------------- the readout spelling, the seed-1 pair, the runner
def test_readout_token_spelling_equals_the_committed_module():
    sys.path.insert(0, os.path.join(REF, "tree", "stack"))
    try:
        from tanitad.data import v8f_dagger as D
    finally:
        sys.path.pop(0)
    for ro, g, px in [("pinned", None, False), ("causal_g", 1.0, True), ("causal_g", 1.0, False), ("causal_g", 0.5, False), ("causal_g", 2, True), ("causal_g", 0.0, False)]:
        assert NEW.readout_token(ro, g, px) == D.readout_token(ro, g, px)
    for bad in [("pinned", 1.0, False), ("pinned", None, True), ("causal_g", None, False), ("nope", 1.0, False)]:
        with pytest.raises(ValueError):
            NEW.readout_token(*bad)
        with pytest.raises(ValueError):
            D.readout_token(*bad)


def test_seed_1_pair_mints(tmp_path):
    r = Rig(tmp_path / "s1", seed=1)
    assert r.mint("RH")[0] == [] and r.mint("B0")[0] == []
    other = Rig(tmp_path / "s1b", seed=1)
    pr, _ = other.mint("RH", rh_toks=other.setf(other.rh_toks, "--init-ckpt", os.path.join(other.tmp, "pod", "runs", "UE_s0", "A", "ckpt_last.pt")))
    assert pr


def test_pair_mismatch_refused(rig):
    b0 = rig.setf(rig.b0_toks, "--lr", "2e-4")                            # B0 moved a lever: refused by C-ARGV (template) and by the pair check
    pr, _ = rig.mint("B0", b0_toks=b0)
    assert pr
    pr, _ = rig.mint("RH", b0_toks=b0)
    assert pr and refused_by(pr, "C-RH-PAIR"), pr


def test_stage_a_refused_for_rh(rig):
    rig.patch_heavy(NEW)
    a = rig.mint_ns("RH_s0", rig.argv_file("RH_s0", rig.rh_toks), "RH_A", pair=rig.argv_file("B0_s0", rig.b0_toks), token="causal_g1x", selfcheck=rig.sc_rh, stage="A")
    assert NEW.mint(a)


def test_selfcheck_levers_must_carry_the_declaration(rig):
    sc = json.load(open(rig.sc_rh))
    sc["levers"]["dagger"]["dagger_bank_readout"] = "pinned"
    pr, _ = rig.mint("RH", sc_rh=put_json(os.path.join(rig.tmp, "sc_bad.json"), sc))
    assert pr and refused_by(pr, "G-DAGGER"), pr
    scb = json.load(open(rig.sc_b0))
    scb["levers"]["dagger"]["dagger_bank_readout"] = "causal_g1x"
    pr, _ = rig.mint("B0", sc_b0=put_json(os.path.join(rig.tmp, "scb_bad.json"), scb))
    assert pr and refused_by(pr, "G-DAGGER-B0"), pr


# ---------------------------------------------------------------------------------------------------------------------------- the CLI the chain calls: mint, then run (the runner starts the trainer ONLY on the token)
def _cli(rig, sub, arm, toks_path, pair_path, gate, token, selfcheck, with_mint_only=True):
    a = [sub, "--arm", arm, "--stage", "B", "--argv", toks_path, "--root", rig.root, "--commit", COMMIT, "--gate-dir", gate, "--py", sys.executable, "--gpu", "0", "--evidence", rig.ev, "--dry", rig.dry,
         "--merge-record", rig.files["merge_rec"], "--union-release", rig.files["urel"], "--tenk-v9", rig.files["tenk"], "--selfcheck", selfcheck, "--template", rig.template, "--pair-argv", pair_path,
         "--pair-check", PAIR_CHECK, "--readout-token", token]
    if sub == "mint":
        a += ["--base-argv", rig.base, "--corpus-join", rig.corpus_join, "--union-join", rig.union_join, "--eval-ids", rig.eval_ids, "--holdout-manifest", rig.holdout_manifest, "--holdout", "0:3"]
    return a


def test_cli_mint_then_run_starts_the_trainer_only_on_the_token(rig, capsys):
    rig.patch_heavy(NEW)
    open(os.path.join(rig.root, "stack", "scripts", "v8f_mini_train.py"), "w").write("import sys, os\nopen(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'STARTED'), 'w').write(' '.join(sys.argv[1:4]))\n")
    rp, bp = rig.argv_file("RH_cli", rig.rh_toks), rig.argv_file("B0_cli", rig.b0_toks)
    gate = os.path.join(rig.tmp, "gate", "RH_cli")
    started = os.path.join(rig.root, "stack", "scripts", "STARTED")
    assert NEW.main(_cli(rig, "mint", "RH_s0", rp, bp, gate, "causal_g1x", rig.sc_rh)) == 0
    assert "TOKEN-WRITTEN RH_s0" in capsys.readouterr().out
    assert NEW.main(_cli(rig, "run", "RH_s0", rp, bp, gate, "causal_g1x", rig.sc_rh)) == 0
    assert os.path.isfile(started)
    os.remove(started)
    # a wrong declared token at run time, an edited argv, and a missing token each REFUSE with rc 2 and start nothing
    assert NEW.main(_cli(rig, "run", "RH_s0", rp, bp, gate, "causal_g1", rig.sc_rh)) == 2
    edited = rig.argv_file("RH_edit", rig.setf(rig.rh_toks, "--lr", "2e-4"))
    assert NEW.main(_cli(rig, "run", "RH_s0", edited, bp, gate, "causal_g1x", rig.sc_rh)) == 2
    assert NEW.main(_cli(rig, "run", "RH_s0", rp, bp, os.path.join(rig.tmp, "gate", "none"), "causal_g1x", rig.sc_rh)) == 2
    assert not os.path.isfile(started)
