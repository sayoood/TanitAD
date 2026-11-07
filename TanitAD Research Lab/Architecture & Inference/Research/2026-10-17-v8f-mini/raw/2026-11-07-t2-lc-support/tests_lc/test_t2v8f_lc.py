"""LC (the LoRA arm) support for the stage-5 driver (AGENT LC): the fold, its pin, the cell install, the manifest rules, the builder, the frozen path.

Every `check_*(mut=None)` is one arm; `mut="<name>"` makes the SAME assertion fail so the arm can be shown RED (the `test_red_*` functions assert exactly that).
  a  identity     B = 0: the folded weights equal the frozen weights and the folded InLoopEncoder's tokens equal the frozen InLoopEncoder's, byte for byte (the LoRA card's identity property)
  b  lever        B != 0: exactly the 48 q/v tensors change (every other parameter identical) and the tokens move
  c  training     the T2 side's merged tokens equal the TRAINING side's `merged_encoder().pooled` on a banked frame, and its folded weights equal `merge()`'s      RED: skip_fold
  d  guards       a skipped fold, a corrupted pin, a wrong blob, a hook that never fires -> each refused by name
  e  frozen       the frozen path is code/'s: same refusals on the real EGAIN manifests, same order, `InLoopEncoder` untouched, unknown kinds still refused
  f  builder      `make_stage5.build(encoder="lora")` on the REAL stage-4 tar with synthetic L blobs: check_manifest5 clean, pins independently recomputed, tar members, LF-only
Run from t2v8f/:  CUDA_VISIBLE_DEVICES=-1 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest tests_lc/test_t2v8f_lc.py -q
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tarfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import lc_common as C  # noqa: E402  (puts code_lc/ first, code/ after)

import pytest  # noqa: E402
import torch  # noqa: E402

import t2v8f_stage5 as D5  # noqa: E402  (code_lc's)
import t2v8f_stage5_cell as CELL  # noqa: E402  (code_lc's)

LM = C.LM
assert os.path.dirname(os.path.abspath(D5.__file__)).replace("\\", "/") == C.LC and os.path.dirname(os.path.abspath(CELL.__file__)).replace("\\", "/") == C.LC
B_STD = 0.01                                                    # N(0, 0.01^2) B: a delta of ~3 % of |W|, large enough to move tokens, small enough to stay finite in fp8
_S = {}


def old_stage5():
    """code/'s t2v8f_stage5 under another name (the file the running MC arm was built from)."""
    if "old" not in _S:
        spec = importlib.util.spec_from_file_location("t2v8f_stage5_OLD", f"{C.CODE}/t2v8f_stage5.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _S["old"] = m
    return _S["old"]


def blobs():
    """Two synthetic L blobs (the real trained heads of two banked checkpoints + a LoRA with B ~ N(0, B_STD^2), different A and B per seed) and the zero-B blob."""
    if "blobs" not in _S:
        d = f"{C.SCRATCH}/blobs_test"
        _S["blobs"] = {"LC_s0": C.write_blob(f"{d}/LC_s0.pt", C.lora_sd(0, B_STD, a_seed=0), 0), "LC_s1": C.write_blob(f"{d}/LC_s1.pt", C.lora_sd(1, B_STD, a_seed=1), 1),
                       "ZERO": C.write_blob(f"{d}/ZERO.pt", C.lora_sd(0, 0.0, a_seed=0), 0)}
    return _S["blobs"]


def pins():
    if "pins" not in _S:
        b = blobs()
        _S["pins"] = {a: LM.build_pin_for(p, base_model=C.base_model()) for a, p in b.items()}
    return _S["pins"]


def block(arms=("LC_s0", "LC_s1")):
    b, p = blobs(), pins()
    return C.make_block({a: b[a] for a in arms}, {a: p[a] for a in arms})


# ============================================================================================================================ a. identity (B = 0)
def check_identity_b0(mut=None):
    b, p = blobs(), pins()
    base_d = LM.model_digest(C.base_model())
    zero = p["LC_s0" if mut == "b_nonzero" else "ZERO"]                 # the control must be able to FAIL: the non-zero LoRA is not the identity
    assert zero["n_b_zero_modules"] == 48 and zero["b_abs_max"] == 0.0, f"B is not zero ({zero['n_b_zero_modules']} zero modules, max|B| {zero['b_abs_max']})"
    blk = C.make_block({"ZERO": b["ZERO"]}, {"ZERO": zero})
    assert zero["merged_qv_tensor_sha256"] == base_d["tensor_sha256"], "B = 0 changed a q/v weight"
    tok, enc = C.lora_tokens(b["ZERO"], blk, "ZERO")
    assert C.fp8_bytes(tok) == C.fp8_bytes(C.frozen_tokens()), "B = 0 folded encoder's tokens are not the frozen encoder's, byte for byte"
    return {"tokens_equal_bytes": True, "report": enc.lora_fold}


# ============================================================================================================================ b. a non-zero B moves exactly the 48 tensors
def check_lever_moves(mut=None):
    p = pins()["LC_s0"]
    base = C.base_model()
    base_d = LM.model_digest(base)
    changed = sorted(k for k, v in p["merged_qv_tensor_sha256"].items() if v != base_d["tensor_sha256"][k])
    assert len(changed) == 48 if mut != "no_change" else len(changed) == 0, f"{len(changed)} of 48 q/v tensors changed"
    m = __import__("copy").deepcopy(base)
    parsed = LM.parse_lora(LM.load_blob(blobs()["LC_s0"])["lora"])
    LM.fold_lora(m, parsed)
    a, b = dict(base.named_parameters()), dict(m.named_parameters())
    other = [n for n in a if not LM._W_NAME.fullmatch(n) or n.split(".")[-2][0] not in "qv"]
    assert other and all(torch.equal(a[n], b[n]) for n in other), "a parameter outside q/v moved"
    tok, _ = C.lora_tokens(blobs()["LC_s0"], block(("LC_s0",)), "LC_s0")
    rl2 = C.rel_l2(tok, C.frozen_tokens())
    assert rl2 > 1e-3, f"the non-zero B did not move the tokens (rel-L2 {rl2})"
    assert bool(torch.isfinite(tok.float()).all())
    return {"changed_tensors": len(changed), "other_params_equal": len(other), "tokens_rel_l2_vs_frozen": rl2}


# ============================================================================================================================ c. equals the training side
def check_merged_equals_training(mut=None):
    """T2 side (`InLoopEncoder` + the fold) vs TRAINING side (`v8f_lora_encoder.merged_encoder()`), same CPU, same frame, same batch (1). EXPECTED exact (epsilon 0): both run the identical bf16 CPU forward on identical
    weights, the same normalise ops and the same fp32 pool -> fp8. The card's tau_device 1e-3 (rel-L2 on pooled tokens) is the bar for a DIFFERENT device; on one CPU any difference is a defect."""
    blob = blobs()["LC_s0"]
    tr = C.train_encoder(0)
    tr.load_lora_state_dict(LM.load_blob(blob)["lora"])
    ref_enc = tr.merged_encoder()
    wd = LM.model_digest(ref_enc.backbone)["merged_qv_sha256"]
    assert wd == pins()["LC_s0"]["merged_qv_sha256"], "the fold's weights are not the training side's merge() weights"
    ref = ref_enc.pooled(C.banked_frame()[None])[0]
    if mut == "skip_fold":                                       # the deliberate regression: the fold is a no-op, the (otherwise valid) pin is not asserted
        orig_fold, orig_assert = LM.fold_lora, LM.assert_pin
        LM.fold_lora, LM.assert_pin = (lambda *a, **k: []), (lambda *a, **k: {"merged_qv_sha256": "skipped"})
        try:
            tok, _ = C.lora_tokens(blob, block(("LC_s0",)), "LC_s0")
        finally:
            LM.fold_lora, LM.assert_pin = orig_fold, orig_assert
    else:
        tok, _ = C.lora_tokens(blob, block(("LC_s0",)), "LC_s0")
    a, b = tok.float(), ref.float()
    rl2, mx = C.rel_l2(a, b), float((a - b).abs().max())
    assert C.fp8_bytes(tok) == C.fp8_bytes(b.to(torch.float8_e4m3fn)), f"T2 tokens != training-side merged tokens: rel-L2 {rl2}, max|d| {mx}"
    return {"epsilon_rel_l2": rl2, "max_abs": mx, "bytes_equal": True}


# ============================================================================================================================ d. guards
def check_guards(mut=None):
    """Each fault must be REFUSED by name before a rollout can use the encoder. `mut="no_guard"` disables the assertions so the same checks fail (RED)."""
    blob, blk = blobs()["LC_s0"], block(("LC_s0",))
    got = []
    import transformers
    import t2a_v8f_encoder as E
    cls_before, fp_before = E.InLoopEncoder, vars(transformers.AutoModel).get("from_pretrained", "ABSENT")

    def refused(label, want, fn):
        try:
            fn()
        except (AssertionError, RuntimeError, ValueError) as e:
            assert want in str(e), f"{label}: refused, but not by {want!r}: {str(e)[:300]}"
            got.append(label)
            return
        raise AssertionError(f"{label}: NOT refused (expected {want})")
    # 1. the fold is skipped (hook never folds): the in-hook digest check names it
    def skip():
        orig = LM.fold_lora
        LM.fold_lora = lambda *a, **k: []
        try:
            C.lora_tokens(blob, blk, "LC_s0")
        finally:
            LM.fold_lora = orig
    refused("fold_skipped", "MERGED_SHA_MISMATCH (after the fold", skip)
    # 2. the in-hook check is bypassed AND the fold skipped: the finished-encoder check still refuses
    def skip_both():
        of, oa = LM.fold_lora, LM.assert_pin
        LM.fold_lora = lambda *a, **k: []
        LM.assert_pin = lambda m, e, arm, where: oa(m, e, arm, where) if where.startswith("on the finished") else {"merged_qv_sha256": "x"}
        try:
            C.lora_tokens(blob, blk, "LC_s0")
        finally:
            LM.fold_lora, LM.assert_pin = of, oa
    refused("fold_skipped_hook_unchecked", "MERGED_SHA_MISMATCH (on the finished encoder)", skip_both)
    # 3. one tensor's pinned digest is wrong: refused, and the tensor is NAMED
    bad = json.loads(json.dumps(blk))
    bad["merged_qv_tensor_sha256"]["LC_s0"]["layer.5.attention.v_proj.weight"] = "0" * 64
    bad["merged_qv_sha256"]["LC_s0"] = LM.digest_of(bad["merged_qv_tensor_sha256"]["LC_s0"])
    refused("pin_tensor_wrong", "layer.5.attention.v_proj.weight", lambda: C.lora_tokens(blob, bad, "LC_s0"))
    # 4. the blob on disk is not the pinned file
    other = blobs()["LC_s1"]
    refused("wrong_blob", "HASH_MISMATCH_LORA_SOURCE", lambda: C.lora_tokens(other, blk, "LC_s0"))
    # 5. a model that is not on CPU/bf16 cannot be folded (the fold must come BEFORE .to(device))
    m32 = __import__("copy").deepcopy(C.base_model()).float()
    refused("wrong_place", "FOLD_WRONG_PLACE", lambda: LM.fold_lora(m32, LM.parse_lora(LM.load_blob(blob)["lora"])))
    # 6. the LoRA entry is malformed
    sd = LM.load_blob(blob)["lora"]
    short = {"meta": sd["meta"], "tensors": dict(list(sd["tensors"].items())[:-2])}
    refused("lora_missing_tensors", "LORA_TENSOR_SET", lambda: LM.parse_lora(short))
    wrong_r = {"meta": dict(sd["meta"], r=16), "tensors": sd["tensors"]}
    refused("lora_meta_r", "LORA_META_MISMATCH", lambda: LM.parse_lora(wrong_r))
    # 7. the encoder restored its class and from_pretrained afterwards
    assert E.InLoopEncoder is cls_before and vars(transformers.AutoModel).get("from_pretrained", "ABSENT") is fp_before, "lora_fold_active did not restore InLoopEncoder / AutoModel.from_pretrained"
    if mut == "no_guard":
        raise AssertionError("no_guard: guards are disabled")
    return {"refused": got}


# ============================================================================================================================ the manifest rules
def manifest_for(enc_block, name="LC"):
    arms = list(D5.arm_names(name))
    P = D5.passes5(name)
    return {"stage": 5, "arms": {"deciding": arms}, "cell_priority": [list(x) for x in P["deciding"]], "beside_order": [list(x) for x in P["beside"]],
            "arm5": {"name": name, "tag": "", "dose": None, "run": name, "arms": arms, "encoder": enc_block, "ckpt_file": {a: f"{a}.pt" for a in arms}, "ckpt_sha256": dict(enc_block.get("source_ckpt_sha256") or {})}}


def check_manifest_rules(mut=None):
    blk = block()
    arms = list(D5.arm_names("LC"))
    assert D5.check_arm5(manifest_for(blk)) == [], D5.check_arm5(manifest_for(blk))
    cp = lambda: json.loads(json.dumps(blk))              # noqa: E731
    faults = []
    a = cp(); a["kind"] = "dinov3_lora"; faults.append((a, "ENCODER_UNSUPPORTED"))
    b = cp(); b["kind"] = None; faults.append((b, "ENCODER_UNSUPPORTED"))
    c = cp(); c["merged_qv_sha256"].pop("LC_s1"); faults.append((c, "LORA_BAD_FIELD merged_qv_sha256"))
    d = cp(); d["targets"] = ["q", "k", "v"]; faults.append((d, "LORA_BAD_FIELD targets"))
    e = cp(); e["base_weights_sha256"] = "1" * 64; faults.append((e, "LORA_BAD_FIELD base_weights_sha256"))
    f = cp(); f["merged_qv_sha256"]["LC_s0"] = "2" * 64; faults.append((f, "LORA_PIN_INCONSISTENT LC_s0"))
    g = cp(); g["merged_qv_tensor_sha256"]["LC_s1"].pop("layer.3.attention.q_proj.weight"); faults.append((g, "LORA_BAD_FIELD merged_qv_tensor_sha256[LC_s1]"))
    h = cp(); h["blocks"] = 12; faults.append((h, "LORA_BAD_FIELD blocks"))
    for bad, want in faults:
        got = D5.check_arm5(manifest_for(bad if mut != "not_applied" else blk))
        assert any(x.startswith(want) for x in got), f"{want} was not named: {got}"
    m = manifest_for(blk)
    m["arm5"]["ckpt_sha256"]["LC_s0"] = "3" * 64
    got = D5.check_arm5(m if mut != "not_applied" else manifest_for(blk))
    assert any(x.startswith("LORA_SOURCE_MISMATCH") for x in got), got
    return {"faults": len(faults) + 1}


# ============================================================================================================================ e. the frozen path is code/'s
def check_frozen_path(mut=None):
    old = old_stage5()
    man = {}
    for dose in ("a05", "a10"):
        with tarfile.open(f"{C.T2}/stage5/EGAIN_{dose}/t2v8f_stage5_overlay_EGAIN_{dose}.tar") as tf:
            man[dose] = json.loads(tf.extractfile("pkg/MANIFEST5.json").read())
    m = man["a10"]
    cp = lambda: json.loads(json.dumps(m))             # noqa: E731
    variants = [m, man["a05"]]
    for fn in (lambda x: x.update(stage=4), lambda x: x["arm5"].update(name="C0C"), lambda x: x["arm5"].update(name="EGAIN_s0"), lambda x: x["arm5"]["ckpt_file"].update(EGAIN_s1="../x.pt"),
               lambda x: x["arm5"]["ckpt_sha256"].pop("EGAIN_s0"), lambda x: x["scenes"][5].update(idx=7), lambda x: x["requires"].pop("code/t2v8f_stage4.py"),
               lambda x: x["cell_priority"].reverse(), lambda x: x["arm5"].pop("encoder"), lambda x: x["arm5"]["encoder"].update(kind="something_else"), lambda x: x["arm5"]["encoder"].update(kind="dinov3_lora")):
        v = cp()
        fn(v)
        variants.append(v)
    for i, v in enumerate(variants):
        new, was = D5.check_manifest5(v), old.check_manifest5(v)
        if any(x.startswith("ENCODER_UNSUPPORTED") for x in was):               # the reason's TEXT differs (it dropped the "not built" clause); the CODE and every other reason must be equal
            assert any(x.startswith("ENCODER_UNSUPPORTED") for x in new), f"variant {i}: an unknown kind is accepted: {new}"
            new, was = [x for x in new if not x.startswith("ENCODER_UNSUPPORTED")], [x for x in was if not x.startswith("ENCODER_UNSUPPORTED")]
        assert (new == was) == (mut != "diverge"), f"variant {i}: code_lc gives {new}, code gives {was}"
    unk = cp(); unk["arm5"]["encoder"]["kind"] = "dinov3_lora"
    assert any(x.startswith("ENCODER_UNSUPPORTED") for x in D5.check_manifest5(unk)), "an unknown encoder kind is accepted by the driver"
    scenes = m["scenes"]
    assert D5.cell_order5(scenes, "C0C") == old.cell_order5(scenes, "C0C") and D5.cell_order5(scenes, "EGAIN") == old.cell_order5(scenes, "EGAIN") and len(D5.cell_order5(scenes, "LC")) == 248
    assert (D5.arm_names("X"), D5.passes5("X"), D5.NAME_RESERVED, D5.REQUIRED_STAGE2, D5.REQUIRED_STAGE4) == (old.arm_names("X"), old.passes5("X"), old.NAME_RESERVED, old.REQUIRED_STAGE2, old.REQUIRED_STAGE4)
    # the cell: a frozen manifest takes the unchanged path -- InLoopEncoder is never replaced, the LoRA module is never imported, a LoRA wrapper is not installed
    import t2a_v8f_encoder as E
    saved_lm = sys.modules.pop("t2v8f_lora_merge", None)
    cls0, init0 = E.InLoopEncoder, E.InLoopEncoder.__init__
    R = types.SimpleNamespace(ARMS=("P10",), CKPT_FILE={}, build_arm=lambda args: types.SimpleNamespace(inner=object()))
    import t2v8f_c0c as C0C
    seen = []
    orig_cp = C0C.CompositePolicy
    C0C.CompositePolicy = lambda inner: seen.append(inner) or ("composite", inner)
    try:
        CELL.install5(R, "EGAIN", {"EGAIN_s0": "a.pt", "EGAIN_s1": "b.pt"}, {"kind": "frozen_dinov3"})
        pol = R.build_arm(types.SimpleNamespace(arm="EGAIN_s0", t2_encoder="dinov3", t2_ckpt_root="x"))
        assert pol.inner[0] == "composite" and len(seen) == 1 and E.InLoopEncoder is cls0 and E.InLoopEncoder.__init__ is init0
        assert "t2v8f_lora_merge" not in sys.modules, "the frozen path imported the LoRA module"
    finally:
        C0C.CompositePolicy = orig_cp
        if saved_lm is not None:
            sys.modules["t2v8f_lora_merge"] = saved_lm
    for bad_kind in ("dinov3_lora", None, "frozen"):
        R2 = types.SimpleNamespace(ARMS=("P10",), CKPT_FILE={}, build_arm=lambda args: None)
        with pytest.raises(SystemExit, match="ENCODER_UNSUPPORTED"):
            CELL.install5(R2, "EGAIN2", {"EGAIN2_s0": "a.pt", "EGAIN2_s1": "b.pt"}, {"kind": bad_kind})
    mp = f"{C.SCRATCH}/m5_unknown_kind.json"
    json.dump({"arm5": {"name": "X", "ckpt_file": {"X_s0": "a"}, "encoder": {"kind": "dinov3_lora"}}}, open(mp, "w"))
    with pytest.raises(SystemExit, match="ENCODER_UNSUPPORTED"):
        CELL.arm5_from_manifest(mp)
    return {"variants": len(variants)}


# ============================================================================================================================ f. the builder, on the REAL stage-4 tar
def lc_make_stage5():
    """code_lc's make_stage5 BY PATH (code/ also holds a make_stage5, and loading code/'s stage-5 driver for the differential test puts code/ first on sys.path)."""
    if "m5" not in _S:
        sys.path.insert(0, C.LC)
        spec = importlib.util.spec_from_file_location("make_stage5_LC", f"{C.LC}/make_stage5.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert os.path.dirname(os.path.abspath(m.D5.__file__)).replace("\\", "/") == C.LC
        _S["m5"] = m
    return _S["m5"]


def check_builder(mut=None):
    M5 = lc_make_stage5()
    b = blobs()
    root = f"{C.SCRATCH}/build_lc_{time.strftime('%H%M%S')}"
    res = M5.build(f"{root}/out", f"{root}/stage", "LC", "t", {0: b["LC_s0"], 1: b["LC_s1"]}, "lora", encoder="lora", base_model=C.base_model())
    with tarfile.open(res["tar"]) as tf:
        names = sorted(tf.getnames())
        man = json.loads(tf.extractfile("pkg/MANIFEST5.json").read())
        code = {n: tf.extractfile(n).read() for n in names if n.startswith("pkg/code/")}
    assert names == sorted(["pkg/MANIFEST5.json", "pkg/ckpt/LC_s0.pt", "pkg/ckpt/LC_s1.pt", "pkg/code/t2v8f_lora_merge.py", "pkg/code/t2v8f_stage5.py", "pkg/code/t2v8f_stage5_cell.py", "pkg/jobs/stage5_LC_t.cmd"]), names
    assert D5.check_manifest5(man) == [], D5.check_manifest5(man)
    enc = man["arm5"]["encoder"]
    assert enc["kind"] == "dinov3_lora_merged" and enc["source_ckpt_sha256"] == man["arm5"]["ckpt_sha256"] == res["ckpt_sha256"] and res["merged_qv_sha256"] == enc["merged_qv_sha256"]
    for arm, blob in (("LC_s0", b["LC_s0"]), ("LC_s1", b["LC_s1"])):             # the pins, independently recomputed with the TRAINING side's merge()
        tr = C.train_encoder(0 if arm == "LC_s0" else 1)
        tr.load_lora_state_dict(LM.load_blob(blob)["lora"])
        assert LM.model_digest(tr.merge())["merged_qv_sha256"] == enc["merged_qv_sha256"][arm], f"{arm}: the builder's pin is not the training side's merge()"
    for n, data in code.items():
        assert b"\r" not in data and hashlib.sha256(data).hexdigest() == next(e["sha256"] for e in man["files"] if f"pkg/{e['path']}" == n), n
        assert data == open(f"{C.LC}/{os.path.basename(n)}", "rb").read()
    assert enc["merged_qv_sha256"]["LC_s0"] != enc["merged_qv_sha256"]["LC_s1"]
    return {"tar": res["tar"], "tar_bytes": res["tar_bytes"], "merged_qv_sha256": enc["merged_qv_sha256"], "ckpt_sha256": res["ckpt_sha256"]}


def check_frozen_build_equals_code_build(mut=None):
    """`code_lc/make_stage5.py build` with the frozen encoder and the EGAIN a10 checkpoints gives the manifest `code/make_stage5.py` gave (stage5/EGAIN_a10), field for field, except the built-at time and the sha256 / size of
    the two shipped code files (code_lc's drivers). The checkpoints, job file, scenes, requires, arm table and cell order are identical."""
    M5 = lc_make_stage5()
    wsp = "E:/Projects/TanitAD-artifacts/v8f-wspedit"
    root = f"{C.SCRATCH}/build_frozen_{time.strftime('%H%M%S')}"
    cks = {0: f"{wsp}/M2b_s0_wspedit_a10/ckpt_last.pt", 1: f"{wsp}/M2b_s1_wspedit_a10/ckpt_last.pt"}
    res = M5.build(f"{root}/out", f"{root}/stage", "EGAIN", "a10", cks, "a10")
    with tarfile.open(res["tar"]) as tf:
        new = json.loads(tf.extractfile("pkg/MANIFEST5.json").read())
        new_names = sorted(tf.getnames())
    with tarfile.open(f"{C.T2}/stage5/EGAIN_a10/t2v8f_stage5_overlay_EGAIN_a10.tar") as tf:
        old = json.loads(tf.extractfile("pkg/MANIFEST5.json").read())
        old_names = sorted(tf.getnames())
    assert new_names == old_names, (new_names, old_names)
    for k in ("arm5", "scenes", "requires", "cell_priority", "beside_order", "arms", "stage1_manifest_sha256", "stage1_tar_sha256", "totals", "start_classes", "ri_tool", "scene_roots_box2", "source_manifests", "t2_conds", "purpose"):
        assert (new[k] == old[k]) == (mut != "diverge"), f"manifest key {k}: code_lc's and code/'s frozen builds {'differ' if new[k] != old[k] else 'agree'}"
    nf, of = {e["path"]: e for e in new["files"]}, {e["path"]: e for e in old["files"]}
    assert sorted(nf) == sorted(of)
    same = [p for p in nf if nf[p] == of[p]]
    differ = sorted(p for p in nf if nf[p] != of[p])
    assert differ == ["code/t2v8f_stage5.py", "code/t2v8f_stage5_cell.py"], f"files that differ: {differ}"
    assert new["arm5"]["encoder"]["kind"] == "frozen_dinov3" and "t2v8f_lora_merge" not in json.dumps(new["files"])
    return {"identical_files": len(same), "differ": differ}


# ============================================================================================================================ the tests
def test_a_identity_b0():
    check_identity_b0()


def test_b_lever_moves_48_tensors():
    check_lever_moves()


def test_c_merged_equals_training_side():
    check_merged_equals_training()


def test_d_guards():
    check_guards()


def test_manifest_rules():
    check_manifest_rules()


def test_e_frozen_path_is_code():
    check_frozen_path()


def test_f_builder():
    check_builder()


def test_f2_frozen_build_equals_code_build():
    check_frozen_build_equals_code_build()


# ---- the deliberate-regression arms: the SAME assertions must fail
def test_red_identity_nonzero_b_is_not_identity():
    with pytest.raises(AssertionError):
        check_identity_b0("b_nonzero")


def test_red_lever_no_change():
    with pytest.raises(AssertionError):
        check_lever_moves("no_change")


def test_red_skip_fold_breaks_token_equality():
    with pytest.raises(AssertionError):
        check_merged_equals_training("skip_fold")


def test_red_guards_disabled():
    with pytest.raises(AssertionError):
        check_guards("no_guard")


def test_red_manifest_faults_not_applied():
    with pytest.raises(AssertionError):
        check_manifest_rules("not_applied")


def test_red_frozen_path_diverged():
    with pytest.raises(AssertionError):
        check_frozen_path("diverge")


def test_red_builder_refuses_mixed_flags():
    b = blobs()
    M5 = lc_make_stage5()
    with pytest.raises(SystemExit):                                   # --encoder lora with a --ckpt-s0: refused before anything is built
        M5.main(["build", "--arm", "LC", "--encoder", "lora", "--lora-ckpt-s0", b["LC_s0"], "--lora-ckpt-s1", b["LC_s1"], "--ckpt-s0", b["LC_s0"]])
    with pytest.raises(SystemExit):                                   # --encoder frozen with a --lora-ckpt
        M5.main(["build", "--arm", "LC", "--ckpt-s0", b["LC_s0"], "--ckpt-s1", b["LC_s1"], "--lora-ckpt-s0", b["LC_s0"]])


def test_red_frozen_build_diverged():
    with pytest.raises(AssertionError):
        check_frozen_build_equals_code_build("diverge")
