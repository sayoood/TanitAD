"""Tests of tok_probe_u.py (card PREREG_V8F_PROOF_ADD45 DSCALE-LEAD with note N1). CPU, torch, tiny synthetic world (6 train clips of 60 frames, 2 eval clips). NOT RUN by the builder: the lead runs it ON THE POD.

Run (the tree is the shipped study tree, the folder that holds stack/):
  CUDA_VISIBLE_DEVICES=-1 DSCALE_TREE=/workspace/v8proof/tree_leadsup PYTHONDONTWRITEBYTECODE=1 /workspace/venv/bin/python -m pytest -q -p no:cacheprovider test_tok_probe_u.py --basetemp <NEW folder two levels deep>
  or: python test_tok_probe_u.py --tree TREE [--tmp NEWFOLDER]
The tool under test is tok_probe_u.py next to this file, or the file named by TOK_PROBE_U_FILE (run_mutations_u.py points it at a mutated copy). The source tool is ../finegrid_r/tok_probe_gr.py
(or TOKPROBE_GR_SRC). Every test holds at least one arm that must FAIL (a planted wrong value) through the helper `must_fail`.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

sys.dont_write_bytecode = True
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
UFILE = Path(os.environ.get("TOK_PROBE_U_FILE", str(HERE / "tok_probe_u.py")))
GFILE = Path(os.environ.get("TOKPROBE_GR_SRC", str(HERE.parent / "finegrid_r" / "tok_probe_gr.py")))
SYN_ARGV = ("--arm M2b --seed 0 --grid pool4x10 --wm-state grid --wm-cell-dim 16 --lead-sup on --w-lead-sup 1.0 --lead-sup-detach on --synthetic --kin-mode yaw "
            "--batch 4 --steps-mult 1 --lr 3e-3 --warmup 20")
NT, NE, TT = 6, 2, 60
STEPS = 12
torch.set_num_threads(4)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def must_fail(fn):
    """The planted-wrong arm of a test: ``fn`` states something false and must raise AssertionError; if it does not, the check under test cannot tell right from wrong."""
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("a planted wrong statement was accepted: the check is blind")


def tree_arg():
    t = os.environ.get("DSCALE_TREE") or os.environ.get("TREE")
    if not t:
        raise RuntimeError("REFUSED: set DSCALE_TREE (the folder that holds stack/) or pass --tree")
    return t


@pytest.fixture(scope="module")
def W(tmp_path_factory):
    """The two tools on the tree, the synthetic world forced to the tiny size, the clip ids of that world, and a work folder."""
    TU, TG = load(UFILE, "tok_probe_u_under_test"), load(GFILE, "tok_probe_gr_source")
    T = TU.use_tree(tree_arg())
    TG.use_tree(tree_arg())
    D = T["D"]
    orig = D.make_synthetic_world
    D.make_synthetic_world = lambda root, *a, **k: orig(root, n_train=NT, n_eval=NE, T=TT, seed=k.get("seed", 0))
    tmp = tmp_path_factory.mktemp("dscale_u", numbered=True)
    names = D.make_synthetic_world(str(tmp / "ids_world"))["names"]
    w = types.SimpleNamespace(TU=TU, TG=TG, D=D, tmp=Path(tmp), train_ids=[str(x) for x in names["train"]], eval_ids=[str(x) for x in names["eval"]], n=[0])
    yield w
    D.make_synthetic_world = orig


def run_train(w, mod, extra, steps=STEPS, arms="TK16_s0"):
    w.n[0] += 1
    out = w.tmp / ("run%02d" % w.n[0])
    rc = mod.main(["train", "--out", str(out), "--tree", tree_arg(), "--trainer-argv", SYN_ARGV, "--steps", str(steps), "--arms", arms, "--grid", "pool4x10", "--device", "cpu", "--threads", "4"] + list(extra))
    assert rc in (0, 2), rc
    return out


def meta(out):
    return json.load(open(out / "train_meta.json", encoding="utf-8"))


def ckpt(out, arm="TK16_s0"):
    return torch.load(str(out / arm / "ckpt_last.pt"), map_location="cpu", weights_only=False)["state_dict"]


def same_weights(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def log_rows(out, arm="TK16_s0"):
    return [json.loads(x) for x in open(out / arm / "train_log.jsonl", encoding="utf-8") if x.strip()]


def idfile(w, name, ids):
    p = w.tmp / name
    p.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------------------------------------ the generated file
def test_the_tool_is_the_generators_output_text_for_text():
    """No hand edit: the generator applied to the source tool gives exactly the bytes of tok_probe_u.py (the generator asserts each old text occurs once)."""
    sys.path.insert(0, str(HERE))
    import make_tok_probe_u as G
    want = G.build_text(str(GFILE)).encode("utf-8")
    got = UFILE.read_bytes()
    assert got == want
    must_fail(lambda: _assert(got + b" " == want))


# ------------------------------------------------------------------------------------------------ default path
def test_default_path_bit_identical_to_the_source_tool(W):
    """Without the new flags the training path equals tok_probe_gr.py: probe weights bit-identical after a few steps, the step losses equal; the log header carries only the new keys, all at their off value."""
    ob, on = run_train(W, W.TG, []), run_train(W, W.TU, [])
    assert same_weights(ckpt(ob), ckpt(on))
    lb, ln = log_rows(ob)[1:], log_rows(on)[1:]
    keys = ("lead_sup", "lead_sup_bce", "lead_sup_gap", "lead_sup_clos", "grad_norm", "g_tok", "g_head")
    assert [[r[k] for k in keys] for r in lb] == [[r[k] for k in keys] for r in ln]
    mn = meta(on)
    assert mn["w_clos"] == 1.0 and mn["shuffle_gap"] is None and mn["clip_list"] is None and mn["eval_guard"] is None
    # planted wrong: a tool trained with the closing part absent must NOT read as bit-identical
    off = run_train(W, W.TU, ["--w-clos", "0.0"])
    must_fail(lambda: _assert(same_weights(ckpt(ob), ckpt(off))))
    assert not same_weights(ckpt(ob), ckpt(off))
    pert = {k: v.clone() for k, v in ckpt(on).items()}
    k0 = next(k for k, v in pert.items() if v.dtype.is_floating_point and v.numel() > 0)
    pert[k0].view(-1)[0] += 1e-7
    must_fail(lambda: _assert(same_weights(ckpt(ob), pert)))


def _assert(x):
    assert x


# ------------------------------------------------------------------------------------------------ clip list and eval guard
def test_clip_list_keeps_exactly_the_listed_clips_and_the_meta_records_count_and_sha(W):
    keep = W.train_ids[:3]
    f = idfile(W, "keep3.txt", keep)
    out = run_train(W, W.TU, ["--clip-list", f])
    m = meta(out)
    assert m["n_clips"] == 3 and m["clip_list"]["n_kept"] == 3 and m["clip_list"]["n_listed"] == 3
    assert m["clip_list"]["sha256"] == hashlib.sha256(Path(f).read_bytes()).hexdigest()
    assert m["n_windows"] < m["n_windows_all_clips"]
    want = hashlib.sha256("\n".join(sorted(keep, key=W.D.sha12)).encode("utf-8")).hexdigest()
    assert m["train_sha256"] == want
    must_fail(lambda: _assert(m["n_clips"] == 4))
    must_fail(lambda: _assert(m["clip_list"]["sha256"] == "0" * 64))
    full = run_train(W, W.TU, [])
    assert meta(full)["n_clips"] == len(W.train_ids) and meta(full)["n_windows"] == meta(out)["n_windows_all_clips"]


def test_clip_list_refusals_missing_id_empty_list_duplicates_and_no_steps(W):
    with pytest.raises(SystemExit) as e:
        run_train(W, W.TU, ["--clip-list", idfile(W, "miss.txt", W.train_ids[:2] + ["no_such_clip"])])
    assert "not among the built clips" in str(e.value)
    with pytest.raises(SystemExit) as e:
        run_train(W, W.TU, ["--clip-list", idfile(W, "empty.txt", [])])
    assert "REFUSED" in str(e.value)
    with pytest.raises(SystemExit) as e:
        run_train(W, W.TU, ["--clip-list", idfile(W, "dup.txt", [W.train_ids[0], W.train_ids[0]])])
    assert "REFUSED" in str(e.value)
    with pytest.raises(SystemExit) as e:                                                   # no --steps: matched compute cannot be kept
        W.n[0] += 1
        W.TU.main(["train", "--out", str(W.tmp / "nosteps"), "--tree", tree_arg(), "--trainer-argv", SYN_ARGV, "--arms", "TK16_s0", "--grid", "pool4x10", "--device", "cpu", "--clip-list", idfile(W, "k.txt", W.train_ids[:2])])
    assert "--steps" in str(e.value)
    # planted wrong: a valid list must not be refused
    ok_out = run_train(W, W.TU, ["--clip-list", idfile(W, "ok.txt", W.train_ids[:2])])
    assert meta(ok_out)["n_clips"] == 2
    must_fail(lambda: _assert(meta(ok_out)["n_clips"] == 3))


def test_eval_guard_refuses_a_shared_id_and_passes_the_real_evaluation_ids(W):
    with pytest.raises(SystemExit) as e:
        run_train(W, W.TU, ["--eval-guard", idfile(W, "g_bad.txt", [W.eval_ids[0], W.train_ids[1]])])
    assert "evaluation clip ids are among the training clips" in str(e.value)
    out = run_train(W, W.TU, ["--eval-guard", idfile(W, "g_ok.txt", W.eval_ids)])
    g = meta(out)["eval_guard"]
    assert g["n_shared"] == 0 and g["n_eval_ids"] == len(W.eval_ids)
    assert not set(W.train_ids) & set(W.eval_ids)
    must_fail(lambda: _assert(g["n_shared"] == 1))
    with pytest.raises(SystemExit):                                                          # the guard acts together with the clip list too
        run_train(W, W.TU, ["--eval-guard", idfile(W, "g_bad2.txt", [W.train_ids[0]]), "--clip-list", idfile(W, "k2.txt", W.train_ids[:2])])


# ------------------------------------------------------------------------------------------------ closing weight 0
def planted_batch(n_tok=40, seed=0, B=6, F=8, d=1024, nan_clos=False):
    rng = np.random.default_rng(seed)
    clos = rng.uniform(-3, 3, size=(B, F)).astype(np.float32)
    if nan_clos:
        clos[:] = np.nan
    feats = torch.as_tensor(rng.normal(size=(B, F, n_tok, d)).astype(np.float32)).half()
    return {"feats": feats, "frame_ok": torch.ones(B, F, dtype=torch.bool), "lead_pres": torch.ones(B, F), "lead_gapm": torch.as_tensor(rng.uniform(15, 45, size=(B, F)).astype(np.float32)),
            "lead_clos": torch.as_tensor(clos)}


def grads_after(W, w_clos, batch):
    import copy
    W.TU._OPTS["w_clos"] = w_clos
    try:
        p = W.TU.build_probe("TK16_s0", grid="pool4x10")
        opt = W.TU.make_opt(p, 3e-4, 1e-2)
        p0 = copy.deepcopy(p)
        T, _, _ = W.TU.step_probe(p, opt, batch, 3e-4, 1.0, 1.0)
        return T, {k: (v.detach().clone() - dict(p0.named_parameters())[k].detach()) for k, v in p.named_parameters()}
    finally:
        W.TU._OPTS["w_clos"] = 1.0


def test_w_clos_zero_gives_the_closing_part_exactly_zero_and_no_gradient_from_the_closing_label(W):
    real, nan = planted_batch(), planted_batch(nan_clos=True)
    T1, d1 = grads_after(W, 1.0, real)
    assert float(T1["lead_sup_clos"]) > 0.0                                               # control: with weight 1.0 the closing part is live
    T0, d0 = grads_after(W, 0.0, real)
    assert float(T0["lead_sup_clos"]) == 0.0
    Tn, dn = grads_after(W, 0.0, nan)                                                     # a batch with NaN closing labels: same update as the real labels under weight 0.0
    assert float(Tn["lead_sup_clos"]) == 0.0 and all(torch.equal(d0[k], dn[k]) for k in d0)
    assert not all(torch.equal(d1[k], d0[k]) for k in d1)                                 # control: weight 1.0 does move the weights differently
    must_fail(lambda: _assert(float(T0["lead_sup_clos"]) > 0.0))
    must_fail(lambda: _assert(all(torch.equal(d1[k], dn[k]) for k in d1)))


def test_w_clos_zero_end_to_end_the_log_shows_a_closing_part_of_exactly_zero(W):
    out0, out1 = run_train(W, W.TU, ["--w-clos", "0.0"]), run_train(W, W.TU, [])
    r0, r1 = log_rows(out0)[1:], log_rows(out1)[1:]
    assert all(r["lead_sup_clos_mean"] == 0.0 for r in r0 if not r["first"])
    assert any(r["lead_sup_clos_mean"] > 0.0 for r in r1)
    assert meta(out0)["w_clos"] == 0.0 and json.loads(open(out0 / "TK16_s0" / "train_log.jsonl", encoding="utf-8").readline())["w_clos"] == 0.0
    must_fail(lambda: _assert(all(r["lead_sup_clos_mean"] == 0.0 for r in r1 if not r["first"])))


# ------------------------------------------------------------------------------------------------ gap shuffle
def fake_ds(n=400, seed=3):
    r = np.random.default_rng(seed)
    gap = r.uniform(5, 60, n).astype(np.float32)
    gap[r.random(n) < 0.4] = np.nan
    pres = (np.isfinite(gap) | (r.random(n) < 0.2)).astype(np.float32)
    clos = np.where(np.isfinite(gap), r.normal(size=n), np.nan).astype(np.float32)
    return types.SimpleNamespace(lead_sup_labels={"pres": pres, "gap": gap, "clos": clos})


def counts_equal(a, b):
    return bool(np.array_equal(np.isfinite(a), np.isfinite(b)) and np.array_equal(np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])))


def test_gap_shuffle_keeps_presence_and_the_gap_mask_and_moves_the_values(W):
    ds = fake_ds()
    g0, p0, c0 = (ds.lead_sup_labels[k].copy() for k in ("gap", "pres", "clos"))
    r = W.TU.shuffle_gap(ds)
    g1 = ds.lead_sup_labels["gap"]
    assert r["n_shuffled"] == int(np.isfinite(g0).sum())
    assert np.array_equal(ds.lead_sup_labels["pres"], p0) and np.array_equal(ds.lead_sup_labels["clos"], c0, equal_nan=True)
    assert counts_equal(g0, g1)                                                           # the same finite values on the same rows' mask
    assert not np.array_equal(g0, g1, equal_nan=True)                                    # and the values really moved
    bad = g1.copy()
    bad[np.flatnonzero(np.isfinite(bad))[0]] = np.nan
    must_fail(lambda: _assert(counts_equal(g0, bad)))                                    # a planted lost label is seen
    must_fail(lambda: _assert(np.array_equal(g0, g1, equal_nan=True)))
    ds2 = fake_ds()
    W.TU.shuffle_gap(ds2)
    assert np.array_equal(ds2.lead_sup_labels["gap"], g1, equal_nan=True)                # fixed rng: reproducible


def test_gap_shuffle_end_to_end_counts_equal_the_unshuffled_run(W):
    a = run_train(W, W.TU, ["--w-clos", "0.0"])
    b = run_train(W, W.TU, ["--w-clos", "0.0", "--shuffle-gap-labels"])
    ma, mb = meta(a), meta(b)
    assert mb["shuffle_gap"]["n_shuffled"] > 0 and ma["shuffle_gap"] is None
    for k in ("n_rows", "n_pres", "n_gap", "n_clos"):
        assert ma["lead_labels"][k] == mb["lead_labels"][k], k
    assert not same_weights(ckpt(a), ckpt(b))
    must_fail(lambda: _assert(ma["lead_labels"]["n_gap"] == mb["lead_labels"]["n_gap"] + 1))
    hdr = log_rows(b)[0]
    assert hdr["shuffle_gap"]["n_shuffled"] == mb["shuffle_gap"]["n_shuffled"]


# ------------------------------------------------------------------------------------------------ exclusivity
def ns(**k):
    base = dict(w_clos=1.0, shuffle_gap_labels=False, shuffle_closing_labels=False, clip_list="", steps=0, eval_guard="")
    base.update(k)
    return types.SimpleNamespace(**base)


def test_exclusivity_and_the_values_the_card_allows(W):
    for bad, word in ((ns(shuffle_gap_labels=True, shuffle_closing_labels=True, w_clos=0.0), "mutually exclusive"), (ns(shuffle_gap_labels=True, w_clos=1.0), "needs --w-clos 0.0"),
                      (ns(w_clos=0.5), "only 1.0"), (ns(clip_list="x.txt", steps=0), "--steps")):
        with pytest.raises(SystemExit) as e:
            W.TU.set_opts(bad)
        assert word in str(e.value), word
    W.TU.set_opts(ns(shuffle_gap_labels=True, w_clos=0.0))                                  # a legal combination is accepted
    assert W.TU._OPTS["w_clos"] == 0.0
    W.TU.set_opts(ns())
    assert W.TU._OPTS == {"clip_list": None, "eval_guard": None, "w_clos": 1.0}
    def raises(args):
        try:
            W.TU.set_opts(args)
        except SystemExit:
            return True
        return False
    assert raises(ns(w_clos=0.5)) and not raises(ns(shuffle_gap_labels=True, w_clos=0.0))
    must_fail(lambda: _assert(raises(ns(shuffle_gap_labels=True, w_clos=0.0))))         # expecting a refusal of the legal combination is the wrong statement
    W.TU.set_opts(ns())


# ------------------------------------------------------------------------------------------------ check_log presence-only form
def wlog(path, arm, bce, gap, clos, shuffle_gap=None, total=3000):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"header": True, "arm": arm, "total": total, "shuffle_gap": shuffle_gap}) + "\n")
        for i in range(len(bce)):
            fh.write(json.dumps({"step": 50 * (i + 1), "first": False, "lead_sup_mean": bce[i] + gap[i] + clos[i], "lead_sup_bce_mean": bce[i], "lead_sup_gap_mean": gap[i], "lead_sup_clos_mean": clos[i],
                                 "n_gap": 10.0, "n_gap_min_interval": 10.0}) + "\n")


def test_check_log_reads_a_gap_permuted_arm_on_its_presence_part_only(W, tmp_path):
    down, up, zero = np.linspace(1.0, 0.5, 60), np.linspace(0.5, 1.0, 60), np.zeros(60)
    wlog(tmp_path / "a.jsonl", "TK16_s0", down, up, zero, {"n_shuffled": 5})            # presence falls, the (permuted) gap part rises: the whole term does not fall
    a = W.TU.check_log(str(tmp_path / "a.jsonl"))
    assert a["pass"] and a["shuffled_closing_arm"] is True and a["pass_intact_parts_fall"] is True
    wlog(tmp_path / "b.jsonl", "TK16_s0", down, up, zero, None)                         # the same numbers without the header key: read on presence + gap, which does not fall
    b = W.TU.check_log(str(tmp_path / "b.jsonl"))
    assert b["pass_intact_parts_fall"] is False and not b["pass"]
    wlog(tmp_path / "c.jsonl", "TK16_s0", up, down, zero, {"n_shuffled": 5})            # presence part rises: a permuted-gap arm that does not learn presence fails
    assert not W.TU.check_log(str(tmp_path / "c.jsonl"))["pass"]
    must_fail(lambda: _assert(not a["pass"]))
    must_fail(lambda: _assert(b["pass"]))


# ------------------------------------------------------------------------------------------------ precheck delegation
def test_precheck_subcommand_delegates_to_the_numpy_only_checker(W, tmp_path):
    with pytest.raises(SystemExit) as e:                                                   # argparse of dscale_precheck.main refuses a call without its required flags (exit 2)
        W.TU.main(["precheck"])
    assert e.value.code == 2
    must_fail(lambda: _assert(e.value.code == 0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--tmp", default="")
    a = ap.parse_args()
    os.environ["DSCALE_TREE"] = a.tree
    args = ["-q", "-p", "no:cacheprovider", __file__]
    if a.tmp:
        args += ["--basetemp", a.tmp]
    sys.exit(pytest.main(args))
