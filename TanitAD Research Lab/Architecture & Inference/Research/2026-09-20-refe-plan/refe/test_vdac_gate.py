#!/usr/bin/env python3
"""VDAC launch-gate tests (CPU, no data).  Prints ZZTEST_PASS.  PREREG_VDAC_P6 (frozen 2026-11-11, sha256 5594c2ae...) + A1 + A2: the drivable heads HA-s0 / HA-s1 / HA-SHUF on P6's frozen trunk (S1c's recipe), H48 excluded from the head's labels.

  V1 EXCLUDE-LOGS (scorer_finetune.read_exclude_logs / aux_excl_audit / aux_excl_refusals / AuxSubset): a wrong declared sha256, an empty and a repeating list are refused; the audit MEASURES through the train set's own label lookup (0 H48 rows reach the loss,
     > 0 labelled frames excluded, the filter == the file); a train set with the filter REMOVED, with a filter that holds one log too many, and with a list that names no log of the index are all refused; with the filter off the class is the pre-edit class bit for bit.
  V2 REACHABILITY from the caller: main() builds the filter from a.aux_drv_exclude_logs, audits the SAME train set object, refuses (rc 4), writes info["aux_excl"], and make_train_set hands the filter to the class; `wired()` is True on the real source and FALSE on 6 source mutants.
  V3 CLI: both flags default OFF; the pair is named together, needs --aux-drv, and the file must hash to the declared sha256 (3 refusals + the good case).
  V4 G-DVB (gate_checks): both flags have registry entries (a table without them is refused as unknown), the new rows are green on a good data event and RED on 8 mutated events (no block, wrong sha, H48 rows reaching the loss, nothing excluded, a filter that is not the file, ...), no row with the flag OFF;
     the HA-SHUF derangement rows (0 fixed points, every key) go RED on a map with one fixed point.
  V5 PROFILE refe_vdac_p6: the three arms legal; argv = C0's argv with the budget / seed / --out replaced + the head tail; refused: every lever dropped / moved (--w-aux-drv, --aux-drv-lr, the exclusion pair, --epochs, tap scene, feed on, labels path, ckpt, seed, shuffle on a non-control arm, shuffle missing on HA-SHUF, a lane flag, a pair / split lever, a wrong H48 sha256);
     the budget is reachable (schedule_reach re-derives 10,000 / 2,000 updates on 60,000 sets, and 7,500 at the parser's --epochs 1.0: the reason --epochs 2 is named).
  V6 FINGERPRINTS: the 25 older profile digests unmoved (24 pinned in test_div_gate + refe_div_p6), the new one stable, 26 profiles.
  V7 LAYER: vdac_layer.peel(trainer) is the tip blob e8b86260; div_layer.peel (which peels VDAC first) still gives d665ed99 and s1zsm_layer.peel 6a16d35c; one changed character inside any reversal makes peel REFUSE.
  V8 SCRIPTS: bash -n x3; vdac_pod.sh vs launch_gate.vdac_argv (3 arms) vs scorer1p6_pod.sh; the chain's python (exclusion audit, run check) on synthetic logs; failure lines carry `REFUSED: stage=` / ZZVDAC_CHAIN_FAIL and no success line carries the bare word REFUSED;
     the pause script's S1a / S1b arms, the S1B_GO condition and the EXIT trap; MUTATIONS RED.
  V9 POST-TRAINING GATE (vdac_postgate.judge): on the REAL head (2,440,450 parameters) after 2 optimiser steps = OK; a dropped state entry, a zeroed first moment, a wrong step count, a wrong parameter count are each RED.
  V10 EXPORT (vdac_export_h48.export) on a synthetic bank; H48 BUILD (vdac_build_h48.build) on a synthetic store; the committed list's sha256 == the profile's == the chain's; HA-SHUF derangement (0 fixed points, identical multiset).
"""
import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import aux_drivable_head as AUX  # noqa: E402
import div_layer as DL  # noqa: E402
import gate_checks as C  # noqa: E402
import gate_token as GT  # noqa: E402
import launch_gate as L  # noqa: E402
import s1zsm_layer as ZSM  # noqa: E402
import scorer_finetune as SF  # noqa: E402
import vdac_layer as VL  # noqa: E402
import vdac_postgate as PG  # noqa: E402

assert "tanitad" not in sys.modules or True
PV = L.PROFILES["refe_vdac_p6"]
ARMS = ("HA-s0", "HA-s1", "HA-SHUF")
SHA = {L.SCORER1P6_CKPT: L.SCORER1P6_CKPT_SHA256, L.VDAC_H48_PATH: L.VDAC_H48_SHA256}
for _f, _p in (("--navtest-logs", L.SCORER1P6_ROOT + "/navtest_ids/navtest_logs.json"), ("--navtest-tokens", L.SCORER1P6_ROOT + "/navtest_ids/navtest_tokens.txt.gz")):
    SHA[_p] = L.SCORER1_NAVTEST_SHA256[_f]


def fake_sha(path):
    if path in SHA:
        return SHA[path]
    raise FileNotFoundError(path)


L.sha256_of_file = fake_sha


def refusals(argv, arm, prof=PV):
    return L.profile_refusals({**prof, "declared_arm": arm}, argv)


def has(rs, needle):
    return any(needle in r for r in rs)


def swap(argv, flag, *vals):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            out += [flag] + list(vals)
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def drop(argv, flag):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith("--"):
                i += 1
            continue
        out.append(argv[i])
        i += 1
    return out


def differing_flags(a, b):
    da, db = {}, {}
    for f, v in GT.flag_pairs(list(a)):
        da.setdefault(f, []).append(tuple(v))
    for f, v in GT.flag_pairs(list(b)):
        db.setdefault(f, []).append(tuple(v))
    return {f for f in set(da) | set(db) if da.get(f) != db.get(f)}


def blob(src: str) -> str:
    b = src.encode("utf-8")
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def load_copy(src_text, name, directory):
    p = os.path.join(directory, name + ".py")
    open(p, "w", encoding="utf-8", newline="").write(src_text)
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


TRAINER = open(os.path.join(HERE, "scorer_finetune.py"), "rb").read().decode("utf-8").replace("\r\n", "\n")
tmp = os.environ.get("SCORER1_TMP") or None
VA = {a: L.vdac_argv(a) for a in ARMS}
parse_ns = lambda argv: C.parse_scorer_argv(SF.build_parser(), argv)[0]  # noqa: E731

# ---------------------------------------------------------------- V1 the exclusion
H48 = open(os.path.join(HERE, "..", "splits", "h48_logs.txt"), "rb").read() if os.path.isfile(os.path.join(HERE, "..", "splits", "h48_logs.txt")) else None


class FakeSrc:
    """the store: every (log, token) is labelled, the label is a constant 1 (inside)"""

    def labels(self, keys):
        return torch.ones(len(keys), 160, 128, dtype=torch.int8), torch.ones(len(keys), dtype=torch.bool)


class FakeDS:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, i):
        return (torch.zeros(1),)


logs = ["logA", "logB", "logC", "logD"]
rows = [{"log_name": lg, "token": "t%d" % k, "step": s, "rank": 0} for lg in logs for k in range(3) for s in (0, 1)]       # 2 rows (steps) per frame
ds_fake = FakeDS(rows)
idx = list(range(len(rows)))
excl = ["logB", "logD"]
with tempfile.TemporaryDirectory(dir=tmp) as td:
    f_ok = os.path.join(td, "ok.txt")
    open(f_ok, "wb").write(("\n".join(excl) + "\n").encode())
    good_sha = hashlib.sha256(open(f_ok, "rb").read()).hexdigest()
    names, got = SF.read_exclude_logs(f_ok, good_sha)
    assert names == sorted(excl) and got == good_sha
    for body, why in ((b"", "empty"), (b"logB\nlogB\n", "repeat"), (b"\n\n", "empty")):
        p = os.path.join(td, "bad.txt")
        open(p, "wb").write(body)
        try:
            SF.read_exclude_logs(p, hashlib.sha256(body).hexdigest())
        except ValueError:
            pass
        else:
            raise AssertionError("a %s list was accepted" % why)
    try:
        SF.read_exclude_logs(f_ok, "0" * 64)
    except ValueError as e:
        assert "declared" in str(e)
    else:
        raise AssertionError("a file that does not hash to the declared sha256 was accepted")
src = FakeSrc()
ts = SF.AuxSubset(ds_fake, idx, src, False, frozenset(excl))
a_ok = SF.aux_excl_audit(ts, rows, idx, names, src)
assert a_ok["reached_loss_rows"] == 0 and a_ok["reached_loss_frames"] == 0 and a_ok["excluded_rows"] == 12 and a_ok["excluded_frames"] == 6 and a_ok["excluded_labelled_frames"] == 6 and a_ok["filter_equals_file"] is True, a_ok
assert SF.aux_excl_refusals(a_ok) == [] and json.dumps(a_ok)
# the lookup the loader uses: excluded rows read zeros + valid False, the others the store's label
lab, val = ts.label_at(idx.index(next(i for i in idx if rows[i]["log_name"] == "logB")))
assert not bool(val) and int(lab.abs().sum()) == 0
assert all(bool(ts.label_at(k)[1]) for k, i in enumerate(idx) if rows[i]["log_name"] in ("logA", "logC"))
# MUTATIONS: the filter REMOVED, a filter with one log too many, a list that names no log of the index
ts_nof = SF.AuxSubset(ds_fake, idx, src, False, frozenset())
a_nof = SF.aux_excl_audit(ts_nof, rows, idx, names, src)
assert a_nof["reached_loss_rows"] == 12 and a_nof["filter_equals_file"] is False and has(SF.aux_excl_refusals(a_nof), "still reach the head's loss"), a_nof
ts_big = SF.AuxSubset(ds_fake, idx, src, False, frozenset(excl + ["logA"]))
a_big = SF.aux_excl_audit(ts_big, rows, idx, names, src)
assert a_big["reached_loss_rows"] == 0 and has(SF.aux_excl_refusals(a_big), "not the file's"), "MUTATION: a filter that holds a log the file does not name is refused"
a_none = SF.aux_excl_audit(ts, rows, idx, ["logZ"], src)
assert a_none["excluded_labelled_frames"] == 0 and has(SF.aux_excl_refusals(a_none), "0 labelled frames"), "an exclusion of nothing is not a test"
# H48 rows that carry NO label in the store are not 'labelled frames excluded'
class HalfSrc(FakeSrc):
    def labels(self, keys):
        lab, val = super().labels(keys)
        return lab, torch.tensor([k[0] != "logD" for k in keys])


a_half = SF.aux_excl_audit(ts, rows, idx, names, HalfSrc())
assert a_half["excluded_frames"] == 6 and a_half["excluded_labelled_frames"] == 3, a_half
# the filter OFF: the class is the pre-edit class (no exclude -> every row reads the store's label); a DS without the attribute gets an empty filter
assert SF.AuxSubset(ds_fake, idx, src, True).exclude == frozenset() and isinstance(SF.make_train_set(type("D", (), {"rows": rows, "aux_src": src, "__getitem__": lambda s, i: (torch.zeros(1),)})(), idx, False), SF.AuxSubset)
d_off = type("D", (), {"rows": rows, "aux_src": src, "__getitem__": lambda s, i: (torch.zeros(1),)})()
assert SF.make_train_set(d_off, idx, False).exclude == frozenset()
d_on = type("D", (), {"rows": rows, "aux_src": src, "aux_exclude": frozenset(excl), "__getitem__": lambda s, i: (torch.zeros(1),)})()
assert SF.make_train_set(d_on, idx, False).exclude == frozenset(excl), "make_train_set hands the dataset's filter to the class"
# the exclusion reaches the LOSS: rows the lookup marks invalid contribute exactly 0 loss and exactly 0.0 gradient, and do not change the loss of the others (AUX.aux_loss is the loss the trainer's aux_backward calls)
class RandSrc(FakeSrc):
    def labels(self, keys):
        g_ = torch.Generator().manual_seed(3)
        return torch.randint(-100, 100, (len(keys), 160, 128), generator=g_, dtype=torch.int8), torch.ones(len(keys), dtype=torch.bool)


ts_r = SF.AuxSubset(ds_fake, idx, RandSrc(), False, frozenset(excl))
pos_ex = [k for k, i in enumerate(idx) if rows[i]["log_name"] == "logB"][:2]
pos_in = [k for k, i in enumerate(idx) if rows[i]["log_name"] == "logA"][:2]
labs, vals = zip(*[ts_r.label_at(k) for k in pos_ex + pos_in])
labs, vals = torch.stack(labs), torch.stack(vals)
assert vals.tolist() == [False, False, True, True]
gl = torch.Generator().manual_seed(5)
lg0, sd0 = torch.randn(4, 40, 32, generator=gl), torch.randn(4, 40, 32, generator=gl) * 3
lg, sd = lg0.clone().requires_grad_(True), sd0.clone().requires_grad_(True)
l_mix, _, _ = AUX.aux_loss(lg, sd, labs, vals)
l_mix.backward()
assert float(lg.grad[:2].abs().sum()) == 0.0 and float(sd.grad[:2].abs().sum()) == 0.0 and float(lg.grad[2:].abs().sum()) > 0.0, "an excluded row's head output gets EXACTLY 0.0 gradient"
l_in, _, _ = AUX.aux_loss(lg0[2:], sd0[2:], labs[2:], vals[2:])
assert abs(float(l_mix) - float(l_in)) < 1e-6, "the excluded rows do not change the loss of the others"
lg1 = lg0.clone()
lg1[:2] += 7.0
assert abs(float(AUX.aux_loss(lg1, sd0, labs, vals)[0]) - float(l_mix)) < 1e-6, "a different head output on an excluded frame changes nothing"
lg_all = lg0.clone().requires_grad_(True)
l_none, _, _ = AUX.aux_loss(lg_all, sd0.clone(), labs, torch.zeros(4, dtype=torch.bool))
l_none.backward()
assert float(l_none) == 0.0 and float(lg_all.grad.abs().sum()) == 0.0, "a micro-batch of only excluded frames: loss 0.0, gradient exactly 0.0"
ts_leak = SF.AuxSubset(ds_fake, idx, RandSrc(), False, frozenset())          # MUTATION: the filter removed -> the same rows now carry labels and the loss differs
labs_l, vals_l = zip(*[ts_leak.label_at(k) for k in pos_ex + pos_in])
assert torch.stack(vals_l).tolist() == [True] * 4 and abs(float(AUX.aux_loss(lg0, sd0, torch.stack(labs_l), torch.stack(vals_l))[0]) - float(l_mix)) > 1e-4, "MUTATION: without the filter the excluded rows move the loss"
print("V1 ok (sha256 refusal; empty / repeating list; the audit measures through the train set's lookup; 3 mutants refused; unlabelled rows are not counted; filter OFF = the old class)")

# ---------------------------------------------------------------- V2 reachability from the caller


def wired(src: str) -> bool:
    """True iff main() refuses an inconsistent exclusion pair, builds the filter from a.aux_drv_exclude_logs under the mutate env, audits the train set make_train_set builds, refuses with rc 4, and the block reaches the data event;
    and the class + make_train_set carry the filter to the label lookup"""
    m = re.search(r"^def main\(\) -> int:\n(.*?)^\S", src + "\nX", re.S | re.M)
    body = m.group(1) if m else ""
    cli = body.find("aux_excl_cli_refusals(a)")
    gate = body.find("a.aux_drv and a.aux_drv_exclude_logs")
    flt = body.find("ds.aux_exclude = ")
    aud = body.find("aux_excl_audit(make_train_set(ds, tr_idx, a.epo)")
    ref = body.find("bad_excl = aux_excl_refusals(aux_excl_info)")
    event = body.find('info = {"event": "data"')
    put = body.find('info["aux_excl"] = aux_excl_info')
    logi = body.find("    log(info)\n")
    loader = body.find("dl = DataLoader(make_train_set(ds, tr_idx")
    fwd = body.find("if aux is not None and h48x_idx:")
    ok_main = (0 <= cli < gate < flt < aud < ref < event < put < logi < loader and "AUX_EXCL_MUTATE_ENV" in body[flt:flt + 300] and "frozenset(excl_names)" in body[flt:flt + 300]
               and "return 4" in body[ref:ref + 300] and "a.aux_drv_exclude_logs" in body[gate:gate + 120] and fwd > loader and "ds_ho = copy.copy(ds)" in body[:flt])
    cls = re.search(r"^class AuxSubset\(Subset\):\n(.*?)^def ", src, re.S | re.M)
    cbody = cls.group(1) if cls else ""
    ok_cls = ('self.exclude' in cbody and 'in self.exclude' in cbody and "def label_at" in cbody and "label_at(idx)" in cbody.split("def __getitem__")[-1])
    mt = re.search(r"^def make_train_set\(.*?\n(.*?)^def ", src, re.S | re.M)
    ok_mk = bool(mt) and 'getattr(ds, "aux_exclude", frozenset())' in mt.group(1)
    return ok_main and ok_cls and ok_mk


assert wired(TRAINER), "main() / AuxSubset / make_train_set do not carry the exclusion to the loss"
muts = {
    "the filter is never built": ("        ds.aux_exclude = frozenset() if os.environ.get(AUX_EXCL_MUTATE_ENV) == \"1\" else frozenset(excl_names)", "        pass"),
    "the audit is skipped": ("aux_excl_audit(make_train_set(ds, tr_idx, a.epo), ds.rows, tr_idx, excl_names, aux_src_eval)", "{}"),
    "the refusal is never raised": ("bad_excl = aux_excl_refusals(aux_excl_info)", "bad_excl = []"),
    "the block never reaches the data event": ('        info["aux_excl"] = aux_excl_info\n', "        pass\n"),
    "the class ignores the filter": ("        if r.get(\"log_name\", \"\") in self.exclude:", "        if False:"),
    "make_train_set drops the filter": ('getattr(ds, "aux_exclude", frozenset())', "frozenset()"),
}
for nm, (old, new) in muts.items():
    assert TRAINER.count(old) == 1, nm
    assert not wired(TRAINER.replace(old, new)), "MUTATION %r must turn wired() RED" % nm
assert "aux_drv_exclude_logs" in C.namespace_reads(TRAINER, C.SCORER_NAMESPACE_FUNCS) and "aux_drv_exclude_sha256" in C.namespace_reads(TRAINER, C.SCORER_NAMESPACE_FUNCS), "the dead-lever check must see the reads"
assert "aux_drv_exclude_logs" not in C.namespace_reads(TRAINER.replace("a.aux_drv_exclude_logs", "a.aux_drv_labels"), C.SCORER_NAMESPACE_FUNCS), "a trainer that never reads the flag is a dead lever"
assert SF.AUX_EXCL_MUTATE_ENV == "REFE_AUXEXCL_MUTATE_NOFILTER"
# the REAL text of main()'s exclusion block, executed on stubs (the block is not a copy: it is cut out of the trainer source): good run, the NOFILTER mutant (env), a wrong list, flag OFF
seg_a = TRAINER.index("    aux_excl_info = h48x_idx = None\n")
seg_b = TRAINER.index("    # ---------------- P6-HN-S (--hn-bank / --ctx-cache; default OFF")
seg = TRAINER[seg_a:seg_b]
fn_src = "def _block(a, ds, tr_idx, aux_src_eval):\n" + seg + "    return aux_excl_info, h48x_idx\n"
ns_g = dict(vars(SF))
exec(compile(fn_src, "main_block", "exec"), ns_g)
blk = ns_g["_block"]


def run_block(excl_names_, env_mut=False, with_flag=True):
    with tempfile.TemporaryDirectory(dir=tmp) as td_:
        pth = os.path.join(td_, "h.txt")
        body = ("\n".join(excl_names_) + "\n").encode()
        open(pth, "wb").write(body)
        a_ = type("A", (), {"aux_drv": True, "aux_drv_exclude_logs": pth if with_flag else "", "aux_drv_exclude_sha256": hashlib.sha256(body).hexdigest(), "epo": False})()
        d_ = FakeDS(rows)
        d_.aux_src = FakeSrc()
        old_env = os.environ.get(SF.AUX_EXCL_MUTATE_ENV)
        if env_mut:
            os.environ[SF.AUX_EXCL_MUTATE_ENV] = "1"
        try:
            return blk(a_, d_, list(idx), FakeSrc()), d_
        finally:
            if env_mut:
                os.environ.pop(SF.AUX_EXCL_MUTATE_ENV, None)
            if old_env is not None:
                os.environ[SF.AUX_EXCL_MUTATE_ENV] = old_env


(r_good, d_good) = run_block(excl)
assert r_good[0]["reached_loss_rows"] == 0 and r_good[0]["excluded_labelled_frames"] == 6 and r_good[0]["filter_equals_file"] is True and d_good.aux_exclude == frozenset(excl), r_good
assert len(r_good[1]) == 6 and len({(rows[i]["log_name"], rows[i]["token"]) for i in r_good[1]}) == 6 and all(rows[i]["log_name"] in excl for i in r_good[1]), "one row per H48 frame for the head's held-out forward"
r_mut, d_mut = run_block(excl, env_mut=True)
assert r_mut == 4, "the NOFILTER mutant (REFE_AUXEXCL_MUTATE_NOFILTER=1) makes the block return rc 4 before the data event: %r" % (r_mut,)
r_none, _ = run_block(["logZ"])
assert r_none == 4, "a list that names no log of the index is refused"
r_off, _ = run_block(excl, with_flag=False)
assert r_off == (None, None), "flag OFF: the block does nothing"
print("V2 ok (main() builds the filter, audits the SAME train set, refuses rc 4, writes aux_excl; the class and make_train_set carry it; wired() RED on 6 mutants; the dead-lever check sees both reads)")

# ---------------------------------------------------------------- V3 CLI
ap = SF.build_parser()
base = ["--ckpt", "c", "--targets", "t", "--images", "i", "--calib", "k", "--onpolicy", "o", "--heldout", "h", "--heldout-logs", "l", "--out", "x"]
ns0 = ap.parse_args(base)
assert ns0.aux_drv_exclude_logs == "" and ns0.aux_drv_exclude_sha256 == "" and SF.aux_excl_cli_refusals(ns0) == [], "both flags default OFF"
with tempfile.TemporaryDirectory(dir=tmp) as td:
    f1 = os.path.join(td, "h.txt")
    open(f1, "wb").write(b"logB\nlogD\n")
    s1 = hashlib.sha256(b"logB\nlogD\n").hexdigest()
    head = base + ["--aux-drv", "--aux-drv-lr", "0.0002", "--aux-drv-labels", "x"]
    assert SF.aux_excl_cli_refusals(ap.parse_args(head + ["--aux-drv-exclude-logs", f1, "--aux-drv-exclude-sha256", s1])) == []
    assert has(SF.aux_excl_cli_refusals(ap.parse_args(base + ["--aux-drv-exclude-logs", f1, "--aux-drv-exclude-sha256", s1])), "silently do nothing")
    assert has(SF.aux_excl_cli_refusals(ap.parse_args(head + ["--aux-drv-exclude-logs", f1])), "named together")
    assert has(SF.aux_excl_cli_refusals(ap.parse_args(head + ["--aux-drv-exclude-sha256", s1])), "named together")
    assert has(SF.aux_excl_cli_refusals(ap.parse_args(head + ["--aux-drv-exclude-logs", f1, "--aux-drv-exclude-sha256", "0" * 64])), "declared")
    assert has(SF.aux_excl_cli_refusals(ap.parse_args(head + ["--aux-drv-exclude-logs", os.path.join(td, "nope.txt"), "--aux-drv-exclude-sha256", s1])), "nope.txt")
print("V3 ok (both flags default OFF; the pair is named together, needs --aux-drv, and the file must hash to the declared sha256)")

# ---------------------------------------------------------------- V4 G-DVB


def nest(d, dotted, v):
    cur = d
    ks = dotted.split(".")
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = v


def good_data(arm, n=60000):
    sp = L.VDAC_ARMS[arm]
    d = {"b_mode": "expreward", "b_compw": "1,1,1,1,1,1", "train_sets": n}
    for k, v in PV["data_expect"].items():
        nest(d, k, v)
    d["aux_drv"] = {"w_aux": 0.1, "tap": "visual", "depth": 2, "feed_built": False, "scorer_in_width": 60, "lr": 0.0002, "n_params": 2440450, "head_params_in_optimiser": 2440450,
                    "shuffle_seed": sp["shuffle"], "store_sha256": AUX.PINNED_STORE_SHA256, "frames": AUX.PINNED_STORE_KEYS, "cell": AUX.CELL_M, "x0": AUX.X0, "y0": AUX.Y0,
                    "nx": int(round((AUX.X1 - AUX.X0) / AUX.CELL_M)), "ny": int(round((AUX.Y1 - AUX.Y0) / AUX.CELL_M))}
    if sp["shuffle"] is not None:
        d["aux_drv"].update({"shuffle_keys": AUX.PINNED_STORE_KEYS, "derangement_fixed_points": 0})
    d["aux_excl"] = {"sha256": L.VDAC_H48_SHA256, "logs": 48, "excluded_rows": 2400, "excluded_frames": 1200, "excluded_labelled_frames": 1200, "reached_loss_rows": 0, "reached_loss_frames": 0,
                     "filter_logs": 48, "filter_equals_file": True}
    return d


def excl_rows(argv, data):
    return [r for r in C.scorer_dvb_rows(parse_ns(argv), data, PV) if r["dest"] in ("aux_drv_exclude_logs", "aux_drv_exclude_sha256")]


# the profile's declared file bytes: scorer_dvb_rows hashes the file the argv names -> the fake path must exist for the comparison; point _file_sha256 at the registered sha
C._file_sha256 = lambda p: L.VDAC_H48_SHA256 if p == L.VDAC_H48_PATH else hashlib.sha256(open(p, "rb").read()).hexdigest() if (p and os.path.isfile(p)) else None
for a in ARMS:
    rr = excl_rows(VA[a], good_data(a))
    assert len(rr) == 7 and all(r["ok"] for r in rr), [r for r in rr if not r["ok"]]
gm = {
    "no block": lambda d: {k: v for k, v in d.items() if k != "aux_excl"},
    "wrong sha": lambda d: {**d, "aux_excl": {**d["aux_excl"], "sha256": "0" * 64}},
    "H48 rows reach the loss": lambda d: {**d, "aux_excl": {**d["aux_excl"], "reached_loss_rows": 40}},
    "H48 frames reach the loss": lambda d: {**d, "aux_excl": {**d["aux_excl"], "reached_loss_frames": 20}},
    "nothing excluded": lambda d: {**d, "aux_excl": {**d["aux_excl"], "excluded_labelled_frames": 0}},
    "filter is not the file": lambda d: {**d, "aux_excl": {**d["aux_excl"], "filter_equals_file": False}},
    "block without a reached count": lambda d: {**d, "aux_excl": {k: v for k, v in d["aux_excl"].items() if k != "reached_loss_rows"}},
    "labelled count not an int": lambda d: {**d, "aux_excl": {**d["aux_excl"], "excluded_labelled_frames": None}},
}
for nm, f in gm.items():
    assert any(not r["ok"] for r in excl_rows(VA["HA-s0"], f(good_data("HA-s0")))), "MUTATION %r must turn an exclusion row RED" % nm
C0A = L.scorer1p6_argv("C0")
assert any(not r["ok"] for r in excl_rows(C0A, good_data("HA-s0"))), "a block under an argv without the flag: RED (`iff`)"
assert excl_rows(C0A, {k: v for k, v in good_data("HA-s0").items() if k != "aux_excl"}) == [], "flag OFF and no block: no exclusion row at all (every other profile's G-DVB is unchanged)"
# HA-SHUF: the derangement rows
sh = good_data("HA-SHUF")
shr = [r for r in C.scorer_dvb_rows(parse_ns(VA["HA-SHUF"]), sh, PV) if r["dest"] == "aux_drv_shuffle_seed"]
assert len(shr) == 3 and all(r["ok"] for r in shr), shr
for nm, f in (("a fixed point", lambda d: {**d, "aux_drv": {**d["aux_drv"], "derangement_fixed_points": 1}}), ("a map over fewer keys", lambda d: {**d, "aux_drv": {**d["aux_drv"], "shuffle_keys": 100}}),
              ("wrong seed built", lambda d: {**d, "aux_drv": {**d["aux_drv"], "shuffle_seed": 1}}), ("no shuffle built", lambda d: {**d, "aux_drv": {**d["aux_drv"], "shuffle_seed": None}})):
    assert any(not r["ok"] for r in C.scorer_dvb_rows(parse_ns(VA["HA-SHUF"]), f(sh), PV) if r["dest"] == "aux_drv_shuffle_seed"), "MUTATION %r" % nm
# the registry entries
assert C.DVB_KINDS_SCORER["aux_drv_exclude_logs"] == "built" == C.DVB_KINDS_SCORER["aux_drv_exclude_sha256"]
dests = {x.dest for x in SF.build_parser()._actions if x.dest != "help"}
assert set(C.DVB_KINDS_SCORER) == dests, (dests ^ set(C.DVB_KINDS_SCORER))
for a in ARMS:
    v_ok = C.check_dvb_scorer(VA[a], {"data": good_data(a)}, {**PV, "declared_arm": a})
    bad = [r for r in (v_ok.get("reasons") or []) if "exclude" in r or "unknown" in r or "NO declared-vs-built" in r or "never read" in r or "aux_drv" in r or "token would NOT" in r]
    assert not bad, (a, bad)
print("   G-DVB verdict on a good HA-s0 data event:", v_ok.get("status") or v_ok.get("verdict"), "; reasons:", [r[:120] for r in (v_ok.get("reasons") or [])][:4])
for flag in ("aux_drv_exclude_logs", "aux_drv_exclude_sha256"):
    saved = dict(C.DVB_KINDS_SCORER)
    del C.DVB_KINDS_SCORER[flag]
    try:
        v_bad = C.check_dvb_scorer(VA["HA-s0"], {"data": good_data("HA-s0")}, {**PV, "declared_arm": "HA-s0"})
    finally:
        C.DVB_KINDS_SCORER.clear()
        C.DVB_KINDS_SCORER.update(saved)
    assert any(flag in r and "NO declared-vs-built" in r for r in (v_bad.get("reasons") or [])), "MUTATION: a flag without a registry entry must be refused: %s" % (v_bad.get("reasons"),)
v_mut = C.check_dvb_scorer(VA["HA-s0"], {"data": gm["H48 rows reach the loss"](good_data("HA-s0"))}, {**PV, "declared_arm": "HA-s0"})
assert any("--aux-drv-exclude-logs" in r for r in (v_mut.get("reasons") or [])), "the full G-DVB check must go RED on the NOFILTER-shaped data event"
# an older profile whose argv carries no exclusion flag: its G-DVB is untouched by the new flags (they are exempt when not named)
v_old = C.check_dvb_scorer(C0A, {"data": {k: v for k, v in good_data("HA-s0").items() if k not in ("aux_excl", "aux_drv")}}, {**L.PROFILES["refe_scorer1p6"], "declared_arm": "C0"})
assert not [r for r in (v_old.get("reasons") or []) if "exclude" in r], v_old.get("reasons")
print("V4 ok (7 rows green on all 3 arms; 8 mutated data events + a block under flag OFF RED; 4 derangement mutants RED; tables without either flag refused; an older profile's G-DVB unchanged)")

# ---------------------------------------------------------------- V5 the profile
P0 = L.scorer1p6_argv("C0")
for a in ARMS:
    assert refusals(VA[a], a) == [], (a, refusals(VA[a], a))
    assert differing_flags(VA[a], P0) >= {"--max-updates", "--eval-every", "--out", "--aux-drv", "--epochs", "--batch", "--accum"}          # (--seed too for HA-s1: C0's seed is 0)
    assert not differing_flags(VA[a], P0) - {"--max-updates", "--eval-every", "--seed", "--out", "--batch", "--accum", "--epochs", "--aux-drv", "--w-aux-drv", "--aux-drv-lr", "--aux-drv-tap", "--aux-drv-depth",
                                           "--aux-drv-labels", "--aux-drv-exclude-logs", "--aux-drv-exclude-sha256", "--aux-drv-shuffle-seed", "--dump-scores"}
    assert VA[a].count("--out") == 1 and VA[a][VA[a].index("--out") + 1] == L.VDAC_ROOT + "/" + a.lower() and "--lane-head" not in VA[a] and "--pair-w" not in VA[a]
assert "--aux-drv-shuffle-seed" in VA["HA-SHUF"] and "--aux-drv-shuffle-seed" not in VA["HA-s0"] and "--aux-drv-shuffle-seed" not in VA["HA-s1"]
assert (VA["HA-s0"][VA["HA-s0"].index("--seed") + 1], VA["HA-s1"][VA["HA-s1"].index("--seed") + 1], VA["HA-SHUF"][VA["HA-SHUF"].index("--seed") + 1]) == ("0", "1", "0")
assert [VA[a][VA[a].index("--max-updates") + 1] for a in ARMS] == ["10000", "10000", "2000"]
A0, A1, AS = VA["HA-s0"], VA["HA-s1"], VA["HA-SHUF"]
cases = 0
for flag in ("--aux-drv", "--w-aux-drv", "--aux-drv-lr", "--aux-drv-labels", "--aux-drv-exclude-logs", "--aux-drv-exclude-sha256", "--epochs", "--batch", "--accum", "--eval-every", "--max-updates", "--aux-drv-tap", "--aux-drv-depth", "--dump-scores"):
    assert refusals(drop(A0, flag), "HA-s0"), "dropping %s must be refused" % flag
    cases += 1
assert has(refusals(drop(A0, "--w-aux-drv"), "HA-s0"), "--w-aux-drv"), "the default 0.0 would disable the head: it must be NAMED"
assert has(refusals(drop(A0, "--aux-drv-lr"), "HA-s0"), "--aux-drv-lr"), "omitting the head's learning rate is refused (aux_s1_refusals)"
assert has(refusals(swap(A0, "--aux-drv-exclude-sha256", "0" * 64), "HA-s0"), "exclude-sha256"), "a wrong declared H48 sha256"
assert has(refusals(swap(A0, "--epochs", "1.0"), "HA-s0"), "--epochs"), "the parser default --epochs 1.0 ends the run at 7,500 updates"
assert has(refusals(swap(A0, "--aux-drv-tap", "scene"), "HA-s0"), "tap"), "tap scene is refused on this host"
assert refusals(A0 + ["--aux-drv-feed", "on"], "HA-s0") and refusals(A0 + ["--aux-drv-feed", "off"], "HA-s0"), "the feed is never named (off by default; on is H2)"
assert refusals(swap(A0, "--aux-drv-labels", "/workspace/data/aux_drivable/label_store_navtest"), "HA-s0"), "the navtest store is not the training store"
assert refusals(swap(A0, "--ckpt", "/workspace/data/rls1/base/model_sft4_B_u400.pt"), "HA-s0"), "u400 is not P6"
assert has(refusals(swap(A0, "--seed", "1"), "HA-s0"), "requires --seed") and has(refusals(swap(A1, "--seed", "0"), "HA-s1"), "requires --seed")
assert has(refusals(A0 + ["--aux-drv-shuffle-seed", "0"], "HA-s0"), "HA-SHUF only"), "a shuffle on a real arm"
assert has(refusals(drop(AS, "--aux-drv-shuffle-seed"), "HA-SHUF"), "missing"), "HA-SHUF without its shuffle"
assert refusals(swap(AS, "--aux-drv-shuffle-seed", "1"), "HA-SHUF"), "another shuffle seed"
assert refusals(swap(AS, "--max-updates", "10000"), "HA-SHUF") and refusals(swap(A0, "--max-updates", "2000"), "HA-s0")
assert refusals(A0 + ["--lane-head"], "HA-s0") and refusals(A0 + ["--pair-w", "0.5", "--pair-heads", "NC"], "HA-s0") and refusals(A0 + ["--split-half", "0"], "HA-s0")
assert refusals(A0 + ["--aux-drv-exclude-logs", L.VDAC_H48_PATH], "HA-s0"), "a flag twice"
assert refusals(VA["HA-s0"], "HA-s1") and refusals(VA["HA-SHUF"], "HA-s0"), "an arm's argv under another arm's name"
assert refusals(P0, "HA-s0") and refusals(L.scorer1p6_argv("C0r"), "HA-s1"), "C0's own argv is not a VDAC arm"
SHA[L.VDAC_H48_PATH] = "0" * 64
assert has(refusals(A0, "HA-s0"), "H48 list"), "an H48 file with other bytes"
SHA[L.VDAC_H48_PATH] = L.VDAC_H48_SHA256
SHA[L.SCORER1P6_CKPT] = "0" * 64
assert has(refusals(A0, "HA-s0"), "P6's model_final.pt")
SHA[L.SCORER1P6_CKPT] = L.SCORER1P6_CKPT_SHA256
assert refusals(A0, "HA-s0", L.PROFILES["refe_aux_s1c"]) and refusals(A0, "H1u10k-s0", L.PROFILES["refe_aux_s1c"]), "the older S1c profile does not take a VDAC argv"
for bad in ("C0", "HA-s2", "ha-s0", ""):
    try:
        L.vdac_argv(bad)
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown arm %r was accepted" % bad)
assert L.VDAC_H48_SHA256 == PV["required_flags"]["--aux-drv-exclude-sha256"] and set(PV["vdac_arms"]) == set(ARMS) and PV["arm_class"] == {"HA-s0": "H1ha", "HA-s1": "H1ha", "HA-SHUF": "H1shuf"}
assert PV["data_expect"] == L.PROFILES["refe_scorer1p6"]["data_expect"] and PV["data_expect_updates"] is None and PV["schedule_reach"] is True
assert "--aux-drv-exclude-logs" in PV["data_flags"] and "--aux-drv-labels" in PV["data_flags"] and "--ckpt" in PV["data_flags"] and "--aux-drv-shuffle-seed" not in PV["forbidden_flags"]
assert PV["suite"][:-1] == L.AUXS1C_SUITE and PV["suite"][-1] == {"script": "test_vdac_gate.py", "token": "ZZTEST_PASS"}
assert "--lane-head" not in PV["required_flags"] and "--teacher-lane-train" not in PV["required_flags"] and PV["required_flags"]["--epochs"] == "2"
# the schedule is reachable: 10,000 / 2,000 updates on 60,000 sets at --epochs 2 (and 7,500 at the parser's 1.0: the reason the flag is named)
for a, want in (("HA-s0", 10000), ("HA-s1", 10000), ("HA-SHUF", 2000)):
    rr = [r for r in C.scorer_schedule_rows(parse_ns(VA[a]), {"train_sets": 60000}) if r["dest"] == "max_updates"]
    assert rr and rr[0]["ok"] and rr[0]["built"] == want, rr
rr = C.scorer_schedule_rows(parse_ns(swap(A0, "--epochs", "1.0")), {"train_sets": 60000})
assert not rr[0]["ok"] and rr[0]["built"] == 7500, rr
print("V5 ok (3 arms legal; argv diff vs C0 = budget / seed / --out + the head tail; %d dropped-flag cases + ~25 moved-lever cases refused; --epochs 2 reaches 10,000 / 2,000 updates, 1.0 would stop at 7,500)" % cases)

# ---------------------------------------------------------------- V6 fingerprints


def fp(n):
    return hashlib.sha256(repr(sorted(L.PROFILES[n].items())).encode()).hexdigest()


TIP_FP = {"refe_p2": "627341286aea3506", "refe_p6": "80fa4d7ec757ad46", "refe_field": "5ddc49053a99fa78", "refe_field_ctrl": "0b6281b1bfa996ed", "refe_rehearsal": "821cd1ef3bf55d4e",
          "refe_rls1": "bce7e25ee8718c08", "refe_aux_s1": "f95f48727b62ccc6", "refe_p7": "94cca167ce121a94", "refe_p2rep": "44ca107f53aceee8", "refe_sft4r": "8972e3c348206b94",
          "refe_scorer1": "88d02fb7b34d4fd8", "refe_scorer1p6": "e74d64617f514d35", "refe_s1z_p6": "5c3bff457aa3bf1e", "refe_rep_p6": "dd348f44c9638f1a", "refe_rep_s1z_p6": "2356afd4efeeb52a",
          "refe_rls1b": "5d6bfbc10152ae96", "refe_rls1c": "26214cf954099d51", "refe_rls1d": "fcc2585f2a5d9b94", "refe_rls1e": "54d96289b19c3cd6", "refe_hnscreen": "d30126bbb5a32957",
          "refe_aux_s1c": "dcdf2dbd2a674f1c", "refe_s1zsm_p6": "86380fbb786daa75", "refe_rep_s1zsm_p6": "27fe236a2fd68ca4", "refe_dose_s1z_p6": "9085f9e5cf59c016",
          "refe_div_p6": "e51f9374a8f29816"}      # the 24 pinned at ee1ba225 (test_div_gate) + refe_div_p6 as landed with DIV: MEASURED with this edit applied
moved = [n for n, want in TIP_FP.items() if not fp(n).startswith(want)]
assert not moved, f"existing profile fingerprints moved: {moved}"
assert set(L.PROFILES) - set(TIP_FP) == {"refe_vdac_p6"} and len(L.PROFILES) == 26
r = repr(sorted(PV.items()))
assert fp("refe_vdac_p6") == fp("refe_vdac_p6") and "function" not in r and "0x" not in r and fp("refe_vdac_p6") != fp("refe_aux_s1c")
base = L.PROFILES["refe_aux_s1c"]
assert PV is not base and PV["required_flags"] is not base["required_flags"] and PV["arm_recipe"] is not base["arm_recipe"] and PV["forbidden_flags"] != base["forbidden_flags"]
print("V6 ok (the 25 older digests unmoved; NEW refe_vdac_p6 = %s...)" % fp("refe_vdac_p6")[:16])

# ---------------------------------------------------------------- V7 the layer
assert VL.has_layer(TRAINER) and VL.MARKER == "AUX_EXCL_MUTATE_ENV = "
pre = VL.peel(TRAINER)
assert not VL.has_layer(pre) and blob(pre) == VL.TIP_TRAINER_BLOB == "e8b862601a97a3910ec712dc5ac75026b16af75f", blob(pre)
assert VL.peel(pre) == pre and "aux_drv_exclude" not in pre and "AuxSubset(ds, idx, src, epo)" in pre
dp = DL.peel(TRAINER)
assert blob(dp) == DL.TIP_TRAINER_BLOB == "d665ed995d064f9e1cae956c3c48867cd348cd3d", "DIV's pin still peels (VDAC first, then DIV)"
assert ZSM.peel(TRAINER) == ZSM.peel(pre) and blob(ZSM.peel(TRAINER)) == ZSM.TIP_TRAINER_BLOB == "6a16d35c1818422ff514e97c4b292c32dde81795", "S1ZSM's pin still peels (VDAC, DIV, then S1ZSM)"
for k, (new, old) in enumerate(VL.REVERSALS):
    bad = TRAINER.replace(new, new.replace("e", "E", 1), 1)
    try:
        VL.peel(bad)
    except AssertionError:
        continue
    raise AssertionError("VDAC reversal %d: a drifted trainer was accepted" % k)
print("V7 ok (peel -> the tip blob e8b86260; DIV's d665ed99 and S1ZSM's 6a16d35c unchanged; every reversal refuses a drifted trainer)")

# ---------------------------------------------------------------- V8 the scripts
bash = None
for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe"):
    if os.path.exists(cand):
        bash = cand
        break
if bash is None:
    import shutil
    bash = shutil.which("bash")
assert bash, "no bash"


def sh(args, env=None, inp=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([bash] + args, capture_output=True, text=True, encoding="utf-8", errors="replace", env=e, cwd=HERE, input=inp)


SCRIPTS = {n: open(os.path.join(HERE, n), "rb").read().decode("utf-8") for n in ("vdac_pod.sh", "vdac_chain.sh", "vdac_pause_p7.sh")}
for n, t in SCRIPTS.items():
    assert "\r" not in t, n + " must be LF-only"
    r_ = sh(["-n", os.path.join(HERE, n)])
    assert r_.returncode == 0, (n, r_.stderr[:300])
for a in ARMS:
    r_ = sh([os.path.join(HERE, "vdac_pod.sh")], env={"ARM": a, "MODE": "argv", "VDAC_WS": "/workspace"})
    assert r_.returncode == 0 and r_.stdout.split() == VA[a], (a, r_.stderr[:200], set(r_.stdout.split()) ^ set(VA[a]))
    pc = sh([os.path.join(HERE, "scorer1p6_pod.sh")], env={"ARM": "C0", "MODE": "argv", "SCORER1P6_WS": "/workspace"}).stdout.split()
    assert differing_flags(r_.stdout.split(), pc) >= {"--aux-drv", "--epochs"}
for env in ({"ARM": "HA-s2", "MODE": "argv"}, {"ARM": "HA-s0", "MODE": "nope"}, {"ARM": "", "MODE": "argv"}):
    r_ = sh([os.path.join(HERE, "vdac_pod.sh")], env=env)
    assert r_.returncode == 2, env
# the pod script's mutations: each makes its argv differ from launch_gate.vdac_argv
for old, new in (("--epochs 2", "--epochs 1"), ("--w-aux-drv 0.1 ", ""), ("--aux-drv-exclude-sha256 $H48_SHA ", ""), ("--aux-drv-exclude-logs $H48 ", ""), ("UPD=10000; EVAL=5000; SHUF=\"\" ;;\n  HA-s1", "UPD=8000; EVAL=5000; SHUF=\"\" ;;\n  HA-s1"),
                 ("SHUF=\"--aux-drv-shuffle-seed 0\"", "SHUF=\"\"")):
    t = SCRIPTS["vdac_pod.sh"]
    assert t.count(old) >= 1, old
    with tempfile.TemporaryDirectory(dir=tmp) as td:
        shutil_dst = os.path.join(td, "vdac_pod.sh")
        open(shutil_dst, "w", encoding="utf-8", newline="\n").write(t.replace(old, new))
        for pn in ("scorer1p6_pod.sh",):
            open(os.path.join(td, pn), "w", encoding="utf-8", newline="\n").write(open(os.path.join(HERE, pn), encoding="utf-8").read())
        diffs = 0
        for a in ARMS:
            r_ = sh([shutil_dst], env={"ARM": a, "MODE": "argv", "VDAC_WS": "/workspace"})
            if r_.returncode != 0 or r_.stdout.split() != VA[a]:
                diffs += 1
        assert diffs >= 1, "MUTATION %r of vdac_pod.sh must move an arm's argv off launch_gate.vdac_argv" % old
# the mutant mode is wired to the env the trainer reads, on the REAL argv, and refuses to train when the audit passes it
assert "REFE_AUXEXCL_MUTATE_NOFILTER=1" in SCRIPTS["vdac_pod.sh"] and SF.AUX_EXCL_MUTATE_ENV in SCRIPTS["vdac_pod.sh"] and "--preflight --out $OUT/mutate_exclude" in SCRIPTS["vdac_pod.sh"]
# ---- the chain
ch = SCRIPTS["vdac_chain.sh"]
for tok in ("ZZVDAC_START", "ZZVDAC_PRECHECK_OK", "ZZVDAC_INPUTS_OK", "ZZVDAC_EGR_OK", "ZZVDAC_PREFLIGHT_OK", "ZZVDAC_GATE_OK", "ZZVDAC_TRAIN_START", "ZZVDAC_TRAIN_DONE", "ZZVDAC_EXPORT_OK", "ZZVDAC_ARM_DONE", "ZZVDAC_CHAIN_DONE", "ZZVDAC_CHAIN_FAIL",
            "vdac_postgate.py", "vdac_export_h48.py", "MODE=mutate", "REFUSED: stage=", "excl_is", "verify --profile $PROFILE", "launch_gate.py run --profile $PROFILE --arm-name", "H48_SHA=" + L.VDAC_H48_SHA256, "5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4",
            "8f121bdf9d6f19acec8927f12c50d3f7be2832c70c7f1fe40140d5eb8cc4ed70", "c3d563586be6479ea11455f92d7a5295c9e1658e2d7ba1cc77b27e87e5c8b417", L.SCORER1P6_CKPT_SHA256, "ZZVDAC_LFS_SHA", "STORE_KEYS=103139"):
    assert tok in ch, tok
assert L.VDAC_CARD_SHA256 in ch and L.VDAC_A1_SHA256 in ch and L.VDAC_A2_SHA256 in ch and "profile\n" not in ch
# no success line carries the bare word REFUSED (a watcher matched DIV's "mutant REFUSED as required" as a failure); failure lines carry the registered prefix
for ln in ch.split("\n"):
    ln_out = ln.replace("'^REFUSED: aux-excl:'", "").replace("s/^REFUSED: //", "")          # the grep / sed PATTERNS name the trainer's refusal; they do not print it
    if re.match(r"^\s*(log|echo) ", ln) and "REFUSED" in ln_out:
        assert "REFUSED: stage=" in ln or "ZZVDAC_CHAIN_FAIL" in ln or "refuse()" in ln, "a success path prints the bare word REFUSED: " + ln[:160]
assert 'log "$ARMN NOFILTER mutant rejected as required' in ch and "mutant REFUSED" not in ch
assert re.search(r'refuse\(\) \{ log "REFUSED: stage=\$STAGE: \$\*"; exit 1; \}', ch)
# the chain's python on synthetic logs: the exclusion audit and the run check
heres = re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", ch, re.S)
excl_py = next(h for h in heres if "aux_excl" in h and "derangement" in h)
run_py = next(h for h in heres if "ZZVDAC_RUN_OK" in h)
with tempfile.TemporaryDirectory(dir=tmp) as td:
    def run_py_code(code, argv):
        p = os.path.join(td, "snippet.py")
        open(p, "w", encoding="utf-8", newline="\n").write(code)
        return subprocess.run([sys.executable, p] + argv, capture_output=True, text=True, encoding="utf-8", errors="replace")

    def write_log(name, data_extra=None, aux_extra=None):
        d = good_data("HA-SHUF" if name.startswith("shuf") else "HA-s0")
        if data_extra:
            d["aux_excl"] = {**d["aux_excl"], **data_extra}
        if aux_extra:
            d["aux_drv"] = {**d["aux_drv"], **aux_extra}
        p = os.path.join(td, name)
        open(p, "w", encoding="utf-8").write("noise\n" + json.dumps({"event": "data", **d}) + "\n")
        return p
    good_p, shuf_p = write_log("real.log"), write_log("shuf.log")
    r_ = run_py_code(excl_py, [good_p, "none", L.VDAC_H48_SHA256, "103139"])
    assert r_.returncode == 0 and "labelled frames excluded 1200" in r_.stdout, r_.stderr
    r_ = run_py_code(excl_py, [shuf_p, "0", L.VDAC_H48_SHA256, "103139"])
    assert r_.returncode == 0, r_.stderr
    for nm, kw, shuf in (("reaching", {"data_extra": {"reached_loss_rows": 3}}, "none"), ("nofilter", {"data_extra": {"filter_equals_file": False}}, "none"), ("zeroexcl", {"data_extra": {"excluded_labelled_frames": 0}}, "none"),
                         ("wrongsha", {"data_extra": {"sha256": "0" * 64}}, "none"), ("shufonreal", {"aux_extra": {"shuffle_seed": 0}}, "none"), ("shufmissing", {}, "0"),
                         ("fixedpoint", {"aux_extra": {"derangement_fixed_points": 2}}, "0")):
        pth = write_log(("shuf_" if shuf == "0" and nm != "shufmissing" else "") + nm + ".log", **kw) if nm != "shufmissing" else good_p
        r_ = run_py_code(excl_py, [pth, shuf, L.VDAC_H48_SHA256, "103139"])
        assert r_.returncode != 0, "MUTATION %r of the chain's exclusion audit must fail" % nm
    r_ = run_py_code(excl_py, [os.path.join(td, "nolog.log"), "none", L.VDAC_H48_SHA256, "103139"])
    assert r_.returncode != 0
    # the run check
    outd = os.path.join(td, "out")
    os.makedirs(os.path.join(outd, "scores"))

    def write_run(upd=10000, mut=None):
        d = good_data("HA-s0")
        ev = [{"event": "start", "updates": upd, "train_sets": 60000}, {"event": "data", **d}, {"event": "aux_saved", "path": "x"}, {"event": "h48x_forward", "frames": 1200, "file": "head_pred_h48x.npz"},
              {"event": "done", "updates": upd}]
        if mut == "reach":
            ev[1]["aux_excl"]["reached_loss_rows"] = 5
        if mut == "frames":
            ev[3]["frames"] = 1199
        if mut == "short":
            ev[4]["updates"] = upd - 1
        if mut == "nosave":
            ev = [e for e in ev if e["event"] != "aux_saved"]
        open(os.path.join(outd, "sft.jsonl"), "w", encoding="utf-8").write("\n".join(json.dumps(e) for e in ev) + "\n")
        open(os.path.join(outd, "sft.log"), "w", encoding="utf-8").write("ZZSFT_DONE\n")
        for fnm in ("aux_drv.pt", "head_pred_h48x.npz", "model_sft_A.pt", "model_sft_B.pt", os.path.join("scores", "head_pred_update%d.npz" % upd)):
            open(os.path.join(outd, fnm), "wb").write(b"x")
    write_run()
    r_ = run_py_code(run_py, [outd, "10000", L.VDAC_H48_SHA256])
    assert r_.returncode == 0 and "ZZVDAC_RUN_OK" in r_.stdout, r_.stderr
    for mut in ("reach", "frames", "short", "nosave"):
        write_run(mut=mut)
        assert run_py_code(run_py, [outd, "10000", L.VDAC_H48_SHA256]).returncode != 0, "MUTATION %r of the run check must fail" % mut
    write_run()
    os.remove(os.path.join(outd, "scores", "head_pred_update10000.npz"))
    assert run_py_code(run_py, [outd, "10000", L.VDAC_H48_SHA256]).returncode != 0, "a missing held-out prediction file"
    write_run(upd=2000)
    assert run_py_code(run_py, [outd, "2000", L.VDAC_H48_SHA256]).returncode == 0
    write_run(upd=2000)
    assert run_py_code(run_py, [outd, "10000", L.VDAC_H48_SHA256]).returncode != 0, "the wrong update count for the arm"
# chain structure: mutants refused-or-no-train, the token before training, the post gate before the export, the export before the egress
i = {k: ch.find(k) for k in ("ARM=$ARMN MODE=preflight", "ARM=$ARMN MODE=mutate", "$PY_C\" $RUN/launch_gate.py run --profile", "token_ok \"$ARMN\" || refuse \"the launch-gate token", "(ZZVDAC_TRAIN_START)", "ARM=$ARMN MODE=run", "$RUN/vdac_postgate.py \"$OUT\"",
                                          "$RUN/vdac_export_h48.py --h48", "$HPY $RUN/hf_egress.py push --repo")}
ks = list(i)
assert all(v > 0 for v in i.values()) and [i[k] for k in ks] == sorted(i[k] for k in ks), i          # preflight < mutant < gate < token before the trainer < train < post gate < export < egress
assert 'grep -q "^REFUSED: aux-excl:"' in ch and '[ "$mrc" = 4 ]' in ch and "mutate_exclude.log" in ch and "no training" in ch
for old in ('[ "$mrc" = 4 ] && ', '! grep -q "^ZZSFT_PREFLIGHT_PASS" $OUT/mutate_exclude.log || refuse'):
    assert ch.count(old) == 1, old      # the chain's own guards exist exactly once (the chain proper runs on the pod only)
# ---- the pause script
pz = SCRIPTS["vdac_pause_p7.sh"]
assert 'S1a) ARMS="HA-s0 HA-SHUF"' in pz and 'S1b) ARMS="HA-s1"' in pz and "S1B_GO" in pz and "trap 'resume_p7' EXIT" in pz and "P7_RESUMED" in pz and "PAUSE_STOPPED_EARLY" in pz and "HOLD_P7" in pz
assert pz.find("S1B_GO") < pz.find("kill -TERM $CHAIN"), "S1b's condition is checked BEFORE P7 is touched"
assert re.search(r'S1b\) ARMS="HA-s1"; \[ -e "\$OUT/S1B_GO" \] \|\| early', pz)
assert "ZZVDAC_CHAIN_DONE $A" in pz and "the remaining arms are NOT run" in pz and "refe-plan-vdac/refe/vdac_chain.sh" in pz
with tempfile.TemporaryDirectory(dir=tmp) as td:
    # S1b without S1B_GO ends before touching P7 (no /proc needed: the guard fires first); S1a with unknown stage name too
    t_ = pz.replace("/workspace/data/refe_vdac", td.replace("\\", "/"))
    pp = os.path.join(td, "vdac_pause_p7.sh")
    open(pp, "w", encoding="utf-8", newline="\n").write(t_)
    r_ = sh([pp, "1", "2", "S1b"])
    logt = open(os.path.join(td, "pause.log"), encoding="utf-8").read()
    assert r_.returncode == 1 and "PAUSE_STOPPED_EARLY S1b needs" in logt and "P7 was not touched" in logt, logt
    r_ = sh([pp, "1", "2", "S9"])
    assert r_.returncode == 1 and "unknown stage S9" in open(os.path.join(td, "pause.log"), encoding="utf-8").read()
    open(os.path.join(td, "S1B_GO"), "w").write("go\n")
    r_ = sh([pp, "1", "2", "S1b"])
    assert "S1b needs" not in open(os.path.join(td, "pause.log"), encoding="utf-8").read().split("unknown stage S9")[-1], "with S1B_GO present the guard passes (the next refusal is the missing token file)"
print("V8 ok (bash -n x3; vdac_pod.sh == launch_gate.vdac_argv on 3 arms, 6 pod-script mutations move an argv; the chain's exclusion audit (9 cases) and run check (7 cases) on synthetic logs; no success line prints REFUSED; stage order; the pause script's S1a / S1b / S1B_GO / trap)")

# ---------------------------------------------------------------- V9 the post-training gate on the REAL head
cfg = AUX.AuxDrivableConfig(tap="visual", depth=2, feed="off", w_aux=0.1, seed=0)
try:
    cfg.aux_drv_feeed = "on"          # G-HYG (card section 7): the head's config refuses an undeclared aux_drv* attribute (the typo of `--aux-drv-feed`)
except Exception as e_:
    assert type(e_).__name__ == "FrozenInstanceError", type(e_)
else:
    raise AssertionError("AuxDrivableConfig accepted an undeclared attribute")
assert "a." + "aux_drv_feeed" not in TRAINER and "aux_drv_feeed" not in C.namespace_reads(TRAINER, C.SCORER_NAMESPACE_FUNCS), "the trainer reads no undeclared aux_drv* attribute"
MISSPELT = TRAINER + chr(10) * 2 + "def main2(a):" + chr(10) + "    return a.aux_drv_feeed" + chr(10)
assert "aux_drv_feeed" in C.namespace_reads(MISSPELT, ("main", "main2")), "MUTATION: a namespace read of a misspelt aux_drv attribute is visible to G-HYG's scan"
aux = AUX.AuxDrivable(cfg)
params = list(aux.head.parameters())
n_real = sum(p.numel() for p in params)
assert n_real == PG.N_PARAMS_CARD == 2440450, n_real          # the card's built head, MEASURED here on CPU


def trained_ckpt(steps, drop_state=None, zero_m=None, step_off=0):
    head = AUX.AuxDrivable(cfg).head
    opt = torch.optim.AdamW(list(head.parameters()), lr=2e-4, weight_decay=0.01)
    g = torch.Generator().manual_seed(1)
    for _ in range(steps):
        for p in head.parameters():
            p.grad = torch.randn(p.shape, generator=g) * 0.01
        opt.step()
        opt.zero_grad(set_to_none=True)
    sd = opt.state_dict()
    if drop_state is not None:
        del sd["state"][drop_state]
    if zero_m is not None:
        sd["state"][zero_m]["exp_avg"] = torch.zeros_like(sd["state"][zero_m]["exp_avg"])
    if step_off:
        sd["state"][0]["step"] = sd["state"][0]["step"] + step_off
    return {"format": "aux_drv/1", "head": head.state_dict(), "feed": None, "opt": sd, "update": steps, "meta": {}}


ck = trained_ckpt(2)
assert PG.judge(ck, 2) == [], PG.judge(ck, 2)
assert has(PG.judge(trained_ckpt(2, drop_state=3), 2), "NO Adam state"), "MUTATION: a head parameter that never received a gradient"
assert has(PG.judge(trained_ckpt(2, zero_m=5), 2), "ZERO first moment"), "MUTATION: only zero gradients were applied"
assert has(PG.judge(trained_ckpt(2, step_off=1), 2), "step count"), "MUTATION: a wrong step count"
assert has(PG.judge(ck, 3), "checkpoint update"), "the arm's update count"
assert has(PG.judge(ck, 2, n_params=2440451), "the card's built head"), "a wrong parameter count"
assert has(PG.judge({"format": "x"}, 2), "not an aux_drv/1") and has(PG.judge({**ck, "opt": None}, 2), "no head optimiser state")
bad_head = dict(ck["head"])
k0 = next(k for k, v in bad_head.items() if v.dtype.is_floating_point)
bad_head[k0] = bad_head[k0] * float("nan")
assert has(PG.judge({**ck, "head": bad_head}, 2), "non-finite")
with tempfile.TemporaryDirectory(dir=tmp) as td:
    torch.save(ck, os.path.join(td, "aux_drv.pt"))
    r_ = subprocess.run([sys.executable, os.path.join(HERE, "vdac_postgate.py"), td, "--updates", "2"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r_.returncode == 0 and "ZZVDAC_POSTGATE_OK" in r_.stdout, r_.stdout + r_.stderr
    r_ = subprocess.run([sys.executable, os.path.join(HERE, "vdac_postgate.py"), td, "--updates", "5"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r_.returncode == 1 and "ZZVDAC_POSTGATE_FAIL" in r_.stdout
    r_ = subprocess.run([sys.executable, os.path.join(HERE, "vdac_postgate.py"), os.path.join(td, "none"), "--updates", "2"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r_.returncode == 1
print("V9 ok (the real head has 2,440,450 parameters; 2 AdamW steps pass; a dropped state, a zeroed first moment, a wrong step / update / parameter count and a NaN tensor are each RED)")

# ---------------------------------------------------------------- V10 export, H48 build, derangement
import train as T  # noqa: E402
import vdac_build_h48 as B  # noqa: E402
import vdac_export_h48 as X  # noqa: E402

with tempfile.TemporaryDirectory(dir=tmp) as td:
    def mk_set(lg, tok, step, rank, ckpt, n_prop=4, hor=3):
        rg = np.random.RandomState(abs(hash((lg, tok, step))) % (2 ** 31))
        return {"kind": "onpolicy_set", "log_name": lg, "token": tok, "step": step, "rank": rank, "ckpt_step": ckpt, "traj": rg.randn(n_prop, hor, 2).tolist(), "yaw": rg.randn(n_prop, hor).tolist(),
                "targets": [{"collision.NuPlanCollision.info": 0.0, "dac.violation": float(k % 2), "progress.ep": 0.5, "ttc.NuPlanTTC.ttc_reward": 1.0} for k in range(n_prop)]}
    for tag, lgs in (("train", ["logA", "logB", "logC"]), ("heldout", ["logH1", "logH2"])):
        sd = os.path.join(td, tag + "_sets")
        os.makedirs(sd)
        with open(os.path.join(sd, "onpolicy_000.jsonl"), "w", encoding="utf-8") as f:
            for lg in lgs:
                for k in range(2):
                    f.write(json.dumps(mk_set(lg, "tok%d" % k, 0, 0, 100)) + "\n")
        pd = os.path.join(td, tag + "_pdm")
        os.makedirs(pd)
        with open(os.path.join(pd, "pdm_000.jsonl"), "w", encoding="utf-8") as f:
            for lg in lgs:
                for k in range(2):
                    if lg == "logC" and k == 1:
                        continue          # a set of an H48 log WITHOUT a PDM label: counted, not exported
                    f.write(json.dumps({"key": [lg, "tok%d" % k, 0, 0], "ckpt_step": 100, **{c: [0.25 * j for j in range(4)] for c in T.OnPolicyBank.PDM_ORDER}}) + "\n")
    arr, rep = X.export([("train", os.path.join(td, "train_sets"), os.path.join(td, "train_pdm")), ("heldout", os.path.join(td, "heldout_sets"), os.path.join(td, "heldout_pdm"))], ["logB", "logC", "logH2", "logZ"], 4, 3)
    assert rep["banks"]["train"]["with_pdm_label"] == 3 and rep["banks"]["train"]["without_pdm_label"] == 1 and rep["banks"]["heldout"]["with_pdm_label"] == 2 and rep["sets"] == 5 and rep["frames"] == 5, rep
    assert arr["traj"].shape == (5, 4, 3, 3) and arr["comp"].shape == (5, 4, 6) and list(arr["comp_order"]) == ["nc", "dac", "ep", "ttc", "c", "ddc"] and set(arr["log"]) == {"logB", "logC", "logH2"}
    assert np.allclose(arr["comp"][0], np.clip(np.stack([np.asarray([0.25 * j for j in range(4)])] * 6, -1), 0, 1)), "the PDM labels replace the six components"
    try:
        X.export([("train", os.path.join(td, "train_sets"), os.path.join(td, "train_pdm"))], ["logZ"], 4, 3)
    except SystemExit:
        pass
    else:
        raise AssertionError("an export of 0 sets was accepted")
    # the CLI refuses a list with other bytes than the declared sha256
    hp = os.path.join(td, "h.txt")
    open(hp, "wb").write(b"logB\n")
    r_ = subprocess.run([sys.executable, os.path.join(HERE, "vdac_export_h48.py"), "--h48", hp, "--h48-sha256", "0" * 64, "--train-sets", "a", "--train-pdm", "b", "--heldout-sets", "c", "--heldout-pdm", "d",
                         "--horizon", "3", "--out", os.path.join(td, "o.npz")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r_.returncode == 1 and "ZZVDAC_EXPORT_FAIL" in r_.stdout
    # H48 build on a synthetic store: 24 given logs + the 8 lowest sha256(log) per SG / BOS / PIT outside them
    rg = np.random.RandomState(7)
    files, ncity = {}, {"us-nv-las-vegas-strip": 60, "us-ma-boston": 30, "us-pa-pittsburgh-hazelwood": 30, "sg-one-north": 30}
    allnames = {}
    for mp, n in ncity.items():
        for k in range(n):
            nm = "2021.%s_%04d" % (mp[:5], k)
            files["shards/%s.aux1" % nm] = {"bytes": 1, "sha256": "0" * 64, "n": 3 + k % 4, "map": mp}
            allnames.setdefault(mp, []).append(nm)
    sft = allnames["us-nv-las-vegas-strip"][:15] + allnames["us-pa-pittsburgh-hazelwood"][:4] + allnames["us-ma-boston"][:4] + allnames["sg-one-north"][:1]
    sdir = os.path.join(td, "store")
    os.makedirs(sdir)
    man = {"store_sha256": "f" * 64, "files": files, "counts": {"logs": len(files), "stored_keys": sum(v["n"] for v in files.values())}}
    json.dump(man, open(os.path.join(sdir, "manifest.json"), "w"))
    sp = os.path.join(td, "sft.txt")
    open(sp, "wb").write(("\n".join(sft) + "\n").encode())
    res = B.build(sdir, sp, pinned="f" * 64)
    assert len(res["h48"]) == 48 and set(sft) <= set(res["h48"]) and res["h48"] == sorted(res["h48"])
    for city, mp in (("SG", "sg-one-north"), ("BOS", "us-ma-boston"), ("PIT", "us-pa-pittsburgh-hazelwood")):
        pool = sorted((n for n in allnames[mp] if n not in sft), key=lambda n: (hashlib.sha256(n.encode()).hexdigest(), n))
        assert res["extra"][city] == pool[:8], city
    assert not (set(res["h48"]) & {n for n in allnames["us-nv-las-vegas-strip"] if n not in sft}), "no extra LV log"
    for bad_args in ((sdir, sp, "0" * 64),):
        try:
            B.build(*bad_args)
        except SystemExit as e:
            assert "REFUSED" in str(e)
        else:
            raise AssertionError("an unpinned store was accepted")
    open(sp, "wb").write(("\n".join(sft[:23]) + "\n").encode())
    try:
        B.build(sdir, sp, pinned="f" * 64)
    except SystemExit as e:
        assert "24" in str(e)
    else:
        raise AssertionError("a 23-log SFT list was accepted")
# the committed list: 48 sorted LF lines, sha256 == the profile's == the chain's == the pod script's
lp = os.path.join(HERE, "..", "splits", "h48_logs.txt")
if os.path.isfile(lp):
    raw = open(lp, "rb").read()
    lines = raw.decode("utf-8").split("\n")
    assert hashlib.sha256(raw).hexdigest() == L.VDAC_H48_SHA256 and b"\r" not in raw and raw.endswith(b"\n") and lines[-1] == "" and len(lines) == 49 and lines[:-1] == sorted(lines[:-1]) and len(set(lines)) == 49
assert L.VDAC_H48_SHA256 in SCRIPTS["vdac_pod.sh"] and L.VDAC_H48_SHA256 in SCRIPTS["vdac_chain.sh"] and L.VDAC_H48_SHA256 == B.__dict__.get("H48_EXPECTED", L.VDAC_H48_SHA256)
# HA-SHUF: a derangement of all keys (0 fixed points, identical multiset), seed-stable, different seeds differ
keys = [("log%03d" % (i // 7), "t%d" % i) for i in range(1000)]
dr = AUX.derangement(keys, 0)
assert sorted(dr) == sorted(keys) and all(a != b for a, b in zip(keys, dr)) and AUX.derangement(keys, 0) == dr and AUX.derangement(keys, 1) != dr
print("V10 ok (export: PDM labels replace the six components, unlabelled sets counted not exported, a 0-set export refused; H48 build: the 8 lowest sha256 per city outside the 24, unpinned store / 23-log list refused; the committed list's sha256 is the profile's, the chain's and the pod script's; HA-SHUF map is a derangement)")

# ---------------------------------------------------------------- V11 the two launch blockers (card section 7): G19 head disjoint from ego / command, G20 label alignment
import torch.nn as nn  # noqa: E402
import vdac_probes as VP  # noqa: E402

torch.manual_seed(0)


class TinyModel(nn.Module):
    """the shape of REFe's forward on a tiny scale: images -> visual_ctx (a function of the images ONLY); ego + command -> ego_tok -> the trajectory"""

    def __init__(self):
        super().__init__()
        self.vis = nn.Linear(5, 256)
        self.ego_enc = nn.Linear(4 + 6, 256)
        self.cmd_embed = nn.Linear(3, 6)
        self.traj = nn.Linear(256, 8)

    def forward(self, img, ego, goal, route_ego_into_head=False, head=None):
        ctx = self.vis(img)                                                # [B, 12, 256]
        ego_tok = self.ego_enc(torch.cat([ego, self.cmd_embed(goal)], -1)).unsqueeze(1)
        traj = self.traj(ego_tok.squeeze(1))
        hctx = ctx + ego_tok if route_ego_into_head else ctx               # the deliberate regression (trainer: REFE_AUX_MUTATE_EGO)
        lg, sd = head(hctx)
        return lg, sd, traj


tm = TinyModel()
head_ = AUX.build_head(AUX.AuxDrivableConfig(tap="visual", depth=1))
img_, ego_, goal_ = torch.randn(2, 12, 5), torch.randn(2, 4), torch.eye(3)[:2]


def mk_run(route):
    def run(e, g):
        lg, sd, tr = tm(img_, e, g, route_ego_into_head=route, head=head_)
        return lg.sum() + sd.sum(), tr.sum()
    return run


g_ok = VP.disjoint_probe(mk_run(False), ego_, goal_)
assert g_ok["head_grad_ego"] is None and g_ok["head_grad_command"] is None and g_ok["control_grad_ego"] > 0 and g_ok["control_grad_command"] > 0 and g_ok["ok"] is True and VP.judge_g19(g_ok) == [], g_ok
g_bad = VP.disjoint_probe(mk_run(True), ego_, goal_)
assert g_bad["ok"] is False and g_bad["head_grad_ego"] > 0 and g_bad["head_grad_command"] > 0 and any("NOT disjoint" in r for r in VP.judge_g19(g_bad)), g_bad     # the regression routes ego_tok into the head: RED
# an exactly-zero gradient is admitted (None or 0.0); a path that merely reaches a tiny value is not
zero_run = lambda e, g: (head_(tm.vis(img_))[0].sum() + 0.0 * e.sum(), tm(img_, e, g, head=head_)[2].sum())  # noqa: E731
g_zero = VP.disjoint_probe(zero_run, ego_, goal_)
assert g_zero["head_grad_ego"] == 0.0 and g_zero["ok"] is True, g_zero
tiny_run = lambda e, g: (head_(tm.vis(img_))[0].sum() + 1e-12 * e.sum(), tm(img_, e, g, head=head_)[2].sum())  # noqa: E731
assert VP.disjoint_probe(tiny_run, ego_, goal_)["ok"] is False
# the control must read: a probe whose control does not depend on ego / command cannot prove anything
dead_ctl = lambda e, g: (head_(tm.vis(img_))[0].sum(), (tm.vis(img_)).sum())  # noqa: E731
assert any("not proved able" in r for r in VP.judge_g19(VP.disjoint_probe(dead_ctl, ego_, goal_)))
# a gate produced under the regression switch can never count; a missing field is refused
assert any("regression switch" in r for r in VP.judge_g19({**g_ok, "mutate": ["REFE_AUX_MUTATE_EGO"]}))
assert VP.judge_g19({k: v for k, v in g_ok.items() if k != "head_grad_command"}) and VP.judge_g19("x")


# ---- G20 on synthetic frames: a drivable corridor along +x (|y| < 3 m)
def corridor(half_w=3.0, y_c=0.0):
    lab = np.full((160, 128), -20, dtype=np.int8)
    ys = VP.Y0 + (np.arange(128) + 0.5) * VP.RES
    lab[:, np.abs(ys - y_c) < half_w] = 30
    return lab


rng = np.random.RandomState(1)
labs, xys, shifted = [], [], []
for k in range(80):
    yc = 8.0 if k % 2 == 0 else -8.0                                      # the road sits left of the ego on half the frames, right on the other half: a flip cannot hide
    labs.append(corridor(3.0, yc))
    xys.append(np.stack([np.linspace(2, 38, 20), yc + rng.uniform(-1, 1, 20)], 1))
    shifted.append(corridor(3.0, -yc))                                    # the next frame's road is on the other side (a lookup shifted by one frame)
g20 = VP.alignment_probe(labs, xys, shifted)
assert g20["ok"] and g20["frac_real"] >= 0.999 and g20["frac_yflip"] < 0.5 and g20["frac_shift1"] < 0.5 and g20["n_frames"] == 80 and VP.judge_g20(g20) == [], g20
# fewer than 64 frames; a misaligned (y-flipped) store; a check whose mutant ALSO passes (a symmetric road, a shift that reads the same label) -- each refused
assert any("frames" in r for r in VP.judge_g20(VP.alignment_probe(labs[:63], xys[:63], shifted[:63])))
assert any("lie inside the label" in r for r in VP.judge_g20(VP.alignment_probe([l[:, ::-1] for l in labs], xys, shifted)))
sym = [corridor(3.0, 0.0)] * 80
sym_x = [np.stack([np.linspace(2, 38, 20), rng.uniform(-1, 1, 20)], 1) for _ in range(80)]
rj = VP.judge_g20(VP.alignment_probe(sym, sym_x, shifted))
assert any("y-flipped" in r for r in rj), rj
assert any("one-frame-shifted" in r for r in VP.judge_g20(VP.alignment_probe(labs, xys, labs))), "a shift mutant that reads the same label passes: refused"
assert any("bar" in r for r in VP.judge_g20({**g20, "bar": 0.5}))
import aux_label_store as ALS  # noqa: E402
assert ALS.target_probe(labs[0], xys[0]) == VP.points_inside(labs[0], xys[0]), "the store's own QC rule"


class FakeStore2:
    def __init__(self, valid_tokens):
        self.v = valid_tokens

    def labels(self, keys):
        lab = np.stack([corridor(3.0, 8.0) if k[1] in self.v else np.zeros((160, 128), np.int8) for k in keys])
        return lab, np.array([k[1] in self.v for k in keys])


rows_ = []
for lg in ("logA", "logB"):
    for st in range(60):
        tok = "%s_%03d" % (lg, st)
        rows_.append({"log_name": lg, "token": tok, "step": st, "rank": 0, "traj": [[1.0 * j, 0.0, 0.0] for j in range(20)]})
        rows_.append({"log_name": lg, "token": tok, "step": st, "rank": 1, "traj": [[1.0 * j, 5.0, 0.0] for j in range(20)]})      # an augmented rank-1 row: never read
fs = FakeStore2({r["token"] for r in rows_ if r["token"] != "logA_010"})
all_idx = list(range(len(rows_)))
L1, X1, S1 = VP.align_frames(rows_, all_idx, fs)
assert 64 <= len(L1) == len(X1) == len(S1) and all(x.shape == (20, 2) and float(x[:, 1].max()) == 0.0 for x in X1), len(L1)       # the rank-1 rows (y = 5) never enter
L2, _, _ = VP.align_frames(rows_, all_idx, fs)
assert len(L1) == len(L2) and all((a == b).all() for a, b in zip(L1, L2)), "deterministic"
gate_ = VP.alignment_gate(rows_, all_idx, fs)
assert gate_["bank"] == {"sets": 240, "distinct_logs": 2, "distinct_frames": 120} and gate_["n_frames"] >= 64, gate_

# ---- the launch gate consumes them: required iff the exclusion lever is named; judged from the raw fields; an older arm is not asked
assert {"G19", "G20"} <= set(C.SCORER_GATES) and "G19" in C.scorer_required_gates(parse_ns(VA["HA-s0"])) and "G20" in C.scorer_required_gates(parse_ns(VA["HA-SHUF"]))
assert not {"G19", "G20"} & set(C.scorer_required_gates(parse_ns(L.scorer1p6_argv("C0"))))
LG_SRC = open(os.path.join(HERE, "launch_gate.py"), encoding="utf-8").read()
assert 'for gx_, judge_ in (("G19", "judge_g19"), ("G20", "judge_g20"))' in LG_SRC and "getattr(VP, judge_)(gv_)" in LG_SRC
# the trainer: the regression switch exists and routes the ego token into the head's context; the gates are built iff the lever is named, on the real fwd; the layer peels them away
assert 'VDAC_MUTATE_EGO_ENV = "REFE_AUX_MUTATE_EGO"' in TRAINER and "model.ego_enc(torch.cat([ego, model.cmd_embed(goal.to(ego.dtype))], -1))" in TRAINER
assert "if aux is not None and a.aux_drv_exclude_logs:" in TRAINER and 'gates["G19_vdac_head_disjoint_from_ego_command"] = VP.disjoint_probe(_vdac_run, batch[1], batch[2]' in TRAINER
assert 'gates["G20_vdac_label_aligned_with_targets"] = VP.alignment_gate(ds.rows, tr_idx, aux_src_eval)' in TRAINER and "VDAC_MUTATE_EGO_ENV" not in pre and "G19_vdac" not in pre
# the pod script and the chain
assert "mutate_ego" in SCRIPTS["vdac_pod.sh"] and "REFE_AUX_MUTATE_EGO=1" in SCRIPTS["vdac_pod.sh"] and "--out $OUT/mutate_ego" in SCRIPTS["vdac_pod.sh"]
for tok in ("MODE=mutate_ego", "ZZVDAC_G19_G20", "EGO-INTO-HEAD mutant rejected by G19", 'refuse "$ARMN preflight: G19/G20 not proved', "NOT disjoint", "ZZVDAC_UPDATES"):
    assert tok in ch, tok
g_py = next(h for h in re.findall(r"<<'PYEOF'[^\n]*\n(.*?)\nPYEOF", ch, re.S) if "G19 head grad wrt ego" in h)
with tempfile.TemporaryDirectory(dir=tmp) as td:
    def pf_log(g19, g20, name):
        pth = os.path.join(td, name)
        open(pth, "w", encoding="utf-8", newline="\n").write(json.dumps({"event": "preflight", "verdict": "PASS", "gates": {"G19_vdac_head_disjoint_from_ego_command": g19, "G20_vdac_label_aligned_with_targets": g20}}) + "\n")
        return pth
    good19 = {"inputs": {"ego": [8, 11], "command": [8, 3]}, "head_grad_ego": None, "head_grad_command": None, "control_grad_ego": 0.3, "control_grad_command": 0.1, "mutate": [], "ok": True}
    good20 = {"n_frames": 200, "n_points": 4000, "bar": 0.95, "min_frames": 64, "frac_real": 0.9999, "frac_yflip": 0.7, "frac_shift1": 0.6, "ok": True, "bank": {"sets": 60000, "distinct_logs": 900, "distinct_frames": 30000}}
    sn = os.path.join(td, "g.py")
    open(sn, "w", encoding="utf-8", newline="\n").write(g_py)

    def run_g(g19, g20):
        return subprocess.run([sys.executable, sn, pf_log(g19, g20, "pf.jsonl"), HERE], capture_output=True, text=True, encoding="utf-8", errors="replace")
    r_ = run_g(good19, good20)
    assert r_.returncode == 0 and "G20 200 frames" in r_.stdout and "'sets': 60000" in r_.stdout, (r_.stdout, r_.stderr)
    for bad19, bad20, why in (({**good19, "head_grad_ego": 0.2, "ok": True}, good20, "NOT disjoint"), (good19, {**good20, "frac_yflip": 0.99, "ok": True}, "y-flipped"), (good19, {**good20, "frac_real": 0.5, "ok": True}, "lie inside the label"),
                              (good19, {**good20, "n_frames": 10}, "frames"), ({**good19, "mutate": ["REFE_AUX_MUTATE_EGO"]}, good20, "regression switch")):
        r_ = run_g(bad19, bad20)
        assert r_.returncode != 0 and why in (r_.stderr + r_.stdout), (why, r_.stdout, r_.stderr)       # a gate claiming ok=True with bad raw fields is REFUSED
    open(os.path.join(td, "empty.jsonl"), "w").write(json.dumps({"event": "preflight", "gates": {}}) + "\n")
    r_ = subprocess.run([sys.executable, sn, os.path.join(td, "empty.jsonl"), HERE], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r_.returncode != 0 and "NOT proved" in (r_.stderr + r_.stdout)
print("V11 ok (G19: head grad None/0.0 vs ego and command, control non-zero, the ego-into-head regression RED; G20: >= 64 frames, real >= 0.95, y-flip and one-frame-shift mutants RED; launch gate + chain wiring)")

print("ZZTEST_PASS")
