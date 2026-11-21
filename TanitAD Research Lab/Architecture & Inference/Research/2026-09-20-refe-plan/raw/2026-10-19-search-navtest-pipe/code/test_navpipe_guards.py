"""Guard tests added after the first mutation pass: the chain-level refusals whose removal the first two suites did not notice (bundle tokens / ctx sha / gate stamp / no-head meta / harness PASS / score alignment).
Each test plants ONE defect in a fake child and requires the chain to stop with the named reason, the ctx kept.  CPU only.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import navpipe_core as NC  # noqa: E402
import navpipe_chain as CH  # noqa: E402
import test_navpipe as T  # noqa: E402
import test_navpipe_s0 as S0  # noqa: E402


class Defective(T.World):
    def __init__(self, tmp, defect=None, **kw):
        super().__init__(tmp, **kw)
        self.defect = defect

    def gpu(self, argv, out):
        rc = super().gpu(argv, out)
        d = os.path.join(self.arg(argv, "--out-dir"), "bundle")
        if self.defect == "swap_token":                                                    # a bundle of the right size whose token list is not the chunk's (ctx.json still names the chunk)
            z = {k: v for k, v in np.load(os.path.join(d, "bundle.npz"), allow_pickle=False).items()}
            tk = [str(t) for t in z["token"]]
            tk[0] = "f" * 16
            z["token"] = np.array(tk)
            np.savez(os.path.join(d, "bundle.npz"), **z)
        if self.defect == "ctx_corrupt":                                                   # ctx.npy no longer hashes to ctx.json
            with open(os.path.join(d, "ctx.npy"), "ab") as f:
                f.write(b"\0")
        return rc

    def gate(self, argv):
        rc = super().gate(argv)
        if self.defect == "stamp_other_bundle":
            p = os.path.join(self.arg(argv, "--out-dir"), "G-REACH-STEPB_PASS_x.json")
            j = json.load(open(p))
            j["bundle_sha256"] = "0" * 64
            json.dump(j, open(p, "w"))
        if self.defect == "stamp_not_pass":
            p = os.path.join(self.arg(argv, "--out-dir"), "G-REACH-STEPB_PASS_x.json")
            j = json.load(open(p))
            j["verdict"] = "FAIL"
            json.dump(j, open(p, "w"))
        if self.defect == "no_stamp":
            os.remove(os.path.join(self.arg(argv, "--out-dir"), "G-REACH-STEPB_PASS_x.json"))
        return rc

    def search(self, argv):
        rc = super().search(argv)
        if self.defect in ("head_in_meta", "arm_in_meta"):
            lo = int(self.arg(argv, "--start"))
            p = os.path.join(self.arg(argv, "--out-dir"), f"shard_P2HEAD_{lo:04d}.npz")
            z = {k: v for k, v in np.load(p, allow_pickle=False).items()}
            meta = json.loads(str(z["meta"]))
            if self.defect == "head_in_meta":
                meta["head_sha256"] = "a" * 64
            else:
                meta["arm"] = "L2s0"
            z["meta"] = np.array(json.dumps(meta))
            np.savez(p, **z)
        return rc


@pytest.mark.parametrize("defect,reason,code", [("swap_token", "G_BUNDLE_INVALID", 3), ("ctx_corrupt", "G_BUNDLE_INVALID", 3), ("stamp_other_bundle", "G_REACH_NOT_PASSED", 3), ("stamp_not_pass", "G_REACH_NOT_PASSED", 3),
                                                  ("no_stamp", "G_REACH_NOT_PASSED", 3), ("head_in_meta", "SEARCH_MERGE_INVALID", 6), ("arm_in_meta", "SEARCH_MERGE_INVALID", 6)])
def test_chain_stops_on_each_planted_defect_and_keeps_the_ctx(tmp_path, defect, reason, code):
    w = Defective(tmp_path, defect)
    ch = w.chain()
    assert ch.run() == code, ch.out.getvalue()
    out = ch.out.getvalue()
    assert f"ZZNAVPIPE_STOP {reason}" in out, out
    assert "ZZNAVPIPE_SCORE_DONE" not in out and "ZZNAVPIPE_BANKED" not in out and "ANALYSIS_DONE" not in out
    assert os.path.exists(os.path.join(w.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))          # the evidence is kept


def run_harness_chain(tmp, status, rc=0, csv=True, monkeypatch=None):
    w = T.World(tmp)
    cfg = w.cfg()
    seen = []

    def runner(argv, env, log_path, cwd=None):
        seen.append(argv)
        label = argv[argv.index("--label") + 1]
        out = os.path.join(argv[argv.index("--out") + 1], label)
        os.makedirs(out, exist_ok=True)
        json.dump({"status": status, "failures": [] if status == "PASS" else ["x"]}, open(os.path.join(out, f"{label}.counts.json"), "w"))
        if csv:
            open(os.path.join(out, f"{label}.csv"), "w").write("token\n")
        return rc
    ch = CH.Chain(cfg, runner=runner, out=io.StringIO(), ram_fn=lambda: 99.0)
    ch.retry_sleep_s = 0
    monkeypatch.setattr(CH.time, "sleep", lambda s: None)
    return ch, seen


def test_run_harness_returns_the_csv_only_on_the_harnesss_own_pass(tmp_path, monkeypatch):
    ch, seen = run_harness_chain(tmp_path / "a", "PASS", monkeypatch=monkeypatch)
    p = ch.run_harness("lab", "seam.npz", "tok.json")
    assert p.endswith("lab.csv") and len(seen) == 1
    for status, rc in (("FAIL", 0), ("PASS", 1), (None, 0)):
        ch, seen = run_harness_chain(tmp_path / f"b{status}{rc}", status, rc=rc, monkeypatch=monkeypatch)
        with pytest.raises(NC.Refusal) as e:
            ch.run_harness("lab", "seam.npz", "tok.json")
        assert e.value.reason == "HARNESS_NOT_PASS" and e.value.code == 7 and len(seen) == 3          # a CSV that exists is NOT a pass; 3 attempts, then a stop


def test_run_harness_waits_for_ram_and_gives_up_after_two_hours(tmp_path, monkeypatch):
    ch, seen = run_harness_chain(tmp_path, "PASS", monkeypatch=monkeypatch)
    ch.ram_fn = lambda: 1.0
    with pytest.raises(NC.Refusal) as e:
        ch.run_harness("lab", "seam.npz", "tok.json")
    assert e.value.reason == "HARNESS_RAM_WAIT" and e.value.code == 4 and seen == []


class NanLabel(S0.S0World):
    def __init__(self, tmp, mode, **kw):
        super().__init__(tmp, **kw)
        self.mode = mode

    def label(self, argv):
        super().label(argv)
        p = self.arg(argv, "--out")
        z = {k: v for k, v in np.load(p, allow_pickle=False).items()}
        if self.mode == "nan":
            z["pdms"] = z["pdms"].copy()
            z["pdms"][0, 0] = np.nan
        if self.mode == "misaligned":
            z["token"] = z["token"][::-1].copy()
        np.savez(p, **z)


@pytest.mark.parametrize("mode", ["nan", "misaligned"])
def test_label_scores_that_are_not_finite_or_not_aligned_to_the_search_are_refused(tmp_path, mode):
    w = NanLabel(tmp_path, mode)
    ch = w.chain0()
    assert ch.run() == 7
    out = ch.out.getvalue()
    assert "ZZNAVPIPE_STOP G_LABEL_NOT_PASSED" in out and "BANKED" not in out
    assert os.path.exists(os.path.join(w.tmp, "work", "chunks", "c00", "bundle", "ctx.npy"))


def test_delete_ctx_refuses_a_chunk_that_is_not_yet_banked_even_when_every_artifact_verifies(tmp_path):
    """G-DELETE-STATUS in isolation (it SURVIVED the first mutation pass because the other delete guards shadowed it): every artifact is recorded and re-verifies from disk and both records are ok, but the chunk's status
    was never advanced to BANKED.  Only the status guard then stands between this chunk and the loss of its ctx."""
    st, files, ctx, cj, man = T.bank_world(tmp_path)
    NC.bank_chunk(st, 0, files, cj, man)
    assert st.status(0) != "BANKED" and NC.verify_banked(st, 0) == [] and st.chunk(0)["records"].get("identity_ok") is True and st.chunk(0)["records"].get("t1id_ok") is True
    e = T.raises("gate", NC.delete_ctx, st, 0, ctx)
    assert e.reason == "DELETE_REFUSED_NOT_BANKED" and os.path.exists(ctx)


def test_loading_the_banked_seam_holds_one_copy_of_the_poses_not_one_per_token(tmp_path):
    """MEASURED 2026-10-20 on the real seam: `{t: z["poses"][i] ...}` re-read the whole `poses` member for every token and each row view kept its copy alive (12,146 x 1.17 MB, about 14 GB resident for the whole run;
    the suite needed 13.5 GB on one test).  Planted seam of 1,500 tokens: the old line holds about 216 MB, the fix about 0.2 MB."""
    import tracemalloc
    n = 1500
    toks = [f"{i:016x}" for i in range(n)]
    path = str(tmp_path / "seam.npz")
    np.savez(path, token=np.array(toks), fingerprint=np.array([f"fp{i}" for i in range(n)]), poses=np.arange(n * 24, dtype=np.float32).reshape(n, 8, 3), arm=np.array("planted"))
    tracemalloc.start()
    try:
        b = NC.load_banked_seam(path)
        cur, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert cur < 20 * 2 ** 20, f"{cur / 2 ** 20:.1f} MiB held for a {n * 96 / 2 ** 20:.2f} MiB array"
    assert len(b["poses"]) == n and b["fp"][toks[7]] == "fp7" and b["poses"][toks[7]].tolist() == np.arange(7 * 24, 8 * 24, dtype=np.float32).reshape(8, 3).tolist() and b["poses"][toks[3]].dtype == np.float32
