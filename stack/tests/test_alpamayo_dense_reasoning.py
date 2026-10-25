"""CPU tests for the dense Alpamayo2-Super reasoning pipeline (stack/scripts/alpamayo_dense_reasoning.py + merge_dense_reasoning.py).

No GPU, no model, no network: a synthetic ClipSource + the MockBackend exercise frame enumeration (vs the real v9 clock when the arrays are
present), timestamps, eligibility, input assembly, sharding, resume / idempotency, output schema, front-only flagging, the merge, and the
output parser against EVERY real meta_action / auto_labeling row of records.parquet (when present)."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import alpamayo_dense_reasoning as adr          # noqa: E402
import merge_dense_reasoning as mdr             # noqa: E402

HF = "/tmp/claude-0/hfdata"
PROBE = "/tmp/claude-0/probe"
V9_EVAL = os.path.join(HF, "v9_labels_eval139.npz")
V9_TRAIN = os.path.join(HF, "v9_labels_train.npz")
RECORDS = os.path.join(PROBE, "records.parquet")
CI = [os.path.join(PROBE, f"clip_index_v7.2_{s}.json") for s in ("train", "eval")]
need_eval = pytest.mark.skipif(not (os.path.exists(V9_EVAL) and all(map(os.path.exists, CI))), reason="v9 eval arrays / clip index absent")
need_records = pytest.mark.skipif(not os.path.exists(RECORDS), reason="records.parquet absent")


# ------------------------------------------------------------------ synthetic data
class FakeSource(adr.ClipSource):
    """Ego drives at 10 m/s with heading psi(t) = psi0 + w t; cameras at 30 fps (per-camera offset); frames encode their index."""

    def __init__(self, clip_id, cams=adr.DRIVING_CAMERA_IDS, ego_range=(0, 20_500_000), cam_end_us=20_400_000, psi0=0.3, w=0.0, hw=(6, 10)):
        super().__init__()
        self.clip_id, self.ego_range, self.psi0, self.w, self.hw = clip_id, ego_range, psi0, w, hw
        self._ts = {c: np.arange(33_333, cam_end_us, 33_333, dtype=np.int64) + 7 * c for c in range(7)}
        self.n_decoded, self.n_calls = 0, 0

    def _ego(self, ts):
        t = ts / 1e6; psi = self.psi0 + self.w * t
        xyz = np.stack([10 * t * np.cos(self.psi0), 10 * t * np.sin(self.psi0), np.zeros_like(t)], 1)
        quat = np.stack([np.zeros_like(t), np.zeros_like(t), np.sin(psi / 2), np.cos(psi / 2)], 1)
        return xyz, quat

    def camera_timestamps(self, c): return self._ts[c]

    def _decode(self, c, idx):
        self.n_decoded += len(idx); self.n_calls += 1
        return np.stack([np.full(self.hw + (3,), (int(i) * 7 + c) % 251, np.uint8) for i in idx])


def synth_v9(tmp, n_clips=5, n_rows=199, dt=0.1006, split="eval139"):
    """Tiny v9-like npz + clip index json; clock rule now_s = grid_start + k dt, k = k0 + j, k0 = 2."""
    uu = [f"00000000-0000-4000-8000-{i:012d}" for i in range(n_clips)]
    sh = [adr.sha12_of(u) for u in uu]
    g0 = np.linspace(0.05, 0.4, n_clips)
    k = np.concatenate([np.arange(2, 2 + n_rows) for _ in uu]).astype(np.int16)
    now = np.concatenate([g0[i] + np.arange(2, 2 + n_rows) * dt for i in range(n_clips)])
    np.savez(os.path.join(tmp, f"v9_labels_{split}.npz"), clip__sha12=np.array(sh), clip__row0=np.arange(n_clips) * n_rows,
             clip__n_rows=np.full(n_clips, n_rows), clip__k0=np.full(n_clips, 2), clip__dt_s=np.full(n_clips, dt), clip__grid_start_s=g0,
             row__now_s=now, row__k=k)
    for s in ("train", "eval"):
        json.dump({"clips": uu if s == "eval" else []}, open(os.path.join(tmp, f"clip_index_v7.2_{s}.json"), "w"))
    return uu, sh


def run(tmp, frames, cfg, backend=None, shard=0, done=None):
    w = adr.ShardWriter(str(tmp), frames.split.iloc[0], shard, flush_every=7)
    done = adr.load_done_keys(str(tmp), frames.split.iloc[0]) if done is None else done
    st = adr.run_frames(frames, lambda u: FakeSource(u, cams=cfg.camera_ids), backend or adr.MockBackend(), w, cfg, done, log=lambda *_: None)
    return w, st


# ------------------------------------------------------------------ enumeration vs the real v9 clock
@need_eval
def test_enumeration_matches_v9_eval_rows_and_clock():
    c = adr.load_v9_clock(V9_EVAL, "eval139"); um = adr.load_uuid_map(CI)
    f = adr.enumerate_frames(c, um, hz=10)
    assert len(f) == 27_664 == int(c.n_rows.sum())                     # manifest n_rows (MEASURED in V9_SCHEMA / manifest)
    assert f.sha12.nunique() == 139
    z = np.load(V9_EVAL, allow_pickle=True)
    for i in range(0, 139, 17):
        s = c.sha12[i]; r0, n = int(c.row0[i]), int(c.n_rows[i]); fi = f[f.sha12 == s]
        assert fi.v9_row.tolist() == list(range(n))                    # v9 row j = episode row j, every row
        assert (fi.v9_row_global.to_numpy() == r0 + np.arange(n)).all()
        k = z["row__k"][r0:r0 + n]
        assert (k == int(z["clip__k0"][i]) + np.arange(n)).all()       # raw k = k0 + j
        now = z["clip__grid_start_s"][i] + k * z["clip__dt_s"][i]      # clock rule
        assert np.allclose(now, c.now_s[r0:r0 + n], atol=1e-9)
        assert (fi.t_us.to_numpy() == np.rint(now * 1e6).astype(np.int64)).all()
        d = np.diff(fi.t_us.to_numpy())
        assert (d > 0).all() and abs(np.median(d) - 100_000) < 2_000  # ~10 Hz, 0.1 s apart
        assert fi.clip_id.iloc[0] == um[s] and adr.sha12_of(fi.clip_id.iloc[0]) == s


@pytest.mark.skipif(not (os.path.exists(V9_TRAIN) and all(map(os.path.exists, CI))), reason="v9 train arrays absent")
def test_enumeration_matches_v9_train_count():
    c = adr.load_v9_clock(V9_TRAIN, "train")
    f = adr.enumerate_frames(c, adr.load_uuid_map(CI), hz=10)
    assert len(f) == 869_278 and f.sha12.nunique() == 4_369


@need_eval
@pytest.mark.parametrize("hz", [5, 2, 1])
def test_subsampling_keeps_the_5p1s_anchor_row(hz):
    c = adr.load_v9_clock(V9_EVAL, "eval139"); f = adr.enumerate_frames(c, adr.load_uuid_map(CI), hz=hz)
    stride = adr.HZ_STRIDE[hz]
    for s, g in list(f.groupby("sha12"))[:20]:
        now = c.clip_now(s); ja = int(np.argmin(np.abs(now - 5.1)))
        assert ja in set(g.v9_row) and (np.diff(g.v9_row) == stride).all()
        assert len(g) == len(range(ja % stride, len(now), stride))


def test_sharding_partitions_exactly():
    sh = [adr.sha12_of(str(i)) for i in range(101)]
    for n in (1, 3, 8):
        parts = [adr.shard_clips(sh, i, n) for i in range(n)]
        flat = [s for p in parts for s in p]
        assert sorted(flat) == sorted(sh) and len(flat) == len(set(flat))
        assert max(map(len, parts)) - min(map(len, parts)) <= 1
    assert adr.shard_clips(sh, 2, 8) == adr.shard_clips(list(reversed(sh)), 2, 8)   # order-independent
    with pytest.raises(ValueError): adr.shard_clips(sh, 8, 8)


def test_sharded_enumeration_is_a_partition(tmp_path):
    uu, sh = synth_v9(str(tmp_path), n_clips=7, n_rows=30)
    c = adr.load_v9_clock(str(tmp_path / "v9_labels_eval139.npz"), "eval139"); um = adr.load_uuid_map([str(tmp_path / "clip_index_v7.2_eval.json")])
    full = adr.enumerate_frames(c, um)
    parts = pd.concat([adr.enumerate_frames(c, um, shard=(i, 3)) for i in range(3)])
    assert len(parts) == len(full) == 210
    assert set(map(tuple, parts[["sha12", "v9_row"]].values)) == set(map(tuple, full[["sha12", "v9_row"]].values))
    sub = adr.enumerate_frames(c, um, clips=adr.parse_clip_filter(uu[1] + "," + sh[2], um))
    assert set(sub.sha12) == {sh[1], sh[2]}


def test_frame_seed():
    assert adr.frame_seed(42, "fixed", "abc", 3, "meta_action") == 42
    a = adr.frame_seed(42, "per_frame", "abc", 3, "meta_action")
    assert a == adr.frame_seed(42, "per_frame", "abc", 3, "meta_action") != adr.frame_seed(42, "per_frame", "abc", 4, "meta_action")
    assert 0 <= a < 2 ** 31


# ------------------------------------------------------------------ eligibility + input assembly
def test_eligibility_rules():
    cam = {c: np.arange(33_333, 20_400_000, 33_333) for c in adr.DRIVING_CAMERA_IDS}
    E = lambda t, task="meta_action", **k: adr.frame_eligibility(t, task, (0, 20_500_000), cam, adr.DRIVING_CAMERA_IDS, **k)
    assert E(1_600_000).startswith("ineligible:t0<=1.6s")                       # NVIDIA: assert t0_us > 1.6e6
    assert E(1_600_001) is None
    assert E(1_600_000, relax_hist=True) is None                                # ego covers t0-1.5 s
    assert E(1_500_000, relax_hist=True) is None and E(1_400_000, relax_hist=True) == "ineligible:ego_history_out_of_range"  # t0-1.5 s < ego start
    assert E(14_100_000, "auto_labeling") is None                               # t0 + 6.4 s = 20.5 s = ego end
    assert E(14_100_001, "auto_labeling").startswith("ineligible:ego_future<6.4s")
    assert E(14_100_001, "auto_labeling", future_source="predicted") is None
    assert E(19_000_000) is None                                                # meta_action needs no future
    assert E(20_450_000).startswith("ineligible:camera")


def test_quaternion_matrix():
    psi = 0.7; R = adr.quat_xyzw_to_mat(np.array([[0, 0, np.sin(psi / 2), np.cos(psi / 2)]]))[0]
    assert np.allclose(R, [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    q = np.random.default_rng(0).normal(size=(50, 4)); M = adr.quat_xyzw_to_mat(q)
    assert np.allclose(M @ np.swapaxes(M, 1, 2), np.eye(3), atol=1e-12) and np.allclose(np.linalg.det(M), 1)


def test_build_frame_data_matches_nvidia_contract():
    src = FakeSource("clip-x", psi0=0.3, w=0.0)
    t0 = 5_100_000
    d = adr.build_frame_data(src, t0, adr.DRIVING_CAMERA_IDS)
    assert d["image_frames"].shape == (6, 4, 3, 6, 10) and d["image_frames"].dtype == torch.uint8
    assert d["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]
    assert d["camera_names"][5] == "camera_front_tele_30fov"
    req = np.array([4_800_000, 4_900_000, 5_000_000, 5_100_000])
    for ci, c in enumerate(adr.DRIVING_CAMERA_IDS):                                 # closest frame AT OR BEFORE each request
        ts = src.camera_timestamps(c); idx = np.searchsorted(ts, req, side="right") - 1
        assert d["absolute_timestamps"][ci].tolist() == ts[idx].tolist()
        assert (ts[idx] <= req).all() and (ts[idx + 1] > req).all()
        assert d["image_frames"][ci, :, 0, 0, 0].tolist() == [(int(i) * 7 + c) % 251 for i in idx]
    assert d["camera_tmin"] == int(d["absolute_timestamps"].min())
    assert torch.allclose(d["relative_timestamps"], (d["absolute_timestamps"] - d["camera_tmin"]).float() * 1e-6)
    assert d["ego_t0_frame_idx"].tolist() == [3] and d["ego_t0"].tolist() == [t0]
    hx, hr, fx = d["ego_history_xyz"][0, 0], d["ego_history_rot"][0, 0], d["ego_future_xyz"][0, 0]
    assert hx.shape == (16, 3) and hr.shape == (16, 3, 3) and fx.shape == (64, 3) and d["ego_future_rot"].shape == (1, 1, 64, 3, 3)
    assert torch.allclose(hx[-1], torch.zeros(3)) and torch.allclose(hr[-1], torch.eye(3), atol=1e-6)
    # straight drive at 10 m/s along the heading -> the t0 ego frame sees x = 10 * dt, y = 0
    assert torch.allclose(hx[:, 0], torch.arange(-15, 1, dtype=torch.float32), atol=1e-4) and hx[:, 1].abs().max() < 1e-4
    assert torch.allclose(fx[:, 0], torch.arange(1, 65, dtype=torch.float32), atol=1e-3)
    assert torch.allclose(d["ego_history_tvals"], torch.arange(-15, 1).float() * 0.1)
    assert torch.allclose(d["ego_future_tvals"], torch.arange(1, 65).float() * 0.1)
    # yawing ego: relative rotation of the first history pose = Rz(-w * 1.5 s)
    d2 = adr.build_frame_data(FakeSource("y", psi0=0.3, w=0.2), t0, adr.DRIVING_CAMERA_IDS)
    a = -0.2 * 1.5
    assert torch.allclose(d2["ego_history_rot"][0, 0, 0, :2, :2], torch.tensor([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]).float(), atol=1e-5)
    # late frame: no future tensors (auto_labeling is then ineligible), meta_action inputs intact
    d3 = adr.build_frame_data(src, 19_000_000, adr.DRIVING_CAMERA_IDS)
    assert "ego_future_xyz" not in d3 and d3["ego_history_xyz"].shape == (1, 1, 16, 3)


def test_consecutive_frames_decode_each_video_frame_once():
    src = FakeSource("c")
    ts = [5_000_000 + 100_000 * i for i in range(10)]
    req = (np.array(ts)[:, None] - np.arange(3, -1, -1)[None] * 100_000).ravel()
    for c in adr.DRIVING_CAMERA_IDS: src.warm(c, req)
    calls = src.n_calls
    for t in ts: adr.build_frame_data(src, t, adr.DRIVING_CAMERA_IDS)
    assert src.n_calls == calls == 6                                   # one batched decode per camera, then all cache hits
    assert src.n_decoded == 6 * len(np.unique(np.searchsorted(src.camera_timestamps(0), req, side="right")))


# ------------------------------------------------------------------ run loop: schema, resume, idempotency, front-only
def _frames(n_clips=3, rows=(10, 13, 50, 130, 150)):
    out = []
    for i in range(n_clips):
        u = f"10000000-0000-4000-8000-{i:012d}"
        for j in rows:
            out.append(dict(split="eval139", sha12=adr.sha12_of(u), clip_id=u, v9_row=j, v9_row_global=1000 * i + j, t_us=300_000 + j * 100_600))
    return pd.DataFrame(out).astype(dict(v9_row="int32", t_us="int64"))


def test_run_schema_ineligible_and_finalize(tmp_path):
    fr = _frames(); cfg = adr.RunCfg(run_id="t")
    w, st = run(tmp_path, fr, cfg)
    assert st["requests"] + st["ineligible"] == len(fr) * 2
    p = w.finalize()
    df = pd.read_parquet(p)
    assert list(df.columns) == adr.COLUMNS and len(df) == len(fr) * 2
    assert {c: str(t) for c, t in df.dtypes.items()}["v9_row"] == "int32" and str(df.t_us.dtype) == "int64"
    assert not df.duplicated(adr.KEY).any()
    assert not os.path.exists(w.parts_dir)
    r10 = df[df.v9_row == 10]                                           # t = 1.306 s <= 1.6 s (row 13 = 1.608 s is eligible)
    assert r10.error.str.startswith("ineligible:t0<=1.6s").all() and r10.raw.isna().all()
    late = df[(df.v9_row == 150) & (df.task == "auto_labeling")]        # t = 15.39 s: < 6.4 s of ego future
    assert late.error.str.startswith("ineligible:ego_future").all()
    ok = df[df.error.isna()]
    assert len(ok) == 3 * (3 * 2 + 1)                                    # rows 13, 50, 130 both tasks + row 150 meta_action only
    ma = ok[ok.task == "meta_action"]; al = ok[ok.task == "auto_labeling"]
    assert ma.meta_action.str.startswith("Longitudinal:").all() and ma.cot.str.startswith("MOCK").all() and ma.coc.isna().all()
    assert al.coc.str.startswith("MOCK").all() and al.meta_action.isna().all()
    assert (ok.input_profile == "a2s_driving_6cam_4f|cams=0,1,2,3,5,6|fut=gt").all() and (~df.front_only).all()
    assert (ok.camera_tmin_us > 0).all() and (ok.camera_tmin_us <= ok.t_us - 300_000).all()
    assert (df.model_id == "MOCK").all() and (df.seed == 42).all()
    # MOCK text carries the cams it was fed: the 6-camera driving profile
    assert ma.cot.str.contains(r"cams \[0, 1, 2, 3, 5, 6\]").all()


def test_resume_is_idempotent_and_complete(tmp_path):
    fr = _frames(); cfg = adr.RunCfg(run_id="a", max_frames=4)
    w, st1 = run(tmp_path, fr, cfg)                                    # "killed" after 4 frames
    assert st1["frames"] == 4
    cfg2 = adr.RunCfg(run_id="b")
    w2, st2 = run(tmp_path, fr, cfg2)
    assert st2["skipped_done"] == 8 and st2["frames"] == len(fr) - 4        # 4 frames x 2 tasks were already written
    w3, st3 = run(tmp_path, fr, cfg2)                                  # nothing left
    assert st3["requests"] == 0 and st3["frames"] == 0 and st3["skipped_done"] == len(fr) * 2
    df = adr.read_outputs(str(tmp_path), "eval139")
    assert len(df) == len(fr) * 2 and not df.duplicated(adr.KEY).any()
    assert set(df.run_id) == {"a", "b"}
    w3.finalize()
    assert len(adr.load_done_keys(str(tmp_path), "eval139")) == len(fr) * 2


def test_retry_errors_reruns_failures_but_never_ineligible(tmp_path):
    fr = _frames(n_clips=1); cfg = adr.RunCfg(run_id="a")
    run(tmp_path, fr, cfg, backend=adr.MockBackend(fail_every=2))
    df = adr.read_outputs(str(tmp_path), "eval139")
    n_fail = int((df.error == "mock_failure").sum()); assert n_fail > 0
    assert len(adr.load_done_keys(str(tmp_path), "eval139", retry_errors=True)) == len(df) - n_fail
    _, st = run(tmp_path, fr, cfg, done=adr.load_done_keys(str(tmp_path), "eval139", retry_errors=True))
    assert st["requests"] == n_fail and st["ineligible"] == 0
    d = adr.dedupe(adr.read_outputs(str(tmp_path), "eval139"))
    assert len(d) == len(fr) * 2 and (d.error.isna() | d.error.str.startswith("ineligible:")).all()


def test_front_only_flagged_in_every_row(tmp_path):
    fr = _frames(n_clips=2); cfg = adr.RunCfg(camera_ids=adr.FRONT_ONLY_CAMERA_IDS, run_id="f")
    w, _ = run(tmp_path, fr, cfg); df = pd.read_parquet(w.finalize())
    assert df.front_only.all() and df.input_profile.str.startswith("a2s_front_only_4f_DEGRADED|cams=1|").all()
    ok = df[df.error.isna() & (df.task == "meta_action")]
    assert ok.cot.str.contains(r"cams \[1\]").all()


def test_source_failure_is_recorded_per_row(tmp_path):
    fr = _frames(n_clips=1); cfg = adr.RunCfg(run_id="s")
    w = adr.ShardWriter(str(tmp_path), "eval139", 0)

    def boom(u): raise FileNotFoundError("member missing")
    adr.run_frames(fr, boom, adr.MockBackend(), w, cfg, set(), log=lambda *_: None); w.flush()
    df = adr.read_outputs(str(tmp_path), "eval139")
    assert len(df) == len(fr) * 2 and df.error.str.startswith("source:FileNotFoundError").all()


def test_dedupe_prefers_ok_then_latest():
    base = dict(split="s", sha12="a", v9_row=1, task="meta_action")
    df = pd.DataFrame([dict(base, error="x", written_unix=3.0), dict(base, error=None, written_unix=1.0), dict(base, error=None, written_unix=2.0)])
    d = adr.dedupe(df)
    assert len(d) == 1 and d.error.isna().all() and d.written_unix.iloc[0] == 2.0


# ------------------------------------------------------------------ parser vs the REAL records (every row)
@need_records
def test_parse_generation_reproduces_every_record():
    rec = pd.read_parquet(RECORDS, columns=["task", "raw_json"])
    n = {"meta_action": 0, "auto_labeling": 0}
    for task, raw_json in zip(rec.task, rec.raw_json):
        if task not in n: continue
        j = json.loads(raw_json); p = adr.parse_generation(task, j["raw_outputs"][0])
        if task == "meta_action":
            assert (p["cot"] or "") == j["cot"][0] and (p["meta_action"] or "") == j["meta_action"][0]
        else:
            assert p["coc"] == j["cot_auto_labeling_json"][0]["chain_of_causation"]
            assert p["coc_json"] == j["cot_auto_labeling_json"][0]
        n[task] += 1
    assert n == {"meta_action": 4729, "auto_labeling": 4729}


@need_records
def test_parse_traj_and_normalise():
    rec = pd.read_parquet(RECORDS); j = json.loads(rec[rec.task == "trajectory"].raw_json.iloc[0])
    assert adr.parse_generation("trajectory", json.dumps({"cot": j["cots"]}))["cot"] == j["cot"]
    assert adr.normalise_raw("a b.<|traj_future_start|><|endoftext|>") == adr.normalise_raw(" a b.<|traj_future_start|>") == "a b."
    with pytest.raises(ValueError): adr.parse_generation("auto_labeling", "no json here<|im_end|>")


@need_records
def test_anchor_agreement_perfect_and_perturbed():
    rec = pd.read_parquet(RECORDS)
    rows = []
    for _, r in rec[rec.task.isin(["meta_action", "auto_labeling"])].head(400).iterrows():
        j = json.loads(r.raw_json); p = adr.parse_generation(r.task, j["raw_outputs"][0])
        traj = json.loads(rec[(rec.clip_id == r.clip_id) & (rec.task == "trajectory")].raw_json.iloc[0])
        rows.append(dict(split="x", sha12=adr.sha12_of(r.clip_id), clip_id=r.clip_id, v9_row=-1, t_us=5_100_000, task=r.task, model_id="m",
                         model_revision="r", quantisation="q", input_profile="p", seed=42, wall_s=1.0, raw=j["raw_outputs"][0], cot=p["cot"],
                         meta_action=p["meta_action"], coc=p["coc"], error=None, front_only=False, camera_tmin_us=traj["camera_tmin"],
                         written_unix=0.0, run_id="t"))
    gen = adr.to_frame(rows)
    rep = adr.anchor_agreement(gen, rec, n_boot=200)
    assert rep["meta_action"]["n"] > 0 and rep["auto_labeling"]["n"] > 0
    for k in ("raw_exact", "cot_exact", "meta_action_exact", "class_Longitudinal", "class_Lane", "v9class_lon", "class_all_three"):
        assert rep["meta_action"][k]["rate"] == 1.0
    assert rep["auto_labeling"]["coc_exact"]["rate"] == 1.0 and rep["input_assembly"]["camera_tmin_equal"]["rate"] == 1.0
    g2 = gen.copy(); m = g2.task == "meta_action"
    g2.loc[m, "meta_action"] = "Longitudinal: Stop.\nLateral: Go Straight.\nLane: Lane Keep."
    g2.loc[m, "camera_tmin_us"] = 1
    rep2 = adr.anchor_agreement(g2, rec, n_boot=200)
    assert rep2["meta_action"]["cot_exact"]["rate"] == 1.0 and rep2["meta_action"]["class_Longitudinal"]["rate"] < 1.0
    lo, hi = rep2["meta_action"]["class_Longitudinal"]["ci95"]; assert lo <= rep2["meta_action"]["class_Longitudinal"]["rate"] <= hi
    assert rep2["input_assembly"]["camera_tmin_equal"]["rate"] < 1.0


@need_records
@need_eval
def test_anchor_frames_from_records():
    rec = pd.read_parquet(RECORDS, columns=["clip_id", "t0_us", "task"])
    clocks = {"eval139": adr.load_v9_clock(V9_EVAL, "eval139")}
    f = adr.anchor_frames(rec, clocks)
    assert len(f) == 4729 and (f.t_us == 5_100_000).all() and f.sha12.is_unique
    ev = f[f.split == "eval139"]
    assert len(ev) == 139 and (ev.v9_row >= 0).all()
    for r in ev.head(10).itertuples():
        now = clocks["eval139"].clip_now(r.sha12); assert abs(now[r.v9_row] - 5.1) <= 0.051
    parts = pd.concat([adr.anchor_frames(rec, clocks, shard=(i, 4)) for i in range(4)])
    assert len(parts) == 4729 and parts.sha12.is_unique


# ------------------------------------------------------------------ merge end to end
def test_merge_end_to_end(tmp_path):
    d = tmp_path / "v9"; d.mkdir(); out = tmp_path / "rd"
    uu, sh = synth_v9(str(d), n_clips=4, n_rows=60)
    c = adr.load_v9_clock(str(d / "v9_labels_eval139.npz"), "eval139"); um = adr.load_uuid_map([str(d / "clip_index_v7.2_eval.json")])
    for i in range(2):                                                   # two shards
        fr = adr.enumerate_frames(c, um, hz=5, shard=(i, 2))
        w, _ = run(out, fr, adr.RunCfg(run_id=f"s{i}"), shard=i)
        if i == 0: w.finalize()                                          # shard 1 left as parts (still merged, reported)
    man = mdr.merge(str(out), "eval139", 5, str(d), str(d), str(tmp_path / "merged"))
    tgt = adr.enumerate_frames(c, um, hz=5)
    assert man["n_rows"] == 2 * len(tgt) and man["n_missing"] == 0 and man["n_unfinalised_part_files"] > 0
    df = pd.read_parquet(tmp_path / "merged" / "v10_reasoning_dense_eval139.parquet")
    cov = pd.read_parquet(tmp_path / "merged" / "v10_reasoning_dense_eval139.coverage.parquet")
    assert len(df) == man["n_rows"] and set(cov.sha12) == set(sh) and (cov.n_missing == 0).all()
    assert (cov.n_ok + cov.n_ineligible + cov.n_error == cov.n_target).all()
    assert man["outputs_sha256"]["v10_reasoning_dense_eval139.parquet"] == adr.sha256_file(str(tmp_path / "merged" / "v10_reasoning_dense_eval139.parquet"))
    assert len(man["coverage_per_clip"]) == 4 * 2 and man["by_task"]["meta_action"]["ineligible"] > 0
    with pytest.raises(SystemExit, match="not target frames"):          # wrong --hz is refused
        mdr.merge(str(out), "eval139", 1, str(d), str(d), str(tmp_path / "m2"))
    # a missing (frame, task) pair is refused unless allowed
    p = adr.output_files(str(out), "eval139")[0]; x = pd.read_parquet(p); x.iloc[1:].to_parquet(p, index=False)
    with pytest.raises(SystemExit, match="missing"):
        mdr.merge(str(out), "eval139", 5, str(d), str(d), str(tmp_path / "m3"))
    assert mdr.merge(str(out), "eval139", 5, str(d), str(d), str(tmp_path / "m3"), allow_missing=True)["n_missing"] == 1


def test_cli_plan_runs(tmp_path, capsys):
    synth_v9(str(tmp_path), n_clips=3, n_rows=40)
    assert adr.main(["plan", "--v9-dir", str(tmp_path), "--clip-index-dir", str(tmp_path), "--split", "eval139", "--hz", "2"]) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["frames"] == 3 * 8 and out["clips"] == 3                 # 40 rows / stride 5, whatever the anchor phase


def test_pose_transform_equals_nvidia_scipy_formulation():
    """load_physical_aiavdataset: xyz_local = R0^-1 (xyz - xyz0); rot_local = R0^-1 * R_i (scipy). Our numpy version must agree."""
    spt = pytest.importorskip("scipy.spatial.transform")
    rng = np.random.default_rng(1)
    for _ in range(50):
        xyz = rng.normal(size=(16, 3)) * 50; q = rng.normal(size=(16, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
        inv = spt.Rotation.from_quat(q[-1]).inv()
        R0 = adr.quat_xyzw_to_mat(q[-1])
        assert np.allclose(inv.apply(xyz - xyz[-1]), (xyz - xyz[-1]) @ R0, atol=1e-9)
        assert np.allclose((inv * spt.Rotation.from_quat(q)).as_matrix(), np.einsum("ji,njk->nik", R0, adr.quat_xyzw_to_mat(q)), atol=1e-12)
