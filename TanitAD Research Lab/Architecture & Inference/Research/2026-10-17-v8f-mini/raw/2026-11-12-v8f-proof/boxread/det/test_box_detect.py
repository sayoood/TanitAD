"""Tests of the BOXREAD-T0 detector pass (card ADD127): stand-in detector, patched decode and render, synthetic files only.
Portable: interpreter from sys.executable; BOXDET_TEST_BASH (a bash), BOXDET_TEST_CWD (cwd of child processes) from the environment."""
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import box_detect as bd   # noqa: E402

QUEUE = os.path.join(HERE, "pod_boxread_queue.sh")
CLIPS = ["aaaaaaaa-0001-xxxx", "bbbbbbbb-0002-xxxx", "cccccccc-0003-xxxx"]
N = 6
FORBIDDEN = ["vo" + "id", "super" + "seded", "re" + "tracted", "for" + "merly", "cor" + "rection", "with" + "drawn", "no " + "longer",
             "closed " + "loop"]


def p(x):
    return str(x).replace("\\", "/")


# ------------------------------------------------------------------------------------------------ stand-ins
class FakeDet:
    """Torchvision call contract: list of float [3,H,W] images in 0..1 -> list of dict(boxes, labels, scores). The marker of an image is the
    pixel value of the (patched) render, i.e. the decoded frame's number."""

    def __init__(self, table):
        self.table, self.batches, self.n_images = table, [], 0

    def __call__(self, imgs):
        self.batches.append(len(imgs))
        out = []
        for im in imgs:
            assert im.dtype == torch.float32 and im.shape[0] == 3 and float(im.min()) >= 0.0 and float(im.max()) <= 1.0
            self.n_images += 1
            v = int(round(float(im[0, 0, 0]) * 255.0))
            rows = self.table.get(v, [])
            out.append({"boxes": torch.tensor([r[0] for r in rows], dtype=torch.float32).reshape(-1, 4),
                        "labels": torch.tensor([r[2] for r in rows], dtype=torch.int64),
                        "scores": torch.tensor([r[1] for r in rows], dtype=torch.float32)})
        return out


WIDE_TABLE = {0: [([10, 20, 30, 40], 0.9, 3)],
              2: [([1, 1, 2, 2], 0.2, 3), ([5, 5, 9, 9], 0.95, 1)],                       # low score car; high score person
              4: [([11, 12, 13, 14], 0.5, 8), ([21, 22, 23, 24], 0.31, 6), ([31, 32, 33, 34], 0.3, 4)],
              6: [([41, 42, 43, 44], 0.9, 3), ([51, 52, 53, 54], 0.8, 3)],
              10: [([61, 62, 63, 64], 0.9, 2)]}                                              # a bicycle
TELE_TABLE = {0: [([1, 2, 3, 4], 0.9, 3)], 2: [([5, 6, 7, 8], 0.9, 8)], 3: [([9, 10, 11, 12], 0.9, 3)], 4: [([13, 14, 15, 16], 0.9, 6)]}


class Rec:
    def __init__(self):
        self.frames, self.grids, self.intr, self.decoded = [], [], [], []
        self.fail_after = None
        self.tele_ts = None


@pytest.fixture
def rec(monkeypatch):
    r = Rec()
    t_frames = np.arange(2 * N, dtype=np.float64) * 33333.0

    monkeypatch.setattr(bd, "frame_plan", lambda ts: {"n_cache": N, "cache_idx": np.arange(N) * 2, "t_frames": t_frames})
    monkeypatch.setattr(bd, "window_frame", lambda size, px=256: (r.frames.append((float(size), int(px))) or ("frame", float(size), int(px))))

    def fake_grid(intr, h, w, frame, az, el, device="cpu"):
        r.grids.append((frame[1], frame[2], float(az), float(el)))
        px = frame[2]
        return torch.zeros(1, px, px, 2), torch.ones(1, px, px, dtype=torch.bool)

    def fake_sample(vid, g, m):
        B, px = vid.shape[0], g.shape[1]
        out = torch.zeros(B, 3, px, px, dtype=torch.uint8)
        for b in range(B):
            out[b] = int(vid[b, 0, 0, 0])
        return out

    def fake_decode(mp4, need):
        for vi in need:
            if r.fail_after is not None and vi > r.fail_after:
                raise RuntimeError("planted decode failure")
            r.decoded.append((p(mp4), int(vi)))
            yield int(vi), torch.full((3, 4, 4), int(vi), dtype=torch.uint8)

    monkeypatch.setattr(bd, "window_grid", fake_grid)
    monkeypatch.setattr(bd, "window_sample", fake_sample)
    monkeypatch.setattr(bd, "decode_selected", fake_decode)
    monkeypatch.setattr(bd, "get_intr", lambda clip, root: r.intr.append(p(root)) or object())
    monkeypatch.setattr(bd, "tele_times", lambda path: np.asarray(r.tele_ts, np.float64))
    return r


def stage(tmp, clips, tele=(), no_wide_video=()):
    w, t = os.path.join(p(tmp), "wide"), os.path.join(p(tmp), "tele")
    for c in clips:
        os.makedirs(f"{w}/ts", exist_ok=True)
        open(f"{w}/ts/{c}.timestamps.parquet", "wb").close()
        if c not in no_wide_video:
            os.makedirs(f"{w}/r0/camera", exist_ok=True)
            open(f"{w}/r0/camera/{c}.mp4", "wb").close()
    for c in tele:
        os.makedirs(f"{t}/ts", exist_ok=True)
        os.makedirs(f"{t}/r0/camera", exist_ok=True)
        open(f"{t}/ts/{c}.timestamps.parquet", "wb").close()
        open(f"{t}/r0/camera/{c}.mp4", "wb").close()
    return w, t


def weights(tmp, content=b"stand-in weights"):
    path = os.path.join(p(tmp), "w.pth")
    open(path, "wb").write(content)
    return path, hashlib.sha256(content).hexdigest()


def write_list(tmp, name, clips):
    path = os.path.join(p(tmp), name)
    open(path, "w", encoding="utf-8").write("\n".join(clips) + "\n")
    return path


def cli(camera, tmp, clips, det, extra=(), prefix=None, win=None):
    wpath, sha = weights(tmp)
    lst = write_list(tmp, "list.txt", clips)
    w, t = os.path.join(p(tmp), "wide"), os.path.join(p(tmp), "tele")
    size, px, el = (32, 768, -6) if camera == "wide" else (16, 1024, 0)
    argv = ["--camera", camera, "--list", lst, "--root", w, "--weights", wpath, "--weights-sha-prefix", prefix or sha[:8],
            "--out-dir", os.path.join(p(tmp), "out"), "--win-size-deg", str(size), "--win-px", str(px), "--az-left-deg=0",
            f"--el-up-deg={el}", "--device", "cpu", "--workers", "1"]
    if camera == "tele":
        argv += ["--tele-root", t]
    argv += list(extra)
    return bd.main(argv, det), sha


def load(tmp, camera, clip):
    return np.load(os.path.join(p(tmp), "out", camera, f"{clip}.npz"))


# ------------------------------------------------------------------------------------------------ tests of the tool
def test_file_keys_shapes_dtypes_and_filtering(tmp_path, rec):
    stage(tmp_path, CLIPS[:2])
    det = FakeDet(WIDE_TABLE)
    rc, sha = cli("wide", tmp_path, CLIPS[:2], det)
    assert rc == 0
    z = load(tmp_path, "wide", CLIPS[0])
    assert sorted(z.files) == sorted(bd.NPZ_KEYS)
    assert z["n_frames"].dtype.kind == "i" and int(z["n_frames"]) == N
    assert z["frame_ok"].dtype == np.bool_ and z["frame_ok"].shape == (N,) and z["frame_ok"].all()
    # kept: j0 car; j2 truck, bus, motorcycle (0.3 is kept at the 0.3 bar); j3 two cars. Dropped: low score, person, bicycle.
    assert z["det_frame"].dtype == np.int32 and z["det_frame"].tolist() == [0, 2, 2, 2, 3, 3]
    assert z["det_label"].dtype == np.int16 and z["det_label"].tolist() == [3, 8, 6, 4, 3, 3]
    assert z["det_score"].dtype == np.float32 and z["det_score"].shape == (6,) and float(z["det_score"].min()) >= 0.3
    assert z["det_box"].dtype == np.float32 and z["det_box"].shape == (6, 4)
    assert z["det_box"][0].tolist() == [10, 20, 30, 40]
    assert z["win"].dtype == np.float64 and z["win"].tolist() == [32.0, 768.0, 0.0, -6.0]
    assert str(z["camera"]) == "wide" and str(z["weights_sha256"]) == sha and len(str(z["weights_sha256"])) == 64
    assert not [f for f in os.listdir(os.path.join(p(tmp_path), "out", "wide")) if f.endswith(".part")]
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    for k in ("arguments", "clips_done", "clips_without_video", "frames", "frames_not_ok", "detections_kept", "detector_seconds",
              "gpu_seconds", "tool_sha256", "weights_sha256", "versions"):
        assert k in r
    assert r["clips_done"] == 2 and r["frames"] == 2 * N and r["detections_kept"] == 12 and r["frames_not_ok"] == 0
    assert r["tool_sha256"] == hashlib.sha256(open(bd.__file__, "rb").read()).hexdigest()
    assert "torch" in r["versions"] and "torchvision" in r["versions"]


def test_score_keep_is_an_argument(tmp_path, rec):
    stage(tmp_path, CLIPS[:1])
    rc, _ = cli("wide", tmp_path, CLIPS[:1], FakeDet(WIDE_TABLE), extra=["--score-keep", "0.5"])
    assert rc == 0
    z = load(tmp_path, "wide", CLIPS[0])
    assert z["det_frame"].tolist() == [0, 2, 3, 3] and z["det_score"].min() >= 0.5


def test_resumable_and_never_overwrites(tmp_path, rec):
    stage(tmp_path, CLIPS[:2])
    det = FakeDet(WIDE_TABLE)
    assert cli("wide", tmp_path, CLIPS[:2], det)[0] == 0
    f0 = os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[0]}.npz")
    before = open(f0, "rb").read()
    open(f0, "wb").write(b"SENTINEL")                       # a file that exists is never replaced
    n_img = det.n_images
    assert cli("wide", tmp_path, CLIPS[:2], det)[0] == 0
    assert open(f0, "rb").read() == b"SENTINEL" and det.n_images == n_img and len(before) > 8
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["clips_done"] == 0 and r["clips_skipped_existing"] == 2
    # a third clip added to the list is the only one run
    stage(tmp_path, CLIPS)
    assert cli("wide", tmp_path, CLIPS, det)[0] == 0
    assert det.n_images == n_img + N and os.path.exists(os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[2]}.npz"))


def test_sha_prefix_refusal_and_missing_file(tmp_path, rec):
    stage(tmp_path, CLIPS[:1])
    det = FakeDet(WIDE_TABLE)
    rc, _ = cli("wide", tmp_path, CLIPS[:1], det, prefix="dd69338a")
    assert rc == 2 and det.n_images == 0
    assert not os.path.exists(os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[0]}.npz"))
    with pytest.raises(ValueError):
        bd.verify_weights(os.path.join(p(tmp_path), "missing.pth"), "dd69338a")
    path, sha = weights(tmp_path)
    assert bd.verify_weights(path, sha[:8]) == sha
    with pytest.raises(ValueError):
        bd.verify_weights(path, "")


def test_detector_is_built_without_any_weight_enum_or_download():
    src = open(bd.__file__, encoding="utf-8").read()
    assert "fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)" in src
    for bad in ("urlopen", "load_state_dict_from_url", "torch.hub", "requests", "wget", "curl", "DEFAULT"):
        assert bad not in src, bad
    assert "weights=" in src and "_V2_COCO" not in src and "COCO_V1" not in src


def test_clip_without_video_gets_no_file_and_is_listed(tmp_path, rec):
    stage(tmp_path, CLIPS, no_wide_video=[CLIPS[1]])
    rc, _ = cli("wide", tmp_path, CLIPS, FakeDet(WIDE_TABLE))
    assert rc == 0
    assert not os.path.exists(os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[1]}.npz"))
    assert os.path.exists(os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[0]}.npz"))
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["clips_without_video"] == 1 and r["clips_without_video_ids"] == [CLIPS[1][:8]] and r["clips_done"] == 2


def test_tele_unmatched_rows_are_not_ok_and_carry_no_detections(tmp_path, rec):
    stage(tmp_path, CLIPS[:2], tele=CLIPS[:1])                      # the second clip has no tele video
    rec.tele_ts = [0, 66666, 199998, 266664, 333330]                # pilot j=2 (t=133332) is 66666 us from both neighbours: no match
    det = FakeDet({**TELE_TABLE, 0: [([1, 2, 3, 4], 0.9, 3)]})
    rc, _ = cli("tele", tmp_path, CLIPS[:2], det)
    assert rc == 0
    z = load(tmp_path, "tele", CLIPS[0])
    assert z["frame_ok"].tolist() == [True, True, False, True, True, True]
    assert z["det_frame"].tolist() == [0, 3, 4, 5] and z["det_label"].tolist() == [3, 8, 3, 6]
    assert 2 not in z["det_frame"].tolist() and det.n_images == 5     # the detector never saw the blank row
    assert z["win"].tolist() == [16.0, 1024.0, 0.0, 0.0] and str(z["camera"]) == "tele"
    assert not os.path.exists(os.path.join(p(tmp_path), "out", "tele", f"{CLIPS[1]}.npz"))
    r = json.load(open(os.path.join(p(tmp_path), "out", "tele", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["clips_without_video"] == 1 and r["frames_not_ok"] == 1 and r["frames_tele_unmatched"] == 1
    assert all("tele" in m for m, _ in rec.decoded)                 # the tele video was decoded, not the wide one


def test_decode_failure_marks_rest_not_ok_and_keeps_the_file(tmp_path, rec):
    stage(tmp_path, CLIPS[:1])
    rec.fail_after = 4
    rc, _ = cli("wide", tmp_path, CLIPS[:1], FakeDet(WIDE_TABLE))
    assert rc == 5
    z = load(tmp_path, "wide", CLIPS[0])
    assert z["frame_ok"].tolist() == [True, True, True, False, False, False]
    assert z["det_frame"].max() <= 2
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["decode_errors"] == 1 and r["frames_not_ok"] == 3


def test_window_parameters_reach_the_render_call(tmp_path, rec):
    stage(tmp_path, CLIPS[:1], tele=CLIPS[:1])
    rec.tele_ts = [0, 66666, 133332, 199998, 266664, 333330]
    wpath, sha = weights(tmp_path)
    lst = write_list(tmp_path, "l.txt", CLIPS[:1])
    base = ["--list", lst, "--root", os.path.join(p(tmp_path), "wide"), "--weights", wpath, "--weights-sha-prefix", sha[:8],
            "--out-dir", os.path.join(p(tmp_path), "o2"), "--device", "cpu", "--workers", "1"]
    assert bd.main(["--camera", "wide", "--win-size-deg", "32", "--win-px", "768", "--az-left-deg", "1.5", "--el-up-deg=-6"] + base,
                   FakeDet(WIDE_TABLE)) == 0
    assert rec.frames[-1] == (32.0, 768) and rec.grids[-1] == (32.0, 768, 1.5, -6.0)
    assert rec.intr[-1].endswith("/wide")
    assert bd.main(["--camera", "tele", "--tele-root", os.path.join(p(tmp_path), "tele"), "--win-size-deg", "16", "--win-px", "1024",
                    "--az-left-deg=-0.5", "--el-up-deg", "0"] + base, FakeDet(TELE_TABLE)) == 0
    assert rec.frames[-1] == (16.0, 1024) and rec.grids[-1] == (16.0, 1024, -0.5, 0.0)
    assert rec.intr[-1].endswith("/tele")


def test_limit_clips_counts_clips_with_video(tmp_path, rec):
    stage(tmp_path, CLIPS, no_wide_video=[CLIPS[0]])
    rc, _ = cli("wide", tmp_path, CLIPS, FakeDet(WIDE_TABLE), extra=["--limit-clips", "1"])
    assert rc == 0
    files = sorted(f for f in os.listdir(os.path.join(p(tmp_path), "out", "wide")) if f.endswith(".npz"))
    assert files == [f"{CLIPS[1]}.npz"]


def test_two_lists_are_merged_and_flags_refused(tmp_path, rec):
    stage(tmp_path, CLIPS)
    wpath, sha = weights(tmp_path)
    l1, l2 = write_list(tmp_path, "a.txt", CLIPS[:2]), write_list(tmp_path, "b.txt", CLIPS[1:])
    base = ["--camera", "wide", "--root", os.path.join(p(tmp_path), "wide"), "--weights", wpath, "--weights-sha-prefix", sha[:8],
            "--out-dir", os.path.join(p(tmp_path), "o3"), "--win-size-deg", "32", "--win-px", "768", "--device", "cpu"]
    assert bd.main(["--list", l1, "--list", l2, "--workers", "1"] + base, FakeDet(WIDE_TABLE)) == 0
    assert len([f for f in os.listdir(os.path.join(p(tmp_path), "o3", "wide")) if f.endswith(".npz")]) == 3
    assert bd.main(["--list", l1, "--workers", "9"] + base, FakeDet(WIDE_TABLE)) == 2           # more than 4 threads refused
    tele_no_root = [a if a != "wide" else "tele" for a in base]
    assert bd.main(["--list", l1] + tele_no_root, FakeDet(TELE_TABLE)) == 2                      # tele without --tele-root refused


def test_projection_of_gpu_hours(tmp_path):
    d = os.path.join(p(tmp_path), "cam")
    os.makedirs(d)
    row = {"frames_detected": 301, "clips_done": 3, "detector_seconds": 31.0, "detector_first_call_seconds": 1.0,
           "detector_first_call_frames": 1}
    open(os.path.join(d, "BOXDET_RUNS.jsonl"), "w", encoding="utf-8").write(json.dumps(row) + "\n")
    h = bd.projected_hours(d, 600)                                    # 30 s / 300 frames = 0.1 s per frame; 301/3 frames per clip
    assert abs(h - 0.1 * (301 / 3) * 600 / 3600.0) < 1e-9
    assert bd.projected_hours(os.path.join(p(tmp_path), "nothing"), 10) is None


def test_no_forbidden_words_in_the_kit_files():
    for name in ("box_detect.py", "test_box_detect.py", "pod_boxread_queue.sh", "BOXDET_DEFS.md"):
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            continue
        txt = open(path, encoding="utf-8").read().lower()
        for w in FORBIDDEN:
            assert w not in txt, (name, w)


# ------------------------------------------------------------------------------------------------ tests of the queue script
def find_bash():
    cand = os.environ.get("BOXDET_TEST_BASH", "")
    if cand and os.path.exists(cand):
        return cand
    if os.name == "nt":
        for c in ("C:/Program Files/Git/usr/bin/bash.exe", "C:/Program Files/Git/bin/bash.exe"):
            if os.path.exists(c):
                return c
    return shutil.which("bash")


def run_queue(tmp, env_extra):
    bash = find_bash()
    if not bash:
        pytest.skip("no bash")
    env = dict(os.environ)
    env.update({"PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1", "PY": p(sys.executable), "FORK_STATE_FILE": "",
                "OUT": os.path.join(p(tmp), "qout"), "CG_DIR": os.path.join(p(tmp), "nocg")})
    env["PATH"] = os.path.dirname(bash) + os.pathsep + env.get("PATH", "")
    env.update(env_extra)
    cwd = os.environ.get("BOXDET_TEST_CWD") or p(tmp)
    r = subprocess.run([bash, p(QUEUE)], env=env, cwd=cwd, capture_output=True, encoding="utf-8", errors="replace", timeout=240)
    return r


STUB = '''import json, os, sys
a = sys.argv[1:]
def val(k):
    for i, x in enumerate(a):
        if x == k: return a[i + 1]
        if x.startswith(k + "="): return x.split("=", 1)[1]
cam, od = val("--camera"), val("--out-dir")
open(os.environ["STUB_LOG"], "a").write(cam + " " + ("smoke" if "--limit-clips" in a else "full") + "\\n")
os.makedirs(os.path.join(od, cam), exist_ok=True)
row = {"frames_detected": 100, "clips_done": 3, "detector_seconds": float(os.environ["STUB_SECS"]),
       "detector_first_call_seconds": 0.0, "detector_first_call_frames": 0}
open(os.path.join(od, cam, "BOXDET_RUNS.jsonl"), "a").write(json.dumps(row) + "\\n")
'''


def queue_env(tmp, secs):
    stub = os.path.join(p(tmp), "stub_det.py")
    open(stub, "w", encoding="utf-8").write(STUB)
    ev = write_list(tmp, "eval.txt", [f"e{i:07d}-clip" for i in range(150)])       # 300 clips in all: the projection scales with it
    tr = write_list(tmp, "train.txt", [f"t{i:07d}-clip" for i in range(150)])
    boxd = os.path.join(p(tmp), "boxes")
    os.makedirs(boxd, exist_ok=True)
    for c in [f"e{i:07d}-clip" for i in range(150)] + [f"t{i:07d}-clip" for i in range(150)]:       # synthetic: 20 lead frames of 101
        np.savez(os.path.join(boxd, f"{c}.npz"), has_lead=np.arange(101) < 20)
    return {"DET": stub, "PROJ": p(bd.__file__), "LIST_EVAL": ev, "LIST_TRAIN": tr, "STUB_LOG": os.path.join(p(tmp), "stub.log"),
            "STUB_SECS": str(secs), "WEIGHTS": os.path.join(p(tmp), "w.pth"), "BOX_DIR": boxd}


def make_box(tmp, clip, has, name="boxes"):
    d = os.path.join(p(tmp), name)
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, f"{clip}.npz"), has_lead=np.asarray(has, bool))
    return d


def lead_args(tmp, name="boxes"):
    return ["--box-dir", os.path.join(p(tmp), name), "--only-lead-frames"]


def test_only_lead_frames_selects_runs_and_keeps_frame_ok(tmp_path, rec):
    stage(tmp_path, CLIPS[:1])
    make_box(tmp_path, CLIPS[0], [1, 0, 0, 1, 0, 0])
    det = FakeDet(WIDE_TABLE)
    rc, _ = cli("wide", tmp_path, CLIPS[:1], det, extra=lead_args(tmp_path))
    assert rc == 0
    z = load(tmp_path, "wide", CLIPS[0])
    assert z["frame_run"].dtype == np.bool_ and z["frame_run"].tolist() == [True, False, False, True, False, False]
    assert z["frame_ok"].all()                                       # frames left out were not tried: not a failure
    assert z["det_frame"].tolist() == [0, 3, 3] and det.n_images == 2   # j2's truck was never looked for
    assert set(z["det_frame"].tolist()) <= set(np.nonzero(z["frame_run"])[0].tolist())
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["frames_selected"] == 2 and r["frames_run"] == 2 and r["only_lead_frames"] is True and r["box_dir"].endswith("boxes")
    assert all(d[1] in (0, 6) for d in rec.decoded)                  # only the selected video frames were decoded


def test_without_the_flag_every_frame_runs_and_frame_run_equals_frame_ok(tmp_path, rec):
    stage(tmp_path, CLIPS[:1])
    rc, _ = cli("wide", tmp_path, CLIPS[:1], FakeDet(WIDE_TABLE))
    assert rc == 0
    z = load(tmp_path, "wide", CLIPS[0])
    assert z["frame_run"].all() and (z["frame_run"] == z["frame_ok"]).all()
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert r["frames_selected"] == N and r["frames_run"] == N and r["only_lead_frames"] is False


def test_only_lead_frames_tele_unmatched_selected_frame_is_not_ok_and_not_run(tmp_path, rec):
    stage(tmp_path, CLIPS[:1], tele=CLIPS[:1])
    rec.tele_ts = [0, 66666, 199998, 266664, 333330]                # j2 unmatched
    make_box(tmp_path, CLIPS[0], [1, 1, 1, 1, 1, 0])
    rc, _ = cli("tele", tmp_path, CLIPS[:1], FakeDet(TELE_TABLE), extra=lead_args(tmp_path))
    assert rc == 0
    z = load(tmp_path, "tele", CLIPS[0])
    assert z["frame_run"].tolist() == [True, True, False, True, True, False]
    assert z["frame_ok"].tolist() == [True, True, False, True, True, True]
    assert z["det_frame"].tolist() == [0, 3, 4]


def test_only_lead_frames_refuses_length_mismatch_missing_file_and_missing_dir(tmp_path, rec):
    stage(tmp_path, CLIPS[:2])
    make_box(tmp_path, CLIPS[0], [1, 0, 1, 0])                       # 4 entries, the clip has 6 cache frames
    det = FakeDet(WIDE_TABLE)
    rc, _ = cli("wide", tmp_path, CLIPS[:1], det, extra=lead_args(tmp_path))
    assert rc == 5 and det.n_images == 0
    assert not os.path.exists(os.path.join(p(tmp_path), "out", "wide", f"{CLIPS[0]}.npz"))
    r = json.load(open(os.path.join(p(tmp_path), "out", "wide", "BOXDET_RECORD.json"), encoding="utf-8"))
    assert "box file has 4 frames" in r["failures"][0][1]
    rc, _ = cli("wide", tmp_path, CLIPS[:2], det, extra=lead_args(tmp_path))      # the second clip has no box file at all
    assert rc == 2 and det.n_images == 0
    rc, _ = cli("wide", tmp_path, CLIPS[:1], det, extra=["--only-lead-frames"])
    assert rc == 2


def test_count_lead_frames_and_projection_from_selected_counts(tmp_path):
    d = make_box(tmp_path, CLIPS[0], [1, 1, 0, 1])
    make_box(tmp_path, CLIPS[1], [0, 0, 1, 0])
    lst = write_list(tmp_path, "c.txt", CLIPS[:2] + [CLIPS[0]])
    assert bd.count_lead_frames(d, [lst]) == (4, 0)
    assert bd.count_lead_frames(d, [write_list(tmp_path, "c2.txt", CLIPS)]) == (4, 1)
    cam = os.path.join(p(tmp_path), "cam")
    os.makedirs(cam)
    row = {"frames_detected": 50, "clips_done": 3, "detector_seconds": 5.5, "detector_first_call_seconds": 0.5,
           "detector_first_call_frames": 1}
    open(os.path.join(cam, "BOXDET_RUNS.jsonl"), "w", encoding="utf-8").write(json.dumps(row) + "\n")
    h = bd.projected_hours(cam, n_frames=7200)                       # 5 s / 49 frames per frame
    assert abs(h - (5.0 / 49) * 7200 / 3600.0) < 1e-9


def test_queue_writes_state_file_lines_busy_then_idle_and_can_switch_off(tmp_path):
    import re
    st = os.path.join(p(tmp_path), "state.txt")
    env = queue_env(tmp_path, 0.01)
    env["FORK_STATE_FILE"] = st
    assert run_queue(tmp_path, env).returncode == 0
    lines = open(st, encoding="utf-8").read().splitlines()
    assert lines and re.fullmatch(r"IDLE \d{4}-\d\d-\d\dT[\d:]+Z", lines[-1])
    busy = [ln for ln in lines if ln.startswith("BUSY")]
    assert len(busy) == 3 and all(re.fullmatch(r"BUSY box-read detector passes of the v8proof fork \d{4}-\d\d-\d\dT[\d:]+Z", ln) for ln in busy)
    tmp2 = tmp_path / "second"
    tmp2.mkdir()
    env2 = queue_env(tmp2, 100.0)                                    # a stop also ends with IDLE
    st2 = os.path.join(p(tmp2), "state.txt")
    env2["FORK_STATE_FILE"] = st2
    assert run_queue(tmp2, env2).returncode == 4
    assert open(st2, encoding="utf-8").read().splitlines()[-1].startswith("IDLE ")
    tmp3 = tmp_path / "third"
    tmp3.mkdir()
    env3 = queue_env(tmp3, 0.01)
    env3["FORK_STATE_FILE"] = ""
    assert run_queue(tmp3, env3).returncode == 0 and not os.path.exists(os.path.join(p(tmp3), "state.txt"))


def test_queue_dry_prints_three_stages_and_starts_nothing(tmp_path):
    env = queue_env(tmp_path, 1.0)
    env["DRY"] = "1"
    r = run_queue(tmp_path, env)
    assert r.returncode == 0, r.stdout + r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("CMD:")]
    assert any("[smoke]" in c and "--camera wide" in c and "--limit-clips 3" in c for c in cmds)
    assert any("[smoke]" in c and "--camera tele" in c and "--tele-root" in c for c in cmds)
    assert any("[wide]" in c and "--win-size-deg 32 --win-px 768" in c and "--el-up-deg=-6" in c for c in cmds)
    assert any("[tele]" in c and "--win-size-deg 16 --win-px 1024" in c and "--weights-sha-prefix dd69338a" in c and "--box-dir" in c and "--only-lead-frames" in c for c in cmds)
    assert not os.path.exists(os.path.join(p(tmp_path), "qout")) and not os.path.exists(env["STUB_LOG"])
    assert r.stdout.strip().splitlines()[-1].startswith("ZZ BOXREAD-QUEUE")


def test_queue_stop_rule_exits_4_on_a_planted_slow_smoke(tmp_path):
    env = queue_env(tmp_path, 100.0)                                  # 1 s per frame: far above 1.0 GPU-h for hundreds of clips
    r = run_queue(tmp_path, env)
    assert r.returncode == 4, r.stdout + r.stderr
    out = os.path.join(p(tmp_path), "qout")
    assert os.path.exists(os.path.join(out, "BOXREAD_STOP.txt")) and not os.path.exists(os.path.join(out, "BOXREAD_DONE.txt"))
    last = r.stdout.strip().splitlines()[-1]
    assert last.startswith("ZZ BOXREAD-QUEUE smoke DONE rc 4 ")
    calls = open(env["STUB_LOG"]).read().split("\n")
    assert [c for c in calls if c] == ["wide smoke", "tele smoke"]    # no full pass was started


def test_queue_runs_all_stages_when_the_projection_is_low_then_refuses_a_second_start(tmp_path):
    env = queue_env(tmp_path, 0.01)
    r = run_queue(tmp_path, env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = os.path.join(p(tmp_path), "qout")
    assert os.path.exists(os.path.join(out, "BOXREAD_START.txt")) and os.path.exists(os.path.join(out, "BOXREAD_DONE.txt"))
    assert not os.path.exists(os.path.join(out, "BOXREAD_STOP.txt"))
    assert r.stdout.strip().splitlines()[-1].startswith("ZZ BOXREAD-QUEUE tele DONE rc 0 ")
    calls = [c for c in open(env["STUB_LOG"]).read().split("\n") if c]
    assert calls == ["wide smoke", "tele smoke", "wide full", "tele full"]
    r2 = run_queue(tmp_path, env)
    assert r2.returncode == 3                                         # a START marker exists: no second start without RESUME=1


def test_queue_refuses_under_the_memory_floor_and_creates_nothing(tmp_path):
    cg = os.path.join(p(tmp_path), "cg")
    os.makedirs(cg)
    open(os.path.join(cg, "memory.max"), "w").write("10000000000\n")
    open(os.path.join(cg, "memory.stat"), "w").write("anon 9000000000\nfile 5\n")
    env = queue_env(tmp_path, 0.01)
    env["CG_DIR"] = cg
    r = run_queue(tmp_path, env)
    assert r.returncode == 2 and "memory free" in r.stdout
    assert not os.path.exists(os.path.join(p(tmp_path), "qout")) and not os.path.exists(env["STUB_LOG"])


def test_queue_script_has_no_rm_mv_or_kill():
    import re
    txt = open(QUEUE, encoding="utf-8").read()
    code = "\n".join(ln for ln in txt.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"(^|[;&|\s])(rm|mv|kill|pkill|killall)(\s|$)", code)
    assert "OMP_NUM_THREADS=4" in code
