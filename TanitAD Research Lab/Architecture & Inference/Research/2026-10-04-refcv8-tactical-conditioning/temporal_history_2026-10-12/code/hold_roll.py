"""ONE roll of refcv7@50,400 on the G3 windows with a camera-history manipulation (arm) -- its own process.
usage: python hold_roll.py <arm: real|hold|st|nt|blind> <seed> <dump_dir> <windows.json> <out.json>
Wraps the battery's r7_roll dataset class: after the trainer's own V3Dataset builds an item, item['frames'] [8,9,H,W] u8
(row r, block k = raw slot r+k, slot 9 = t0) is rebuilt from the 10 raw slots per the arm. Everything else is the battery roll."""
import json, os, sys
from pathlib import Path
HERE = Path(os.environ.get("BATT_CODE", "E:/hold_temporal/work/battery"))
sys.path.insert(0, str(HERE)); sys.stdout.reconfigure(encoding="utf-8")
from tanitad.eval import refcv7_loader as L  # noqa: E402
L.bootstrap()
import torch  # noqa: E402
import r7_roll as RR  # noqa: E402
import run_battery_r7 as RB  # noqa: E402

ARM, SEED, DUMP, WIN, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
STATS = {"n": 0, "nt_clamped": 0, "real_roundtrip_ok": 0}
_orig = RR.make_dataset_cls


def build_stack(raw):                       # raw: list of 10 [3,H,W] u8, slot 0..9 -> [8,9,H,W]
    return torch.stack([torch.cat([raw[r], raw[r + 1], raw[r + 2]], 0) for r in range(8)])


def transform(ds, i, item):
    fr = item["frames"]
    assert fr.dtype == torch.uint8 and tuple(fr.shape[:2]) == (8, 9), (fr.dtype, tuple(fr.shape))
    raw = [fr[0, 0:3], fr[0, 3:6]] + [fr[r, 6:9] for r in range(8)]
    if STATS["real_roundtrip_ok"] < 3:
        assert torch.equal(build_stack(raw), fr), "row/slot model of the stack is wrong"
        STATS["real_roundtrip_ok"] += 1
    STATS["n"] += 1
    if ARM == "real":
        return item
    if ARM == "hold":
        new = torch.stack([torch.cat([raw[r + 2]] * 3, 0) for r in range(8)])
    elif ARM == "st":
        new = build_stack([raw[9]] * 10)
    elif ARM == "blind":
        new = torch.full_like(fr, 128)
    elif ARM == "nt":
        e_i, t = ds.index[i]
        j = t + 8 - 1 - 10                  # row whose NEWEST raw frame is t0-10 (t0 = newest of row t+7)
        if j >= 0:
            a = ds.episodes[e_i].frames[j:j + 1][0, 6:9].clone()
        else:
            STATS["nt_clamped"] += 1
            a = ds.episodes[e_i].frames[0:1][0, 0:3].clone()
        b, c = raw[4], raw[9]
        new = build_stack([a, a, b, b, b, b, b, c, c, c])
    else:
        raise SystemExit("unknown arm " + ARM)
    assert new.shape == fr.shape and new.dtype == fr.dtype
    item["frames"] = new.contiguous()
    return item


def patched(tr, ctx):
    base = _orig(tr, ctx)

    class HistWindows(base):
        def _build(self, i):
            with type(self)._lock:
                return transform(self, i, tr.V3Dataset.__getitem__(self, i))
    return HistWindows


RR.make_dataset_cls = patched
windows = json.load(open(WIN, encoding="utf-8"))
ckpt, cfg = "D:/refcv7_eval_kit/ckpt/ckpt_50400.pt", "D:/refcv7_eval_kit/ckpt/config.json"
rec = RB.roll_one(ckpt, cfg, SEED, DUMP, windows, None, device="cuda", full=False)
rec["hist_arm"] = ARM; rec["hist_stats"] = STATS
json.dump(rec, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
print("[hold_roll] done", ARM, SEED, STATS, "wall_s", rec["wall_s"], "cuda_max_gib", rec["cuda_max_memory_allocated_gib"])
