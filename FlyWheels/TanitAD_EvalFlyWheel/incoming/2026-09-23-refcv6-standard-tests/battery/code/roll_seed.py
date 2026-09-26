"""ONE inference-seed roll in its OWN process (called by run_battery.py): its CUDA context dies with
it, so the orchestrating parent never holds the card while it waits on the gate."""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.stdout.reconfigure(encoding="utf-8")
import refcv6_loader as L  # noqa: E402

L.bootstrap()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--windows-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--smoke-cpu-fp32-trunk", action="store_true")
    a = ap.parse_args()
    if a.smoke_cpu_fp32_trunk:
        _orig = L.build_model

        def _fp32(*aa, **kk):
            m, c, ar, r = _orig(*aa, **kk)
            for mod in m.modules():
                lv = getattr(mod, "memory_levers", None)
                if isinstance(lv, dict) and "bf16" in lv:
                    lv["bf16"] = False
                    lv["channels_last"] = False
            r["smoke_note"] = "trunk bf16/NHWC OFF (--smoke-cpu-fp32-trunk)"
            return m, c, ar, r
        L.build_model = _fp32
    import run_battery as RB
    windows = json.load(open(a.windows_json, encoding="utf-8"))
    rec = RB.roll_one(a.ckpt, a.config, a.seed, a.dump_dir, windows, None, device=a.device)
    rec["device"] = a.device
    json.dump(rec, open(a.out_json, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[roll_seed] seed {a.seed} -> {a.out_json}")


if __name__ == "__main__":
    main()
