from pathlib import Path
p = Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/a3_keq_roll.py"); s = p.read_text(encoding="utf-8")
rep = [('ap.add_argument("--min-windows", type=int, default=500)', 'ap.add_argument("--min-windows", type=int, default=500)\nap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))'),
       ('"--windows-json", str(WIN), "--device", "cuda",', '"--windows-json", str(WIN), "--device", A.device,'),
       ('"seed": A.seed, "head": ST["attached"], "pose_clock": POSE,', '"seed": A.seed, "device": A.device, "head": ST["attached"], "pose_clock": POSE,')]
for a, b in rep:
    assert s.count(a) == 1, a
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8"); print("ok")
