"""max|delta| between the new units-declaring bank and the LIVE bank."""
import hashlib, torch
from pathlib import Path
A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
new = torch.load(A/"refc_anchors_6s_v0cond_alat_117.pt", map_location="cpu", weights_only=False)
for live_p in [Path(r"C:/Users/Admin/navcomp/ckpt/anchors.pt"),
               Path(r"C:/Users/Admin/refcv4b_egodrop/pull/anchors.pt")]:
    live = torch.load(live_p, map_location="cpu", weights_only=False)
    print("LIVE", live_p, "md5", hashlib.md5(live_p.read_bytes()).hexdigest(),
          "bytes", live_p.stat().st_size)
    print("   live keys:", sorted(live.keys()) if isinstance(live, dict) else type(live))
    for k in ("anchors", "controls"):
        a, b = new[k], live[k]
        if a.shape != b.shape:
            print(f"   {k}: SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}"); continue
        dmax = float((a.double() - b.double()).abs().max())
        print(f"   {k}: shape {tuple(a.shape)}  max|delta| = {dmax!r}   IDENTICAL={dmax==0.0}")
    # NEGATIVE CONTROL: the comparison must be able to see a difference
    pert = new["controls"].clone(); pert[0,0] += 1e-6
    print("   NEGATIVE CONTROL (perturb one element by 1e-6): max|delta| = %r  (non-vacuous=%s)"
          % (float((pert.double()-live["controls"].double()).abs().max()),
             float((pert.double()-live["controls"].double()).abs().max()) != 0.0))
    # does the LIVE bank declare units?
    print("   live control_units:", live.get("control_units", "<<ABSENT>>") if isinstance(live, dict) else "n/a")
    print()
