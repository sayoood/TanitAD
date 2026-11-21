"""THE ECHO TEST for E-DEC-49 — is `action -> delta-speed` PHYSICS or ARITHMETIC?

The panel found the action predicts the ego's own speed change at r +0.3386
(t 2.65) while the latent and the prediction both read ~0. Before that can be
called a finding I must rule out the case that makes it worthless:

  IF the corpus's action channel IS the ego's acceleration, and speed is
  INTEGRATED FROM IT, then predicting delta-speed from the action is READING THE
  LABEL'S OWN SOURCE -- the same defect as the nav-echo (route head scored 1.0000
  on a bijection of its own input) and the sitclf leak (labels derived from ego,
  classifier given ego).

This probe is CPU-only, so it costs nothing and does not contend with the GPU.

  r(act1 , dv_1tick)   ~1.0  => the channel IS the derivative. ECHO.
                       <<1.0 => partly independent; the signal is real but the
                                arithmetic component must be stated.
  r(act1 , dv_4tick)         => the actual quantity the panel predicted.

It also prints what the channels ARE (ranges, units) so the claim can name them.
"""
import pathlib, sys, os, numpy as np, torch
sys.stdout.reconfigure(encoding="utf-8")
SP = pathlib.Path(__file__).resolve().parent
LEAD = pathlib.Path(os.environ.get("SPD_CORPUS", str(SP / "sp2/cache/physicalai-val130-heldout")))
clips = sorted(LEAD.glob("*.v2ep.pt"))[:20]
A1, A0, DV1, DV4, V = [], [], [], [], []
for c in clips:
    d = torch.load(c, map_location="cpu", weights_only=False)
    a = np.asarray(d["actions"], dtype=np.float64)
    v = np.asarray(d["poses"], dtype=np.float64)[:, 3].ravel()
    m = min(len(a), len(v)) - 5
    if m < 20: continue
    i = np.arange(m)
    A0.append(a[i, 0]); A1.append(a[i, 1])
    DV1.append(v[i+1] - v[i]); DV4.append(v[i+4] - v[i]); V.append(v[i])
A0, A1 = np.concatenate(A0), np.concatenate(A1)
DV1, DV4, V = np.concatenate(DV1), np.concatenate(DV4), np.concatenate(V)
def r(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
print("\n  THE ECHO TEST - is action->delta-speed PHYSICS or ARITHMETIC?")
print(f"  n = {len(A1)} frames over {len(clips)} held-out clips\n")
print(f"  {'channel':<26}{'min':>10}{'mean':>10}{'max':>10}{'sd':>10}")
for nm, x in (("act[:,0] (steer)", A0), ("act[:,1] (accel)", A1),
              ("speed", V), ("dv over 1 tick", DV1), ("dv over 4 ticks", DV4)):
    print(f"  {nm:<26}{x.min():>10.4f}{x.mean():>10.4f}{x.max():>10.4f}{x.std():>10.4f}")
r1, r4 = r(A1, DV1), r(A1, DV4)
rs1 = r(A0, DV1)
print(f"\n  r( act[:,1] , dv_1tick )  = {r1:+.4f}   <- ~1.0 would mean ECHO")
print(f"  r( act[:,1] , dv_4tick )  = {r4:+.4f}   <- the panel's actual target")
print(f"  r( act[:,0] , dv_1tick )  = {rs1:+.4f}   <- steer, expected ~0 (sanity)")
print(f"  r( speed    , dv_4tick )  = {r(V, DV4):+.4f}   <- why spd_t failed as a control")
print()
if abs(r1) > 0.95:
    print("  => ECHO. The action channel IS the speed derivative; the panel's")
    print("     +0.3386 is arithmetic and CANNOT be quoted as a finding.")
elif abs(r1) > 0.5:
    print(f"  => PARTLY ARITHMETIC (r {r1:+.3f}). The channel is strongly related to")
    print("     the derivative. The finding must state this explicitly, and the")
    print("     load-bearing claim becomes the ZERO on z_t / zhat, not the +0.34.")
else:
    print(f"  => NOT AN ECHO (r {r1:+.3f}). The action channel is largely independent")
    print("     of the measured speed change; the +0.3386 is a real relationship.")
print()
