"""Re-read the anchor artifact; prove a units-stripped file is REFUSED."""
import copy, json, torch
from pathlib import Path
from tanitad.refs import anchor_meta as am

A = Path(r"C:/Users/Admin/tanitad-review-20260906/art")
p = A/"refc_anchors_6s_v0cond_alat_117.pt"
d = torch.load(p, map_location="cpu", weights_only=False)
print("keys:", sorted(d.keys()))
print("anchors", tuple(d["anchors"].shape), d["anchors"].dtype,
      "controls", tuple(d["controls"].shape), d["controls"].dtype)

res = am.read_anchor_artifact(str(p))
print("read_anchor_artifact ->", type(res).__name__)
try:
    print("  control_units =", res.control_units, " source =", getattr(res,"units_source",None))
except AttributeError:
    print("  repr:", repr(res)[:400])
print("  declared:", {k: d.get(k) for k in
      ("control_units","horizon_s","dt","ref_speed_ms","kappa_cap","alat_v_floor")})

# ---- units column meaning, read from the FILE not from memory --------------
print("  controls_columns (in file):", d.get("controls_columns"))
c = d["controls"]
print("  controls col0 range %.4f..%.4f   col1 range %.4f..%.4f"
      % (c[:,0].min(), c[:,0].max(), c[:,1].min(), c[:,1].max()))

# ---- THE KAMM CONTROL: alat reading vs the curvature misreading ------------
V = 36.0
alat_g = float(c[:,1].abs().max())/9.81
kappa_g = (V**2 * float(c[:,1].abs().max()))/9.81
print("  IF col1 is a_lat (m/s^2): peak %.3f g   over mu=0.7: %d/117"
      % (alat_g, int((c[:,1].abs()/9.81 > 0.7).sum())))
print("  IF col1 were curvature (1/m) at v=36: peak %.1f g   over mu=0.7: %d/117"
      % (kappa_g, int(((V**2*c[:,1].abs())/9.81 > 0.7).sum())))

# ---- DELIBERATE REGRESSION: strip the units, prove REFUSAL -----------------
strip = copy.deepcopy(d)
for k in ("control_units","controls_columns","schema","horizon_s","dt",
          "ref_speed_ms","kappa_cap","alat_v_floor","provenance"):
    strip.pop(k, None)
q = A/"_REGRESSION_units_stripped.pt"
torch.save(strip, q)
print()
print("DELIBERATE REGRESSION (units stripped, keys now: %s)" % sorted(strip.keys()))
try:
    r2 = am.read_anchor_artifact(str(q))
    print("  ⛔ NOT REFUSED -> guard is decoration. got:", repr(r2)[:200])
except Exception as e:
    print("  ✅ REFUSED:", type(e).__name__, str(e)[:220])
# CONVERSE CONTROL: the untouched file must NOT be refused
try:
    am.read_anchor_artifact(str(p)); print("  ✅ CONVERSE CONTROL: untouched file NOT refused (guard not over-broad)")
except Exception as e:
    print("  ⛔ CONVERSE CONTROL FAILED — guard refuses a legal file:", e)
q.unlink()
