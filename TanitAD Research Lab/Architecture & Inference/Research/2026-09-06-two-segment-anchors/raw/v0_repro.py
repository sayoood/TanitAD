"""V0 - can we REPRODUCE the live refcv4b bank from its controls?

⛔ POSITIVE ASSERTIONS ONLY. Every check prints VERIFIED/MISMATCH with the full
64-char sha, and refuses to call a comparison a match when either side is not
64 chars (the empty-string-equals-empty-string hole, CLAUDE.md 2026-09-04).
"""
import hashlib
import sys

import torch

import _env  # noqa: F401
from tanitad.refs import anchor_twoseg as ts   # noqa: E402

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"
EXPECT_A = "51f930dc6f3564ff8f21c9070ca97b805d4e82908e42d0ef1f99be9f5a3a66df"
EXPECT_C = "b072f4c052331beb79bef117c7b233f702802b517429cc9157a841294e089664"
HORIZONS = (5, 10, 15, 20, 30, 40, 50, 60)
SLOTS = [h - 1 for h in HORIZONS]


def sha(t):
    return hashlib.sha256(
        t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes()
    ).hexdigest()


def cmp(name, got, want):
    if len(got) != 64 or len(want) != 64:
        print(f"  {name:34s} INCONCLUSIVE (a sha is not 64 chars)")
        return False
    ok = got == want
    print(f"  {name:34s} {'VERIFIED' if ok else 'MISMATCH'} {got}")
    if not ok:
        print(f"  {'':34s} expected                 {want}")
    return ok


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    d = sd.get("model", sd)
    A = d["core.decoder.anchors"].float()
    C = d["core.decoder.anchor_controls"].float()
    print(f"[src] anchors {tuple(A.shape)} controls {tuple(C.shape)}")
    ok = True
    ok &= cmp("checkpoint anchors", sha(A), EXPECT_A)
    ok &= cmp("checkpoint controls", sha(C), EXPECT_C)

    # 1. reproduce `anchors` by rolling `controls` at ref_speed_ms
    print("\n[1] reproduce the stored `anchors` from `controls` at v_ref")
    for vref in (10.0,):
        r = ts.roll_bank(C, torch.tensor([vref]), control_units="alat",
                         steps=60, slots=SLOTS, dt=0.1,
                         alat_v_floor=4.0, kappa_cap=0.12)[0]
        cmp(f"roll(controls, v={vref}) == anchors", sha(r), EXPECT_A)

    # 2. B2 - the SINGLE-SEGMENT LIMIT is bit-identical
    print("\n[2] B2 single-segment limit: t_split = horizon_s reproduces the "
          "constant roll BIT-IDENTICALLY")
    speeds = torch.tensor([0.0, 1.0, 4.0, 8.0, 10.0, 16.0, 22.0, 30.0, 36.0])
    base2 = ts.roll_bank(C, speeds, control_units="alat", steps=60,
                         slots=SLOTS)
    C3 = ts.as_three_column(C, 6.0)
    base3 = ts.roll_bank(C3, speeds, control_units="alat", steps=60,
                         slots=SLOTS)
    same = torch.equal(base2, base3)
    print(f"  torch.equal over {tuple(base2.shape)} "
          f"({base2.numel():,} floats, {len(speeds)} speeds): "
          f"{'BIT-IDENTICAL' if same else 'DIFFERS'}")
    if not same:
        print(f"  max |diff| = {float((base2 - base3).abs().max()):.3e}")
    ok &= same
    # and a split INSIDE the horizon must NOT be identical -- the mutation
    # control: if this also read identical the test above would be vacuous.
    C3b = C3.clone()
    C3b[:, 2] = 3.0
    mid = ts.roll_bank(C3b, speeds, control_units="alat", steps=60,
                       slots=SLOTS)
    diff = not torch.equal(base2, mid)
    print(f"  MUTATION CONTROL t_split=3.0 differs: "
          f"{'YES (the check can fail)' if diff else 'NO -- THE TEST IS VACUOUS'}"
          f"   max |diff| = {float((base2 - mid).abs().max()):.4f} m")
    ok &= diff
    # ...but NOT on the 2 s prefix (slots 0..3): B5's structural zero
    p = float((base2[:, :, :4] - mid[:, :, :4]).abs().max())
    print(f"  B5 2 s prefix (slots 0-3) max |diff| = {p:.10e} m  "
          f"{'EXACT ZERO' if p == 0.0 else 'NON-ZERO'}")
    ok &= (p == 0.0)

    # 3. the family itself
    print("\n[3] the default two-segment family")
    new = ts.two_segment_controls()
    print(f"  {tuple(new.shape)}")
    for i, row in enumerate(new.tolist()):
        print(f"   [{i}] a_lon={row[0]:+.4f}  a_lat={row[1]:+.4f}  "
              f"t_split={row[2]:.1f}s")
    ext = ts.extend_controls(C, 6.0)
    print(f"  extended controls {tuple(ext.shape)}")
    print(f"  {ts.describe_family(ext, 117)}")
    # the first 117 rows must be UNTOUCHED
    cmp("extended[:117, :2] == controls",
        sha(ext[:117, :2].contiguous()), EXPECT_C)

    print("\nV0_EXIT=%d" % (0 if ok else 1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
