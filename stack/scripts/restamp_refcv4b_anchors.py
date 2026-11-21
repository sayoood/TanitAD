#!/usr/bin/env python3
"""Re-stamp refcv4b's 117 v0-conditioned anchor bank with DECLARED units.

The shipped `anchors.pt` is a bare {anchors, controls} dict: nothing in the file
says that column 1 of `controls` is LATERAL ACCELERATION. Read as curvature the
SAME BYTES peak at 396 g; read correctly, 0.31 g. `read_anchor_artifact` refuses
it, which is the guard working -- but every consumer then has to pass
`--anchor-control-units alat` by name, and one that forgets gets a refusal
rather than a number.

This writes the declared artifact. It does NOT modify the live run's file.

⛔ The constants are NOT invented. They are read from `anchors.units.json`,
the units record written beside the bank on 2026-09-04, and every one of them
is corroborated by MODEL_REGISTRY.md 4.6. The tensors are asserted BIT-IDENTICAL
to the live bank by sha256 before anything is written.
"""
import hashlib
import json
import sys

import torch

sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.refs import anchor_meta  # noqa: E402

SRC = "/workspace/experiments/refcv4b-b1-v72-40k/anchors.pt"
UNITS = "/workspace/anchors.units.json"
OUT = "/workspace/anchors_117_alat_declared.pt"

# the values the run was ACTUALLY trained under, from the units record
EXPECT_FILE_SHA = "e86cf507d55a4585435025fe52f33817d08dab879e1f65ff6a1fc9b0eb81e8fb"
EXPECT_ANCHORS_SHA = "51f930dc6f3564ff8f21c9070ca97b805d4e82908e42d0ef1f99be9f5a3a66df"
EXPECT_CONTROLS_SHA = "b072f4c052331beb79bef117c7b233f702802b517429cc9157a841294e089664"


def main() -> int:
    u = json.load(open(UNITS))

    # 1. the SOURCE bytes must be the bank refcv4b trained on -- shape-asserted
    raw = open(SRC, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if len(got) != 64 or len(EXPECT_FILE_SHA) != 64:
        print("INCONCLUSIVE: a sha256 is not 64 chars")
        return 3
    if got != EXPECT_FILE_SHA:
        print("MISMATCH: source file sha256 %s != %s" % (got, EXPECT_FILE_SHA))
        return 1
    print("source file sha256 VERIFIED %s" % got)

    # 2. the units record must describe THIS file
    if u["file_sha256"] != got:
        print("MISMATCH: units record describes a different file")
        return 1
    print("units record matches the file it describes")

    d = torch.load(SRC, map_location="cpu", weights_only=True)
    a, c = d["anchors"], d["controls"]

    art = anchor_meta.build_anchor_artifact(
        a, c,
        control_units=u["control_units"],
        horizons=u["horizons_steps"],
        dt=u["dt_s"],
        ref_speed_ms=u["ref_speed_ms"],
        kappa_cap=u["kappa_cap_inv_m"],
        alat_v_floor=u["alat_v_floor_ms"],
        builder=u["builder"],
        extra={
            "restamped_from": SRC,
            "restamped_from_file_sha256": got,
            "restamp_reason": (
                "the shipped bank declared no control_units; the constants "
                "come from anchors.units.json written 2026-09-04 beside it, "
                "corroborated by MODEL_REGISTRY.md 4.6 and by refcv4b's own "
                "config.json['argv'] (--anchor-control-units alat)"),
            "restamped_by": "Arch+Inference, 2026-09-06 refcv5 launch",
        },
    )

    # 3. THE TENSORS MUST NOT HAVE MOVED
    if art["anchors_sha256"] != EXPECT_ANCHORS_SHA:
        print("MISMATCH: anchors tensor changed")
        return 1
    if art["controls_sha256"] != EXPECT_CONTROLS_SHA:
        print("MISMATCH: controls tensor changed")
        return 1
    print("anchors  sha256 VERIFIED %s" % art["anchors_sha256"])
    print("controls sha256 VERIFIED %s" % art["controls_sha256"])

    torch.save(art, OUT)

    # 4. READ IT BACK -- units must now resolve from the FILE, no override
    ra = anchor_meta.read_anchor_artifact(OUT)
    print("readback: control_units=%s source=%s horizon_s=%s dt=%s "
          "ref_speed_ms=%s kappa_cap=%s alat_v_floor=%s"
          % (ra.control_units, ra.control_units_source,
             ra.meta.get("horizon_s"), ra.meta.get("dt"),
             ra.meta.get("ref_speed_ms"), ra.meta.get("kappa_cap"),
             ra.meta.get("alat_v_floor")))
    if ra.control_units_source != "file":
        print("MISMATCH: units still not resolved from the file")
        return 1
    # and the tensors it hands back are still the same bytes
    if anchor_meta.sha256_of_tensor(ra.anchors) != EXPECT_ANCHORS_SHA:
        print("MISMATCH: readback anchors differ")
        return 1
    if anchor_meta.sha256_of_tensor(ra.controls) != EXPECT_CONTROLS_SHA:
        print("MISMATCH: readback controls differ")
        return 1
    print("readback tensors BIT-IDENTICAL to the live bank")

    # 5. same-breath NEGATIVE control: the ORIGINAL must still be refused
    try:
        anchor_meta.read_anchor_artifact(SRC)
        print("CONTROL FAILED: the undeclared bank was NOT refused")
        return 1
    except anchor_meta.AnchorUnitsMissing:
        print("CONTROL OK: the undeclared source bank is still REFUSED")

    print("WROTE %s (%d bytes)" % (OUT, len(open(OUT, "rb").read())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
