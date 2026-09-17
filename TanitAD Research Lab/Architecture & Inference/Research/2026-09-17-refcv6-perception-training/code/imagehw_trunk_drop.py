"""D-REFCV6-IMAGEHW-DROPS-TRUNK-FIELDS -- measured on BOTH sides.

The claim: on tip `c16b7f1`, `--image-hw` reconstructs `CNNEncoderConfig` from a
HAND-LISTED subset of its fields, so four trunk fields revert to their dataclass
defaults -- including two CONTROL ARMS.

⛔ The evidence is the CONFIG THE TRAINER WOULD BUILD, read from both trees
through each tree's OWN `_pin_trainer_cfg`. A comparison against my own
expectation would measure my expectation; a comparison against the ARGV is the
independent reference, because argv is what the operator asked for.
"""
import io
import importlib
import json
import sys
from pathlib import Path

CASES = [
    # (flag list, dest, asked-for value, why it matters)
    (["--trunk-name", "resnet34.a1_in1k"], "trunk_name", "resnet34.a1_in1k",
     "the PI's SECOND COMPARISON RUN; 256 vs 1024 channels at stride 16"),
    (["--trunk-mode", "inflate"], "trunk_mode", "inflate",
     "the cheaper alternative arm (one pass, inflated stem)"),
    (["--trunk-fuse", "last"], "trunk_fuse", "last",
     "⛔ THE SINGLE-FRAME CONTROL ARM"),
    (["--trunk-fuse-plain-init"], "trunk_fuse_identity", False,
     "⛔ THE DELIBERATE REGRESSION of the identity init"),
    (["--no-trunk-pretrained"], "trunk_pretrained", False,
     "⛔ THE IMAGENET KNOCKOUT ARM (this flag is new; absent on the tip)"),
]
BASE = ["--arm", "hier", "--size", "tiny", "--out", "X", "--trunk", "timm"]
rep = {}
for tag, wt in (("tip", r"C:/Users/Admin/tanitad-wt-perctrain-tip"),
                ("patched", r"C:/Users/Admin/tanitad-wt-perctrain")):
    for p in list(sys.path):
        if "tanitad-wt-perctrain" in p:
            sys.path.remove(p)
    for p in (wt + "/stack/scripts", wt + "/taniteval", wt, wt + "/stack"):
        sys.path.insert(0, p)
    for m in [m for m in list(sys.modules)
              if m == "refc_v3_train" or m.startswith("tanitad")]:
        del sys.modules[m]
    T = importlib.import_module("refc_v3_train")
    assert Path(T.__file__).is_relative_to(Path(wt)), (T.__file__, wt)
    rows = []
    for flags, dest, want, why in CASES:
        row = {"flags": flags, "field": dest, "asked": want, "why": why}
        for hw, key in ((None, "without_image_hw"),
                        (["--image-hw", "256", "1024"], "with_image_hw")):
            try:
                a = T.build_parser().parse_args(BASE + flags + (hw or []))
                c = T._pin_trainer_cfg(
                    T.v3.refc_v3_sized_config(a.size, hier=True), a)
                row[key] = getattr(c.core.encoder, dest)
            except SystemExit as e:
                row[key] = f"<SystemExit: {e}>"
            except Exception as e:                       # noqa: BLE001
                row[key] = f"<{type(e).__name__}: {str(e)[:120]}>"
        row["honoured_without_image_hw"] = row["without_image_hw"] == want
        row["honoured_with_image_hw"] = row["with_image_hw"] == want
        row["DROPPED_BY_image_hw"] = bool(row["honoured_without_image_hw"]
                                          and not row["honoured_with_image_hw"])
        rows.append(row)
    rep[tag] = {"trainer": T.__file__, "cases": rows,
                "n_dropped": sum(r["DROPPED_BY_image_hw"] for r in rows)}

rep["verdict"] = {
    "tip_fields_dropped_by_image_hw":
        [r["field"] for r in rep["tip"]["cases"] if r["DROPPED_BY_image_hw"]],
    "patched_fields_dropped_by_image_hw":
        [r["field"] for r in rep["patched"]["cases"]
         if r["DROPPED_BY_image_hw"]],
    # ⭐ THE DISCRIMINATING CONTROL: the probe must SEE the defect on the tip.
    # A probe that reads "nothing dropped" on both trees has proven nothing.
    "probe_detects_the_defect_on_the_tip": rep["tip"]["n_dropped"] > 0,
    "patched_is_clean": rep["patched"]["n_dropped"] == 0,
}
io.open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(rep, indent=1))
print(json.dumps(rep["verdict"], indent=1))
for t in ("tip", "patched"):
    print("\n==", t)
    for r in rep[t]["cases"]:
        print(" %-22s asked=%-18s without=%-18s with=%-18s dropped=%s"
              % (r["field"], r["asked"], r["without_image_hw"],
                 r["with_image_hw"], r["DROPPED_BY_image_hw"]))
