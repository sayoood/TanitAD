"""stack/ops/runs.d/refcv7-r101-s0.argv.json: the box head's tokens join the canonical argv (the Master Mind 2026-09-27,
after the landing 28d8365): the A9 flags + the placed VIS-1 sidecar + ``--slot-query-select learned_ref``; the box items
move from ``todo_box_head`` into ``changes_vs_refcv6`` with their SPEC sources; ``launch_prep`` names the sidecar.
Serialised exactly as the file is (json indent=1, ensure_ascii=False, trailing newline -- round-trip verified)."""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
raw = src.read_bytes()
o = json.loads(raw.decode("utf-8"))
assert (json.dumps(o, indent=1, ensure_ascii=False) + "\n").encode("utf-8") == raw, "the serialisation style moved"
SIDECAR = "/home/nvidia/data/refcv7/vis1_sidecar_refcv6b1_train4369_eval139.npz"
SIDECAR_SHA = "278443b3356bca054c15753e7c08d465d71b0327a2ce564e143d091261349dd4"
BOX = ["--slot-presence-loss", "focal", "--slot-presence-prior", "0.01", "--slot-deep-supervision", "--slot-vis1",
       "--vis1-sidecar", SIDECAR, "--slot-query-select", "learned_ref"]
argv = list(o["argv"])
for f in ("--slot-presence-loss", "--slot-presence-prior", "--slot-deep-supervision", "--slot-vis1",
          "--vis1-sidecar", "--slot-query-select", "--slot-dn-groups"):
    assert f not in argv, f"{f} is already in the canonical argv"
i = argv.index("--join3d")                      # the box head's own inputs sit together: after the 3-D join
argv = argv[:i + 2] + BOX + argv[i + 2:]
o["argv"] = argv
changes = list(o["changes_vs_refcv6"])
j = [k for k, c in enumerate(changes) if c.get("flag") == "--out"]
at = j[0] if j else len(changes)
box_changes = [
    {"flag": "--slot-presence-loss", "refcv6": None, "refcv7": ["focal"],
     "why": "SPEC_REFCV7 14 (A9 R1): sigmoid focal presence (alpha 0.25, gamma 2, weight 2.0, normalised per matched "
            "GT) + the focal matching cost, REPLACING refcv6's BCE with NO_OBJECT_W 0.1 -- on BOTH slot heads"},
    {"flag": "--slot-presence-prior", "refcv6": None, "refcv7": ["0.01"],
     "why": "SPEC_REFCV7 14 (A9 R1): the presence logit initialised at p = 0.01 (refcv6 built 0.05)"},
    {"flag": "--slot-deep-supervision", "refcv6": None, "refcv7": [],
     "why": "SPEC_REFCV7 14 (A9 R2): per-layer supervision through the shared norm + head, re-matched per layer "
            "(3 decoder layers, 0 parameters)"},
    {"flag": "--slot-vis1", "refcv6": None, "refcv7": [],
     "why": "SPEC_REFCV7 14 + 15.1 (A9 R3, A10): VIS-1 -- POSITIVE = vis_frac >= 0.30 AND >= 100 px; the rest of "
            "the visible-target filter is IGNORE (out of matching; no presence penalty within 2 m; DontCare at eval)"},
    {"flag": "--vis1-sidecar", "refcv6": None, "refcv7": [SIDECAR],
     "why": f"SPEC_REFCV7 14 (A9 R3): the precomputed VIS-1 sidecar, TRAIN 4,369 + EVAL 139 clips, sha256 {SIDECAR_SHA} "
            "(the box-head package's raw/thor/vis1_full_record.json; two controls vs the audit's banked rows, 0 "
            "mismatches); the dataset REFUSES a missing clip, frame or row"},
    {"flag": "--slot-query-select", "refcv6": None, "refcv7": ["learned_ref"],
     "why": "SPEC_REFCV7 19 + 19.1 (A14, A14.1): the BOX head's queries are learned reference points -- the full-model "
            "one-frame test PASSES (matched presence 0.873, anchor kept 1.000), the unanchored red arm and heatmap "
            "selection fail; +R6 (A10.1, --slot-dn-groups) is NOT adopted"},
    {"flag": None, "lever": "R4 (the code default, no flag)", "refcv6": 100, "refcv7": 300,
     "why": "SPEC_REFCV7 14 (A9 R4): N_QUERIES_DEFAULT = 300 on BOTH slot heads (agent_slots.py, one spelling)"},
]
o["changes_vs_refcv6"] = changes[:at] + box_changes + changes[at:]
assert "todo_box_head" in o
del o["todo_box_head"]
prep = list(o["launch_prep"])
prep.insert(2, f"{SIDECAR} := the box-head builder's VIS-1 sidecar /home/nvidia/bx_0252/full/"
               f"vis1_sidecar_refcv6b1_train4369_eval139.npz, placed by the Master Mind 2026-09-27 (sha256 "
               f"{SIDECAR_SHA}, read on Thor)")
o["launch_prep"] = prep
out = (json.dumps(o, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_bytes(out)
print(f"argv {len(argv)} tokens (+{len(BOX)}), changes {len(o['changes_vs_refcv6'])} (+{len(box_changes)}), "
      f"todo_box_head removed, launch_prep {len(prep)}")
