"""Write the CANONICAL refcv7 launch argv (the file the launch gate binds and sup_refcv7.sh runs).

    python make_refcv7_argv.py <refcv6-r101-s0 argv.json> <out .argv.json>

Built FROM refcv6-r101-s0's own argv (every carried flag is refcv6's, byte for byte), with each
change stated and sourced. The file is a JSON object; the gate and the supervisor read `argv`
(`launch_gate.load_argv_file`, `sup_refcv7.sh`). `todo_box_head` and `launch_prep` are NOT argv:
they list what must happen before this file can PASS the gate.
"""
import json
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
r6 = json.loads(src.read_text(encoding="utf-8"))
D = "/home/nvidia/data"
TAU = "0.18063741505146028"          # the banked record's `tau`, verbatim (sha256 10ca19db...)


def pairs(argv):
    out_, i = [], 0
    while i < len(argv):
        f, i = argv[i], i + 1
        vals = []
        while i < len(argv) and not argv[i].startswith("--"):
            vals.append(argv[i])
            i += 1
        out_.append((f, vals))
    return out_


p = pairs(r6)
flags = [f for f, _ in p]
assert len(flags) == len(set(flags)), "refcv6 argv repeats a flag"
changes = []


def set_(flag, vals, why):
    global p
    for k, (f, v) in enumerate(p):
        if f == flag:
            changes.append({"flag": flag, "refcv6": v, "refcv7": vals, "why": why})
            p[k] = (flag, vals)
            return
    changes.append({"flag": flag, "refcv6": None, "refcv7": vals, "why": why})
    p.insert(next(k for k, (f, _) in enumerate(p) if f == "--opt"), (flag, vals))


# ---- NEW-1 (SPEC 1, A5 sec. 10): the plan is a residual on the ha0_ext_pose prior ---------------
assert ("--ego-history", []) in p, "refcv6 argv has no --ego-history (A5 needs it)"
set_("--residual-prior", ["ha0_ext_pose"], "SPEC_REFCV7 10 (A5): past poses only; needs --ego-history")
# ---- FIX-4 / A2 (sec. 7): all three selection mechanisms ON, tau from the BANKED file ----------
set_("--graft-tac8-prior", [], "SPEC_REFCV7 7 (A2, PI ruling E1)")
set_("--graft-nav-compliance", [], "SPEC_REFCV7 7 (A2, PI ruling E1)")
set_("--nav-compliance-tau-rad", [TAU], "the banked TRAIN-split tau (ab436ee raw/nav_compliance_tau_train.json)")
set_("--nav-compliance-tau-file", [f"{D}/refcv7/nav_compliance_tau_train.json"],
     "fixes batch 2 (b4a59b9): the trainer reads tau from the file and stamps {path, sha256, tau}")
set_("--speed-ceiling-filter", [], "SPEC_REFCV7 7 (A2): inference-only max-speed ceiling")
# ---- NEW-2 (A3 sec. 8, A6 sec. 11, A7 sec. 12, A8 sec. 13): the map at 10 cm, one lift --------
set_("--w-map", ["0"], "SPEC_REFCV7 11.1 (A6): the 0.5 m map head and loss are REMOVED")
set_("--map-gt-root", [f"{D}/sam3_gt_v3"], "SPEC_REFCV7 11.2/12 item 3: the /3 export at 100 x +-30 m")
set_("--map-hires", ["on"], "SPEC_REFCV7 8.1 (A3): the 10 cm map is REQUIRED")
set_("--w-map-hires", ["1.0"], "NEW-2 BUILD.md sec. 6: inherits refcv6's map loss budget (0.984/1.048 measured)")
set_("--map-hires-class-weights", [f"{D}/refcv7/map_hires_class_weights_train_100x30.json"],
     "SPEC_REFCV7 13 item 1 (A8): sqrt_mf, TRAIN, at the 100 x +-30 m extent")
set_("--map-hires-decision-rule", ["prior_corrected"], "SPEC_REFCV7 9 item 1 (A4)")
set_("--map-hires-x-max-m", ["100"], "SPEC_REFCV7 12 (A7)")
set_("--map-hires-y-half-m", ["30"], "SPEC_REFCV7 12 (A7)")
set_("--map-hires-grad-ckpt", ["on"], "SPEC_REFCV7 12 item 4 (A7)")
set_("--bev-source", ["map_hires_pool"], "SPEC_REFCV7 11.1 (A6 option c): every consumer reads the pooled BEV")
set_("--bev-planner-crop-m", ["60", "16"], "SPEC_REFCV7 12 item 2 (A7): planner grid unchanged")
# ---- the run itself ---------------------------------------------------------------------------
set_("--out", ["/home/nvidia/refcv7_run/runs/refcv7-r101-s0"], "the refcv7 run dir")

argv = [t for f, v in p for t in (f, *v)]
rec = {
    "schema": "tanitad.launch_argv/1",
    "arm": "refcv7-r101-s0",
    "what": "THE refcv7 launch argv. The launch gate binds `argv` (sha256 of the ordered compact "
            "JSON array); sup_refcv7.sh runs exactly `argv` through `launch_gate.py exec`. Any "
            "edit is a NEW gate run and a new token.",
    "derived_from": {"file": "refcv6-r101-s0 argv (config.json argv of the run the PI stopped)",
                     "n_tokens": len(r6)},
    "changes_vs_refcv6": changes,
    "argv": argv,
    "todo_box_head": {
        "status": "NOT LANDED -- the box-head builder (a2e8b42841442809f, package "
                  "…/2026-09-27-refcv7-box-head/) owns the flag names; they are ADDED here when it "
                  "lands, and the gate REFUSES this argv for refcv7 until they are (G-BOX-OVERFIT, "
                  "G-LIVE-PRES).",
        "levers": [
            {"id": "R1", "what": "sigmoid focal presence loss (alpha 0.25, gamma 2, weight 2.0, "
                                 "normalised per matched GT) + focal matching cost + prior 0.01, "
                                 "REPLACING the BCE with NO_OBJECT_W 0.1", "flag": "TBD"},
            {"id": "R2", "what": "per-layer supervision, re-matched per layer (3 layers)",
             "flag": "TBD"},
            {"id": "R3", "what": "VIS-1 IGNORE semantics: POSITIVE = vis_frac >= 0.30 AND >= 100 px; "
                                 "every other real GT object IGNORE; train + eval sidecars (sha256 "
                                 "in config.json)", "flag": "TBD (+ the two sidecar paths)"},
            {"id": "R4", "what": "300 queries on BOTH slot heads", "flag": "TBD"},
            {"id": "R6 (A10.1, conditional)", "what": "denoising queries `--slot-dn-groups 5` -- "
             "only if the Master Mind decides +R6 on the early G-BOX-OVERFIT verdicts",
             "flag": "--slot-dn-groups 5 (if decided)"},
        ],
        "gate": "PROFILES['refcv7']['open_items'] BOX-HEAD -- the gate cannot PASS while it is open",
    },
    "todo_map_lift": {
        "status": "OPEN -- the EARLY (non-binding) G-MAP-OVERFIT MAIN FAILED at step 1,000: lane "
                  "0.417 and non-drivable edge 0.076 vs the 0.50 bar; the other six classes pass "
                  "(INHERITED, Master Mind 2026-09-27). The prereg's sec. 9 lever order applies once "
                  "R2/C1-C3 validate the harness, and the next lever is the LIFT, so the map path "
                  "changes once more before launch. One or two lift flags are ADDED here (and as "
                  "required values in the gate profile) when NEW-2's lift change lands.",
        "levers": [
            {"id": "LIFT", "what": "the prereg sec. 9 LIFT lever: Z = 0 plane / 0.1 m near lift / "
                                   "stride 4", "flag": "TBD (one or two flags)"},
        ],
        "gate": "PROFILES['refcv7']['open_items'] MAP-LIFT -- the gate cannot PASS while it is open; "
                "the BINDING G-MAP-OVERFIT run is made on the final map path",
    },
    "launch_prep": [
        f"{D}/refcv7/nav_compliance_tau_train.json := the banked repo file TanitAD Research Lab/"
        "Architecture & Inference/Research/2026-09-26-declared-vs-built/raw/"
        "nav_compliance_tau_train.json (sha256 10ca19db42ed0a49083978a4cb6d0914664d28bc3231ed7058a01d5161e938ca)",
        f"{D}/refcv7/map_hires_class_weights_train_100x30.json := compute_map_class_weights.py "
        f"--v2-cache {D}/refcv6-b1-416x1024-train --gt-root {D}/sam3_gt_v3 --split train (sqrt_mf, "
        "pre_registered; NEW-2 BUILD.md sec. 10 step 1)",
        "G-MAP-OVERFIT and G-BOX-OVERFIT: each BINDING harness run wrapped by "
        "`stack/scripts/closure_run.py --binding --out <closure.json> --result <PASS json> "
        "--data-root <data dir> ... -- <harness> ... --launch-argv-sha256 <sha256 of THIS argv>` "
        "(SPEC_REFCV7 A11); the gate takes the pair (--map-overfit-record + --map-overfit-closure, "
        "--box-overfit-record + --box-overfit-closure) and recomputes every closure blob on the "
        "launch tree",
        "the refcv7 Training Watch: DECLARED in PROFILES['refcv7']['map_hires']['watch_builder'] "
        "(Master Mind 2026-09-27); the gate builds it from the smoke's metrics.jsonl",
    ],
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(f"{len(argv)} argv tokens, {len(changes)} changes vs refcv6 -> {out}")
for c in changes:
    print(f"  {c['flag']:28s} {c['refcv6']!s:40s} -> {c['refcv7']}")
