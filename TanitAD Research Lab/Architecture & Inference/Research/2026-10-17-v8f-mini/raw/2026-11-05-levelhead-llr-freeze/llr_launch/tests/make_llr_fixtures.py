"""Generate tests/fixtures/llr_real_records.json: the REAL G-LIVE records the shipped trainer writes for the lead head, for BOTH wirings, on the dev box (CPU, the real C0 init checkpoint, a synthetic world and a synthetic lead file).
The numbers of the step-0 record (and so the printed W_lead-for-a-tenth) are meaningless (synthetic batch); what the fixtures carry is the KEY SET, the value types, the exactly-zero controls and the built-lever keys exactly as the shipped code writes them.
usage: python -I tests/make_llr_fixtures.py [<patched tree root>]      (default F:/TanitAD-scratch/motion_c/lead_build/new; its model/trainer/data sha256 must equal the package's stack_ship pins)
The test suite uses the fixtures to build gate stubs: the REAL gate selfcheck of arm CL (level head on) + RAPR-P01's recovery blocks + these lead records."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
TREE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("F:/TanitAD-scratch/motion_c/lead_build/new")
for rel in ("stack/scripts/v8f_mini_train.py", "stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py"):
    a, b = (TREE / rel).read_bytes(), (PKG / "stack_ship" / rel).read_bytes()
    assert a == b, f"{rel}: the patched tree is not the package's shipped file"
for p in (TREE / "stack", TREE / "taniteval", TREE / "stack" / "scripts", TREE / "stack" / "tests"):
    sys.path.insert(0, str(p))
import torch  # noqa: E402
import test_v8f_lead_head as T  # noqa: E402
from tanitad.data import v8f_mini_data as D  # noqa: E402
from tanitad.models import v8f_mini as M  # noqa: E402
import v8f_mini_train as TR  # noqa: E402

KEYS = ("level_head", "w_level", "level_token_zero_init", "level_vision_only_source", "lead_head", "w_lead", "lead_vision_only_source", "lead_token_zero_init", "aux_cond", "aux_vision_only_source", "aux_token_dim")
ck = torch.load(T.C0_INIT, map_location="cpu", weights_only=False)
cfg_d = {**ck["cfg"]}
out = {"made_by": "tests/make_llr_fixtures.py on the shipped trainer (sha256 equal to stack_ship)", "wirings": {}}
with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    w = D.make_synthetic_world(root, n_train=6, n_eval=2, T=60, seed=0)
    npz = T.write_lead_npz(root / "lead_train.npz", w["names"]["train"])
    world = {"root": root, "npz": npz, **w}
    b, J = T.make_batch(world, hmax=cfg_d["hmax"])
    LL = T.LLof(world, J)
    mu, sd = M.cons_stats(J)
    bb = T.with_both(b, LL)
    for wiring in ("add", "token"):
        torch.manual_seed(0)
        on = M.V8FMini(M.V8FMiniConfig(**{**cfg_d, "level_head": "on", "w_level": 0.2027, "lead_head": "on", "w_lead": 1.0, "aux_cond": wiring}), M.P10Prior.synthetic(0), mu, sd)
        rec = TR.load_init_ckpt(str(T.C0_INIT), on, on.cfg)
        on.train()
        lv = on.built_levers()
        ldp = TR.lead_head_probe(on, bb, on.cfg)
        lvp = TR.level_head_probe(on, bb, on.cfg)
        out["wirings"][wiring] = {"levers": {k: lv[k] for k in KEYS if k in lv}, "lead_head_probe": ldp, "level_head_probe": lvp, "fresh_keys": sorted(rec["fresh_keys"]), "declared_new_keys": sorted(rec["declared_new_keys"]),
                                  "lead_labels": {"path": "D:\\tanitad\\data\\v9\\lead_train4572.npz", "sha256": hashlib.sha256(Path(npz).read_bytes()).hexdigest(), **{k: LL.get(k) if isinstance(LL, dict) else getattr(LL, k) for k in ("n_rows", "n_has", "n_near", "n_clips", "n_clips_with_labels")}}}
        print(wiring, "fresh", len(rec["fresh_keys"]), "levers", out["wirings"][wiring]["levers"])
(HERE / "fixtures").mkdir(exist_ok=True)
(HERE / "fixtures" / "llr_real_records.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8", newline="\n")
print("FIXTURES-DONE")
