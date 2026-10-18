"""Write template_pins.json: sha256 of the source segment of every GENERIC object of the SR-TTC R1 template (`srttc_r1.py`) that epr1.py must keep byte-identical.
    python -I template_pins_make.py <path to srttc_r1.py> <out json>
`test_epr1.py::TestTemplatePins` re-derives the segments from epr1.py and compares; the pins are what 'mirrors SR-TTC R1 exactly' means for the generic machinery."""
import ast, hashlib, json, sys
PINNED = ("_cmp", "_under", "PathGuardError", "BankError", "PathGuard", "_io_read_bytes", "_io_sha256", "_io_npz", "_io_listdir", "_io_exists", "_io_stat", "_io_torch_load",
          "_io_makedirs", "_io_write_bytes", "sha256_bytes", "canon_json", "fold_of_rank", "log_key", "load_ranked_logs", "select_logs", "load_manifest", "aggregate_np", "top8_np",
          "picks_np", "load_p2_row", "boot_draws", "paired_delta", "lc_call")
def segments(src: str) -> dict:
    tree, out = ast.parse(src), {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in PINNED:
            out[n.name] = hashlib.sha256(ast.get_source_segment(src, n).encode("utf-8")).hexdigest()
    return out
if __name__ == "__main__":
    raw = open(sys.argv[1], "rb").read()
    segs = segments(raw.decode("utf-8"))
    missing = [p for p in PINNED if p not in segs]
    assert not missing, missing
    json.dump({"template": "raw/2026-10-18-sel-rank/code/srttc_r1.py", "template_file_sha256": hashlib.sha256(raw).hexdigest(), "pinned": segs}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
    print(len(segs), "pins; template sha256", hashlib.sha256(raw).hexdigest()[:16])
