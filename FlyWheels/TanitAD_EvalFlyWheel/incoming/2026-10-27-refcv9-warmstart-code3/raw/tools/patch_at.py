"""assert_map_only_finetune.py (LF working copy), round 3: generalised to `--r9-train-only`'s sets. The legacy call
(`--src --dst`, no `--sets`) is unchanged, key for key."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
E = []


def edit(old, new):
    E.append((old, new))


edit('''⛔ INDEPENDENT of the trainer by construction: the map-only module paths are a LITERAL here (the PI's three, the same
literal the trainer's ``map_only_ft.MAP_ONLY_MODULES`` carries -- a test holds the two equal), and bit equality is a
comparison of raw BYTES (a -0.0 differs from a +0.0), not ``torch.equal``.
"""''', '''⛔ INDEPENDENT of the trainer by construction: the map-only module paths are a LITERAL here (the PI's three, the same
literal the trainer's ``map_only_ft.MAP_ONLY_MODULES`` carries -- a test holds the two equal), and bit equality is a
comparison of raw BYTES (a -0.0 differs from a +0.0), not ``torch.equal``.

THE r9 FORM (refcv9 WARM START, ``--r9-train-only <sets>``; MM 2026-10-27):

    python stack/scripts/assert_map_only_finetune.py --src ckpt_ema_48852.pt --dst <run>/ckpt.pt \\
        --sets planner,tactical,map [--also-frozen core.decoder.offset_head ...] [--require-each-module-moved]

  * ``--sets``: the TRAINED modules are the union of the named sets -- :data:`TRAIN_ONLY_SETS`, a LITERAL here (a test
    holds it equal to ``map_only_ft.TRAIN_ONLY_SETS``). Every tensor OUTSIDE them -- the PERCEPTION -- must be
    bit-identical in every view, exactly as the map-only form demands of the planner.
  * ``--also-frozen PREFIX`` (repeatable): a subtree INSIDE a set that must ALSO be bit-identical (the build's
    declared grad-unreachable subtrees: the launch gate's ``map_only["declared_frozen"]`` literal). It is checked as
    frozen and named in the output.
  * the output NAMES the changed tensors PER SET and per module (``per_set``); ``--require-each-module-moved`` fails
    a view in which some module of the sets has NO changed tensor (the tool alone passes a run that trained nothing).
"""''')
edit('''MAP_ONLY_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
EMA_SCHEMA = "weight_ema/1"


def is_map_only(key: str) -> bool:
    return any(key == p or key.startswith(p + ".") for p in MAP_ONLY_MODULES)
''', '''MAP_ONLY_MODULES = ("_map_hires.refine", "_map_hires.near", "_map_hires.near_refine")
#: ``--r9-train-only``'s sets (refcv9 warm start), in their canonical order -- LITERALS (the trainer's
#: ``map_only_ft.TRAIN_ONLY_SETS`` is held equal by a test, never read)
TRAIN_ONLY_SETS = {
    "planner": ("core.decoder",),
    "tactical": ("phi_tac", "nav_to_tac", "lat_head_tac", "lon_head_tac", "tac_goal_head", "tac_latent_proj",
                 "tac_decoder_v6", "tac_behaviour_gate_v6", "core.tactical_trunk", "core.lat_head", "core.lon_head"),
    "map": MAP_ONLY_MODULES}
EMA_SCHEMA = "weight_ema/1"


def _under(key: str, prefixes) -> bool:
    return any(key == p or key.startswith(p + ".") for p in prefixes)


def is_map_only(key: str) -> bool:
    return _under(key, MAP_ONLY_MODULES)


def parse_sets(spec: str) -> tuple:
    """``"planner,map"`` -> ``("planner", "map")`` in the canonical order; refuses an empty, unknown or repeated name."""
    parts = [x.strip() for x in str(spec).split(",")]
    bad = [x for x in parts if x not in TRAIN_ONLY_SETS]
    if not parts or bad or len(set(parts)) != len(parts):
        raise ValueError(f"--sets {spec!r}: the sets are {list(TRAIN_ONLY_SETS)}, each named once (bad: {bad})")
    return tuple(x for x in TRAIN_ONLY_SETS if x in parts)
''')
edit('''def compare(ref: dict, got: dict) -> dict:
    """One view against the reference -> {"ok", "missing", "extra", "non_map_changed", "map_changed", ...}."""
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    non_map, map_changed, n_non_map, n_map = [], [], 0, 0
    for k in sorted(set(ref) & set(got)):
        a, b = ref[k], got[k]
        if is_map_only(k):''', '''def compare(ref: dict, got: dict, sets: tuple | None = None, also_frozen: tuple = (),
            require_moved: bool = False) -> dict:
    """One view against the reference -> {"ok", "missing", "extra", "non_map_changed", "map_changed", ...}. With
    ``sets`` the trained modules are the sets' union (minus ``also_frozen``), every other key is FROZEN (the
    perception), and ``per_set`` names the changed tensors of each set and module."""
    mods = MAP_ONLY_MODULES if sets is None else tuple(m for s_ in sets for m in TRAIN_ONLY_SETS[s_])
    also = tuple(also_frozen)

    def trained(k):
        return _under(k, mods) and not _under(k, also)
    missing = sorted(set(ref) - set(got))
    extra = sorted(set(got) - set(ref))
    non_map, map_changed, n_non_map, n_map = [], [], 0, 0
    for k in sorted(set(ref) & set(got)):
        a, b = ref[k], got[k]
        if trained(k):''')
edit('''    return {"ok": not (missing or extra or non_map), "n_non_map_checked": n_non_map, "n_map_only": n_map,
            "missing": missing[:20], "n_missing": len(missing), "extra": extra[:20], "n_extra": len(extra),
            "non_map_changed": non_map[:50], "n_non_map_changed": len(non_map),
            "map_changed": map_changed, "n_map_changed": len(map_changed)}''', '''    res = {"ok": not (missing or extra or non_map), "n_non_map_checked": n_non_map, "n_map_only": n_map,
           "missing": missing[:20], "n_missing": len(missing), "extra": extra[:20], "n_extra": len(extra),
           "non_map_changed": non_map[:50], "n_non_map_changed": len(non_map),
           "map_changed": map_changed, "n_map_changed": len(map_changed)}
    if sets is None:
        return res
    keys = sorted(set(ref) & set(got))
    changed = {r["key"] for r in map_changed}
    per_set, unmoved = {}, []
    for s_ in sets:
        pm = {}
        for m in TRAIN_ONLY_SETS[s_]:
            km = [k for k in keys if _under(k, (m,)) and trained(k)]
            pm[m] = {"n_tensors": len(km), "n_changed": sum(1 for k in km if k in changed)}
            if not pm[m]["n_changed"]:
                unmoved.append(m)
        per_set[s_] = {"modules": list(TRAIN_ONLY_SETS[s_]), "per_module": pm,
                       "n_tensors": sum(v["n_tensors"] for v in pm.values()),
                       "changed": [r["key"] for r in map_changed if _under(r["key"], TRAIN_ONLY_SETS[s_])],
                       "n_changed": sum(v["n_changed"] for v in pm.values())}
    also_rows = {a: sum(1 for k in keys if _under(k, (a,))) for a in also}
    res.update({"sets": list(sets), "per_set": per_set, "also_frozen": also_rows,
                "n_frozen_checked": n_non_map, "n_frozen_changed": len(non_map), "frozen_changed": non_map[:50],
                "modules_unmoved": unmoved})
    if require_moved and unmoved:
        res["ok"] = False
    return res''')
edit('''def run(src: str, dst: str, src_view: str | None = None, dst_views=None) -> dict:
    sv, srec = views(src)''', '''def run(src: str, dst: str, src_view: str | None = None, dst_views=None, sets: tuple | None = None,
        also_frozen: tuple = (), require_moved: bool = False) -> dict:
    sv, srec = views(src)''')
edit('''    out = {"tool": "assert_map_only_finetune.py", "map_only_modules": list(MAP_ONLY_MODULES),
           "src": srec, "src_view": src_view, "dst": drec, "views": {}}''', '''    out = {"tool": "assert_map_only_finetune.py", "map_only_modules": list(MAP_ONLY_MODULES),
           "src": srec, "src_view": src_view, "dst": drec, "views": {}}
    if sets is not None:
        out.update({"sets": list(sets), "trained_modules": [m for s_ in sets for m in TRAIN_ONLY_SETS[s_]],
                    "also_frozen": list(also_frozen), "require_each_module_moved": bool(require_moved)})''')
edit('''        r = compare(sv[src_view], dv[name])
        out["views"][name] = r''', '''        r = (compare(sv[src_view], dv[name]) if sets is None else
             compare(sv[src_view], dv[name], sets, tuple(also_frozen), require_moved))
        out["views"][name] = r''')
edit('''        for row in r["map_changed"]:
            print(f"    map changed: {row['key']} {row['shape']} max|d|={row.get('max_abs_diff')} "
                  f"rel={row.get('rel_norm_change')}")
    print(f"[assert-map-only] {'PASS' if res['ok'] else 'FAIL'}")''', '''        if "per_set" not in r:
            for row in r["map_changed"]:
                print(f"    map changed: {row['key']} {row['shape']} max|d|={row.get('max_abs_diff')} "
                      f"rel={row.get('rel_norm_change')}")
            continue
        print(f"    perception (every tensor outside the sets {r['sets']}) + also-frozen {r['also_frozen']}: "
              f"{r['n_frozen_checked']} checked, {r['n_frozen_changed']} changed")
        for s_, e in r["per_set"].items():
            print(f"    set `{s_}`: {e['n_changed']} of {e['n_tensors']} tensors changed; per module "
                  f"{ {m: v['n_changed'] for m, v in e['per_module'].items()} }")
            for k in e["changed"]:
                print(f"      {s_} changed: {k}")
        if r["modules_unmoved"]:
            print(f"    {'FAIL' if res.get('require_each_module_moved') else 'NOTE'}: modules with NO changed tensor: "
                  f"{r['modules_unmoved']}")
    print(f"[assert-map-only] {'PASS' if res['ok'] else 'FAIL'}")''')
edit('''    ap.add_argument("--json", default=None, help="also write the full result here")
    a = ap.parse_args(argv)
    try:
        res = run(a.src, a.dst, a.src_view, a.dst_view)''', '''    ap.add_argument("--json", default=None, help="also write the full result here")
    ap.add_argument("--sets", default=None, metavar="SET[,SET]",
                    help="--r9-train-only's sets (planner / tactical / map): the trained modules are their union and "
                         "every other tensor (the perception) must be bit-identical. Absent: the map-only form")
    ap.add_argument("--also-frozen", action="append", default=None, metavar="PREFIX",
                    help="with --sets: a subtree INSIDE a set that must ALSO be bit-identical (declared frozen)")
    ap.add_argument("--require-each-module-moved", action="store_true",
                    help="with --sets: FAIL a view in which some module of the sets has no changed tensor")
    a = ap.parse_args(argv)
    if a.sets is None and (a.also_frozen or a.require_each_module_moved):
        print("[assert-map-only] ERROR: --also-frozen / --require-each-module-moved need --sets", file=sys.stderr)
        return 2
    try:
        sets = None if a.sets is None else parse_sets(a.sets)
        res = run(a.src, a.dst, a.src_view, a.dst_view, sets, tuple(a.also_frozen or ()),
                  bool(a.require_each_module_moved))''')

for old, new in E:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor matched {n} times:\n{old[:400]}")
    s = s.replace(old, new)
p.write_text(s, encoding="utf-8", newline="\n")
print("applied", len(E), "edits")
