#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate Media/media_index.js from Media/media_index.json, and validate both.

The media browser (Media/tool/index.html) runs from file:// where fetch() of a
local .json is blocked, so the index is embedded as a .js global:

    window.MEDIA_INDEX = {...};   // exact content of media_index.json

Usage (from anywhere inside the repo):
    python Media/tool/build_index_js.py            # validate + (re)write media_index.js
    python Media/tool/build_index_js.py --check    # validate only (no write); nonzero exit on failure

Validation:
  * media_index.json parses and has the expected top-level shape
  * every record carries every schema field, ids are unique,
    type in {video,image}, eval_tier in {T0,T1,none}
  * every asset path EXISTS on disk (repo-relative)
  * with --check: media_index.js exists and byte-matches a regeneration
"""
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # .../Media/tool
MEDIA = os.path.dirname(HERE)                              # .../Media
REPO = os.path.dirname(MEDIA)                              # repo root
JSON_PATH = os.path.join(MEDIA, "media_index.json")
JS_PATH = os.path.join(MEDIA, "media_index.js")

SCHEMA = ["id", "path", "type", "title", "date", "topic", "model_arm", "eval_tier",
          "source_run", "caption", "tags", "size_bytes", "in_manifest"]
TIERS = {"T0", "T1", "none"}
TYPES = {"video", "image"}


def read_json_with_retry(path, tries=8):
    """G: (Drive) mounts flap with transient OSErrors — retry reads."""
    import time
    last = None
    for i in range(tries):
        try:
            with io.open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except OSError as e:
            last = e
            time.sleep(min(3 * (i + 1), 20))
    raise last


def render_js(index):
    body = json.dumps(index, indent=1, ensure_ascii=False)
    return ("// GENERATED FILE - do not edit by hand.\n"
            "// Rebuild with: python Media/tool/build_index_js.py  (source: Media/media_index.json)\n"
            "window.MEDIA_INDEX = " + body + ";\n")


def validate(index):
    errors = []
    assets = index.get("assets")
    if not isinstance(assets, list) or not assets:
        return ["media_index.json: 'assets' missing or empty"]
    if index.get("count") != len(assets):
        errors.append(f"count field ({index.get('count')}) != len(assets) ({len(assets)})")
    seen = set()
    for i, r in enumerate(assets):
        where = f"assets[{i}] ({r.get('id', '?')})"
        for k in SCHEMA:
            if k not in r:
                errors.append(f"{where}: missing field '{k}'")
        rid = r.get("id")
        if rid in seen:
            errors.append(f"{where}: duplicate id '{rid}'")
        seen.add(rid)
        if r.get("type") not in TYPES:
            errors.append(f"{where}: bad type {r.get('type')!r}")
        if r.get("eval_tier") not in TIERS:
            errors.append(f"{where}: bad eval_tier {r.get('eval_tier')!r}")
        if not r.get("caption"):
            errors.append(f"{where}: empty caption")
        p = r.get("path", "")
        if "\\" in p:
            errors.append(f"{where}: path must use forward slashes: {p}")
        full = os.path.join(REPO, p.replace("/", os.sep))
        if not os.path.isfile(full):
            errors.append(f"{where}: path does not exist on disk: {p}")
        if r.get("in_manifest"):
            sha = r.get("manifest_sha256") or ""
            if len(sha) != 64:
                errors.append(f"{where}: in_manifest record without a 64-hex manifest_sha256")
            if not r.get("campaign"):
                errors.append(f"{where}: in_manifest record without a campaign")
            if not p.startswith("Media/"):
                errors.append(f"{where}: in_manifest record should point at its Media/ media_path: {p}")
    # cross-check against MEDIA_MANIFEST.json: every manifest asset appears exactly once
    man_path = os.path.join(MEDIA, "MEDIA_MANIFEST.json")
    if os.path.isfile(man_path):
        try:
            man = read_json_with_retry(man_path)
            man_shas = [a["sha256"] for a in man.get("assets", [])]
            idx_shas = [r.get("manifest_sha256") for r in assets if r.get("in_manifest")]
            missing = set(man_shas) - set(idx_shas)
            extra = set(x for x in idx_shas if x) - set(man_shas)
            if missing:
                errors.append(f"{len(missing)} manifest asset(s) missing from the derived index (join by sha256)")
            if extra:
                errors.append(f"{len(extra)} derived video record(s) carry a sha256 not in MEDIA_MANIFEST.json")
            dupes = [s for s, n in __import__('collections').Counter(idx_shas).items() if n > 1]
            if dupes:
                errors.append(f"{len(dupes)} manifest sha256(s) appear more than once in the derived index")
        except OSError as e:
            errors.append(f"could not read MEDIA_MANIFEST.json for the cross-check: {e}")
    return errors


def main(argv):
    check_only = "--check" in argv
    index = read_json_with_retry(JSON_PATH)
    errors = validate(index)
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} problem(s):")
        for e in errors[:60]:
            print("  -", e)
        if len(errors) > 60:
            print(f"  ... +{len(errors) - 60} more")
        return 1
    js = render_js(index)
    if check_only:
        if not os.path.isfile(JS_PATH):
            print("CHECK FAILED: media_index.js does not exist (run without --check to create it)")
            return 1
        with io.open(JS_PATH, "r", encoding="utf-8") as f:
            current = f.read()
        if current != js:
            print("CHECK FAILED: media_index.js is STALE - rerun: python Media/tool/build_index_js.py")
            return 1
        print(f"OK: {len(index['assets'])} assets, every path exists, media_index.js in sync.")
        return 0
    with io.open(JS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(js)
    print(f"OK: wrote media_index.js ({len(js):,} bytes) - {len(index['assets'])} assets, every path verified on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
