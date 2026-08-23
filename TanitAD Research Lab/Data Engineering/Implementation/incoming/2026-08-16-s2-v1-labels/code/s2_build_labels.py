"""S2 v1 label build — Engine A recompute + geometry-primary labels + schema.

Executes S2_STRATEGIC_GAP.md §6 items 1-5 for the 801 already-labeled clips
(201 fused aug120 + 600 fused w120val), CPU-only, no model re-run:

  1. recompute Engine A from the bridged ego npz (t0_idx=80, the exact
     ph0_v2 call: engine_a_for_prompt(engine_a_summary(poses, 80))), strip
     `situations` (goal/situation disjointness) — persisted as JSONL;
  2. CROSS-CHECK the recompute against the prompt-recovered Engine A of all
     201 aug120 clips (parsed from the B4 prompts) — the recompute is only
     trusted for val600 because it reproduces the shipped geometry exactly;
  3. derive geometry-primary g_str/a_str via stack/scripts/s2_derive.py and
     emit `s2-strategic-v1` records via colab/s2_schema.py (validated,
     disjointness-asserted, ROUTE_TO refused by the validator);
  4. exclude the 4 triple-empty val records WITH REASONS (never silently);
  5. write clip_index.json — clip UUID <-> v2ep shard <-> legacy 16-bit
     episode_id <-> stable 63-bit id, with the collision census computed
     over the FULL 2400/600 corpus lists so ambiguous legacy ids are NAMED.

Counts are counts of RECORDS, never files (C18).
"""
import json
import os
import re
import sys
from collections import Counter

REPO = r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD"
SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
PULL = os.path.join(SP, "s2_pull")
EGO = os.path.join(SP, "s2_ego")
PKG = os.path.join(REPO, "TanitAD Research Hub", "Data Engineering",
                   "Implementation", "incoming", "2026-08-16-s2-v1-labels")
LABELS = os.path.join(PKG, "labels")
RAW = os.path.join(PKG, "raw")
os.makedirs(LABELS, exist_ok=True)
os.makedirs(RAW, exist_ok=True)

sys.path.insert(0, os.path.join(REPO, "stack"))
sys.path.insert(0, os.path.join(REPO, "stack", "scripts"))
sys.path.insert(0, os.path.join(REPO, "colab"))
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

import s2_derive                                     # noqa: E402
import s2_schema                                     # noqa: E402
from ph0_pilot import engine_a_for_prompt, engine_a_summary  # noqa: E402

try:
    from tanitad.data.v2_dataset import stable_episode_id
except ModuleNotFoundError:
    # v2_dataset drags torchvision (absent on the dev box). The id is pure
    # blake2b>>1 — same fallback train_p8_occupancy ships, parity with the
    # real function pinned by test_p8.py::test_episode_uid_matches_stable_…
    import hashlib

    def stable_episode_id(clip_id: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(clip_id.encode("utf-8"), digest_size=8).digest(),
            "big") >> 1

assert s2_derive.check_vocab_drift() == "checked"
assert s2_schema.check_v6_drift() == "checked"

T0_IDX = 80                       # int(round(8.0 * POSE_HZ)), ph0_v2.py:784
#: §2 / §4.6 — triple-empty val records (VLM+SAM3+Alpamayo all absent, ego
#: state null); their NONE_ABSTAIN was the default-of-absence, not a
#: judgement. Excluded from the S2 label set BY ID, with reasons.
TRIPLE_EMPTY_VAL = (
    "1d4dcb4e-5117-4e84-9eac-59690879c7d6",
    "a26a627a-caf4-4f23-a02c-9a4e558fc867",
    "b02c28ce-e2c7-4f37-86f6-9888d519fe43",
    "b0388541-b7de-465d-8411-998cf5881bee",
)


def load_fused(dirname: str) -> dict[str, dict]:
    out = {}
    d = os.path.join(PULL, dirname)
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and not f.startswith("_"):
            rec = json.load(open(os.path.join(d, f), encoding="utf-8"))
            out[rec["clip_id"]] = rec
    return out


def recompute_engine_a(npz_path: str) -> dict:
    poses = torch.as_tensor(np.load(npz_path)["poses"], dtype=torch.float32)
    ea = engine_a_for_prompt(engine_a_summary(poses, T0_IDX))
    ea.pop("situations", None)        # disjointness: never persisted here
    ea.pop("polyline_xy", None)
    return json.loads(json.dumps(ea))  # plain python types only


# --------------------------------------------------------------------------- #
# 1+2. Engine A recompute + cross-check against the prompt-recovered blocks    #
# --------------------------------------------------------------------------- #
def prompt_engine_a(v2rec: dict):
    for c in v2rec.get("_calls", []):
        if c.get("call") == "B4_symbols":
            m = re.search(r"ENGINE_A = (\{.*?\})\n", c.get("prompt") or "",
                          re.S)
            if m:
                try:
                    return json.loads(m.group(1))
                except Exception:                     # noqa: BLE001
                    return None
    return None


def crosscheck(ea: dict, pea: dict) -> list[str]:
    """Field-by-field recompute vs prompt-embedded block ((_fmt_engine_a
    flattens route/speed_profile and truncates events to [:3])."""
    diffs = []
    r, sp = ea.get("route", {}), ea.get("speed_profile", {})

    def close(a, b, tol=1e-4):
        if a is None or b is None:
            return a is None and b is None
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))

    for name, mine, theirs, tol in (
            ("route_token", r.get("token"), pea.get("route_token"), None),
            ("route_valid", bool(r.get("token_valid")),
             bool(pea.get("route_valid")), None),
            ("route_dist_m", r.get("dist_m"), pea.get("route_dist_m"), 1e-4),
            ("route_arc_m", r.get("arc_m"), pea.get("route_arc_m"), 1e-4),
            ("maneuver_dyaw_rad", r.get("maneuver_dyaw_rad"),
             pea.get("maneuver_dyaw_rad"), 1e-3),
            ("v_min_future_ms", sp.get("v_min_future_ms"),
             pea.get("v_min_future_ms"), 1e-3),
            ("v_max_future_ms", sp.get("v_max_future_ms"),
             pea.get("v_max_future_ms"), 1e-3),
            ("net_dv_ms", sp.get("net_dv_ms"), pea.get("net_dv_ms"), 1e-3),
            ("stops", bool(sp.get("stops")), bool(pea.get("stops")), None),
            ("peak_kappa_per_m", ea.get("peak_kappa_per_m"),
             pea.get("peak_kappa_per_m"), 1e-3)):
        ok = close(mine, theirs, tol) if tol else mine == theirs
        if not ok:
            diffs.append(f"{name}: mine={mine!r} prompt={theirs!r}")
    for name in ("lane_change_events", "speed_events"):
        if json.dumps((ea.get(name) or [])[:3]) != \
                json.dumps(pea.get(name) or []):
            diffs.append(f"{name}[:3] differ")
    return diffs


def main() -> int:
    fused_aug = load_fused("fused_aug120")
    fused_val = load_fused("fused_w120val")
    print(f"fused records: aug120 {len(fused_aug)} val {len(fused_val)}",
          flush=True)

    # prompt-recovered EA per aug120 clip (from the 353 batch v2 records)
    v2_by: dict[str, dict] = {}
    import glob as _g
    for p in sorted(_g.glob(os.path.join(PULL, "batch_*", "v2", "*.json"))):
        for r in json.load(open(p, encoding="utf-8")).get("clips", []):
            pea = prompt_engine_a(r)
            if pea is not None:
                v2_by[r["clip_id"]] = pea

    # ---- recompute + cross-check ---------------------------------------- #
    ea_all: dict[str, dict] = {}
    check = {"n_checked": 0, "n_match": 0, "mismatches": {}}
    for split, ids in (("aug120", sorted(fused_aug)),
                       ("w120val", sorted(fused_val))):
        with open(os.path.join(LABELS, f"engine_a_{split}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for i, cid in enumerate(ids):
                ea = recompute_engine_a(
                    os.path.join(EGO, split, f"{cid}.npz"))
                ea_all[cid] = ea
                fh.write(json.dumps({"clip_id": cid, "t0_idx": T0_IDX,
                                     "engine_a": ea}) + "\n")
                if split == "aug120" and cid in v2_by:
                    check["n_checked"] += 1
                    diffs = crosscheck(ea, v2_by[cid])
                    if diffs:
                        check["mismatches"][cid] = diffs
                    else:
                        check["n_match"] += 1
                if (i + 1) % 100 == 0:
                    print(f"[{split}] {i + 1}/{len(ids)}", flush=True)
    json.dump(check, open(os.path.join(RAW, "engineA_recompute_check.json"),
                          "w"), indent=1)
    print(f"CROSSCHECK {check['n_match']}/{check['n_checked']} exact "
          f"({len(check['mismatches'])} mismatched)", flush=True)

    # ---- 3+4. derive + emit --------------------------------------------- #
    censi: dict[str, dict] = {}
    excluded = []
    review_rows = []
    for split, fused_by in (("aug120", fused_aug), ("w120val", fused_val)):
        cnt_g, cnt_a, cnt_prov = Counter(), Counter(), Counter()
        agree_g = Counter()
        remapped = 0
        sources_seen = Counter()
        n_out = 0
        with open(os.path.join(LABELS, f"s2_labels_{split}.jsonl"), "w",
                  encoding="utf-8") as fh:
            for cid in sorted(fused_by):
                frec = fused_by[cid]
                if split == "w120val" and cid in TRIPLE_EMPTY_VAL:
                    excluded.append({
                        "clip_id": cid,
                        "reason": ("triple-empty record: VLM/SAM3/Alpamayo "
                                   "layers all absent, ego_state null — its "
                                   "fused NONE_ABSTAIN was the "
                                   "default-of-absence, not a judgement "
                                   "(S2_STRATEGIC_GAP §2/§4.6)"),
                        "engine_a_route": (ea_all[cid].get("route") or {})
                        .get("token"),
                        "note": ("Engine A geometry EXISTS for this clip "
                                 "(banked in engine_a_w120val.jsonl) — "
                                 "recoverable if the standing val600 "
                                 "re-fuse lands a real VLM layer"),
                    })
                    continue
                sym = ((frec.get("semantics") or {}).get("symbols")) or {}
                ea = ea_all[cid]
                g = s2_derive.derive_g_str(ea, sym)
                a = s2_derive.derive_a_str(ea, sym)
                rec = s2_schema.build_record(
                    cid, g, a,
                    provenance_notes={
                        "engine_a": "recomputed from bridged ego npz "
                                    "(refb_labels route_v3/latmaneuver/"
                                    "lonmode @ t0_idx=80)",
                        "vlm": "corroboration only — ph0-v2.2 "
                               "(vision+ego-past-prompt+engineA-prompt)",
                    })
                fh.write(json.dumps(rec) + "\n")
                n_out += 1
                cnt_g[g["token"]] += 1
                cnt_a[a["token"]] += 1
                cnt_prov[f"g:{g['provenance']}"] += 1
                cnt_prov[f"a:{a['provenance']}"] += 1
                for s in g["sources"]:
                    sources_seen[s] += 1
                for s in a["sources"]:
                    sources_seen[s] += 1
                agr = g["corroboration"].get("agrees")
                agree_g["agree" if agr else
                        ("disagree" if agr is False else "n/a")] += 1
                if g["corroboration"].get("remapped_from_route_to"):
                    remapped += 1
                if split == "aug120":
                    review_rows.append({
                        "clip_id": cid,
                        "vlm_goal": sym.get("goal_kind"),
                        "vlm_actions": [
                            f"{x.get('verb')}({x.get('direction')})"
                            for x in (sym.get("actions") or [])],
                        "route": (ea.get("route") or {}).get("token"),
                        "route_valid": (ea.get("route") or {})
                        .get("token_valid"),
                        "dyaw": (ea.get("route") or {})
                        .get("maneuver_dyaw_rad"),
                        "dist_m": (ea.get("route") or {}).get("dist_m"),
                        "stops": (ea.get("speed_profile") or {}).get("stops"),
                        "net_dv": (ea.get("speed_profile") or {})
                        .get("net_dv_ms"),
                        "g_str": g["token"],
                        "g_args": {s2_schema.GOAL_ARG_NAMES[i]: g["args"][i]
                                   for i in range(8) if g["arg_mask"][i]},
                        "a_str": a["token"],
                        "a_args": {s2_schema.GOAL_ARG_NAMES[i]: a["args"][i]
                                   for i in range(8) if a["arg_mask"][i]},
                        "remapped_route_to": bool(
                            g["corroboration"].get("remapped_from_route_to")),
                        "g_reason": g.get("reason"),
                        "scenario": frec.get("scenario_description"),
                    })
        censi[split] = {
            "n_records": n_out, "g_str": dict(cnt_g), "a_str": dict(cnt_a),
            "provenance": dict(cnt_prov),
            "g_vlm_agreement": dict(agree_g),
            "route_to_remapped": remapped,
            "sources": dict(sources_seen),
        }
        print(f"[{split}] {n_out} records  g_str={dict(cnt_g)}", flush=True)

    json.dump(excluded, open(os.path.join(
        LABELS, "s2_excluded_w120val.json"), "w"), indent=1)

    # ---- goal/situation disjointness, verified on the OUTPUT -------------- #
    # (a) per-record assert_disjoint already ran inside build_record;
    # (b) corpus-level re-scan of every emitted payload;
    # (c) the sources census must contain ONLY engine_a.* / vlm.* / declared
    #     defaults — anything else is a leak candidate and fails the build.
    allowed_src = re.compile(
        r"^(engine_a\.[a-z_0-9=.]+|vlm\.[a-z_]+|pi_default:.+|default:.+|"
        r"route_to_gate)$")
    bad_sources = [s for split in censi
                   for s in censi[split]["sources"]
                   if not allowed_src.match(s)]
    assert not bad_sources, f"unexpected label sources: {bad_sources}"
    n_scanned = 0
    for split in ("aug120", "w120val"):
        with open(os.path.join(LABELS, f"s2_labels_{split}.jsonl"),
                  encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                s2_schema.assert_disjoint(rec)
                errs = s2_schema.validate(rec)
                assert not errs, f"{rec['clip_id']}: {errs}"
                n_scanned += 1
    print(f"DISJOINT+VALID re-scan over {n_scanned} emitted records: PASS",
          flush=True)

    # ---- 5. clip_index.json ---------------------------------------------- #
    def legacy_id(cid: str) -> int:
        return int.from_bytes(cid.encode()[:4].ljust(4, b"\0"), "big")

    corpus_lists = {
        "bridged_w120train_2400": json.load(open(os.path.join(
            EGO, "bridged_w120train_2400__clips.json"))),
        "w120val_600": json.load(open(os.path.join(
            EGO, "w120val_600__clips.json"))),
    }
    # legacy-id collision census over the FULL corpus lists (the ambiguity a
    # legacy-id join would hit), not just the labeled subset
    coll = {}
    for name, lst in corpus_lists.items():
        by = Counter(legacy_id(c) for c in lst)
        coll[name] = {"n_clips": len(lst),
                      "n_legacy_ids": len(by),
                      "n_clips_in_colliding_ids":
                          sum(v for v in by.values() if v > 1)}
    idx = {"_doc": ("clip UUID -> shard + episode ids for the S2 join. "
                    "`episode_id_legacy` is the 16-bit-entropy id BAKED "
                    "INTO the v2ep payloads (first 4 chars; COLLIDES — see "
                    "_legacy_collisions; a lookup through it must refuse "
                    "ambiguous keys). `episode_id_stable` is "
                    "tanitad.data.v2_dataset.stable_episode_id (blake2b>>1, "
                    "collision-free), derived at LOAD time by the trainer."),
           "_t0_s": 8.0, "_valid_window_s": [-2.0, 2.0],
           "_legacy_collisions": coll, "clips": {}}
    for split, corpus in (("aug120", "bridged_w120train_2400"),
                          ("w120val", "w120val_600")):
        members = set(corpus_lists[corpus])
        for cid in sorted(load_fused(f"fused_{split}")):
            assert cid in members, f"{cid} not in {corpus} clips.json"
            idx["clips"][cid] = {
                "label_split": split,
                "corpus": corpus,
                "v2ep_file": f"{cid}.v2ep.pt",
                "ego_npz": f"{corpus}/ego/{cid}.npz",
                "episode_id_legacy": legacy_id(cid),
                "episode_id_stable": stable_episode_id(cid),
                "excluded": cid in TRIPLE_EMPTY_VAL,
            }
    json.dump(idx, open(os.path.join(LABELS, "clip_index.json"), "w"),
              indent=1)
    print(f"clip_index: {len(idx['clips'])} clips  collisions={coll}",
          flush=True)

    json.dump(censi, open(os.path.join(RAW, "build_censuses.json"), "w"),
              indent=1)
    json.dump(review_rows, open(os.path.join(RAW, "review_rows_aug120.json"),
                                "w"), indent=1)
    print("S2_BUILD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
