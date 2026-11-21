#!/usr/bin/env python3
"""T-B driver: run the REAL `taniteval/tools/refcv3_arm.py` with the lead mask installed.

⛔ ONE INSTRUMENT, NOT TWO. This file imports `refcv3_arm.py` from the roll tree and
replaces exactly ONE factory -- `make_eval_dataset_class` -- so the model rebuild, the
window grid, the four families, the bootstrap and the dump schema are the harness's own
and are bit-comparable with the banked refcv5-v2 dump (proved by `code/provgate.py`:
`os` reproduces the bank EXACTLY on ep000-001).

    python run_mask_roll.py --mask-mode lead|rand|zero --mask-seed 0 \
        -- <every refcv3_arm.py argument, verbatim>

The masker's summary is written next to the dump as `MASK_SUMMARY.json`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

REPO = os.environ.get("D3_REPO", r"C:\Users\Admin\refcv5cmp\repo")
for p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval"), REPO,
          os.path.join(REPO, "stack", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leadmask as LM  # noqa: E402

ARM_PATH = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")
LEAD_BLOCK = r"C:\Users\Admin\refcv5cmp\data\b1_eval_lead_block.npz"
AGENT_JOIN = r"C:\Users\Admin\tanitad-caches\b1-agent-join-20260906\b1eval_agents.jsonl.xz"
EXTRINSICS = r"C:\Users\Admin\refcv5v2_final\extrinsics141.json"


def _load_arm():
    spec = importlib.util.spec_from_file_location("refcv3_arm_real", ARM_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["refcv3_arm_real"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-mode", required=True, choices=("lead", "rand", "zero"))
    ap.add_argument("--mask-seed", type=int, default=0)
    ap.add_argument("--gap-max-m", type=float, default=30.0)
    ap.add_argument("--agent-margin-px", type=int, default=8)
    ap.add_argument("--episodes-dir", default=r"C:\Users\Admin\refcv5cmp\data\eval")
    a, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]

    arm = _load_arm()
    from tanitad.data.v2_dataset import load_or_build_manifest, stable_episode_id

    man = load_or_build_manifest(a.episodes_dir, verbose=False)
    clip_ids = [str(c) for c in man["clip_id"]]
    n_stack = int(man["n_stack"][0])
    sid_to_clip = {int(stable_episode_id(c)): c for c in clip_ids}
    keep = set(clip_ids)

    print(f"[mask] {len(keep)} clips, n_stack={n_stack}, mode={a.mask_mode}, "
          f"seed={a.mask_seed}, gap_max={a.gap_max_m} m", flush=True)
    agents = LM.load_agents(AGENT_JOIN, keep)
    astats = agents.pop("_stats")
    leads = LM.load_leads(LEAD_BLOCK, keep)
    proj = LM.Projector(EXTRINSICS)
    print(f"[mask] join {astats} · lead rows {len(leads)} · cams {len(proj.cams)}",
          flush=True)
    masker = LM.LeadMasker(a.mask_mode, agents, leads, proj, n_stack=n_stack,
                           gap_max_m=a.gap_max_m,
                           agent_margin_px=a.agent_margin_px, seed=a.mask_seed)

    # ---- the ONE replacement --------------------------------------------- #
    real_factory = arm.make_eval_dataset_class

    def patched_factory():
        Base = real_factory()

        class MaskedEvalV3Windows(Base):
            def _window_u8(self, i: int) -> dict:
                d = Base._window_u8(self, i)
                e_i, t = self.index[i]
                cid = sid_to_clip.get(int(self.episodes[e_i].episode_id))
                if cid is not None:
                    d["frames"] = masker.mask_window(cid, int(t), d["frames"])
                return d

        return MaskedEvalV3Windows

    arm.make_eval_dataset_class = patched_factory

    # ---- the dump dir, so the summary lands beside it --------------------- #
    dump_dir = None
    for i, x in enumerate(rest):
        if x == "--dump-dir":
            dump_dir = rest[i + 1]
    try:
        arm.main(rest)
    finally:
        s = masker.summary()
        s["mask_seed"] = a.mask_seed
        s["agent_join"] = AGENT_JOIN
        s["lead_block"] = LEAD_BLOCK
        s["extrinsics"] = EXTRINSICS
        s["agent_join_load"] = astats
        s["n_stack"] = n_stack
        txt = json.dumps(s, indent=1)
        print("[mask-summary] " + json.dumps(
            {k: v for k, v in s.items()
             if k not in ("class_heights_m", "agent_join_load")}), flush=True)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
            with open(os.path.join(dump_dir, "MASK_SUMMARY.json"), "w",
                      encoding="utf-8") as fh:
                fh.write(txt)


if __name__ == "__main__":
    main()
