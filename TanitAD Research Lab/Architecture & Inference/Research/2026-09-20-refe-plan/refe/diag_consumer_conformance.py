"""Do the MODEL, the TARGET BANK, the TRAINER and the PLANNER still agree on the interface?

⛔ THE STRUCTURAL HOLE THIS CLOSES (Review 5, §6). Every other instrument in this package builds its
inputs FROM the config it is testing -- `randn(B, cfg.n_cameras, 3, cfg.img_h, cfg.img_w)`,
`randn(B, cfg.ego_dim)`. That makes each of them internally consistent and structurally incapable
of noticing that a CONSUMER disagrees with the config. It is the same shape as a check whose
expected value is an expression over the code under test: it measures determinism, not correctness.

MEASURED, twice in one evening, while every guard stayed green:
  * `ego_dim` moved 8 -> 9 in `model.py` and the 4,146 banked rows, `train.py` and `planner.py`
    were all left behind. The package's own control died with
    `mat1 and mat2 shapes cannot be multiplied (2x44 and 45x256)` -- a matmul error three frames
    deep that names neither the bank nor the field.
  * `n_cameras` moved 1 -> 4 and `build_targets.index_images`, `train.FrameStore` and
    `planner.FrameResolver` stayed single-camera. The builder crashed on its own SUCCESS path.

⭐ THE RULE THIS FILE FOLLOWS: **an expectation here is a LITERAL, or it is read from a DIFFERENT
consumer -- never from `REFeConfig`.** Where a real artifact exists (a banked target file) it is
read from disk. Where the consumer is code, its shape is recovered from the SOURCE by AST, not by
calling it with config-derived inputs.

Arms (each with a mutation that must turn it RED -- run with `--self-test`):
  1 the bank's ego width          == the model's ego_dim
  2 the bank's camera list        == the model's camera tuple, in order
  3 the bank's image list length  == the model's n_cameras
  4 the trainer's CAMERAS tuple   == the model's camera tuple
  5 the planner's ego literal     == the model's ego_dim          (by AST, not by import)
  6 the bank's goal width         >= the model's goal input width
  7 CONTROL: a bank row really was read (a count of zero is not a pass)

Usage:
  python diag_consumer_conformance.py [--targets <dir-or-jsonl>] [--self-test]
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def planner_ego_width(path: Path) -> int | None:
    """Count the elements of `ego_vec = torch.tensor([[ ... ]])` in planner.py, from SOURCE.

    Deliberately NOT by importing and calling: the planner needs nuplan, a checkpoint and a live
    EgoState, and any stub we built would be shaped by our own assumption about the width -- which
    is the assumption under test. The literal in the source is the thing the planner will actually
    build at runtime, and it is independent of `REFeConfig`.
    """
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "ego_vec" for t in node.targets):
            continue
        for sub in ast.walk(node.value):
            if isinstance(sub, ast.List) and sub.elts and isinstance(sub.elts[0], ast.List):
                return len(sub.elts[0].elts)      # the inner row of the [[ ... ]] batch literal
    return None


def module_tuple(path: Path, name: str) -> tuple | None:
    """Read a module-level tuple/list constant out of SOURCE without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return tuple(ast.literal_eval(node.value))
            except Exception:
                return None
    return None


def bank_rows(targets: str) -> list[tuple[dict, str]]:
    """The first row of EVERY target file the TRAINER would load, and where each came from.

    ⛔⛔ THIS READ `**/*.jsonl` AND RETURNED THE FIRST ROW OF THE FIRST FILE UNTIL 2026-09-23 --
    and in the assembled training directory the first file alphabetically is
    `scorer_targets_rank1.jsonl`, a SCORER row with no ego, cameras or goal. The gate reported
    CONSUMERS_DIVERGED on a correct bank while its "a real bank row was read" control PASSED,
    because it had read a row -- just not one the trainer ever consumes. The consumer's glob is
    `targets_rank*.jsonl` (train.TargetBank), so that is the gate's glob; and EVERY rank file is
    checked, because rank 1 comes from a different producer (augment_search) than rank 0.
    """
    if os.path.isdir(targets):
        cands = sorted(glob.glob(os.path.join(targets, "targets_rank*.jsonl")))
        if not cands:   # a sharded bank keeps its rank files one level down
            cands = sorted(glob.glob(os.path.join(targets, "**", "targets_rank*.jsonl"),
                                     recursive=True))
    else:
        cands = [targets] if targets else []
    out = []
    for p in cands:
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        out.append((json.loads(line), p))
                        break
        except Exception:
            continue
    return out


def arms(row, ego_dim, cameras, n_goal_points, planner_ego, train_cams, build_cams) -> dict:
    """The arms, as a PURE FUNCTION of every component's belief.

    ⭐ Written this way so the self-test can feed it a deliberately diverged input and require the
    owning arm to read False. An arm evaluated only against live data can never be shown to be
    capable of failing -- which is the whole complaint this file answers.
    """
    n_img = len(row["image"]) if row and isinstance(row.get("image"), list) else (1 if row else 0)
    goal_w = len(row.get("goal", [])) if row else 0
    return {
        "1 bank ego width == model ego_dim":
            row is not None and len(row.get("ego", [])) == ego_dim,
        "2 bank camera list == model cameras, in order":
            row is not None and tuple(row.get("cameras", ())) == tuple(cameras),
        "3 bank image count == model n_cameras":
            row is not None and n_img == len(cameras),
        "4 trainer CAMERAS == model cameras":
            train_cams is not None and tuple(train_cams) == tuple(cameras),
        "4b builder CAMERAS == model cameras":
            build_cams is not None and tuple(build_cams) == tuple(cameras),
        "5 planner ego literal == model ego_dim": planner_ego == ego_dim,
        "6 bank goal width >= model goal input": goal_w >= 2 * n_goal_points,
        # ⛔ WITHOUT THIS, AN UNREADABLE BANK PASSES EVERY ARM ABOVE BY VACUUM. "0 rows read" is a
        # claim about the READ, not about the corpus -- the same hole as a grep -c of 0 on a file
        # that could not be opened.
        "7 CONTROL a real bank row was actually read": row is not None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    # ⛔ THE DEFAULT POINTED AT THE STALE SINGLE-CAMERA, 8-D-EGO BANK UNTIL 2026-09-21, so running
    # this file with no arguments reported `CONSUMERS_DIVERGED` against a corpus nobody uses --
    # a guard whose default makes it cry wolf gets ignored, which is how a real divergence hides.
    ap.add_argument("--targets", default="D:/Projects/TanitAD/data/refe_targets_4cam")
    ap.add_argument("--self-test", action="store_true",
                    help="reintroduce each historical divergence and require the arm to go RED")
    a = ap.parse_args(argv)

    import model as M
    cfg = M.REFeConfig()

    rows = bank_rows(a.targets) or [(None, "")]
    p_ego = planner_ego_width(HERE / "planner.py")
    t_cams = module_tuple(HERE / "train.py", "CAMERAS")
    b_cams = module_tuple(HERE / "build_targets.py", "CAMERAS")

    print("== what each component believes ==")
    print(f"  model.REFeConfig      ego_dim {cfg.ego_dim}   n_cameras {cfg.n_cameras}   "
          f"cameras {tuple(cfg.cameras)}")
    print(f"  planner.py  (AST)     ego literal {p_ego}")
    print(f"  train.py    (AST)     CAMERAS {t_cams}")
    print(f"  build_targets (AST)   CAMERAS {b_cams}")
    checks: dict = {}
    for row, src in rows:
        if row is None:
            print(f"  target bank           NO ROW READ from {a.targets!r} (no targets_rank*.jsonl)")
        else:
            print(f"  target bank           ego {len(row.get('ego', []))}-D   "
                  f"images {len(row['image']) if isinstance(row.get('image'), list) else 1}   "
                  f"cameras {row.get('cameras')}   goal {len(row.get('goal', []))}-D   "
                  f"rank {row.get('rank')}")
            print(f"                        <- {src}")
        # every file must conform: an arm is PASS only if it passes on every rank file
        for k, v in arms(row, cfg.ego_dim, tuple(cfg.cameras), cfg.n_goal_points,
                         p_ego, t_cams, b_cams).items():
            checks[k] = checks.get(k, True) and v

    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    if a.self_test:
        # ⭐ A REAL MUTATION TEST, NOT A RESTATEMENT. Each case below feeds `arms()` -- the SAME
        # function that produced the table above -- one deliberately diverged input, and requires
        # the arm that owns it to read False. ⚠️ My first version instead re-derived a condition
        # from the live bank ("is the bank 8-D?"), which is an expression over the data rather than
        # a mutation of it: arm 1 read GREEN and announced itself inert. That is exactly the
        # defect class this file exists for, committed inside the file itself.
        print("\n== SELF-TEST: each arm must go RED under the divergence it exists for ==")
        good_row = {"ego": [0.0] * 7, "cameras": ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"],
                    "image": ["a", "b", "c", "d"], "goal": [0.0] * 4}
        GC, GE, GN = tuple(good_row["cameras"]), 7, 2
        base = arms(good_row, GE, GC, GN, GE, GC, GC)
        cases = [
            ("1", "a bank banked at 8-D against a 7-D model",
             lambda: arms({**good_row, "ego": [0.0] * 8}, GE, GC, GN, GE, GC, GC)),
            ("2", "a bank that names only CAM_F0",
             lambda: arms({**good_row, "cameras": ["CAM_F0"]}, GE, GC, GN, GE, GC, GC)),
            ("3", "a bank row carrying one image for a 4-camera rig",
             lambda: arms({**good_row, "image": ["a"]}, GE, GC, GN, GE, GC, GC)),
            ("4", "a trainer left behind on one camera",
             lambda: arms(good_row, GE, GC, GN, GE, ("CAM_F0",), GC)),
            ("4b", "a builder left behind on one camera",
             lambda: arms(good_row, GE, GC, GN, GE, GC, ("CAM_F0",))),
            ("5", "a planner still building the 8-element ego literal",
             lambda: arms(good_row, GE, GC, GN, 8, GC, GC)),
            ("6", "a bank whose goal is narrower than the model's input",
             lambda: arms({**good_row, "goal": [0.0] * 2}, GE, GC, GN, GE, GC, GC)),
            # ⚠️ ARM 7 IS THE ONE CASE WITH LEGITIMATE COLLATERAL, and it is listed here as a
            # LITERAL rather than waived. An unreadable bank genuinely makes arms 1/2/3/6
            # unanswerable, and that is the control DOING ITS JOB: without it those four would
            # have reported four quiet PASSes over a bank nobody opened. Writing the expected
            # collateral out means a CHANGE in it fails this test instead of being absorbed.
            ("7", "a bank that could not be read at all",
             lambda: arms(None, GE, GC, GN, GE, GC, GC),
             ["1 bank ego width == model ego_dim",
              "2 bank camera list == model cameras, in order",
              "3 bank image count == model n_cameras",
              "6 bank goal width >= model goal input"]),
        ]
        ok = all(base.values())
        print(f"  [{'PASS' if ok else 'FAIL'}] CONTROL: the un-mutated inputs pass every arm")
        inert = [] if ok else ["the control itself fails -- the arms are miscalibrated"]
        for case in cases:
            tag, what, mut = case[0], case[1], case[2]
            expected_collateral = case[3] if len(case) > 3 else []
            res = mut()
            owner = next(k for k in res if k.startswith(tag + " "))
            went_red = not res[owner]
            # ⚠️ and it must go red for the RIGHT reason: no arm beyond the DECLARED collateral may
            # collapse with it, or the instrument reports one divergence as several and localises
            # nothing. Missing declared collateral is a failure too -- it means an arm that should
            # have noticed the same broken input stayed green.
            others = sorted(k for k, v in res.items() if not v and k != owner)
            tag_ok = went_red and others == sorted(expected_collateral)
            print(f"  [{'RED ' if tag_ok else 'FAIL'}] arm {tag}: {what}"
                  + ("" if tag_ok else f"   -> red={went_red} collateral={others}"))
            if not tag_ok:
                inert.append(f"arm {tag} ({what})")
        if inert:
            print("\nSELF_TEST_FAILED -- " + "; ".join(inert))
            return 2
        print("\nSELF_TEST_OK -- every arm goes red under its own historical divergence, alone")

    good = all(checks.values())
    print("\n" + ("CONSUMERS_CONFORM" if good else "CONSUMERS_DIVERGED"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
