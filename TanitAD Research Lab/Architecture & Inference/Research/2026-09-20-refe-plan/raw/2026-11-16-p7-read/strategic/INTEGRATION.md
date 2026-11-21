# Integration: STRATEGIC family beside the pinned P5-C read (no pinned file edited)

Status: **reported, not a decision bar.** PREREG_P5C froze its bars before the PI's 2026-11-16 ruling ("strategic is not paused anymore"); this family adds NO bar and gates nothing.

1. Land `eval/instruments/strategic_family.py` and `test_strategic_family.py` (new files, next to `families_run.py`). It imports the pinned `eval/command_adherence.py` and, through it, `proxy_eval.py`; it edits neither.
2. In the P5-C read, after the arms' seams exist (the same seams `p5c_read_prep` feeds the families stage), add one extra, separate command, with CPU only and the free-commit guard built into the CLI:

       python eval/instruments/strategic_family.py read --arm C1=<C1 seam.npz> --base C0=<C0 seam.npz> --out <read_dir>/strategic_family_C1_C0.json
   (add `--arm C0r=...` as a second arm if C0r is read). Seams are the `stage2_<name>.npz` / `refe_<name>.npz` files with `token`, `poses` [N,8,3], optional `source`.
3. Wire it in `p5c_render.py` (or the read's report step) as the STRATEGIC block that today prints `p5c_read.STRATEGIC` = PAUSED: print the S-DEC and S-GOAL tables from the JSON instead, labelled "REPORTED, no bar". Pinned `p5c_read.py` / `p5c_families.py` stay byte-identical; if a renderer is pinned, run the command separately and attach the JSON/markdown next to the read's output and cite it from the register row.
4. Controls travel with the JSON (`controls`: logged future = 0, mirror lowers adherence, self-vs-self = 0); a read whose JSON has `INVALID` must not be quoted.
5. Env needed: `REFE_EVAL_DIR` only if the file is run from outside `eval/`; `TANITAD_REPO` (stack/, taniteval/); venv `C:/Users/Admin/venvs/tanitad`; semicolons in PYTHONPATH on Windows.
