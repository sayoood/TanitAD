# Thor suite gate — the producer of G-SUITE's verdict

`launch_gate.py` profiles with `suite_mode="consume"` (refcv7, refcv8) do not run pytest themselves. G-SUITE reads a
**full-suite verdict, bound to the launch commit and tree** (`launch_gate.py suite-bind`). These are the tools that
produce that verdict. Until 2026-10-12 they lived only on the dev box (`qland/work/refcv7/r4/`,
`qland/work/thorgate_n1/`) and on Thor (`/home/nvidia/gate_*/`). The pre-gate audit could not find them in the repo
(`…/2026-10-04-refcv8-tactical-conditioning/pregate_2026-10-12/RESULT.md`, B6).

| file | where it runs | what it does |
|---|---|---|
| `make_thor_suite_plan.py` | dev box | builds `<dir>`: TIP archive, CAND overlay, `plan_thor.json`, plus copies of the two scripts below |
| `thor_gate_runner2.py` | Thor (CPU) | extracts TIP and CAND trees, md5-checks the overlay, runs an import probe, then runs one `pytest` per file with junit; writes `results/{TIP,CAND}.json` and `results/ALLDONE` |
| `gate_verdict.py` | dev box or Thor | reads the junit files and writes `verdict.json` (regressions, new failures, pre-existing failures, TIP-skip/CAND-fail, crashes without junit) |

Provenance. Both scripts are banked import-guarded: their module-level code moved into `main()`, because the
landing's import probe executes every new module. Nothing else changed.
* `thor_gate_runner2.py`: from the Thor copy in `/home/nvidia/gate_final_1855/` (md5 `8d550602…`, CRLF). Banked as
  LF, md5 `6c981e02…`.
* `gate_verdict.py`: from `qland/work/thorgate_n1/` (md5 `a14b4ad6…`). Banked md5 `d8897a80…`. On the pin-source run
  (`qland/work/thorgate_b2_0201/res`), the banked copy reproduces that run's `verdict.json` **byte for byte** (sha256
  `70c66cba…`, the value `launch_gate_thor_env_failures.json` cites as its source).

## Recipe (refcv8 launch)

```
# dev box: TIP = the launch commit's parent (or the commit itself), CAND = the launch commit
python stack/ops/gate/make_thor_suite_plan.py --git-dir <repo>/.git --tip <parent> --cand <launch> --out <dir>
# ship <dir> to Thor (e.g. /home/nvidia/gate_<c12>), then on Thor, CPU only, after the ladder has stopped:
cd /home/nvidia/gate_<c12> && nohup nice -n 15 /home/nvidia/venvs/tanitad-train/bin/python thor_gate_runner2.py \
    /home/nvidia/gate_<c12> /home/nvidia/gate_fix_2224/pytest_pkgs > runner.out 2> runner.err < /dev/null 200>&- &
# wait for results/ALLDONE (assert on the FILE, never on a status), then:
mkdir -p /home/nvidia/gate_<c12>/v/res && ln -s /home/nvidia/gate_<c12>/results /home/nvidia/gate_<c12>/v/res/results
python gate_verdict.py /home/nvidia/gate_<c12>/v      # -> v/verdict.json
python stack/scripts/launch_gate.py suite-bind --verdict v/verdict.json --cand-tree /home/nvidia/gate_<c12>/CAND \
    --commit <launch sha> --out suite_bound.json
# refcv8_prelaunch.sh run ... --suite-verdict suite_bound.json
```

⚠️ `gate_verdict.py` expects `<dir>/res/results/{CAND,TIP}` (the layout after pulling the results back as
`res.tar`); on Thor the symlink above gives it that layout.

⛔ **Scope is the pinned protocol, not "every test".** G-SUITE admits a Thor failure only if its id is in
`stack/ops/launch_gate_thor_env_failures.json`. That list was classified on plans that ran **all stack tests plus 7
taniteval files**. `make_thor_suite_plan.py` therefore defaults to those 7 (`TANITEVAL_PROTOCOL`). `--taniteval-all`
widens the run and needs a re-classified pin before its verdict can pass.

⛔ The CAND tree's digest must equal the launched tree's (`tree_manifest`: `stack/` + `taniteval/`, caches
excluded). The archive carries committed bytes (`core.autocrlf=false`); a CRLF tree digests differently.

⛔ A CAND that deletes or renames a file under `stack/` or `taniteval/` is refused by the builder. The runner can only
overlay, so the CAND tree would still hold TIP's copy.
