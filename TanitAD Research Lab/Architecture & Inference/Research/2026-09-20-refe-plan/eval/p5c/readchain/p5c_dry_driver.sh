#!/usr/bin/env bash
# DRY RUN driver (serial): 1) the seal REGRESSION arm (must refuse, exit 4)  2) the identity run (C1 = C0 = P7). Started by dry_run_p5c.ps1 through run_p5c_read.cmd.
HOLD="E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p5c_readchain"
export P5C_COMMIT_WAIT_MIN="${P5C_COMMIT_WAIT_MIN:-720}"
D=$P5C_DRYWORK_ROOT
P5C_WORK="$D/dryrun_regression_seal" P5C_REGRESSION=seal bash "$HOLD/p5c_read_chain.sh" > "$D/dryrun_regression_seal.stdout.log" 2>&1 < /dev/null
echo "regression arm exit $?" >> "$D/driver.log"
P5C_WORK="$D/dryrun_identity" bash "$HOLD/p5c_read_chain.sh" > "$D/dryrun_identity.stdout.log" 2>&1 < /dev/null
echo "identity run exit $?" >> "$D/driver.log"
