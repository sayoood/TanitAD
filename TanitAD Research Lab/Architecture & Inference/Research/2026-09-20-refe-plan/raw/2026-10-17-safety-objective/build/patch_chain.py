#!/usr/bin/env python3
"""patch_chain: the SAFEOBJ chain wiring (refe/rls1_checks.py + refe/rls1_pod.sh).  usage: patch_chain.py <PK dir>

rls1_checks.py: `monitor|replay --observe-only` (R1o's K3 is LOGGED -- ZZRLS1_K3_OBSERVED -- and never kills; K2, K4, K6 stay live), `safeobj-gate --dirs ...`
                (R1o runs only if R0d and R1d both read bar S = PASS).
rls1_pod.sh:    arms R0d R1d R1o (registered, profile refe_rls1d, argv = R0's + the levers), each read against its OWN arm A, the seed-matched C<n> as the second control,
                R1o's S gate and its --observe-only monitor/replay.
"""
import os

import patchlib as PL

RC_BASE = "f28a106e"      # prefix of the tip blob (checked by startswith below)
PS_BASE = "d1ba34a6"

RC_EDITS = [
    # ---- judge: K3 observe-only
    ('def judge(arm, rec, hist, k3_enabled=True):\n    """(code or None, why).  rec/hist items: {\'tag\',\'g3_ok\',\'g5_ok\',\'k4_breach\',\'k5_breach\'}.  K3 and K4 need TWO CONSECUTIVE breaches, K5 (R2 only) one."""\n',
     'def judge(arm, rec, hist, k3_enabled=True, observe_only=False):\n    """(code or None, why).  rec/hist items: {\'tag\',\'g3_ok\',\'g5_ok\',\'k4_breach\',\'k5_breach\'}.  K3 and K4 need TWO CONSECUTIVE breaches, K5 (R2 only) one.\n'
     '    observe_only (SAFEOBJ R1o, PREREG_SAFEOBJ.md section 9): K3 is NOT a kill -- the caller logs it (ZZRLS1_K3_OBSERVED); K4 and K5 are judged exactly as before."""\n'),
    ('    if k3_enabled and prev is not None and k3(rec) and k3(prev):\n        return "K3",',
     '    if k3_enabled and not observe_only and prev is not None and k3(rec) and k3(prev):\n        return "K3",'),
    # ---- monitor
    ('def monitor(arm, out_dir, check_fn, pid, alive_fn, kill_fn, sleep_fn=time.sleep, now_fn=time.time, poll=20.0, max_hours=6.0, start=None):',
     'def monitor(arm, out_dir, check_fn, pid, alive_fn, kill_fn, sleep_fn=time.sleep, now_fn=time.time, poll=20.0, max_hours=6.0, start=None, observe_only=False):'),
    ('            code, why = judge(arm, rec, hist, k3_enabled)\n            hist.append(rec)\n',
     '            code, why = judge(arm, rec, hist, k3_enabled, observe_only)\n'
     '            if observe_only:                                             # SAFEOBJ R1o: the K3 it WOULD have been killed by is logged, never acted on\n'
     '                would, would_why = judge(arm, rec, hist, k3_enabled)\n'
     '                if would == "K3":\n'
     '                    say("ZZRLS1_K3_OBSERVED", arm, r["tag"], would_why)\n'
     '            hist.append(rec)\n'),
    # ---- replay
    ('def replay(arm, out_dir, check_fn):\n', 'def replay(arm, out_dir, check_fn, observe_only=False):\n'),
    ('        code, why = judge(arm, rec, hist, k3_enabled)\n        hist.append(rec)\n        recs.append({**rec, "k3_enabled": k3_enabled})\n',
     '        code, why = judge(arm, rec, hist, k3_enabled, observe_only)\n        hist.append(rec)\n        recs.append({**rec, "k3_enabled": k3_enabled, **({"observe_only": True} if observe_only else {})})\n'),
    # ---- the R1o gate (new function, after replay)
    ('    return None, "", recs\n',
     '    return None, "", recs\n'
     '\n\n'
     'def safeobj_gate(dirs):\n'
     '    """SAFEOBJ (PREREG_SAFEOBJ.md section 4 / outcome table): R1o runs only if EVERY constrained arm read bar S = PASS.  Each <dir>/safeobj_S.json must exist, parse, and carry "S": "PASS"\n'
     '    (written by the reader, never by hand).  -> (ok, why); a missing, unparsable or non-PASS file is a refusal."""\n'
     '    bad = []\n'
     '    for d in dirs:\n'
     '        p = os.path.join(d, "safeobj_S.json")\n'
     '        try:\n'
     '            with open(p, encoding="utf-8") as f:\n'
     '                s = json.load(f).get("S")\n'
     '        except (OSError, ValueError, AttributeError) as e:\n'
     '            bad.append(f"{p}: {type(e).__name__}")\n'
     '            continue\n'
     '        if s != "PASS":\n'
     '            bad.append(f"{p}: S={s!r}")\n'
     '    return (not bad), (f"S = PASS on {len(dirs)} constrained arm(s)" if not bad else "; ".join(bad))\n'),
    # ---- CLI
    ('    p = sub.add_parser("updates-check"); p.add_argument("--args", required=True); p.add_argument("--n-sets", type=int, required=True)\n',
     '    p = sub.add_parser("updates-check"); p.add_argument("--args", required=True); p.add_argument("--n-sets", type=int, required=True)\n'
     '    p = sub.add_parser("safeobj-gate"); p.add_argument("--dirs", nargs="+", required=True)\n'),
    ('        p.add_argument("--rule", default="navsim_v1_lane")\n',
     '        p.add_argument("--rule", default="navsim_v1_lane")\n'
     '        p.add_argument("--observe-only", action="store_true")                 # SAFEOBJ: R1o only (refused for any other arm below)\n'),
    ('    if a.cmd == "updates-check":\n',
     '    if a.cmd == "safeobj-gate":\n'
     '        ok, why = safeobj_gate(a.dirs)\n'
     '        say("ZZRLS1_SAFEOBJ_S_PASS" if ok else "ZZRLS1_SAFEOBJ_S_REFUSED", why)\n'
     '        return 0 if ok else 1\n'
     '    if a.cmd == "updates-check":\n'),
    ('    if a.cmd == "monitor":\n        say("ZZRLS1_MONITOR_UP"',
     '    if a.cmd in ("monitor", "replay") and a.observe_only and a.arm != "R1o":\n'
     '        say("ZZRLS1_OBSERVE_ONLY_REFUSED", a.arm, "--observe-only is R1o\'s alone (the matched lever-off control); on any other arm it would disable K3")\n'
     '        return 1\n'
     '    if a.cmd == "monitor":\n        say("ZZRLS1_MONITOR_UP"'),
    ('        return monitor(a.arm, a.out, chk, a.pid, _alive, _kill_group, poll=a.poll, max_hours=a.max_hours)\n    code, why, recs = replay(a.arm, a.out, chk)\n',
     '        return monitor(a.arm, a.out, chk, a.pid, _alive, _kill_group, poll=a.poll, max_hours=a.max_hours, observe_only=a.observe_only)\n'
     '    code, why, recs = replay(a.arm, a.out, chk, observe_only=a.observe_only)\n'),
]

SAFEOBJ_ARGV_ON = "--epo-con g5 --epo-con-eps 0 --epo-con-lr 4 --epo-con-max 10 --epo-con-lam0 0"

PS_EDITS = [
    ('REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c"\n',
     '# SAFEOBJ (2026-10-17, from eval/rls1d_arms.json; PREREG_SAFEOBJ.md): R0d / R1d = R0 / R1\'s argv + the constrained-EPO levers ' + SAFEOBJ_ARGV_ON + '; R1o = seed 1 with --epo-con off NAMED.\n'
     '# Gate profile refe_rls1d; each is read against its OWN arm A (the K3 / K4 monitor, the replay, rlval_vs_A0.json), the seed-matched C<n> is the SECOND control (read when its dumps are there, announced when not).\n'
     '# Run as  RLS1_ARMS="R0d R1d R1o"  (a K3 kill of R0d or R1d STOPS the chain; R1o refuses unless R0d and R1d both read S = PASS: rls1_checks.py safeobj-gate; R1o\'s monitor and replay are --observe-only).\n'
     'REGISTERED_ARMS="C0 C1 R0 R1 R2 R0b R1b R0c R0d R1d R1o"\n'),
    ('    R0c) echo refe_rls1c ;;\n', '    R0c) echo refe_rls1c ;;\n    R0d|R1d|R1o) echo refe_rls1d ;;\n'),
    ('--epo-terms-heldout $TERMS_VA --epo-ref armA" ;;\n    *) return 1 ;;\n',
     '--epo-terms-heldout $TERMS_VA --epo-ref armA" ;;\n'
     '    # SAFEOBJ (eval/rls1d_arms.json, token for token; launch_gate.py profile refe_rls1d, which requires every --epo-con* flag NAMED: the parser default `off` must never carry a lever)\n'
     '    R0d) seed=0; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA ' + SAFEOBJ_ARGV_ON + '" ;;\n'
     '    R1d) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA ' + SAFEOBJ_ARGV_ON + '" ;;\n'
     '    R1o) seed=1; epo="--epo --epo-terms-train $TERMS_TR --epo-terms-heldout $TERMS_VA --epo-con off" ;;\n'
     '    *) return 1 ;;\n'),
    ('refe/test_rl_r0c.py \\\n', 'refe/test_rl_r0c.py refe/test_rl_safeobj.py \\\n'),
    ('  PROFILE="$(arm_profile "$ARM")"\n  mkdir -p "$ARMDIR" "$LG"\n',
     '  PROFILE="$(arm_profile "$ARM")"\n'
     '  # SAFEOBJ (PREREG_SAFEOBJ.md section 4): R1o -- the lever-off control -- runs only if BOTH constrained arms read bar S = PASS (the reader writes <arm>/safeobj_S.json); refused here, before any GPU work.\n'
     '  OBS=""\n'
     '  if [ "$ARM" = R1o ]; then\n'
     '    OBS="--observe-only"                  # R1o\'s K3 is logged (ZZRLS1_K3_OBSERVED), never a kill: a kill at u500 would delete the matched control at the checks where the lever is read; K2 / K6 stay live\n'
     '    env -u HF_TOKEN "$PY" rls1_checks.py safeobj-gate --dirs "$R/arms/R0d" "$R/arms/R1d" > "$R/safeobj_gate_R1o.log" 2>&1 200>&-\n'
     '    grep -q "^ZZRLS1_SAFEOBJ_S_PASS" "$R/safeobj_gate_R1o.log" || refuse "R1o runs only if R0d and R1d both read bar S = PASS: $(tail -n 1 "$R/safeobj_gate_R1o.log" | cut -c1-300)"\n'
     '  fi\n'
     '  mkdir -p "$ARMDIR" "$LG"\n'),
    ('  [ "$ARM" = R0c ] && RD=C0        # RL-S1c: C0 is R0c\'s SECOND control (the 9d read below); named here, not left to ${ARM:1:1}\n',
     '  [ "$ARM" = R0c ] && RD=C0        # RL-S1c: C0 is R0c\'s SECOND control (the 9d read below); named here, not left to ${ARM:1:1}\n'
     '  case "$ARM" in R0d) RD=C0;; R1d|R1o) RD=C1;; esac        # SAFEOBJ: the seed-matched control is the SECOND control (read when its dumps are there, announced when not); named here, not left to ${ARM:1:1}\n'),
    ('  [ "$ARM" = R0c ] && REFDIR="$ARMDIR/scores"\n',
     '  [ "$ARM" = R0c ] && REFDIR="$ARMDIR/scores"\n'
     '  case "$ARM" in R0d|R1d|R1o) REFDIR="$ARMDIR/scores";; esac     # SAFEOBJ (PREREG_SAFEOBJ.md section 9: "K3 and K4 read against the arm\'s OWN A"): the monitor and replay reference is its own arm A, whether or not C<n>\'s dumps are on the pod\n'),
    ('--sets "$SETS_VA" --tsv "$VAL_TSV" --pid "$TPID" > "$ARMDIR/monitor.log" 2>&1 200>&- &\n',
     '--sets "$SETS_VA" --tsv "$VAL_TSV" --pid "$TPID" ${OBS:-} > "$ARMDIR/monitor.log" 2>&1 200>&- &\n'),
    ('--sets "$SETS_VA" --tsv "$VAL_TSV" > "$ARMDIR/replay.log" 2>&1 200>&-\n',
     '--sets "$SETS_VA" --tsv "$VAL_TSV" ${OBS:-} > "$ARMDIR/replay.log" 2>&1 200>&-\n'),
    ('      fi\n    fi\n    grep -q "^ZZRLVAL_OK" "$ARMDIR/rlval.log"',
     '      elif [ "$ARM" = R0d ] || [ "$ARM" = R1d ] || [ "$ARM" = R1o ]; then\n'
     '        log "ZZRLS1_SAFEOBJ_NO_SECOND_CONTROL arm $ARM: $RD\'s banked dumps $R/arms/$RD/scores/scores_update1000_A.npz are not on the pod, so rlval_vs_$RD.json was NOT produced: the read against its own arm A (rlval_vs_A0.json) stands alone"\n'
     '      fi\n    fi\n    grep -q "^ZZRLVAL_OK" "$ARMDIR/rlval.log"'),
]


def main():
    pk = PL.pk_arg()
    for fn, edits, base in (("rls1_checks.py", RC_EDITS, RC_BASE), ("rls1_pod.sh", PS_EDITS, PS_BASE)):
        path = os.path.join(pk, "refe", fn)
        src = PL.read(path)
        blob = PL.git_blob(src)
        if not blob.startswith(base):
            raise SystemExit(f"[{fn}] base blob {blob} does not start with {base}")
        out, rev = PL.apply_edits(src, edits, fn)
        PL.write(path, out)
        print({"file": fn, "blob_before": blob, "blob_after": PL.git_blob(out), "edits": len(edits)})


if __name__ == "__main__":
    main()
