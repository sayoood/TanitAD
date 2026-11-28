# HANDOFF: OPTB read chain build (2026-11-18). State at agent end.

Package: `F:/TanitAD-scratch/actrsp/optb/readchain/` (staged only; nothing committed or pushed; box 2, pod, git, other jobs untouched).

## Done (MEASURED, 14 tests green: `test_route.py` 9 + `test_tools.py` 5)
* Loading route `optb_route.py` (+ `make_ft_ckpt.py`, `route_probe.py`): ladder tree REFUSES a real FT ckpt (TypeError unexpected keyword `wm_gamma_levels`); the route loads it strictly (no missing / unexpected key), remaps the non-existent probe path to the local copy, sha256 still checked; R35 rollouts bit-equal ladder vs route; wrong-sha / tampered probe / unknown basename refuse.
* Chain `optb_readchain.sh` + `optb_readchain.cmd` + `launch_wmi.ps1`; `readchain_tools.py` (evidence C2/C3/C5, delta, bandmet, manifest). Dry run B (tampered bank): refused at evidence, rc 3, sentinel in own file (MEASURED, 4 s).

## RUNNING / PENDING when I ended (WMI pid 1320, `dry_all.cmd`; output `tmp/dry_all.out`)
1. Dry run A (`tmp/dryA/`): FAKEa_s0 = R35 copy through ALL stages (planner `--n 8`, 1 seed) then FAKEc_s0 unbanked + box-2 STOP.txt -> chain must end with `out/ZZ_ENDED_BY_BOX2_STOP.txt`, rc 0 (`tmp/dryA/chain.rc`). Per-stage wall: `out/state/FAKEa_s0.{plan,ia3pk,bandmet}.wall`.
2. Then `time_plan.sh`: ONE full real planner seed on R35_s0 through the route -> `tmp/timing/timing.txt` (`rc=0 wall_s=...`, ESTIMATED ~35 min before).
Next step for whoever picks up: read those files; if rc/sentinels are as stated, copy the MEASURED minutes into `BUILD_READCHAIN.md` section 4 (marked PENDING), then delete `tmp/` (log path + size). If the dry run failed, the stage log named in `tmp/dryA/out/ZZ_STOP.txt` has the cause.

## Not done / UNVERIFIED
* No real FT bank exists: evidence C2/C3 were exercised on a real trainer log + selfcheck of the synthetic recipe; C5 against R35 init only in refusal form (dry run).
* ia_read (BAND-MET) and the planner were never run on a real FT checkpoint (none exists).
