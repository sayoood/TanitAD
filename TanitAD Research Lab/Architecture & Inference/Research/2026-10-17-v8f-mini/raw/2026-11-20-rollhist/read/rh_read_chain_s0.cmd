@echo off
rem rh_read_chain_s0.cmd -- the PROVISIONAL one-seed (seed 0) ROLLHIST read (Amendment 2 item 6). Launch only through WMI (CreateFlags 0x08000200), stdin NUL.
rem After it ends, the MM moves the cross-seed step markers aside (t0evals gt0 headway q3 strat read) so the two-seed read recomputes them; per-seed pulls and rollouts are reused.
cd /d F:\TanitAD-scratch\rollhist\read
set PATH=C:\Program Files\Git\usr\bin;%PATH%
set RH_SEEDS=0
"C:\Program Files\Git\usr\bin\bash.exe" F:/TanitAD-scratch/rollhist/read/rh_read_chain.sh < NUL
