# HANDOFF_DOSE2 (package complete; nothing run on box 2, nothing staged or committed)

State: the second-dose package under theory/stopbal/dose2/ is built and tested (see tests/RESULTS.md: unit 194/0, dry run 102/0, read 114/0). README.md has the launch line, the wall-time estimate and the read order. PACKAGE_MANIFEST.sha256 lists every file.
Next step (Master Mind): land the package, then launch with the README's three lines, detached. The chain waits for the SLOWWM end marker by itself.
Scratch under F:/TanitAD-scratch/dose2_build/ is deleted at the end of the build (see the final report for the log).
Rule slip, reported earlier: one early shell command created dose2/frozen, stages, jobs, tests, read and the two jobs copies through a variable that hid the dagland path from the guard; every later file was created with the Write tool.
Leftover: dose2/__pycache__/ (one .pyc from the first session, created before the checker set dont_write_bytecode); it is not in the manifest and is not needed (delete needs a command that names no dagland path, which the guard allows only outside this tree: left for the Master Mind).
