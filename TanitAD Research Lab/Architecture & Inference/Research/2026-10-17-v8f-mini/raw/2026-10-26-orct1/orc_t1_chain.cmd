@echo off
rem Detached launcher for orc_t1_chain.sh (start it through WMI Win32_Process.Create, as hmc_chain.cmd and t1hd_chain.cmd). Arguments pass through (--dry-run).
rem Optional environment goes HERE (a WMI-created process does not inherit the caller's shell): ORCT_B2_OUT, ORCT_M2B_SHA_JSON, ORCT_MAX_WAIT_H, ORCT_READER_TREE, ORCT_TOOL_TREE.
"C:\Program Files\Git\bin\bash.exe" -lc "bash /c/Users/Admin/orct1/orc_t1_chain.sh %* < /dev/null" < NUL > C:\Users\Admin\orct1\logs\orc_t1_chain.launch.out 2>&1
