@echo off
set CUDA_VISIBLE_DEVICES=
set PYTHONIOENCODING=utf-8
set "T=C:\Users\Admin\par18\c2run\tree\TanitAD Research Lab\Architecture & Inference\Research\2026-09-20-refe-plan\raw"
cd /d "%T%\2026-10-19-search-navtest-pipe\code"
echo ZZC2RUN_LAUNCH %DATE% %TIME% >> C:\Users\Admin\par18\c2run\logs\c2_run.log
"C:\Users\Admin\venvs\tanitad\Scripts\python.exe" -u navpipe_chain.py --mode navtest --work C:\Users\Admin\c2_navtest --amendment "%T%\2026-10-19-search-p2-confirm\c2\AMENDMENT_C2.md" --amendment-sha256 a248ec27f7976af9d7b577cae24dedd8e7113b2384151f404cf75e5858c8db5b --c1-result C:\Users\Admin\spc_c1\RESULT_SPC.json --c1-result-sha256 d6278e9d37788e954672015e35c8619140d72ea9c8e9cfd992bd8c22f30cabbe --dose-result C:\Users\Admin\spc_dose\RESULT_DOSE.json --dose-result-sha256 1a3c6ac39d154e5d6f0deef18d3d62e7f6af8cd60c9606d550e47ed0f49670d5 --config "%T%\2026-10-19-l2-stepB\raw\stepB_config.json" --fixture-record "%T%\2026-10-19-search-p2-confirm\c2\FIXTURE_RECORD.json" --table-ref D:\Projects\TanitAD-artifacts\p2_full_table\table_p2_full.npz < NUL >> C:\Users\Admin\par18\c2run\logs\c2_run.log 2>&1
echo ZZC2RUN_EXIT %ERRORLEVEL% %DATE% %TIME% >> C:\Users\Admin\par18\c2run\logs\c2_run.log
