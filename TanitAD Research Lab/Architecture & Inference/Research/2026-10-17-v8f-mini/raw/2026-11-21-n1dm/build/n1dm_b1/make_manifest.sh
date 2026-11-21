#!/bin/bash
# make_manifest.sh -- MANIFEST_SHA256.txt of the N1DM build deliverables (run again after logs/finish.log ends with FINISHED; the patch, the delta tar and the mutation table are written by the finisher).
B=F:/TanitAD-scratch/n1dc/build/n1dm_b1; cd $B || exit 1
{ for f in HANDOFF_N1DM.md changed_files.txt mkpatch_n1dm.py run_mutants_n1dm.py mk_tiny_arm.py dry_post_lines.sh finish_n1dm.sh run_regress.sh run_regress_rest.sh regress_cmp.py runpt.sh envrun.sh patch_scripts/*.py n1dm.patch n1dm_delta.tar n1dm_tree.sha256 all_blobs_n1dm.txt mutation_table_n1dm.json REGRESSION.json TREE_FILES_SHA256.txt pod/* logs/*.log regress_order.txt; do [ -f "$f" ] && sha256sum "$f"; done
  sha256sum F:/TanitAD-scratch/n1dc/next/PREREG_V8F_N1DM_DRAFT.md; cd F:/TanitAD-scratch/n1dc/read_n1dm && for f in *.py *.sh *.cmd eval141/*.py eval141/*.sh eval141/*.cmd test/*; do [ -f "$f" ] && sha256sum "$f" 2>/dev/null | sed "s# *# *read_n1dm/#" || true; done; } > $B/MANIFEST_SHA256.txt
wc -l $B/MANIFEST_SHA256.txt
