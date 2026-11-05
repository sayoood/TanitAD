# BASH_ENV for the shell tests: bash functions win over any PATH command, so a stray ssh / scp / rsync / sftp / curl / wget in a script under test cannot reach a remote. Each leaves a tripwire file and fails.
for _n in ssh scp sftp rsync curl wget ftp telnet nc ping; do
  eval "$_n() { echo NETWORK-BLOCKED $_n \"\$*\" >> \"\${CLR_TRIPWIRE:-/tmp/clr_tripwire}\"; echo NETWORK-BLOCKED $_n >&2; return 99; }"
done
unset _n
