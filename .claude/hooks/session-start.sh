#!/bin/bash
# SessionStart hook for Claude Code cloud sessions.
#  1. Installs the Python dependencies when requirements/pyproject change.
#  2. Starts `python -m stockdata sync` fully detached, so the session is not
#     held up. sync restores the encrypted snapshot when STOCKDATA_SNAPSHOT_KEY
#     is set (else builds from scratch; timings in DATA.md) and then tops up.
#     Progress: data/logs/sync.log, or `python -m stockdata status`.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Python packages go into the system Python, so the hash file lives there too.
pyhashes="$(python3 -I -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/.stockdata-session-start.sha256"
if ! sha256sum --check --status "$pyhashes" 2>/dev/null; then
  rm -f "$pyhashes"
  python3 -m pip install --quiet --disable-pip-version-check -r requirements.txt
  python3 -m pip install --quiet --disable-pip-version-check --no-deps --no-build-isolation -e .
  sha256sum .claude/hooks/session-start.sh requirements.txt pyproject.toml > "$pyhashes" || rm -f "$pyhashes"
fi

# Data sync in the background: new session id and no inherited stdio, so it
# outlives the hook and does not keep the hook's output pipe open.
logdir="${STOCKDATA_DIR:-$PWD/data}/logs"
mkdir -p "$logdir"
setsid nohup python3 -m stockdata sync >> "$logdir/sync.log" 2>&1 < /dev/null &

echo "stockdata: data sync running in the background (log: $logdir/sync.log)." \
     "Check with 'python -m stockdata status'; load data with stockdata.load_panel(start, end)."
