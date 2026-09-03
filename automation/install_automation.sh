#!/bin/bash
# Install the daily paper-trading schedule as macOS launchd jobs.
#
# Jobs (all paper trading — logs to mlb_betting/logs/):
#   morning  10:00  refresh data, retrain, log ML/F5/prop picks
#   lineups  15:30  second props pass once confirmed lineups are posted
#   clv      18:30  capture closing lines for CLV
#   settle   00:15  settle finished games, update bankroll
#
# Usage:
#   bash automation/install_automation.sh            # install / reinstall
#   bash automation/install_automation.sh uninstall  # remove all jobs

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="$(command -v python3 || command -v python)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
JOBS="morning lineups clv settle"

if [ "$1" = "uninstall" ]; then
    for job in $JOBS; do
        launchctl unload "$AGENTS_DIR/com.mlbbetting.$job.plist" 2>/dev/null || true
        rm -f "$AGENTS_DIR/com.mlbbetting.$job.plist"
    done
    echo "All mlbbetting launchd jobs removed."
    exit 0
fi

mkdir -p "$PROJECT_DIR/logs" "$AGENTS_DIR"

echo "Project: $PROJECT_DIR"
echo "Python:  $PYTHON_BIN"
echo

for job in $JOBS; do
    template="$PROJECT_DIR/automation/com.mlbbetting.$job.plist.template"
    target="$AGENTS_DIR/com.mlbbetting.$job.plist"
    sed -e "s|__PYTHON__|$PYTHON_BIN|g" \
        -e "s|__PROJECT__|$PROJECT_DIR|g" \
        "$template" > "$target"
    launchctl unload "$target" 2>/dev/null || true
    launchctl load "$target"
    echo "  ✓ $job installed"
done

echo
echo "Done. Schedule: morning 10:00 / lineups 15:30 / clv 18:30 / settle 00:15"
echo "Logs: $PROJECT_DIR/logs/"
echo "Note: jobs only run while the Mac is awake. If it sleeps through a"
echo "time, run the script manually or consider 'caffeinate' / pmset wake."
