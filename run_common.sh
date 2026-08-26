#!/usr/bin/env bash
# Shared prelude for every run_*.sh in this repo. Source it, never execute it.
#
# It exists because the batch scripts used to disagree with each other about
# basic facts: run_all_experiments.sh activated ~/.virtualenvs/WM-model while
# run_all_analysis.sh activated ~/.venv/WM-model, so the same repo ran under two
# different interpreters depending on which script you happened to invoke. The
# venv is now resolved once, here, by probing for whichever one exists.
#
# Provides:
#   REPO_ROOT              absolute path to the repo
#   start_provenance NAME  creates logs/NAME_<ts>/ and records commit + branch
#   run_step NAME CMD...   runs CMD, logs to $LOGDIR/NAME.log, records rc
#   latest_exp PREFIX      newest experiments/PREFIX_<timestamp> dir, or empty

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- venv resolution -----------------------------------------------------------
# GPU server uses ~/.venv/WM-model; local machines use ~/.virtualenvs/WM-model.
# Respect an already-active venv so callers can override by activating first.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  for _candidate in "$HOME/.venv/WM-model" "$HOME/.virtualenvs/WM-model" "$REPO_ROOT/venv"; do
    if [[ -f "$_candidate/bin/activate" ]]; then
      # shellcheck disable=SC1091
      source "$_candidate/bin/activate"
      break
    fi
  done
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "ERROR: no WM-model virtualenv found." >&2
  echo "  Looked in: ~/.venv/WM-model, ~/.virtualenvs/WM-model, ./venv" >&2
  echo "  Activate one yourself and re-run, or see the table in AGENTS.md." >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export MPLCONFIGDIR="${REPO_ROOT}/.matplotlib"
mkdir -p "$MPLCONFIGDIR" "${REPO_ROOT}/logs"

# --- provenance ----------------------------------------------------------------
# Every run records the commit it ran under. The neural-efficiency audit had to
# reconstruct this from file mtimes across two timezones; don't repeat that.
LOGDIR=""
PROV=""

start_provenance () {
  local name="$1"
  LOGDIR="logs/${name}_$(date -u +%Y%m%d_%H%M%S)"
  mkdir -p "$LOGDIR"
  PROV="$LOGDIR/00_provenance.log"
  {
    echo "commit:  $(git rev-parse --short HEAD 2>/dev/null || echo 'not-a-git-repo')"
    echo "branch:  $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
    echo "python:  $(python -c 'import sys; print(sys.executable)')"
    echo "host:    $(hostname)"
    echo "started: $(date -u)"
  } | tee "$PROV"
  echo "Logs: $LOGDIR"
}

run_step () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$PROV"
  "$@" > "$LOGDIR/${name}.log" 2>&1
  local rc=$?
  echo "=== [$(date -u +%H:%M:%S)] END   $name rc=$rc ===" | tee -a "$PROV"
  return 0   # a failed step must not abort the remaining batch
}

finish_provenance () {
  echo "finished: $(date -u)" | tee -a "$PROV"
  echo
  echo "Provenance: $PROV"
  if grep -q 'rc=[^0]' "$PROV"; then
    echo "WARNING: at least one step exited non-zero -- check $PROV" >&2
  fi
}

# --- experiment discovery ------------------------------------------------------
latest_exp () {
  local prefix="$1"
  find "${REPO_ROOT}/experiments" -maxdepth 1 -type d \
       -name "${prefix}_[0-9]*" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -1 | cut -d' ' -f2-
}
