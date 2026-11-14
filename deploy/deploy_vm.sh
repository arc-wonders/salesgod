#!/usr/bin/env bash
# Safe deploy of a git ref to the SalesGod production VM.
# See deploy/DEPLOY.md.  Usage:
#   deploy_vm.sh <git-ref> [--dry-run] [--force-overwrite-vm-only] [--accept PATH]...
#   deploy_vm.sh <git-ref> --seed      # only record DEPLOYED_REF (VM must already match the ref)
#   deploy_vm.sh --status              # deployed ref + VM-only drift right now
set -uo pipefail

HOST="${DEPLOY_HOST:-lakshay@34.143.141.218}"
APP=/opt/salesgod/app
BASE=/opt/salesgod
SVC=salesgod-voice
PORT=8080
LIVE_TIMEOUT="${LIVE_TIMEOUT:-600}"     # seconds to wait for calls to end
READY_TIMEOUT="${READY_TIMEOUT:-300}"   # seconds to wait for the 'ready:' log line
STALE_CALL="${STALE_CALL:-1800}"        # an accepted call with no hangup older than this counts as dead
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 "$HOST")

REF="" DRY=0 FORCE=0 STATUS=0 SEED=0 ACCEPT=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --force-overwrite-vm-only) FORCE=1 ;;
    --status) STATUS=1 ;;
    --seed) SEED=1 ;;
    --accept) shift; ACCEPT+=(--accept "$1") ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 64 ;;
    *) REF="$1" ;;
  esac
  shift
done
[ "$STATUS" = 1 ] || [ -n "$REF" ] || { echo "usage: $0 <git-ref> [--dry-run] [--force-overwrite-vm-only]  |  $0 --status" >&2; exit 64; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
log() { echo "[deploy] $*"; }
die() { echo "[deploy] ERROR: $*" >&2; exit 1; }


vm_manifest() {
  "${SSH[@]}" "sudo bash -s" <<REMOTE | LC_ALL=C sort -k2
set -e
cd $APP
find . -path ./data -prune -o \\( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \\) -prune -o -type f ! -name '*.pyc' -print0 | xargs -0 -r sha256sum
REMOTE
}

local_manifest() {  # $1 = dir
  ( cd "$1" && find . -path ./data -prune -o \( -name __pycache__ -o -name '*.egg-info' -o -name .pytest_cache \) -prune -o -type f ! -name '*.pyc' -print0 \
      | xargs -0 -r sha256sum | LC_ALL=C sort -k2 )
}

vm_cat() { "${SSH[@]}" "sudo cat $BASE/$1 2>/dev/null" 2>/dev/null; }

# prints "<last_accept> <last_hangup> <last_sim> <now>" (epoch seconds, 0 = none since last restart)
call_state() {
  "${SSH[@]}" "sudo journalctl -u $SVC --since '-3 hours' -o short-unix --no-pager | awk -v now=\$(date +%s) '
    /voice.server ready:/ {la=0; lh=0}
    /WebSocket \\/plivo\\/stream.*\\[accepted\\]/ {la=\$1}
    /POST \\/plivo\\/hangup/ {lh=\$1}
    /\\/api\\/sim/ {ls=\$1}
    END {printf \"%d %d %d %d\\n\", la, lh, ls, now}'"
}

# echoes a reason and returns 0 when a call is live, 1 when idle
call_live() {
  local la lh ls now
  read -r la lh ls now < <(call_state) || return 1
  [ -n "${now:-}" ] || { echo "could not read journal"; return 0; }   # unknown = treat as live
  if [ "$la" -gt "$lh" ] && [ $((now - la)) -lt "$STALE_CALL" ]; then
    echo "a Plivo call was accepted $((now - la))s ago with no hangup yet"; return 0
  fi
  if [ "$ls" -gt 0 ] && [ $((now - ls)) -lt 60 ]; then
    echo "/api/sim used $((now - ls))s ago"; return 0
  fi
  return 1
}

wait_idle() {  # $1 = timeout
  local t0=$SECONDS why
  while why=$(call_live); do
    [ $((SECONDS - t0)) -lt "$1" ] || { echo "$why"; return 1; }
    log "waiting for calls to finish: $why"
    sleep 10
  done
  return 0
}

fetch_state() {  # fills $WORK/vm.txt, $WORK/last.txt (if any), LAST_REF
  log "reading VM tree manifest..."
  vm_manifest > "$WORK/vm.txt" || die "could not read the VM manifest over ssh"
  [ -s "$WORK/vm.txt" ] || die "VM manifest is empty (ssh/sudo problem?)"
  LAST_REF="$(vm_cat DEPLOYED_REF | head -1)"
  if vm_cat DEPLOYED_MANIFEST > "$WORK/last.txt" && [ -s "$WORK/last.txt" ]; then
    LASTARG=(--last "$WORK/last.txt")
  else
    LASTARG=(); LAST_REF=""
  fi
}

# ---------------------------------------------------------------- status
if [ "$STATUS" = 1 ]; then
  fetch_state
  echo "Host:         $HOST"
  if [ ${#LASTARG[@]} -eq 0 ]; then
    echo "Deployed ref: UNKNOWN (no DEPLOYED_REF/DEPLOYED_MANIFEST yet - first run seeds it)"
    echo "Cannot compute drift without a record. The VM tree has $(wc -l < "$WORK/vm.txt") files."
    exit 0
  fi
  echo "Deployed ref: $LAST_REF"
  # build == last: anything different on the VM is drift
  python3 "$HERE/vm_diff.py" --build "$WORK/last.txt" --vm "$WORK/vm.txt" --last "$WORK/last.txt" --last-ref "${LAST_REF%% *}" --vm-host "$HOST"
  rc=$?
  [ $rc = 0 ] && echo "No VM-only drift."
  if why=$(call_live); then echo "Live call: YES ($why)"; else echo "Live call: no"; fi
  exit $rc
fi

# ---------------------------------------------------------------- export + build manifest
git -C "$REPO" rev-parse --verify -q "$REF^{commit}" >/dev/null || die "unknown git ref: $REF"
FULL="$(git -C "$REPO" rev-parse "$REF^{commit}")"
SHORT="$(git -C "$REPO" rev-parse --short "$FULL")"
log "exporting $REF ($SHORT) without data/..."
git -C "$REPO" archive --format=tar -o "$WORK/build.tar" "$FULL" -- . ':(exclude)data' || die "git archive failed"
mkdir "$WORK/build" && tar xf "$WORK/build.tar" -C "$WORK/build"
local_manifest "$WORK/build" > "$WORK/build.txt"
log "build has $(wc -l < "$WORK/build.txt") files"

fetch_state
if [ ${#LASTARG[@]} -eq 0 ]; then
  log "no DEPLOYED_MANIFEST on the VM: last deploy UNKNOWN - every differing file must be identical to the ref or --accept'ed"
else
  log "last deployed ref: $LAST_REF"
fi

# ---------------------------------------------------------------- classify
python3 "$HERE/vm_diff.py" --build "$WORK/build.txt" --vm "$WORK/vm.txt" "${LASTARG[@]}" "${ACCEPT[@]}" \
  --last-ref "${LAST_REF%% *}" --vm-host "$HOST" --write-lists "$WORK/lists" | tee "$WORK/report.txt"
DIFF_RC=${PIPESTATUS[0]}
if [ "$DIFF_RC" -ne 0 ] && [ "$DIFF_RC" -ne 2 ]; then die "vm_diff.py failed"; fi

if [ "$DIFF_RC" = 2 ]; then
  if [ "$FORCE" = 1 ]; then
    log "DANGER files present but --force-overwrite-vm-only given: continuing (VM copies stay in the backup)"
  else
    echo
    echo "[deploy] ABORTED: the VM has work that is not in $REF. Nothing was changed."
    echo "[deploy] Save it first (commands above), merge it, and re-run; or pass --accept PATH for"
    echo "[deploy] files you are sure can be discarded; or --force-overwrite-vm-only to override all."
    exit 2
  fi
fi

TS="$(date -u +%Y%m%d-%H%M%S)"
BACKUP="$BASE/backup-app-$TS-$SHORT.tgz"

if [ "$SEED" = 1 ]; then
  [ "$DIFF_RC" = 0 ] || [ "$FORCE" = 1 ] || exit 2
  [ ! -s "$WORK/lists/changed.txt" ] || die "cannot seed: the VM differs from $REF (see build_only/new_in_build above)"
  [ "$DRY" = 0 ] || { log "dry-run: would record DEPLOYED_REF=$FULL"; exit 0; }
  printf '%s %s\n' "$FULL" "$REF" | "${SSH[@]}" "sudo -u salesgod tee $BASE/DEPLOYED_REF >/dev/null"
  "${SSH[@]}" "sudo -u salesgod tee $BASE/DEPLOYED_MANIFEST >/dev/null" < "$WORK/vm.txt"
  log "seeded DEPLOYED_REF=$FULL (VM files untouched, nothing restarted)"
  exit 0
fi

# ---------------------------------------------------------------- dry run stops here
if [ "$DRY" = 1 ]; then
  echo
  if why=$(call_live); then log "live call right now: $why (a real deploy would wait up to ${LIVE_TIMEOUT}s)"
  else log "no live call right now"; fi
  log "DRY RUN - would back up to $BACKUP"
  log "DRY RUN - would extract $(wc -l < "$WORK/build.txt") files as salesgod, restart $SVC, verify, roll back on failure"
  log "DRY RUN - nothing was changed on the VM."
  exit "$DIFF_RC"
fi

# ---------------------------------------------------------------- real deploy
log "waiting for the line to be quiet (timeout ${LIVE_TIMEOUT}s)..."
why=$(wait_idle "$LIVE_TIMEOUT") || die "a call is still live after ${LIVE_TIMEOUT}s ($why); nothing changed"

log "backing up $APP (excluding data/) to $BACKUP"
"${SSH[@]}" "sudo tar czf $BACKUP --exclude='app/data' -C $BASE app && sudo test -s $BACKUP" || die "backup failed; nothing changed"

log "uploading build"
scp -q -o BatchMode=yes "$WORK/build.tar" "$HOST:/tmp/salesgod-build-$TS.tar" || die "upload failed; nothing changed"

restore() {  # $1 = restart? (1/0)
  log "ROLLBACK: restoring $BACKUP"
  if [ -s "$WORK/lists/added.txt" ]; then
    "${SSH[@]}" "cd $APP && sudo -u salesgod xargs -d '\n' -r rm -f --" < "$WORK/lists/added.txt"
  fi
  "${SSH[@]}" "sudo -u salesgod tar xzf $BACKUP -C $BASE" || { echo "[deploy] ROLLBACK FAILED - restore by hand from $BACKUP" >&2; return 1; }
  if [ "$1" = 1 ]; then
    "${SSH[@]}" "sudo systemctl restart $SVC"
    log "rolled back and restarted"
  else
    log "rolled back (service was never restarted)"
  fi
}

log "extracting as salesgod"
"${SSH[@]}" "sudo -u salesgod tar xf /tmp/salesgod-build-$TS.tar -C $APP --no-same-owner --no-same-permissions --touch; rc=\$?; rm -f /tmp/salesgod-build-$TS.tar; exit \$rc" \
  || { restore 0; die "extract failed"; }

if ! why=$(wait_idle "$LIVE_TIMEOUT"); then
  restore 0; die "a call started during the deploy and is still live ($why); files rolled back, no restart"
fi

T0="$("${SSH[@]}" 'date "+%Y-%m-%d %H:%M:%S"')"
log "restarting $SVC (since $T0)"
"${SSH[@]}" "sudo systemctl restart $SVC" || { restore 1; die "restart failed"; }

FAIL=""
ready=0; t0=$SECONDS
while [ $((SECONDS - t0)) -lt "$READY_TIMEOUT" ]; do
  if "${SSH[@]}" "sudo journalctl -u $SVC --since '$T0' --no-pager | grep -q 'voice.server ready:'"; then ready=1; break; fi
  sleep 5
done
[ "$ready" = 1 ] || FAIL="no 'ready:' log line within ${READY_TIMEOUT}s"

http_check() {  # path, needle
  local out code
  out=$("${SSH[@]}" "curl -s -m 15 -w '\n%{http_code}' http://127.0.0.1:$PORT$1") || return 1
  code="${out##*$'\n'}"
  [ "$code" = 200 ] || { echo "GET $1 returned $code"; return 1; }
  if [ -n "${2:-}" ] && ! grep -qi "$2" <<<"$out"; then echo "GET $1 is missing '$2'"; return 1; fi
  return 0
}
if [ -z "$FAIL" ]; then FAIL="$(http_check / '<title>[^<]*SalesGod Console')" || true; fi
if [ -z "$FAIL" ]; then FAIL="$(http_check /classic)" || true; fi
if [ -z "$FAIL" ]; then
  tb=$("${SSH[@]}" "sudo journalctl -u $SVC --since '$T0' --no-pager | grep -c Traceback" || true)
  [ "${tb:-0}" = 0 ] || FAIL="$tb Traceback line(s) in the journal since restart"
fi

if [ -n "$FAIL" ]; then
  echo "[deploy] VERIFY FAILED: $FAIL" >&2
  restore 1
  exit 1
fi

# ---------------------------------------------------------------- record success
vm_manifest > "$WORK/after.txt" || die "deployed OK but could not read the VM manifest to record it"
printf '%s %s\n' "$FULL" "$REF" | "${SSH[@]}" "sudo -u salesgod tee $BASE/DEPLOYED_REF >/dev/null"
"${SSH[@]}" "sudo -u salesgod tee $BASE/DEPLOYED_MANIFEST >/dev/null" < "$WORK/after.txt"
"${SSH[@]}" "echo '$(date -u +%FT%TZ) $FULL $REF backup=$BACKUP' | sudo -u salesgod tee -a $BASE/DEPLOYED_LOG >/dev/null"

echo
log "SUCCESS: $REF ($SHORT) is live"
log "  files changed: $(wc -l < "$WORK/lists/changed.txt") (+ $(wc -l < "$WORK/lists/added.txt") new)"
log "  overwritten VM-only work: $(wc -l < "$WORK/lists/danger.txt") file(s) (kept in the backup)"
log "  backup: $BACKUP"
log "  rollback by hand: sudo -u salesgod tar xzf $BACKUP -C $BASE && sudo systemctl restart $SVC"
