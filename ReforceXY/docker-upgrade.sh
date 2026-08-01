#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
FREQTRADE_CONFIG="${FREQTRADE_CONFIG:-${SCRIPT_DIR}/user_data/config.json}"
LOCAL_DOCKER_IMAGE="${LOCAL_DOCKER_IMAGE:-reforcexy-freqtrade}"
REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-freqtradeorg/freqtrade:stable_freqairl}"

################################

echo_timestamped() {
  printf '%s - %s\n' "$(date +"%Y-%m-%d %H:%M:%S")" "$*"
}

is_pid_running() {
  _pid="$1"
  [ -n "$_pid" ] && kill -0 "$_pid" 2>/dev/null
}

create_lock() {
  _dir="$1"
  umask 077
  if command mkdir "$_dir" 2>/dev/null; then
    if ! printf '%d\n' "$$" >"$_dir/pid"; then
      rm -rf "$_dir" 2>/dev/null || true
      return 1
    fi
    return 0
  fi
  return 1
}

LOCK_TAG=$(printf '%s' "$LOCAL_DOCKER_IMAGE" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')
LOCKDIR="${TMPDIR:-/tmp}/docker-upgrade.${LOCK_TAG}.lock.d"

if [ -d "$LOCKDIR" ]; then
  _oldpid=$(command sed -n '1p' "$LOCKDIR/pid" 2>/dev/null | tr -cd '0-9' || true)
  if [ -n "$_oldpid" ] && is_pid_running "$_oldpid"; then
    echo_timestamped "Error: already running for ${LOCAL_DOCKER_IMAGE} (pid ${_oldpid})"
    exit 1
  fi
  echo_timestamped "Warning: removing stale lock ${LOCKDIR} (pid ${_oldpid:-unknown})"
  rm -rf "$LOCKDIR" || true
fi

trap 'rm -rf "$LOCKDIR"' 0
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 131' QUIT

if ! create_lock "$LOCKDIR"; then
  echo_timestamped "Error: already running for ${LOCAL_DOCKER_IMAGE}"
  exit 1
fi

jsonc_to_json() {
  awk '
    BEGIN{in_str=0;esc=0;in_block=0;have_prev=0}
    {
      line=$0; len=length(line); i=1; out="";
      sub(/\r$/, "", line)
      while(i<=len){
        c=substr(line,i,1);
        nextc = (i<len)?substr(line,i+1,1):"\n";
        if(in_block){
          if(c=="*" && nextc=="/"){ in_block=0; i+=2; }
          else { i++; }
          continue;
        }
        if(!in_str){
          if(c=="/" && nextc=="/"){ break; }
          if(c=="/" && nextc=="*"){ in_block=1; i+=2; continue; }
          if(c=="\""){ in_str=1; out=out c; i++; continue; }
          out=out c; i++;
        } else {
          out=out c;
          if(esc){ esc=0; }
          else if(c=="\\"){ esc=1; }
          else if(c=="\""){ in_str=0; }
          i++;
        }
      }
      sub(/[[:space:]]+$/, "", out)
      if (out ~ /^[[:space:]]*$/) next
      cur = out
      if (have_prev){
        sub(/,[[:space:]]*\}[[:space:]]*$/, "}", prev)
        sub(/,[[:space:]]*\][[:space:]]*$/, "]", prev)
        if (prev ~ /,[[:space:]]*$/ && cur ~ /^[[:space:]]*[}\]]/) {
          sub(/,[[:space:]]*$/, "", prev)
        }
        key = (cur ~ /^[[:space:]]*"[^"]+"[[:space:]]*:/)
        openval = (cur ~ /^[[:space:]]*[{[]/)
        strval = (cur ~ /^[[:space:]]*"/) && !(key)
        numval = (cur ~ /^[[:space:]]*-?[0-9]/)
        boolnull = (cur ~ /^[[:space:]]*(true|false|null)([[:space:]]|,|]|\}|$)/)
        prev_value_end = (prev ~ /[}\]][[:space:]]*$/) || (prev ~ /"[[:space:]]*$/) || (prev ~ /-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?[[:space:]]*$/) || (prev ~ /(true|false|null)[[:space:]]*$/)
        if (prev_value_end && (key || openval || strval || numval || boolnull)) {
          prev = prev ","
        }
        print prev
      }
      prev = cur
      have_prev=1
    }
    END {
      if (have_prev){
        sub(/,[[:space:]]*\}[[:space:]]*$/, "}", prev)
        sub(/,[[:space:]]*\][[:space:]]*$/, "]", prev)
        sub(/,[[:space:]]*$/, "", prev)
        print prev
      }
    }
  ' "$1" | jq -c '.'
}

short_digest() {
  _d="$1"
  if [ -z "$_d" ] || [ "$_d" = "none" ]; then
    printf '%s\n' "$_d"
    return 0
  fi
  case "$_d" in
    sha256:*) _h=${_d#sha256:} ;;
    *) _h="$_d" ;;
  esac
  printf '%s' "$_h" | LC_ALL=C command cut -c1-12
  printf '\n'
}

escape_telegram_markdown() {
  printf '%s' "$1" | \
  command sed \
    -e 's/\\\([][_*()~`>#+=|{}.!-]\)/MDV2ESC\1/g' | \
  command sed \
    -e 's/`\([^`]*\)`/MDV2COPEN\1MDV2CCLOSE/g' \
    -e 's/\[\([^]]*\)\](\([^)]*\))/MDV2LOPEN\1MDV2LMID\2MDV2LCLOSE/g' \
    -e 's/!\[\([^]]*\)\](\([^)]*\))/MDV2EOPEN\1MDV2EMID\2MDV2ECLOSE/g' \
    -e 's/__\([^_]*\)__/MDV2UOPEN\1MDV2UCLOSE/g' \
    -e 's/\*\([^*]*\)\*/MDV2BOPEN\1MDV2BCLOSE/g' \
    -e 's/_\([^_]*\)_/MDV2IOPEN\1MDV2ICLOSE/g' \
    -e 's/~\([^~]*\)~/MDV2SOPEN\1MDV2SCLOSE/g' \
    -e 's/||\([^|]*\)||/MDV2POPEN\1MDV2PCLOSE/g' | \
  command sed \
    -e 's/\\/\\\\/g' \
    -e 's/[][_*()~`>#+=|{}.!-]/\\&/g' | \
  command sed \
    -e 's/MDV2COPEN/`/g'      -e 's/MDV2CCLOSE/`/g' \
    -e 's/MDV2LOPEN/[/g'      -e 's/MDV2LMID/](/g'     -e 's/MDV2LCLOSE/)/g' \
    -e 's/MDV2EOPEN/!\[/g'    -e 's/MDV2EMID/](/g'     -e 's/MDV2ECLOSE/)/g' \
    -e 's/MDV2UOPEN/__/g'     -e 's/MDV2UCLOSE/__/g' \
    -e 's/MDV2BOPEN/*/g'      -e 's/MDV2BCLOSE/*/g' \
    -e 's/MDV2IOPEN/_/g'      -e 's/MDV2ICLOSE/_/g' \
    -e 's/MDV2SOPEN/~/g'      -e 's/MDV2SCLOSE/~/g' \
    -e 's/MDV2POPEN/||/g'     -e 's/MDV2PCLOSE/||/g' \
    -e 's/MDV2ESC\\\([][_*()~`>#+=|{}.!-]\)/\\\1/g'
}

send_telegram_message() {
  if ! command -v jq >/dev/null 2>&1; then
    echo_timestamped "Warning: jq not found, skipping telegram notification"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo_timestamped "Warning: curl not found, skipping telegram notification"
    return 0
  fi

  if [ -z "${FREQTRADE_CONFIG_JSON:-}" ]; then
    FREQTRADE_CONFIG_JSON=$(jsonc_to_json "$FREQTRADE_CONFIG" 2>/dev/null || printf '%s\n' '')
  fi
  printf '%s' "$FREQTRADE_CONFIG_JSON" | jq empty 2>/dev/null || { echo_timestamped "Warning: invalid JSON configuration, skipping telegram notification"; return 0; }

  freqtrade_telegram_enabled=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.enabled // "false"' 2>/dev/null || printf '%s\n' 'false')
  if [ "$freqtrade_telegram_enabled" = "false" ]; then
    return 0
  fi

  telegram_message=$(escape_telegram_markdown "$1")
  if [ -z "$telegram_message" ]; then
    echo_timestamped "Warning: message variable is empty, skipping telegram notification"
    return 0
  fi

  freqtrade_telegram_token=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.token // ""' 2>/dev/null || printf '%s\n' '')
  freqtrade_telegram_chat_id=$(printf '%s' "$FREQTRADE_CONFIG_JSON" | jq -r '.telegram.chat_id // ""' 2>/dev/null || printf '%s\n' '')
  if [ -n "$freqtrade_telegram_token" ] && [ -n "$freqtrade_telegram_chat_id" ]; then
    set +e
    curl_error=$({ command curl -sS --max-time 10 -X POST \
      --data-urlencode "text=${telegram_message}" \
      --data-urlencode "parse_mode=MarkdownV2" \
      --data "chat_id=${freqtrade_telegram_chat_id}" \
      "https://api.telegram.org/bot${freqtrade_telegram_token}/sendMessage" 1>/dev/null; } 2>&1)
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
      echo_timestamped "Warning: failed to send telegram message: $curl_error"
      return 0
    fi
  fi
}

if ! command -v docker >/dev/null 2>&1; then
  echo_timestamped "Error: docker not found in PATH"
  exit 1
fi

if [ ! -f "${SCRIPT_DIR}/docker-compose.yml" ] && [ ! -f "${SCRIPT_DIR}/docker-compose.yaml" ]; then
  echo_timestamped "Error: docker-compose.yml or docker-compose.yaml file not found in ${SCRIPT_DIR}"
  exit 1
fi

if [ ! -f "$FREQTRADE_CONFIG" ]; then
  echo_timestamped "Error: ${FREQTRADE_CONFIG} file not found"
  exit 1
fi

echo_timestamped "Info: docker image pull for ${REMOTE_DOCKER_IMAGE}"
local_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || printf '%s\n' 'none')
if ! command docker image pull --quiet "$REMOTE_DOCKER_IMAGE" >/dev/null 2>&1; then
  echo_timestamped "Error: docker image pull failed for ${REMOTE_DOCKER_IMAGE}"
  exit 1
fi
remote_digest=$(command docker image inspect --format='{{.Id}}' "$REMOTE_DOCKER_IMAGE" 2>/dev/null || printf '%s\n' 'none')

rebuild_local_image=false
if [ "$local_digest" != "$remote_digest" ]; then
  rebuild_local_image=true
  short_local_digest=$(short_digest "$local_digest")
  short_remote_digest=$(short_digest "$remote_digest")
  message="docker image ${REMOTE_DOCKER_IMAGE} was updated (${short_local_digest} -> ${short_remote_digest})"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
else
  echo_timestamped "Info: docker image ${REMOTE_DOCKER_IMAGE} is up to date"
fi

if [ "$rebuild_local_image" = true ]; then
  message="rebuilding and restarting docker image ${LOCAL_DOCKER_IMAGE}"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
  cd -- "$SCRIPT_DIR" || exit 1
  compose_image=$(command docker compose config --images | command sed -n '1p')
  if [ -z "$compose_image" ]; then
    echo_timestamped "Error: ReforceXY image name was not resolved"
    exit 1
  fi
  previous_runtime_image_id=$(
    command docker image inspect --format='{{.Id}}' "$compose_image" 2>/dev/null ||
      printf '%s\n' 'none'
  )
  if [ "$previous_runtime_image_id" != "none" ]; then
    archive_tag="${compose_image}-archive:${previous_runtime_image_id#sha256:}"
    if ! command docker image tag "$previous_runtime_image_id" "$archive_tag"; then
      echo_timestamped "Error: failed to archive ${previous_runtime_image_id}"
      exit 1
    fi
    echo_timestamped "Info: archived previous runtime image as ${archive_tag}"
  fi
  if ! command "${SCRIPT_DIR}/docker-compose.sh" pull --quiet >/dev/null 2>&1; then
    echo_timestamped "Warning: docker compose pull failed"
  fi
  if ! command "${SCRIPT_DIR}/docker-compose.sh" --progress quiet down; then
    echo_timestamped "Error: docker compose down failed"
    exit 1
  fi
  if ! command docker image rm "$compose_image" >/dev/null 2>&1; then
    echo_timestamped "Warning: docker image rm failed for ${compose_image}"
  fi
  if ! command "${SCRIPT_DIR}/docker-compose.sh" --progress quiet up -d --build; then
    echo_timestamped "Error: docker compose up failed"
    exit 1
  fi
  message="rebuilt and restarted docker image ${LOCAL_DOCKER_IMAGE}"
  echo_timestamped "Info: $message"
  send_telegram_message "$message"
  echo_timestamped "Info: pruning unused docker images"
  command docker image prune -f >/dev/null 2>&1 || true
else
  echo_timestamped "Info: no rebuild and restart needed for docker image ${LOCAL_DOCKER_IMAGE}"
fi

exit 0
