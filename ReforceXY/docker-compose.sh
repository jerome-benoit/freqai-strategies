#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPOSITORY_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-freqtradeorg/freqtrade:stable_freqairl}"

if ! command -v git >/dev/null 2>&1; then
  printf '%s\n' "Error: git not found in PATH" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  printf '%s\n' "Error: docker not found in PATH" >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null 2>&1; then
  printf '%s\n' "Error: sha256sum not found in PATH" >&2
  exit 1
fi
if [ "$#" -eq 0 ]; then
  printf '%s\n' "Usage: ./docker-compose.sh <docker compose arguments>" >&2
  exit 2
fi

case " $* " in
  *" exec "*)
    printf '%s\n' \
      "Error: docker compose exec can reuse stale runtime metadata; use run or recreate the service" >&2
    exit 1
    ;;
  *" start "* | *" restart "* | *" unpause "* | *" --no-recreate "*)
    printf '%s\n' \
      "Error: this command can reuse stale ReforceXY code; use run or up to recreate when needed" >&2
    exit 1
    ;;
esac

git_commit=$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" rev-parse --verify HEAD)
if [ -n "$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" status --porcelain --untracked-files=normal -- ReforceXY)" ]; then
  printf '%s\n' "Error: ReforceXY files are dirty; commit them before a reproducible run" >&2
  exit 1
fi

refresh_image=false
case " $* " in
  *" --build "* | *" build "*)
    refresh_image=true
    docker image pull --quiet "$REMOTE_DOCKER_IMAGE" >/dev/null
    ;;
esac

freqtrade_image=$(
  docker image inspect --format='{{range .RepoDigests}}{{println .}}{{end}}' \
    "$REMOTE_DOCKER_IMAGE" 2>/dev/null |
    command sed -n '1p'
)
if [ -z "$freqtrade_image" ]; then
  printf '%s\n' "Error: no immutable digest found for ${REMOTE_DOCKER_IMAGE}" >&2
  printf '%s\n' "Run: docker image pull ${REMOTE_DOCKER_IMAGE}" >&2
  exit 1
fi

export REFORCEXY_FREQTRADE_IMAGE="$freqtrade_image"
export REFORCEXY_GIT_COMMIT="$git_commit"
REFORCEXY_MODEL_SOURCE_SHA256=$(
  sha256sum "${SCRIPT_DIR}/user_data/freqaimodels/ReforceXY.py" | command cut -d' ' -f1
)
REFORCEXY_MANIFEST_SOURCE_SHA256=$(
  sha256sum "${SCRIPT_DIR}/user_data/freqaimodels/reproducibility.py" |
    command cut -d' ' -f1
)
export REFORCEXY_MODEL_SOURCE_SHA256
export REFORCEXY_MANIFEST_SOURCE_SHA256
cd -- "$SCRIPT_DIR"

needs_runtime_image=false
case " $* " in
  *" create "* | *" run "* | *" up "*) needs_runtime_image=true ;;
esac
if [ "$needs_runtime_image" = true ]; then
  compose_image=$(docker compose config --images | command sed -n '1p')
  if [ -z "$compose_image" ]; then
    printf '%s\n' "Error: ReforceXY image name was not resolved" >&2
    exit 1
  fi
  if [ "$refresh_image" = true ] || ! docker image inspect "$compose_image" >/dev/null 2>&1; then
    docker compose build freqtrade
  fi
  REFORCEXY_RUNTIME_IMAGE_ID=$(
    docker image inspect --format='{{.Id}}' "$compose_image"
  )
  export REFORCEXY_RUNTIME_IMAGE_ID
fi

exec docker compose "$@"
