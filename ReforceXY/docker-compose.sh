#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
REPOSITORY_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
WRAPPER_PATH="${SCRIPT_DIR}/docker-compose.sh"
COMPOSE_FILE_PATH="${SCRIPT_DIR}/docker-compose.yml"
RUNTIME_COMPOSE_FILE_PATH="${SCRIPT_DIR}/docker-compose.runtime.yml"
REMOTE_DOCKER_IMAGE="${REMOTE_DOCKER_IMAGE:-freqtradeorg/freqtrade:stable_freqairl}"
REFORCEXY_GYMNASIUM_VERSION=1.3.0
REFORCEXY_MATPLOTLIB_VERSION=3.11.0
REFORCEXY_SB3_CONTRIB_VERSION=2.9.0
REFORCEXY_SCIPY_VERSION=1.18.0
REFORCEXY_STABLE_BASELINES3_VERSION=2.9.0

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

policy_error() {
  printf '%s\n' "Error: $1 is not supported by the ReforceXY provenance wrapper" >&2
  exit 1
}

syntax_error() {
  printf '%s\n' "Error: $1" >&2
  exit 2
}

forbidden_compose_environment=
if [ -n "${COMPOSE_FILE:-}" ]; then
  forbidden_compose_environment=COMPOSE_FILE
elif [ -n "${COMPOSE_ENV_FILES:-}" ]; then
  forbidden_compose_environment=COMPOSE_ENV_FILES
elif [ -n "${COMPOSE_PROJECT_NAME:-}" ]; then
  forbidden_compose_environment=COMPOSE_PROJECT_NAME
elif [ -n "${COMPOSE_PROFILES:-}" ]; then
  forbidden_compose_environment=COMPOSE_PROFILES
elif [ -n "${COMPOSE_PATH_SEPARATOR:-}" ]; then
  forbidden_compose_environment=COMPOSE_PATH_SEPARATOR
fi
if [ -n "$forbidden_compose_environment" ]; then
  policy_error "the ${forbidden_compose_environment} environment override"
fi
export COMPOSE_DISABLE_ENV_FILE=1
export COMPOSE_MENU=0

primary_command=
controlled_build=false
refresh_base=false
needs_runtime_image=false
parser_state=global
run_service_found=false
run_help=false
global_help=false
command_help=false
build_seen=false
build_value=false
no_build_seen=false
no_build_value=false
dry_run_seen=false
dry_run_value=false

parse_compose_arguments() {
  original_argc=$#
  scanned=0
  while [ "$scanned" -lt "$original_argc" ]; do
    argument=$1
    shift
    scanned=$((scanned + 1))
    keep_argument=true

    case "$parser_state" in
      global)
        case "$argument" in
          -f | -f?* | --file | --file=* | \
            --project-directory | --project-directory=* | \
            --env-file | --env-file=* | \
            -p | -p?* | --project-name | --project-name=* | \
            --profile | --profile=*)
            policy_error "a Compose configuration or identity override"
            ;;
          --compatibility | --all-resources)
            policy_error "a Compose compatibility or expanded-resource mode"
            ;;
          --dry-run)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=1 | --dry-run=t | --dry-run=T | --dry-run=TRUE | \
            --dry-run=true | --dry-run=True)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=0 | --dry-run=f | --dry-run=F | --dry-run=FALSE | \
            --dry-run=false | --dry-run=False)
            dry_run_seen=true
            dry_run_value=false
            keep_argument=false
            ;;
          --dry-run=* | --dry-run?*)
            syntax_error "invalid docker compose --dry-run boolean"
            ;;
          --ansi | --parallel | --progress)
            [ "$scanned" -lt "$original_argc" ] || \
              syntax_error "${argument} requires a value"
            option_value=$1
            shift
            scanned=$((scanned + 1))
            set -- "$@" "$argument" "$option_value"
            continue
            ;;
          --ansi=* | --parallel=* | --progress=*)
            ;;
          -h | --help | --version)
            global_help=true
            ;;
          --)
            syntax_error "-- before the primary Compose command is unsupported"
            ;;
          -*)
            syntax_error "unsupported global Compose option"
            ;;
          *)
            primary_command=$argument
            case "$argument" in
              exec)
                policy_error "docker compose exec because it can reuse stale runtime metadata"
                ;;
              start | restart | unpause)
                policy_error "a command that can reuse stale ReforceXY code"
                ;;
              watch | scale)
                policy_error "a command that can start an unsealed runtime"
                ;;
              run)
                parser_state=run_options
                needs_runtime_image=true
                ;;
              create | up)
                parser_state=command_arguments
                needs_runtime_image=true
                ;;
              build)
                parser_state=command_arguments
                refresh_base=true
                ;;
              *)
                parser_state=command_arguments
                ;;
            esac
            ;;
        esac
        ;;
      run_options)
        case "$argument" in
          -f | -f?* | --file | --file=* | \
            --project-directory | --project-directory=* | \
            --env-file | --env-file=* | \
            --project-name | --project-name=* | \
            --profile | --profile=* | \
            --compatibility | --all-resources)
            policy_error "a Compose configuration or identity override"
            ;;
          -e | -e?* | --env | --env=* | \
            --env-from-file | --env-from-file=* | \
            -v | -v?* | --volume | --volume=*)
            policy_error "a docker compose run environment or volume override"
            ;;
          --pull | --pull=*)
            policy_error "a docker compose run pull policy that can replace the verified runtime image"
            ;;
          --dry-run)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=1 | --dry-run=t | --dry-run=T | --dry-run=TRUE | \
            --dry-run=true | --dry-run=True)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=0 | --dry-run=f | --dry-run=F | --dry-run=FALSE | \
            --dry-run=false | --dry-run=False)
            dry_run_seen=true
            dry_run_value=false
            keep_argument=false
            ;;
          --dry-run=* | --dry-run?*)
            syntax_error "invalid docker compose run --dry-run boolean"
            ;;
          --entrypoint | --cap-add | --cap-drop | --label | --name | \
            --publish | --user | --workdir | \
            -l | -p | -u | -w)
            [ "$scanned" -lt "$original_argc" ] || \
              syntax_error "${argument} requires a value"
            option_value=$1
            shift
            scanned=$((scanned + 1))
            set -- "$@" "$argument" "$option_value"
            continue
            ;;
          --entrypoint=* | --cap-add=* | --cap-drop=* | --label=* | \
            --name=* | --publish=* | --user=* | --workdir=* | \
            -l?* | -p?* | -u?* | -w?*)
            ;;
          --build)
            build_seen=true
            build_value=true
            keep_argument=false
            ;;
          --build=1 | --build=t | --build=T | --build=TRUE | \
            --build=true | --build=True)
            build_seen=true
            build_value=true
            keep_argument=false
            ;;
          --build=0 | --build=f | --build=F | --build=FALSE | \
            --build=false | --build=False)
            build_seen=true
            build_value=false
            keep_argument=false
            ;;
          --build=* | --build?*)
            syntax_error "invalid docker compose run --build boolean"
            ;;
          -d | --detach | -i | --interactive | --no-deps | -T | --no-tty | \
            -q | --quiet | --quiet-build | --quiet-pull | --remove-orphans | \
            --rm | -P | --service-ports | --use-aliases)
            ;;
          -h | --help)
            run_help=true
            ;;
          --)
            [ "$scanned" -lt "$original_argc" ] || \
              syntax_error "docker compose run -- requires a service"
            service=$1
            shift
            scanned=$((scanned + 1))
            [ "$service" = freqtrade ] || \
              policy_error "a docker compose run service other than freqtrade"
            set -- "$@" "$argument" "$service"
            run_service_found=true
            parser_state=container_arguments
            continue
            ;;
          -*)
            syntax_error "unsupported docker compose run option"
            ;;
          *)
            [ "$argument" = freqtrade ] || \
              policy_error "a docker compose run service other than freqtrade"
            run_service_found=true
            parser_state=container_arguments
            ;;
        esac
        ;;
      command_arguments)
        case "$argument" in
          --)
            parser_state=container_arguments
            ;;
          --file | --file=* | --project-directory | --project-directory=* | \
            --env-file | --env-file=* | --project-name | --project-name=* | \
            --profile | --profile=* | --compatibility | --all-resources | \
            -p | -p?*)
            policy_error "a Compose configuration or identity override"
            ;;
          -f)
            [ "$primary_command" = logs ] || \
              policy_error "a Compose file override after the primary command"
            ;;
          -f?*)
            policy_error "an attached Compose file override after the primary command"
            ;;
          --no-recreate | --no-recreate=* | --watch | --watch=* | \
            --scale | --scale=*)
            policy_error "a command option that can reuse or start an unsealed runtime"
            ;;
          --pull | --pull=*)
            case "$primary_command" in
              create | up)
                policy_error "a command pull policy that can replace the verified runtime image"
                ;;
            esac
            ;;
          --dry-run)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=1 | --dry-run=t | --dry-run=T | --dry-run=TRUE | \
            --dry-run=true | --dry-run=True)
            dry_run_seen=true
            dry_run_value=true
            keep_argument=false
            ;;
          --dry-run=0 | --dry-run=f | --dry-run=F | --dry-run=FALSE | \
            --dry-run=false | --dry-run=False)
            dry_run_seen=true
            dry_run_value=false
            keep_argument=false
            ;;
          --dry-run=* | --dry-run?*)
            syntax_error "invalid docker compose ${primary_command} --dry-run boolean"
            ;;
          --build-arg | --builder | -m | --memory | --provenance | --sbom | \
            --ssh)
            case "$primary_command" in
              build)
                [ "$scanned" -lt "$original_argc" ] || \
                  syntax_error "${argument} requires a value"
                option_value=$1
                shift
                scanned=$((scanned + 1))
                set -- "$@" "$argument" "$option_value"
                continue
                ;;
            esac
            ;;
          --build-arg=* | --builder=* | -m?* | --memory=* | \
            --provenance=* | --sbom=* | --ssh=*)
            ;;
          --policy)
            case "$primary_command" in
              pull)
                [ "$scanned" -lt "$original_argc" ] || \
                  syntax_error "${argument} requires a value"
                option_value=$1
                shift
                scanned=$((scanned + 1))
                set -- "$@" "$argument" "$option_value"
                continue
                ;;
            esac
            ;;
          --policy=*)
            ;;
          --attach | --no-attach | --exit-code-from | -t | --timeout | \
            --wait-timeout)
            case "$primary_command" in
              up)
                [ "$scanned" -lt "$original_argc" ] || \
                  syntax_error "${argument} requires a value"
                option_value=$1
                shift
                scanned=$((scanned + 1))
                set -- "$@" "$argument" "$option_value"
                continue
                ;;
            esac
            ;;
          -h | --help)
            command_help=true
            ;;
          --build)
            case "$primary_command" in
              create | up)
                build_seen=true
                build_value=true
                keep_argument=false
                ;;
            esac
            ;;
          --build=1 | --build=t | --build=T | --build=TRUE | \
            --build=true | --build=True)
            case "$primary_command" in
              create | up)
                build_seen=true
                build_value=true
                keep_argument=false
                ;;
            esac
            ;;
          --build=0 | --build=f | --build=F | --build=FALSE | \
            --build=false | --build=False)
            case "$primary_command" in
              create | up)
                build_seen=true
                build_value=false
                keep_argument=false
                ;;
            esac
            ;;
          --build=* | --build?*)
            case "$primary_command" in
              create | up)
                syntax_error "invalid docker compose ${primary_command} --build boolean"
                ;;
            esac
            ;;
          --no-build)
            case "$primary_command" in
              create | up)
                no_build_seen=true
                no_build_value=true
                ;;
            esac
            ;;
          --no-build=1 | --no-build=t | --no-build=T | --no-build=TRUE | \
            --no-build=true | --no-build=True)
            case "$primary_command" in
              create | up)
                no_build_seen=true
                no_build_value=true
                ;;
            esac
            ;;
          --no-build=0 | --no-build=f | --no-build=F | --no-build=FALSE | \
            --no-build=false | --no-build=False)
            case "$primary_command" in
              create | up)
                no_build_seen=true
                no_build_value=false
                ;;
            esac
            ;;
          --no-build=* | --no-build?*)
            case "$primary_command" in
              create | up)
                syntax_error "invalid docker compose ${primary_command} --no-build boolean"
                ;;
            esac
            ;;
        esac
        ;;
      container_arguments)
        ;;
    esac

    if [ "$keep_argument" = true ]; then
      set -- "$@" "$argument"
    fi
  done

  if [ -z "$primary_command" ] && [ "$global_help" != true ]; then
    syntax_error "missing docker compose command"
  fi
  if [ "$primary_command" = run ] && \
    [ "$run_service_found" != true ] && [ "$run_help" != true ]; then
    syntax_error "docker compose run requires the freqtrade service"
  fi
  if [ "$needs_runtime_image" = true ] && [ "$build_seen" = true ]; then
    controlled_build=$build_value
    refresh_base=$build_value
  fi
  if [ "$global_help" != true ] && [ "$run_help" != true ] && \
    [ "$command_help" != true ] && [ "$build_value" = true ] && \
    [ "$no_build_seen" = true ] && \
    [ "$no_build_value" = true ]; then
    syntax_error "--build and --no-build cannot be combined"
  fi
  if [ "$global_help" != true ] && [ "$run_help" != true ] && \
    [ "$command_help" != true ] && [ "$dry_run_seen" = true ] && \
    [ "$dry_run_value" = true ]; then
    policy_error "Compose dry-run because wrapper-owned provenance effects cannot be simulated"
  fi

  execute_compose "$@"
}

execute_compose() {
if [ "$global_help" = true ] || [ "$run_help" = true ] || \
  [ "$command_help" = true ]; then
  exec docker compose \
    --file "$COMPOSE_FILE_PATH" \
    --project-directory "$SCRIPT_DIR" \
    "$@"
fi

git_commit=$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" rev-parse --verify HEAD)
if [ -n "$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" status --porcelain --untracked-files=normal -- ReforceXY)" ]; then
  printf '%s\n' "Error: ReforceXY files are dirty; commit them before a reproducible run" >&2
  exit 1
fi

if [ "$refresh_base" = true ]; then
  docker image pull --quiet "$REMOTE_DOCKER_IMAGE" >/dev/null
fi

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
REFORCEXY_RUNTIME_REQUIREMENTS_SHA256=$(
  sha256sum "${SCRIPT_DIR}/runtime-requirements.txt" | command cut -d' ' -f1
)
export REFORCEXY_MODEL_SOURCE_SHA256
export REFORCEXY_MANIFEST_SOURCE_SHA256
export REFORCEXY_RUNTIME_REQUIREMENTS_SHA256
MODEL_SOURCE_SHA256_AT_CHECK=$REFORCEXY_MODEL_SOURCE_SHA256
MANIFEST_SOURCE_SHA256_AT_CHECK=$REFORCEXY_MANIFEST_SOURCE_SHA256
RUNTIME_REQUIREMENTS_SHA256_AT_CHECK=$REFORCEXY_RUNTIME_REQUIREMENTS_SHA256
COMPOSE_FILE_SHA256_AT_CHECK=$(sha256sum "$COMPOSE_FILE_PATH" | command cut -d' ' -f1)
RUNTIME_COMPOSE_FILE_SHA256_AT_CHECK=$(
  sha256sum "$RUNTIME_COMPOSE_FILE_PATH" | command cut -d' ' -f1
)
WRAPPER_SHA256_AT_CHECK=$(sha256sum "$WRAPPER_PATH" | command cut -d' ' -f1)
cd -- "$SCRIPT_DIR"

if [ "$needs_runtime_image" = true ]; then
  compose_image=$(
    docker compose \
      --file "$COMPOSE_FILE_PATH" \
      --project-directory "$SCRIPT_DIR" \
      config --images | command sed -n '1p'
  )
  if [ -z "$compose_image" ]; then
    printf '%s\n' "Error: ReforceXY image name was not resolved" >&2
    exit 1
  fi
  image_matches_runtime() {
    image_reference=$1
    [ "$(docker image inspect --format='{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image_reference" 2>/dev/null)" = "$git_commit" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "org.opencontainers.image.base.name"}}' "$image_reference" 2>/dev/null)" = "$freqtrade_image" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.gymnasium"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_GYMNASIUM_VERSION" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.matplotlib"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_MATPLOTLIB_VERSION" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.runtime-requirements-sha256"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_RUNTIME_REQUIREMENTS_SHA256" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.sb3-contrib"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_SB3_CONTRIB_VERSION" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.scipy"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_SCIPY_VERSION" ] &&
      [ "$(docker image inspect --format='{{index .Config.Labels "io.github.freqai-strategies.reforcexy.stable-baselines3"}}' "$image_reference" 2>/dev/null)" = "$REFORCEXY_STABLE_BASELINES3_VERSION" ]
  }
  image_is_current=false
  if image_matches_runtime "$compose_image"; then
    image_is_current=true
  fi
  if [ "$no_build_seen" = true ] && [ "$no_build_value" = true ] && \
    [ "$image_is_current" != true ]; then
    printf '%s\n' \
      "Error: the ReforceXY image is absent or stale and --no-build forbids rebuilding it" >&2
    exit 1
  fi
  if [ "$controlled_build" = true ] || [ "$image_is_current" != true ]; then
    docker compose \
      --file "$COMPOSE_FILE_PATH" \
      --project-directory "$SCRIPT_DIR" \
      build freqtrade
  fi
  REFORCEXY_RUNTIME_IMAGE_ID=$(
    docker image inspect --format='{{.Id}}' "$compose_image"
  )
  if [ -z "$REFORCEXY_RUNTIME_IMAGE_ID" ] || \
    ! image_matches_runtime "$REFORCEXY_RUNTIME_IMAGE_ID"; then
    printf '%s\n' \
      "Error: ReforceXY image labels do not match the clean commit, immutable base, and locked runtime" >&2
    exit 1
  fi
  export REFORCEXY_RUNTIME_IMAGE_ID
fi

if [ "$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" rev-parse --verify HEAD)" != "$git_commit" ] || \
  [ -n "$(git -c core.fsmonitor=false -C "$REPOSITORY_ROOT" status --porcelain --untracked-files=normal -- ReforceXY)" ] || \
  [ "$(sha256sum "${SCRIPT_DIR}/user_data/freqaimodels/ReforceXY.py" | command cut -d' ' -f1)" != "$MODEL_SOURCE_SHA256_AT_CHECK" ] || \
  [ "$(sha256sum "${SCRIPT_DIR}/user_data/freqaimodels/reproducibility.py" | command cut -d' ' -f1)" != "$MANIFEST_SOURCE_SHA256_AT_CHECK" ] || \
  [ "$(sha256sum "${SCRIPT_DIR}/runtime-requirements.txt" | command cut -d' ' -f1)" != "$RUNTIME_REQUIREMENTS_SHA256_AT_CHECK" ] || \
  [ "$(sha256sum "$COMPOSE_FILE_PATH" | command cut -d' ' -f1)" != "$COMPOSE_FILE_SHA256_AT_CHECK" ] || \
  [ "$(sha256sum "$RUNTIME_COMPOSE_FILE_PATH" | command cut -d' ' -f1)" != "$RUNTIME_COMPOSE_FILE_SHA256_AT_CHECK" ] || \
  [ "$(sha256sum "$WRAPPER_PATH" | command cut -d' ' -f1)" != "$WRAPPER_SHA256_AT_CHECK" ]; then
  printf '%s\n' "Error: ReforceXY provenance inputs changed during wrapper execution" >&2
  exit 1
fi

if [ "$needs_runtime_image" = true ]; then
  exec docker compose \
    --file "$COMPOSE_FILE_PATH" \
    --file "$RUNTIME_COMPOSE_FILE_PATH" \
    --project-directory "$SCRIPT_DIR" \
    "$@"
fi

exec docker compose \
  --file "$COMPOSE_FILE_PATH" \
  --project-directory "$SCRIPT_DIR" \
  "$@"
}

parse_compose_arguments "$@"
