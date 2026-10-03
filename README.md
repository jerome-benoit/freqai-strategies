# FreqAI strategies

Two Freqtrade strategies with different decision models:

| Strategy     | Decision model                                                                           | Supplied profile                                                | Operator guide                         |
| ------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------------- | -------------------------------------- |
| QuickAdapter | Regress smoothed Zigzag morphology; enter at calibrated extrema with price confirmation. | Dry-run, spot, 5m; XGBoost; HPO enabled.                        | [QuickAdapter](quickadapter/README.md) |
| ReforceXY    | Reinforcement-learning policy chooses neutral, entry and exit actions.                   | Dry-run, spot, 5m; MaskablePPO; frame stacking and HPO enabled. | [ReforceXY](ReforceXY/README.md)       |

These profiles are starting configurations, not evidence of profitability.
Training losses and episode rewards are not portfolio performance measures.

## Documentation

- [QuickAdapter operation and configuration](quickadapter/README.md): decisions,
  prediction warmup, model selection, continuation and all strategy tunables.
- [ReforceXY operation and configuration](ReforceXY/README.md): actions, model and
  policy compatibility, masking, HPO, continuation and observation constraints.
- [Evaluation protocol](docs/evaluation.md): chronological comparisons, costs,
  economic outcomes, uncertainty and limits of native backtesting.
- [Reward-space analysis](ReforceXY/reward_space_analysis/README.md): synthetic
  reward diagnostics and trusted real-data imports; not strategy profitability.
- [Contribution guide](CONTRIBUTING.md): matching runtime images, coverage, lint,
  types and [analysis-suite maintenance](ReforceXY/reward_space_analysis/tests/README.md).

## Start safely

Follow the relevant guide's quick start. Both supplied configurations enable
`dry_run` and disable the API server. Review exchange, pair universe, sizing,
protections and costs before changing trading mode or disabling dry-run. Private
exchange credentials are needed only for operations that require them; never
commit secrets or the working `user_data/config.json`.

Before enabling the API, replace its public template username, password, JWT
secret and WebSocket token. Compose binds the exposed API port to localhost;
retain that restriction unless access is protected by a VPN or SSH tunnel.
Review the Compose timezone for your location.

The builds follow moving Freqtrade `stable_freqai` / `stable_freqairl` tags and
resolve dependencies at build time. Record resolved image digests and dependency
versions for reproducible evaluations.

## Common workflows

Run Compose commands from the selected strategy directory, using the same
Compose project name and environment as when the stack was started.

```shell
docker compose ps
docker compose exec freqtrade /bin/sh
docker compose logs -f freqtrade
```

`freqtrade` is the service name in both supplied stacks. The container names are
`freqtrade-quickadapter` and `freqtrade-ReforceXY`; for example:

```shell
docker logs -f freqtrade-quickadapter
```

`docker compose down` stops the bot and removes its containers. It does not
close positions or necessarily cancel exchange orders. Plan maintenance around
open exposure and back up configuration, database, models and HPO state.

## Image updates

The updater must be copied into the selected strategy directory: it locates
Compose and the default configuration relative to itself. Its built-in image
values are ReforceXY-specific; QuickAdapter requires a different remote base.

### QuickAdapter

From the repository root:

```shell
cd quickadapter
cp ../scripts/docker-upgrade.sh .
LOCAL_DOCKER_IMAGE="$(docker compose config --images)" \
REMOTE_DOCKER_IMAGE=freqtradeorg/freqtrade:stable_freqai \
./docker-upgrade.sh
```

### ReforceXY

From the repository root:

```shell
cd ReforceXY
cp ../scripts/docker-upgrade.sh .
LOCAL_DOCKER_IMAGE="$(docker compose config --images)" \
REMOTE_DOCKER_IMAGE=freqtradeorg/freqtrade:stable_freqairl \
./docker-upgrade.sh
```

The shipped stacks each resolve one local image: `quickadapter-freqtrade` and
`reforcexy-freqtrade` with their default project names. Deriving it from Compose
also handles `COMPOSE_PROJECT_NAME` overrides. With a customized multi-image
stack, inspect `docker compose config --images` and set only the `freqtrade`
service's image. Keep the same project/environment for the updater.

| Variable              | Built-in fallback                          | Meaning                                                                                |
| --------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------- |
| `FREQTRADE_CONFIG`    | `<script directory>/user_data/config.json` | Configuration used for optional Telegram notifications; override for a different path. |
| `LOCAL_DOCKER_IMAGE`  | `reforcexy-freqtrade`                      | Local Compose image to archive and remove before recreation.                           |
| `REMOTE_DOCKER_IMAGE` | `freqtradeorg/freqtrade:stable_freqairl`   | Base-image tag whose image ID is checked for updates.                                  |

When that remote tag changes, the script stops the stack, archives the previous
local image, attempts to remove its current tag, recreates the stack and prunes
unused images. This introduces downtime. Read warnings: a failed image removal
can leave the old image available to Compose. The script does not detect source
or dependency updates when the remote image ID is unchanged. For an explicit
rebuild, use `docker compose build --pull`, then `docker compose up -d`.

The latest successful archive tag is logged as
`<local image repository>-archive:<previous image ID without sha256:>`; older
archives are removed. Record it before relying on rollback. Stop the affected
stack, retag that archived image to its resolved local image name, then use
`docker compose up -d --no-build`. An image rollback does not undo trades,
configuration changes or database/model migrations; retain compatible state
backups and inspect exposure before restarting.

For example, a daily QuickAdapter check at 03:00 (after creating
`user_data/logs` and adapting the absolute checkout path):

```cron
0 3 * * * cd /path/to/freqai-strategies/quickadapter && LOCAL_DOCKER_IMAGE="$(docker compose config --images)" REMOTE_DOCKER_IMAGE=freqtradeorg/freqtrade:stable_freqai ./docker-upgrade.sh >> user_data/logs/docker-upgrade.log 2>&1
```

For ReforceXY, use its directory and `stable_freqairl`. Schedule automated
restarts only when their downtime and open-position risk are acceptable.

## Note

> Do not expect any support of any kind on the Internet. Nevertheless, PRs
> implementing documentation, bug fixes, cleanups or sensible features will be
> discussed and might get merged.
