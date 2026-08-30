ARG freqtrade_image=freqtradeorg/freqtrade@sha256:e808ec0d6c14cf5a2ea8406093b05e8b9230ac363ebdac573101988cd1450410
FROM ${freqtrade_image} AS runtime

ARG optuna_version=4.9.0
ARG optuna_dashboard_version=0.20.0
ARG optunahub_version=0.4.1
ARG cmaes_version=0.13.0
ARG freqtrade_image
ARG scipy_version=1.17.1
ARG matplotlib_version=3.11.1
ARG gymnasium_version=1.3.0
ARG sb3_contrib_version=2.9.0
ARG stable_baselines3_version=2.9.0

ENV MPLBACKEND=Agg \
    REFORCEXY_FREQTRADE_IMAGE=${freqtrade_image}

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libatomic1 \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
COPY --chown=ftuser:ftuser runtime-requirements.txt /tmp/reforcexy-runtime-requirements.txt
RUN pip install --user --no-cache-dir \
    cmaes==${cmaes_version} \
    optuna==${optuna_version} \
    optuna-dashboard==${optuna_dashboard_version} \
    optunahub==${optunahub_version} \
 && python -m pip install --user --no-cache-dir --no-deps \
    --only-binary=:all: --require-hashes \
    -r /tmp/reforcexy-runtime-requirements.txt \
 && python -c "from importlib import metadata; from pathlib import Path; import contourpy, cycler, fontTools, gymnasium, kiwisolver, matplotlib, pyparsing, sb3_contrib, scipy, stable_baselines3; lock=Path('/tmp/reforcexy-runtime-requirements.txt'); expected={line.split('==',1)[0]:line.split('==',1)[1].split()[0] for line in lock.read_text().splitlines() if '==' in line}; assert all(metadata.version(name)==version for name,version in expected.items()); modules=(contourpy,cycler,fontTools,gymnasium,kiwisolver,matplotlib,pyparsing,sb3_contrib,scipy,stable_baselines3); assert all('/home/ftuser/.local/' in str(Path(module.__file__).resolve()) and '/site-packages/' in str(Path(module.__file__).resolve()) for module in modules)" \
 && python -m pip check

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-reforcexy" \
      maintainer="auto-generated"

FROM runtime AS qa
ENV FREQAI_QUALITY_PROJECT=reforcexy
COPY --chown=ftuser:ftuser .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements-dev.txt \
 && python -m basedpyright --version \
 && rm /tmp/requirements-dev.txt

FROM runtime AS production
ARG freqtrade_image
ARG reforcexy_git_commit
ARG runtime_requirements_sha256
ARG scipy_version=1.17.1
ARG matplotlib_version=3.11.1
ARG gymnasium_version=1.3.0
ARG sb3_contrib_version=2.9.0
ARG stable_baselines3_version=2.9.0
RUN printf '%s' "${reforcexy_git_commit}" | grep -Eq '^[0-9a-f]{40}$' \
 && printf '%s' "${runtime_requirements_sha256}" | grep -Eq '^[0-9a-f]{64}$' \
 && printf '%s  %s\n' "${runtime_requirements_sha256}" /tmp/reforcexy-runtime-requirements.txt | sha256sum -c -
ENV REFORCEXY_GIT_COMMIT=${reforcexy_git_commit} \
    REFORCEXY_RUNTIME_REQUIREMENTS_SHA256=${runtime_requirements_sha256}
LABEL org.opencontainers.image.revision="${reforcexy_git_commit}" \
      org.opencontainers.image.base.name="${freqtrade_image}" \
      io.github.freqai-strategies.reforcexy.gymnasium="${gymnasium_version}" \
      io.github.freqai-strategies.reforcexy.matplotlib="${matplotlib_version}" \
      io.github.freqai-strategies.reforcexy.runtime-requirements-sha256="${runtime_requirements_sha256}" \
      io.github.freqai-strategies.reforcexy.sb3-contrib="${sb3_contrib_version}" \
      io.github.freqai-strategies.reforcexy.scipy="${scipy_version}" \
      io.github.freqai-strategies.reforcexy.stable-baselines3="${stable_baselines3_version}"
