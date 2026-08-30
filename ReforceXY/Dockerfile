FROM freqtradeorg/freqtrade:stable_freqairl AS runtime

ARG optuna_version=4.9.0
ARG optuna_dashboard_version=0.20.0
ARG optunahub_version=0.4.1
ARG matplotlib_version=3.11.1

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
RUN pip install --user --no-cache-dir \
    optuna==${optuna_version} \
    optuna-dashboard==${optuna_dashboard_version} \
    optunahub==${optunahub_version} \
    -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt \
    matplotlib==${matplotlib_version}

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-reforcexy" \
      maintainer="auto-generated"

FROM runtime AS qa

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends libatomic1 \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
COPY --chown=ftuser:ftuser .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements-dev.txt \
 && python -m basedpyright --version \
 && rm /tmp/requirements-dev.txt

FROM runtime AS production
