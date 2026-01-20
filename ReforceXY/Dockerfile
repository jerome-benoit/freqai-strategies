FROM freqtradeorg/freqtrade:stable_freqairl

ARG optuna_version=4.7.0

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
RUN pip install --user --no-cache-dir \
    optuna==${optuna_version} \
    optuna-dashboard \
    optunahub \
    -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-reforcexy" \
      maintainer="auto-generated"
