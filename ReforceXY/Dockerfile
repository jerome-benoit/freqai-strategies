ARG freqtrade_image=freqtradeorg/freqtrade@sha256:9caa7cb0a8a13f9639ad3980114f3af0ffbd896d43736eeb65f8d0db8c009093
FROM ${freqtrade_image}

ARG optuna_version=4.9.0
ARG optuna_dashboard_version=0.20.0
ARG optunahub_version=0.4.0
ARG cmaes_version=0.13.0
ARG freqtrade_image

ENV REFORCEXY_FREQTRADE_IMAGE=${freqtrade_image}

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
RUN pip install --user --no-cache-dir \
    cmaes==${cmaes_version} \
    optuna==${optuna_version} \
    optuna-dashboard==${optuna_dashboard_version} \
    optunahub==${optunahub_version}

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-reforcexy" \
      maintainer="auto-generated"
