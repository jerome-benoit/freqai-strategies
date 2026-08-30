FROM freqtradeorg/freqtrade:stable_freqai AS runtime

ARG optuna_version=4.9.0
ARG optuna_dashboard_version=0.20.0
ARG optunahub_version=0.4.1
ARG scikit_learn_extra_version=0.3.0
ARG scikit_image_version=0.26.0
ARG ngboost_version=0.5.11
ARG catboost_version=1.2.10

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
RUN pip install --user --no-cache-dir \
    "pandas>=3.0" \
    optuna==${optuna_version} \
    optuna-integration==${optuna_version} \
    optuna-dashboard==${optuna_dashboard_version} \
    optunahub==${optunahub_version} \
    -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt \
    scikit-learn-extra==${scikit_learn_extra_version} \
    scikit-image==${scikit_image_version} \
    ngboost==${ngboost_version} \
    catboost==${catboost_version}

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-quickadapter" \
      maintainer="auto-generated"

FROM runtime AS qa
ENV FREQAI_QUALITY_PROJECT=quickadapter
COPY --chown=ftuser:ftuser .devcontainer/requirements-dev.txt /tmp/requirements-dev.txt
RUN pip install --user --no-cache-dir -r /tmp/requirements-dev.txt \
 && python -m basedpyright --version \
 && rm /tmp/requirements-dev.txt

FROM runtime AS production
