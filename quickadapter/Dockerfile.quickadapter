FROM freqtradeorg/freqtrade:stable_freqai

ARG optuna_version=4.6.0
ARG scikit_learn_extra_version=0.3.0
ARG scikit_image_version=0.25.2

USER root
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential git \
 && rm -rf /var/lib/apt/lists/*
USER ftuser
RUN pip install --user --no-cache-dir \
    optuna==${optuna_version} \
    optuna-integration==${optuna_version} \
    optuna-dashboard \
    optunahub \
    -r https://hub.optuna.org/samplers/auto_sampler/requirements.txt \
    scikit-learn-extra==${scikit_learn_extra_version} \
    scikit-image==${scikit_image_version}

LABEL org.opencontainers.image.source="freqai-strategies" \
      org.opencontainers.image.title="freqtrade-quickadapter" \
      maintainer="auto-generated"
