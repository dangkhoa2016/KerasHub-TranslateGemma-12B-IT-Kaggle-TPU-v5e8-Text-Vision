# Release Notes — v1.0.0

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE_NOTES_v1.0.0.vi.md)

## Overview

`v1.0.0` is the first public release of the TranslateGemma 12B IT text + vision REST server for Kaggle TPU v5e-8.

The release focuses on reproducible serving, clear operational contracts, safe packaging, and a GitHub-import-first Kaggle workflow.

## Validated runtime

The end-to-end Kaggle validation for this release demonstrated:

```text
unit tests                 93/93 PASS
TPU devices                8
logical workers            1
mesh                       [1,8]
mesh axes                  [batch, model]
dtype                      bfloat16
generation                 split_compile
strict weight loading      true
model weights              1065
trainable weights          1065
vision                     enabled
text smoke test            PASS
multipart vision           PASS
jobs                       2 completed / 0 failed
structured log             clean
```

The validated environment used Python 3.12.x, Keras 3.15.1, KerasHub 0.31.0, JAX 0.10.2, jaxlib 0.10.2, and `libtpu` 0.0.17.

## Frozen TPU engine integrity

The TPU inference core is tracked by these SHA256 values:

```text
121c6adce3f79094dff1a3509fc9592b06a864692a4539d60c39552b3b97d70d  src/translategemma_server/tpu/engine.py
e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13  src/translategemma_server/tpu/distribution.py
5ed47b7760f98064523e38476d11e98ff59feee65f08457f742338e253c4a511  src/translategemma_server/tpu/generation.py
```

## Serving architecture

- one logical TranslateGemma 12B IT model;
- one TPU worker spanning all 8 TPU devices;
- ModelParallel mesh `[1,8]` with `[batch, model]` axes;
- BF16 inference and strict checkpoint loading;
- split prefill/decode JIT with a Python autoregressive loop;
- CPU-only Flask coordinator with bounded jobs and lifecycle supervision.

## Public API

The release provides authenticated text and image translation, sync/async job endpoints, health/readiness/runtime metadata, restart supervision, Python and Node.js clients, and an optional Cloudflare Quick Tunnel.

## Kaggle startup hardening

Setup preserves the JAX/JAXLIB versions supplied by Kaggle. If `libtpu` is absent, the helper installs `libtpu==0.0.17` with `--no-deps`; otherwise the installed runtime is retained. Real TPU runs use `TPU_PREFLIGHT_MODE=required` so exactly 8 TPU devices remain a hard gate.

## Documentation and repository hygiene

The public repository contains paired English/Vietnamese documentation, community templates, CPU-friendly CI, notebook JSON validation, documentation parity checks, source packaging, SHA256 manifests, and secret scanning.

## Scope

This repository is a Kaggle-oriented serving implementation. It does not bundle TranslateGemma model weights and it does not present temporary tunnel endpoints as production infrastructure.
