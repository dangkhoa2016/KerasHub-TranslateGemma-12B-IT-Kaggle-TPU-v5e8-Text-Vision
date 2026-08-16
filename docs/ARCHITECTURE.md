# Architecture

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](ARCHITECTURE.vi.md)

## Goal

Serve one TranslateGemma 12B IT multimodal model on the complete Kaggle TPU v5e-8 8-device slice while keeping HTTP coordination outside the accelerator process.

## Runtime topology

```text
Kaggle Notebook
      |
      v
Flask coordinator (CPU)
      |
 bounded job queue
      |
      v
single TPU worker
      |
TranslateGemma 12B IT
Keras / KerasHub / JAX
      |
ModelParallel mesh [1,8]
      |
TPU0 TPU1 TPU2 TPU3 TPU4 TPU5 TPU6 TPU7
```

## Coordinator responsibilities

The Flask coordinator owns HTTP authentication, validation, request IDs, queue/job lifecycle, health endpoints, restart supervision, structured logging, and client-facing polling behavior.

It intentionally does not import JAX, Keras, or KerasHub during normal coordinator startup.

## TPU worker responsibilities

The worker discovers the attached Keras checkpoint, configures the 8-device mesh, loads weights strictly, initializes text/vision preprocessing, performs compilation, and executes inference.

Exactly one worker owns the complete TPU mesh. HTTP concurrency is absorbed by the coordinator queue rather than by launching multiple 12B model processes.

## Model parallelism

The device mesh has shape `[1,8]` and axes `[batch, model]`. Large model tensors are sharded across the model axis while the batch axis remains size 1.

This is model parallelism, not eight independent model replicas.

## Generation

Generation is deliberately split into:

```text
prefill JIT
decode-step JIT
Python autoregressive loop
```

The split design keeps compilation and host-memory behavior predictable for this Kaggle environment.

## Readiness contract

Coordinator liveness is separate from model readiness. `/health/live` means the HTTP process is alive; `/health/ready` becomes `200` only after the TPU worker has initialized the required 8-device runtime and model.

## Vision path

The same logical model handles text translation and text extraction/translation from images. Image requests support multipart upload and JSON/base64 transport.

## Failure boundaries

Startup problems are separated into source/configuration, dependency/bootstrap, TPU preflight, model/worker load, and REST/inference phases. Changes should target the phase that actually failed instead of rewriting stable lower layers.

## Security boundary

Secrets are generated locally at runtime, excluded from source packaging, and never required in notebook Markdown or repository documentation.
