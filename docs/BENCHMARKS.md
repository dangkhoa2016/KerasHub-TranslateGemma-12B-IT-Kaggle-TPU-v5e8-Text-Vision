# Validation and performance notes

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARKS.vi.md)

These observations come from the validated Kaggle TPU v5e-8 workflow and are reference values, not performance guarantees.

## Functional acceptance

The validated release reached:

```text
TPU devices                8
mesh                       [1,8]
dtype                      bfloat16
strict weights             1065/1065
generation                 split_compile
text translation           PASS
multipart vision           PASS
jobs                       2 completed / 0 failed
```

## Startup and readiness

A representative validated run required roughly four minutes for model loading/readiness. During this period `/health/live` can succeed while `/health/ready` correctly returns `503`.

This distinction is intentional and should be preserved.

## First-call compilation

The first text and vision requests can take several minutes because the reported duration includes JAX compilation. Later requests with compatible shapes are expected to avoid much of that cold-start cost.

Representative first-call observations were approximately:

```text
text request total         ~255 s
text prefill compile       ~159 s
text decode first call     ~94 s
vision request total       ~241 s
vision prefill compile     ~145 s
vision decode first call   ~95 s
```

Do not interpret these values as service-level objectives. Kaggle images, model mounts, cache state, and runtime versions can change.

## Memory-oriented generation design

The server uses separate prefill/decode compilation plus a Python autoregressive loop rather than a single fused generation path. The design prioritizes stable memory behavior for this 12B TPU workload.

## Benchmark policy

Ordinary documentation or API changes should not consume TPU quota for repeated performance measurements. Re-run accelerator measurements when a change can materially affect inference, compilation, memory use, or device topology.
