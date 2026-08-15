# Changelog

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CHANGELOG.vi.md)

All notable public releases are documented here.

## v1.0.0 — 2026-08-17

Initial public release of the TranslateGemma 12B IT text + vision REST server for Kaggle TPU v5e-8.

### Runtime

- One logical TranslateGemma `translategemma_12b_it` model sharded across 8 TPU devices.
- Keras ModelParallel mesh `[1,8]` with axes `[batch, model]`.
- BF16 inference, strict checkpoint loading, and split prefill/decode compilation.
- Kaggle TPU bootstrap preserves the installed JAX/JAXLIB stack and installs `libtpu==0.0.17` only when `libtpu` is absent.

### REST service

- Authenticated Flask coordinator with text and image translation.
- Synchronous and asynchronous job APIs with `202 + /result/<job_id>` polling.
- JSON/base64 and `multipart/form-data` image transport.
- Liveness, readiness, runtime information, restart supervision, request IDs, and structured logs.

### Developer experience

- GitHub-import-first Kaggle notebook workflow.
- Python and Node.js clients.
- CPU-friendly unit/static/security CI plus a separate real-TPU validation workflow on Kaggle.
- Bilingual English/Vietnamese documentation with automated parity checks.
- Clean source packaging with SHA256 manifest and secret scanning.
