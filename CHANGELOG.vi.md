# Changelog

> 🌐 Language / Ngôn ngữ: [English](CHANGELOG.md) | **Tiếng Việt**

Tất cả public release đáng chú ý được ghi tại đây.

## v1.0.0 — 2026-08-17

Public release đầu tiên của REST server TranslateGemma 12B IT text + vision cho Kaggle TPU v5e-8.

### Runtime

- Một logical model TranslateGemma `translategemma_12b_it` được shard trên 8 TPU devices.
- Keras ModelParallel mesh `[1,8]` với axes `[batch, model]`.
- BF16 inference, strict checkpoint loading và split prefill/decode compilation.
- Kaggle TPU bootstrap giữ nguyên JAX/JAXLIB có sẵn và chỉ cài `libtpu==0.0.17` khi `libtpu` chưa tồn tại.

### REST service

- Flask coordinator có authentication cho dịch text và image.
- Sync/async job API với cơ chế polling `202 + /result/<job_id>`.
- Image transport bằng JSON/base64 và `multipart/form-data`.
- Liveness, readiness, runtime information, restart supervision, request ID và structured log.

### Trải nghiệm developer

- Workflow Kaggle ưu tiên import notebook trực tiếp từ GitHub.
- Python và Node.js clients.
- CI unit/static/security chạy CPU và workflow validation TPU thật riêng trên Kaggle.
- Tài liệu song ngữ English/Vietnamese với kiểm tra parity tự động.
- Clean source packaging có SHA256 manifest và secret scanning.
