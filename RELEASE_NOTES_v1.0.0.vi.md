# Ghi chú phát hành — v1.0.0

> 🌐 Language / Ngôn ngữ: [English](RELEASE_NOTES_v1.0.0.md) | **Tiếng Việt**

## Tổng quan

`v1.0.0` là public release đầu tiên của REST server TranslateGemma 12B IT text + vision cho Kaggle TPU v5e-8.

Release tập trung vào reproducible serving, operational contracts rõ ràng, safe packaging và workflow Kaggle ưu tiên import trực tiếp từ GitHub.

## Runtime đã validation

End-to-end Kaggle validation của release này đã chứng minh:

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

Môi trường validation dùng Python 3.12.x, Keras 3.15.1, KerasHub 0.31.0, JAX 0.10.2, jaxlib 0.10.2 và `libtpu` 0.0.17.

## Integrity của TPU engine đã khóa

TPU inference core được theo dõi bằng các SHA256 sau:

```text
121c6adce3f79094dff1a3509fc9592b06a864692a4539d60c39552b3b97d70d  src/translategemma_server/tpu/engine.py
e07b7ac54b600a5cbfdaede8c2daa534797bcb7bcea70dfcb8f19ab1b9ac8d13  src/translategemma_server/tpu/distribution.py
5ed47b7760f98064523e38476d11e98ff59feee65f08457f742338e253c4a511  src/translategemma_server/tpu/generation.py
```

## Kiến trúc serving

- một logical model TranslateGemma 12B IT;
- một TPU worker trải trên đủ 8 TPU devices;
- ModelParallel mesh `[1,8]` với axes `[batch, model]`;
- BF16 inference và strict checkpoint loading;
- split prefill/decode JIT với Python autoregressive loop;
- Flask coordinator chỉ chạy CPU với bounded jobs và lifecycle supervision.

## Public API

Release cung cấp text/image translation có authentication, sync/async job endpoints, health/readiness/runtime metadata, restart supervision, Python và Node.js clients cùng Cloudflare Quick Tunnel tùy chọn.

## Kaggle startup hardening

Setup giữ nguyên JAX/JAXLIB do Kaggle cung cấp. Nếu `libtpu` chưa tồn tại, helper cài `libtpu==0.0.17` bằng `--no-deps`; nếu đã có thì giữ runtime hiện tại. TPU run thật dùng `TPU_PREFLIGHT_MODE=required` để đúng 8 TPU devices luôn là hard gate.

## Documentation và repository hygiene

Public repository có tài liệu English/Vietnamese theo cặp, community templates, CI CPU-friendly, notebook JSON validation, kiểm tra documentation parity, source packaging, SHA256 manifests và secret scanning.

## Phạm vi

Repository này là implementation serving hướng Kaggle. Nó không bundle TranslateGemma model weights và không xem temporary tunnel endpoints là production infrastructure.
