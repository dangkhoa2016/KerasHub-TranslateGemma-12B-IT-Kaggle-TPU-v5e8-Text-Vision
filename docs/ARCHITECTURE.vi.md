# Kiến trúc

> 🌐 Language / Ngôn ngữ: [English](ARCHITECTURE.md) | **Tiếng Việt**

## Mục tiêu

Serve một multimodal model TranslateGemma 12B IT trên toàn bộ slice Kaggle TPU v5e-8 8 thiết bị trong khi giữ HTTP coordination bên ngoài accelerator process.

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

## Trách nhiệm của coordinator

Flask coordinator quản lý HTTP authentication, validation, request IDs, queue/job lifecycle, health endpoints, restart supervision, structured logging và polling behavior phía client.

Nó chủ ý không import JAX, Keras hoặc KerasHub trong normal coordinator startup.

## Trách nhiệm của TPU worker

Worker discover Keras checkpoint đã attach, cấu hình 8-device mesh, load weights strict, khởi tạo text/vision preprocessing, thực hiện compilation và chạy inference.

Đúng một worker sở hữu toàn bộ TPU mesh. HTTP concurrency được coordinator queue hấp thụ thay vì launch nhiều process model 12B.

## Model parallelism

Device mesh có shape `[1,8]` và axes `[batch, model]`. Các tensor model lớn được shard theo model axis trong khi batch axis giữ size 1.

Đây là model parallelism, không phải tám model replica độc lập.

## Generation

Generation chủ ý tách thành:

```text
prefill JIT
decode-step JIT
Python autoregressive loop
```

Split design giúp compilation và host-memory behavior dễ dự đoán trong môi trường Kaggle này.

## Readiness contract

Coordinator liveness tách biệt với model readiness. `/health/live` nghĩa HTTP process đang sống; `/health/ready` chỉ thành `200` sau khi TPU worker khởi tạo runtime 8 thiết bị bắt buộc và model.

## Vision path

Cùng một logical model xử lý text translation và text extraction/translation từ image. Image request hỗ trợ multipart upload và JSON/base64 transport.

## Failure boundaries

Startup problem được tách thành source/configuration, dependency/bootstrap, TPU preflight, model/worker load và REST/inference phases. Change nên nhắm đúng phase thực sự fail thay vì rewrite stable lower layers.

## Security boundary

Secret được tạo local ở runtime, bị loại khỏi source packaging và không bao giờ cần xuất hiện trong notebook Markdown hoặc repository documentation.
