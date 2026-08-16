# Ghi chú validation và performance

> 🌐 Language / Ngôn ngữ: [English](BENCHMARKS.md) | **Tiếng Việt**

Các observation này đến từ workflow Kaggle TPU v5e-8 đã validation và chỉ là reference values, không phải performance guarantees.

## Functional acceptance

Validated release đạt:

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

## Startup và readiness

Một validated run đại diện cần khoảng bốn phút để model loading/readiness. Trong thời gian này `/health/live` có thể thành công trong khi `/health/ready` đúng thiết kế vẫn trả `503`.

Sự tách biệt này là có chủ ý và cần được giữ nguyên.

## First-call compilation

Text và vision request đầu tiên có thể mất vài phút vì duration được báo gồm JAX compilation. Request sau với shape tương thích được kỳ vọng tránh phần lớn cold-start cost đó.

Các first-call observation đại diện xấp xỉ:

```text
text request total         ~255 s
text prefill compile       ~159 s
text decode first call     ~94 s
vision request total       ~241 s
vision prefill compile     ~145 s
vision decode first call   ~95 s
```

Không xem các giá trị này là service-level objectives. Kaggle images, model mounts, cache state và runtime versions có thể thay đổi.

## Generation design ưu tiên memory

Server dùng separate prefill/decode compilation cộng Python autoregressive loop thay vì một fused generation path duy nhất. Thiết kế ưu tiên stable memory behavior cho workload TPU 12B này.

## Chính sách benchmark

Ordinary documentation hoặc API changes không nên tiêu TPU quota cho repeated performance measurements. Chạy lại accelerator measurements khi change có thể ảnh hưởng đáng kể tới inference, compilation, memory use hoặc device topology.
