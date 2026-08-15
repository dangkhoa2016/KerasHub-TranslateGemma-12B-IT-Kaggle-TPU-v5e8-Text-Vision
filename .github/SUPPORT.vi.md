# Hỗ trợ

> 🌐 Language / Ngôn ngữ: [English](SUPPORT.md) | **Tiếng Việt**

## Trước khi yêu cầu trợ giúp

Vui lòng đọc [README.vi.md](../README.vi.md), [docs/KAGGLE.vi.md](../docs/KAGGLE.vi.md) và [docs/API.vi.md](../docs/API.vi.md).

Với Kaggle runtime problem, hãy cung cấp:

- exact Git commit notebook đã in;
- `RUN_TPU_VALIDATION` là `True` hay `False`;
- phase bị fail: setup, TPU preflight, model load, readiness, text hay vision;
- error message liên quan sau khi loại secret;
- Kaggle có báo đủ 8 TPU devices hay không.

## Hỏi ở đâu

Dùng GitHub issue cho reproducible bug hoặc feature request. Chọn issue template có sẵn để report chứa đủ thông tin điều tra.

Không dùng public issue cho security vulnerability hoặc sensitive runtime data; hãy theo [SECURITY.vi.md](../SECURITY.vi.md).

## Phạm vi hỗ trợ

Repository này tập trung vào workflow Kaggle TPU v5e-8 được cung cấp. Cloud khác, TPU generation khác, local TPU setup, model family khác và production hosting có thể cần adaptation riêng.
