# Đóng góp

> 🌐 Language / Ngôn ngữ: [English](CONTRIBUTING.md) | **Tiếng Việt**

Cảm ơn bạn đã giúp cải thiện dự án.

## Phạm vi

Hoan nghênh đóng góp về reliability, documentation, tests, API ergonomics, security hardening, khả năng sử dụng trên Kaggle và các runtime defect được tái hiện rõ ràng.

Vui lòng tránh thay đổi kiến trúc không liên quan đến một vấn đề đã được tái hiện.

## Các bất biến runtime

Giữ các thuộc tính sau trừ khi một lỗi Kaggle TPU thực tế chứng minh cần thay đổi:

- một logical model TranslateGemma 12B;
- một TPU inference worker;
- đúng 8 TPU devices;
- ModelParallel mesh `[1,8]`;
- BF16 inference;
- strict checkpoint loading;
- split prefill/decode compilation;
- Flask coordinator phía CPU;
- `202 + /result/<job_id>` cho request chạy lâu.

## Workflow phát triển

1. Tạo branch có phạm vi rõ ràng.
2. Thêm hoặc cập nhật test trước khi thay đổi behavior.
3. Giữ notebook checkout tại `/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision`.
4. Chạy bộ verification CPU-friendly ở local.
5. Dùng Kaggle TPU v5e-8 cho acceptance test liên quan accelerator khi thay đổi runtime behavior.
6. Giữ commit đủ nhỏ để review độc lập.

## Kiểm tra bắt buộc

```bash
python3 -m pip install -r requirements-ci.txt
bash scripts/test_unit.sh
python3 scripts/check_docs.py
python3 -m compileall -q src scripts tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
node --check clients/node/translategemma-client.mjs
python3 scripts/secret_scan.py .
```

## Tài liệu

Mỗi tài liệu Markdown phải có cặp English/Vietnamese với số dòng bằng nhau. Giữ technical facts, commands, paths, endpoint names, environment variables và release identity đồng bộ giữa hai ngôn ngữ.

## Pull requests

Mô tả vấn đề, giải pháp nhỏ nhất, các test đã chạy và việc có chạy Kaggle TPU thật hay không. Không claim accelerator validation khi mới chỉ chạy CPU/static checks.

Xem [.github/PULL_REQUEST_TEMPLATE.vi.md](.github/PULL_REQUEST_TEMPLATE.vi.md) để dùng review checklist.
