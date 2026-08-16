# Hướng dẫn Kaggle

> 🌐 Language / Ngôn ngữ: [English](KAGGLE.md) | **Tiếng Việt**

## Workflow khuyến nghị: import từ GitHub

1. Tạo Kaggle Notebook mới.
2. Mở **File → Import Notebook → GitHub**.
3. Tìm `dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision`.
4. Chọn `notebooks/kaggle-tpu-v5e8-text-vision.ipynb`.
5. Nhấn **Import**.
6. Bật **Internet**.
7. Chọn **TPU v5e-8 / `v5litepod-8`**.
8. Dùng **Add Input / Models** để attach Keras TranslateGemma model chứa `translategemma_12b_it`.
9. Giữ `RUN_TPU_VALIDATION=True`.
10. Dùng **Restart Session → Run All**.

## Vì sao notebook clone repository lần nữa

GitHub Import đưa notebook cells vào Kaggle. Code cell đầu sau đó clone hoặc hard-refresh `main` vào:

```text
/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
```

Cách này giữ server source, scripts, clients, tests và documentation neo vào repository thay vì copied notebook code.

## Run All validation những gì

Notebook:

1. kiểm tra official Git checkout và in exact HEAD;
2. yêu cầu working tree sạch;
3. chạy dependency/setup checks;
4. chạy CPU-friendly unit tests;
5. yêu cầu đúng 8 TPU devices khi bật TPU validation;
6. start Flask coordinator và single TPU worker;
7. chờ mesh `[1,8]` ready;
8. query authenticated runtime information;
9. chạy text translation;
10. chạy multipart vision translation;
11. tùy chọn start Cloudflare Quick Tunnel;
12. in final service status.

## Chế độ chỉ CPU/source

Đặt:

```python
RUN_TPU_VALIDATION = False
```

chỉ khi bạn chủ động muốn dependency, unit và static checks mà không initialize TPU hoặc start server. Chế độ này không phải evidence của accelerator inference.

## Clone thủ công cho developer

```bash
git clone https://github.com/dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision.git \
  /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
cd /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
cp .env.example .env
INSTALL_PYTHON_DEPS=auto TPU_PREFLIGHT_MODE=required bash scripts/setup.sh
bash scripts/start.sh
```

## Thứ tự troubleshooting

Nếu run fail, trước hết xác định failure xảy ra trong Git/source checks, dependency/bootstrap, TPU preflight, model loading hay REST/inference. Tránh thay TPU engine cho failure xảy ra trước model initialization.

## TPU quota

Dùng CPU/source-only mode cho documentation và ordinary unit-test work. Dành TPU sessions cho change cần real accelerator acceptance testing.
