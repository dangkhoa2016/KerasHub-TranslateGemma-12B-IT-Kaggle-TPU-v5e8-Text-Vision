# TranslateGemma 12B clients

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Cả hai client triển khai contract `200` hoặc `202 + /result/<job_id>` của server, gửi authentication mà không in key và không cần thêm client package.

## Python

Dịch text:

```bash
python3 clients/python/translategemma_client.py \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  text "Good morning! How are you?"
```

Dịch image bằng multipart:

```bash
python3 clients/python/translategemma_client.py \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  image assets/sample-image-with-text.png \
  --source-lang English \
  --target-lang Vietnamese \
  --max-new-tokens 256 \
  --multipart
```

Dùng `--json-base64` khi bạn chủ động muốn kiểm tra image transport JSON/base64.

## Node.js 18+

```bash
node clients/node/translategemma-client.mjs text \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  --text "Good morning! How are you?"
```

```bash
node clients/node/translategemma-client.mjs image \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  --image assets/sample-image-with-text.png
```

## Hành vi polling

Khi server trả HTTP `202`, cả hai client poll `GET /result/<job_id>` tới khi job đạt `completed` hoặc `failed`. Điều này đặc biệt quan trọng trong cold JAX compilation.
