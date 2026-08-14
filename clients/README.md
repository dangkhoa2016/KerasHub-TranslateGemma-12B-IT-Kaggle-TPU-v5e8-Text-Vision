# TranslateGemma 12B clients

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Both clients implement the server's `200` or `202 + /result/<job_id>` contract, send authentication without printing the key, and require no extra client package.

## Python

Text translation:

```bash
python3 clients/python/translategemma_client.py \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  text "Good morning! How are you?"
```

Multipart image translation:

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

Use `--json-base64` when you specifically want to exercise JSON/base64 image transport.

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

## Polling behavior

When the server returns HTTP `202`, both clients poll `GET /result/<job_id>` until the job reaches `completed` or `failed`. This is especially important during cold JAX compilation.
