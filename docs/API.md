# REST API

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](API.vi.md)

Default local base URL: `http://127.0.0.1:7860`.

## Authentication

Unless `API_AUTH_REQUIRED=false`, protected endpoints accept either:

```text
Authorization: Bearer <API_KEY>
```

or:

```text
X-API-Key: <API_KEY>
```

`POST /restart` additionally requires the separate restart secret.

## Request IDs

Clients may send `X-Request-ID`. Otherwise the server generates one and returns it in the response headers and job metadata.

## Health and metadata

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health/live` | Coordinator liveness |
| `GET` | `/health/ready` | TPU/model readiness |
| `GET` | `/health/ready?details=1` | Authenticated worker details |
| `GET` | `/info` | Authenticated safe runtime summary |

`/health/live` does not imply TPU readiness. `/health/ready` returns `503` while the worker is loading, restarting, or unavailable and becomes `200` only after the model is ready.

## Text translation

`POST /translate` and `POST /translate/async` accept JSON:

```json
{
  "text": "Good morning! How are you?",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "max_new_tokens": 128
}
```

## Image translation

`POST /translate/image` and `POST /translate/image/async` accept JSON/base64 or `multipart/form-data`.

Multipart fields:

```text
image=<binary file>
source_lang=English
target_lang=Vietnamese
max_new_tokens=256
```

## Async and cold-compile polling

A synchronous request may exceed the HTTP wait window during first compilation. In that case the server can return HTTP `202` with a `job_id`.

Poll:

```text
GET /result/<job_id>
```

until the job becomes `completed` or `failed`.

## Restart

`POST /restart` is protected by the API authentication layer and the separate restart secret. Use it for managed worker recovery rather than killing processes manually.

## Common status codes

| Status | Meaning |
|---|---|
| `200` | Success or completed job |
| `202` | Accepted; poll the job result |
| `400` | Invalid request |
| `401` | Authentication failed |
| `413` | Request/image too large |
| `429` | Queue is full |
| `503` | Model loading or worker unavailable |
