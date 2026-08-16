# REST API

> 🌐 Language / Ngôn ngữ: [English](API.md) | **Tiếng Việt**

Default local base URL: `http://127.0.0.1:7860`.

## Authentication

Trừ khi `API_AUTH_REQUIRED=false`, protected endpoints chấp nhận một trong hai cách:

```text
Authorization: Bearer <API_KEY>
```

hoặc:

```text
X-API-Key: <API_KEY>
```

`POST /restart` còn yêu cầu restart secret riêng.

## Request IDs

Client có thể gửi `X-Request-ID`. Nếu không, server tự tạo và trả lại trong response headers cùng job metadata.

## Health và metadata

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health/live` | Coordinator liveness |
| `GET` | `/health/ready` | TPU/model readiness |
| `GET` | `/health/ready?details=1` | Worker details có authentication |
| `GET` | `/info` | Safe runtime summary có authentication |

`/health/live` không đồng nghĩa TPU đã ready. `/health/ready` trả `503` khi worker đang loading, restarting hoặc unavailable và chỉ thành `200` sau khi model ready.

## Dịch text

`POST /translate` và `POST /translate/async` nhận JSON:

```json
{
  "text": "Good morning! How are you?",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "max_new_tokens": 128
}
```

## Dịch image

`POST /translate/image` và `POST /translate/image/async` nhận JSON/base64 hoặc `multipart/form-data`.

Multipart fields:

```text
image=<binary file>
source_lang=English
target_lang=Vietnamese
max_new_tokens=256
```

## Async và cold-compile polling

Synchronous request có thể vượt HTTP wait window trong lần compile đầu. Khi đó server có thể trả HTTP `202` cùng `job_id`.

Poll:

```text
GET /result/<job_id>
```

cho tới khi job thành `completed` hoặc `failed`.

## Restart

`POST /restart` được bảo vệ bởi API authentication layer và restart secret riêng. Dùng endpoint này để managed worker recovery thay vì kill process thủ công.

## Status code thường gặp

| Status | Ý nghĩa |
|---|---|
| `200` | Thành công hoặc job đã hoàn tất |
| `202` | Đã nhận; poll kết quả job |
| `400` | Request không hợp lệ |
| `401` | Authentication thất bại |
| `413` | Request/image quá lớn |
| `429` | Queue đầy |
| `503` | Model đang load hoặc worker unavailable |
