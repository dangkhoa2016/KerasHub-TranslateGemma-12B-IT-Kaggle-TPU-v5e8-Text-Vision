# KerasHub TranslateGemma 12B IT trên Kaggle TPU v5e-8 — REST Server Text + Vision

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

Chạy **TranslateGemma 12B IT** dưới dạng REST service có xác thực cho text và vision trên Kaggle **TPU v5e-8 / `v5litepod-8`** bằng **Keras 3, KerasHub và JAX**.

Project dùng một logical model trong một TPU worker được spawn riêng, shard trên toàn bộ **8 TPU devices** bằng Keras ModelParallel mesh `[1,8]`. Flask coordinator nhẹ chạy phía CPU và phụ trách HTTP, authentication, queue, async result polling, request ID, health check, structured logging và worker supervision.

## Trạng thái

**Release:** `v1.0.0`

Kiến trúc serving đã được kiểm chứng end-to-end trên Kaggle TPU v5e-8 với:

```text
TPU devices             8
logical TPU workers     1
mesh                    [1,8]
mesh axes               [batch, model]
dtype                   bfloat16
generation              split_compile
strict model weights    1065/1065
text translation        PASS
multipart vision        PASS
final jobs              2 completed / 0 failed
```

Public release hiện tại giữ nguyên TPU inference core đã proven. Phần bootstrap Kaggle cũng giữ nguyên JAX/JAXLIB đang có và chỉ cài bản đã proven `libtpu==0.0.17` khi `libtpu` hoàn toàn bị thiếu.

## Repository này cung cấp gì

- Serving TranslateGemma `translategemma_12b_it` multimodal trên đủ 8 TPU devices.
- Keras ModelParallel mesh `[1,8]` với BF16 inference.
- Strict checkpoint loading cho monolithic weights hoặc sharded weights được hỗ trợ.
- Split prefill JIT + decode-step JIT + Python autoregressive loop.
- Dịch text và OCR/dịch nội dung trong ảnh.
- Input ảnh qua JSON/base64 hoặc `multipart/form-data`.
- Endpoint sync và async với `202 + /result/<job_id>` polling.
- API-key authentication và restart secret riêng.
- `/health/live`, `/health/ready` và `/info` có authentication.
- Request ID và structured log gọn.
- Python client không cần dependency ngoài và Node.js 18+ client.
- Cloudflare Quick Tunnel tùy chọn nhưng vẫn giữ authentication.
- Unit/static/security checks chạy CPU-only cho development và CI.

## Kiến trúc

```text
HTTP / Python / Node clients
          |
          v
Flask coordinator (CPU-only; không import JAX/Keras)
          |
   bounded queue + job/result store
          |
          v
một spawned TPU worker
          |
TranslateGemmaTPUEngine
          |
Keras ModelParallel mesh [1,8]
          |
TPU0 ... TPU7
```

Service cố ý chỉ dùng **một TPU inference worker**. Nhiều HTTP request có thể xếp hàng, nhưng một logical TranslateGemma 12B model sở hữu toàn bộ TPU mesh.

### Vì sao dùng split compile

Runtime ổn định compile prefill và decode riêng:

```text
prefill JIT
   +
decode-step JIT
   +
Python autoregressive loop
```

Thiết kế này tránh mức host-RAM cao hơn rất nhiều đã quan sát với fused generation, đồng thời vẫn giữ layout model-parallel 8 thiết bị.

## Yêu cầu

Để chạy full runtime trên Kaggle:

1. Kaggle Notebook với **TPU v5e-8** được bật.
2. Bật **Internet** để checkout Git và tải dependency còn thiếu nếu cần.
3. Attach Keras TranslateGemma model có preset `translategemma_12b_it`.
4. Dành đủ thời gian cho lần model load và JAX compile đầu tiên. Request đầu tiên chậm hơn nhiều so với các request sau là bình thường.

Nếu chỉ muốn kiểm tra source/unit, notebook public có thể chạy với TPU validation tắt.

## Chạy nhanh trên Kaggle

### Khuyến nghị: import notebook trực tiếp từ GitHub

Bạn **không cần** tải source ZIP hoặc clone repository thủ công trước khi mở notebook.

1. Trên Kaggle, tạo **New Notebook**.
2. Mở **File → Import Notebook → GitHub**.
3. Tại ô **Search by user, organization and/or repository**, nhập:

   ```text
   dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
   ```

4. Chọn notebook trong repository:

   ```text
   notebooks/kaggle-tpu-v5e8-text-vision.ipynb
   ```

5. Nhấn **Import**.
6. Trong phần settings của Kaggle Notebook, bật **Internet** và chọn **TPU v5e-8 / `v5litepod-8`**.
7. Dùng **Add Input / Models** để attach Keras TranslateGemma model có preset `translategemma_12b_it`.
8. Giữ:

   ```python
   RUN_TPU_VALIDATION = True
   ```

9. Chạy lại từ session sạch bằng **Restart Session → Run All**.

GitHub Import chỉ đưa notebook vào Kaggle. Code cell đầu tiên của notebook sau đó sẽ clone hoặc hard-refresh official repository vào:

```text
/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
```

Nhờ vậy GitHub repository vẫn là single source of truth cho server source, scripts, clients và tests.

### Run All sẽ làm gì

Notebook sẽ:

1. clone hoặc hard-refresh `main` từ repository này;
2. kiểm tra Git tree clean và in exact HEAD;
3. chạy setup và unit/source checks;
4. yêu cầu TPU preflight đủ 8 devices;
5. start Flask coordinator và một TPU worker;
6. chờ mesh `[1,8]` ready;
7. đọc runtime metadata an toàn;
8. chạy smoke test text và multipart vision;
9. tùy chọn start Cloudflare Quick Tunnel;
10. in trạng thái service cuối cùng.

Chỉ đặt `RUN_TPU_VALIDATION=False` khi bạn chủ động muốn chạy dependency/unit/static validation mà không initialize TPU và không start server.

### Phương án khác: clone thủ công cho developer

Dùng cách này khi bạn muốn xem hoặc chỉnh sửa source trực tiếp thay vì bắt đầu từ GitHub notebook importer của Kaggle:

```bash
git clone https://github.com/dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision.git \
  /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision

cd /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
cp .env.example .env
INSTALL_PYTHON_DEPS=auto TPU_PREFLIGHT_MODE=required bash scripts/setup.sh
bash scripts/start.sh
python3 scripts/wait_ready.py \
  --api-key-file data/api_key.txt \
  --timeout 1800 \
  --expected-devices 8 \
  --expected-mesh 1,8
```

## Xác minh service đang chạy

### Runtime info an toàn

```bash
python3 clients/python/translategemma_client.py \
  --api-key-file data/api_key.txt info
```

Một TPU worker đúng cấu hình phải báo 8 devices, mesh `[1,8]`, BF16, split-compile generation, strict weight loading và vision support.

### Text smoke test

```bash
bash scripts/test.sh
```

### Multipart vision smoke test

```bash
python3 clients/python/translategemma_client.py \
  --api-key-file data/api_key.txt \
  image assets/sample-image-with-text.png \
  --source-lang English \
  --target-lang Vietnamese \
  --max-new-tokens 256 \
  --multipart
```

## REST API

| Method | Endpoint | Mục đích |
|---|---|---|
| `GET` | `/` | Metadata của service |
| `GET` | `/health/live` | Coordinator liveness |
| `GET` | `/health/ready` | Model/worker readiness |
| `GET` | `/info` | Runtime information an toàn, có auth |
| `POST` | `/translate` | Dịch text |
| `POST` | `/translate/async` | Dịch text async |
| `POST` | `/translate/image` | OCR/dịch ảnh |
| `POST` | `/translate/image/async` | OCR/dịch ảnh async |
| `GET` | `/result/<job_id>` | Poll async/cold-compile job |
| `POST` | `/restart` | Restart worker có bảo vệ |

Authentication chấp nhận:

```text
Authorization: Bearer <API_KEY>
```

hoặc:

```text
X-API-Key: <API_KEY>
```

`POST /restart` còn yêu cầu restart secret riêng.

### Text request

```json
{
  "text": "Good morning! How are you?",
  "source_lang": "English",
  "target_lang": "Vietnamese",
  "max_new_tokens": 128
}
```

### Multipart image request

```bash
curl -X POST http://127.0.0.1:7860/translate/image \
  -H "Authorization: Bearer $(cat data/api_key.txt)" \
  -H "X-Request-ID: demo-image-001" \
  -F "image=@assets/sample-image-with-text.png" \
  -F "source_lang=English" \
  -F "target_lang=Vietnamese" \
  -F "max_new_tokens=256"
```

Image transport bằng JSON/base64 vẫn được hỗ trợ.

## Clients

### Python

Không cần package client bên thứ ba:

```bash
python3 clients/python/translategemma_client.py \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  text "Good morning! How are you?"
```

Client tự poll `/result/<job_id>` khi server trả `202`.

### Node.js

Yêu cầu Node.js 18+ và dùng built-in `fetch`, `FormData`, `Blob`:

```bash
node clients/node/translategemma-client.mjs text \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  --text "Good morning! How are you?"
```

Xem thêm [clients/README.md](clients/README.md) và [docs/API.md](docs/API.md).

## Hành vi runtime

### Readiness

`/health/live` chỉ có nghĩa coordinator process còn sống. `/health/ready` chưa ready trong lúc TPU worker đang load/compile và chỉ trở thành `200` khi model thực sự sẵn sàng.

### Cold compilation

Request text hoặc vision đầu tiên có thể mất vài phút vì JAX compile các prefill/decode shape tương ứng. Đây là hành vi dự kiến. Cold request có thể trả `202`; client nên poll result endpoint được trả về thay vì coi đó là lỗi.

### Chính sách Kaggle JAX / libtpu

Project **không** tự ý thay toàn bộ accelerator stack của Kaggle.

- Giữ nguyên JAX và JAXLIB hiện có.
- Giữ nguyên `libtpu` hiện có, kể cả khi version khác known reference.
- Nếu `libtpu` hoàn toàn bị thiếu, `scripts/ensure_libtpu.py` cài `libtpu==0.0.17` bằng `--no-deps`.
- Functional gate cuối vẫn là TPU preflight yêu cầu đúng 8 TPU devices.

### Startup noise đã biết

Một run TPU thành công vẫn có thể xuất hiện một số dòng startup cấp thấp liên quan TPU `SliceBuilder` port 8471 hoặc CUDA `cuInit 303`. Chỉ riêng các dòng này không chứng minh inference lỗi; hãy dựa vào TPU preflight, readiness và worker metadata. stderr lạ vẫn cần được kiểm tra.

## Cấu hình

Runtime defaults được mô tả trong `.env.example`. Một số giá trị quan trọng:

```text
model                     translategemma_12b_it
backend                   JAX
framework                 Keras / KerasHub
dtype                     bfloat16
expected TPU devices      8
mesh                      [1,8]
mesh axes                 [batch,model]
weight loading            strict
generation                split prefill/decode compile
vision                    enabled
worker load timeout       1200 s
worker restart budget     1
TPU preflight             required
```

## Vận hành

```bash
bash scripts/status.sh
bash scripts/restart.sh
bash scripts/stop.sh
```

Tunnel tùy chọn:

```bash
bash scripts/run_tunnel.sh
python3 scripts/demo_info.py
bash scripts/stop_tunnel.sh
```

Luôn giữ API authentication khi service có thể truy cập từ bên ngoài localhost.

## Development và validation

CPU-only checks:

```bash
python3 -m pip install -r requirements-ci.txt
bash scripts/test_unit.sh
python3 -m compileall -q src scripts
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
node --check clients/node/translategemma-client.mjs
python3 scripts/secret_scan.py .
```

Các checks này xác minh source, contracts, packaging và security nhưng **không** thay thế một lần chạy Kaggle TPU thật.

### Build clean source archive

```bash
python3 scripts/package_source.py
```

Packager loại generated credentials, `.env`, runtime logs, PID/tunnel state, caches và artifact phát triển nội bộ.

### Tagged GitHub releases

Khi push release tag như `v1.0.0`, `.github/workflows/release.yml` sẽ chạy CPU verification, build public artifacts, tạo SHA256, upload artifacts và tạo GitHub Release.

## Tài liệu

- [Kiến trúc](docs/ARCHITECTURE.vi.md)
- [Hướng dẫn Kaggle](docs/KAGGLE.vi.md)
- [REST API](docs/API.vi.md)
- [Ghi chú validation](docs/BENCHMARKS.vi.md)
- [Ví dụ client](clients/README.vi.md)
- [Đóng góp](CONTRIBUTING.vi.md)
- [Bảo mật](SECURITY.vi.md)
- [Changelog](CHANGELOG.vi.md)

## Bảo mật

Không commit hoặc public:

- `.env`
- `data/api_key.txt`
- `data/restart_secret.txt`
- bearer/API-key values
- request body chứa nội dung nhạy cảm
- image base64 payload
- tunnel credentials
- SSH secret keys

Xem [SECURITY.md](SECURITY.md).

## License và third-party software

Source code gốc của repository được phát hành theo [MIT License](LICENSE). Model weights TranslateGemma và thư viện/tool bên thứ ba giữ license/terms riêng; xem [NOTICE.vi.md](NOTICE.vi.md) và [THIRD_PARTY_NOTICES.vi.md](THIRD_PARTY_NOTICES.vi.md).
