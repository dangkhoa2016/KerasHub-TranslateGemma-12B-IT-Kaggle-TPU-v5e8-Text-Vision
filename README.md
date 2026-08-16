# KerasHub TranslateGemma 12B IT on Kaggle TPU v5e-8 — Text + Vision REST Server

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

Run **TranslateGemma 12B IT** as an authenticated text-and-vision REST service on a Kaggle **TPU v5e-8 / `v5litepod-8`** session using **Keras 3, KerasHub, and JAX**.

The project uses one logical model in one spawned TPU worker, sharded across all **8 TPU devices** with a Keras ModelParallel mesh `[1,8]`. A lightweight Flask coordinator remains CPU-only and handles HTTP, authentication, queuing, asynchronous result polling, request IDs, health checks, structured logging, and worker supervision.

## Status

**Release:** `v1.0.0`

The serving architecture has been validated end-to-end on Kaggle TPU v5e-8 with:

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

The current public release keeps the proven TPU inference core unchanged. The Kaggle bootstrap also preserves the existing JAX/JAXLIB stack and only installs the proven `libtpu==0.0.17` when `libtpu` is completely absent.

## What this repository provides

- TranslateGemma `translategemma_12b_it` multimodal serving on all 8 TPU devices.
- Keras ModelParallel mesh `[1,8]` with BF16 inference.
- Strict checkpoint loading for monolithic or supported sharded weights.
- Split prefill JIT + decode-step JIT + Python autoregressive loop.
- Text translation and image OCR/translation.
- JSON/base64 and `multipart/form-data` image input.
- Synchronous endpoints plus asynchronous `202 + /result/<job_id>` polling.
- API-key authentication and a separate restart secret.
- `/health/live`, `/health/ready`, and authenticated `/info` endpoints.
- Request IDs and compact structured logs.
- Dependency-free Python client and Node.js 18+ client.
- Optional Cloudflare Quick Tunnel with authentication kept enabled.
- CPU-only unit/static/security checks for development and CI.

## Architecture

```text
HTTP / Python / Node clients
          |
          v
Flask coordinator (CPU-only; no JAX/Keras import)
          |
   bounded queue + job/result store
          |
          v
one spawned TPU worker
          |
TranslateGemmaTPUEngine
          |
Keras ModelParallel mesh [1,8]
          |
TPU0 ... TPU7
```

The service intentionally uses **one TPU inference worker**. Multiple HTTP requests may queue, but one logical TranslateGemma 12B model owns the complete TPU mesh.

### Why generation is split-compiled

The stable runtime compiles prefill and decode separately:

```text
prefill JIT
   +
decode-step JIT
   +
Python autoregressive loop
```

This avoids the much higher host-memory pressure observed with fused generation while preserving the 8-device model-parallel layout.

## Requirements

For the full Kaggle runtime:

1. A Kaggle Notebook session with **TPU v5e-8** enabled.
2. **Internet** enabled for Git checkout and any missing dependency download.
3. The Keras TranslateGemma model containing the `translategemma_12b_it` preset attached to the notebook.
4. Enough time for the first model load and JAX compilation. Initial requests are expected to be much slower than later calls.

For source/unit validation only, the public notebook can run with TPU validation disabled.

## Kaggle quick start

### Recommended: import the notebook directly from GitHub

You do **not** need to download the source ZIP or clone the repository manually before opening the notebook.

1. In Kaggle, create a **New Notebook**.
2. Open **File → Import Notebook → GitHub**.
3. In **Search by user, organization and/or repository**, enter:

   ```text
   dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
   ```

4. Select the notebook from the repository:

   ```text
   notebooks/kaggle-tpu-v5e8-text-vision.ipynb
   ```

5. Click **Import**.
6. In the Kaggle notebook settings, enable **Internet** and select **TPU v5e-8 / `v5litepod-8`**.
7. Use **Add Input / Models** to attach the Keras TranslateGemma model containing the `translategemma_12b_it` preset.
8. Keep:

   ```python
   RUN_TPU_VALIDATION = True
   ```

9. Run the notebook from a clean session with **Restart Session → Run All**.

Importing from GitHub gives Kaggle the notebook itself. The notebook's first code cell then clones or hard-refreshes the official repository into:

```text
/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
```

This keeps GitHub as the single source of truth for the server source, scripts, clients, and tests.

### What Run All does

The notebook will:

1. clone or hard-refresh `main` from this repository;
2. verify a clean Git checkout and print the exact HEAD;
3. run setup and unit/source checks;
4. require the 8-device TPU preflight;
5. start the Flask coordinator and one TPU worker;
6. wait for mesh `[1,8]` readiness;
7. query safe runtime metadata;
8. run text and multipart vision smoke tests;
9. optionally start a Cloudflare Quick Tunnel;
10. print final service status.

Set `RUN_TPU_VALIDATION=False` only when you intentionally want dependency/unit/static validation without initializing TPU or starting the server.

### Alternative: manual clone for developers

Use this workflow when you want to inspect or modify the source directly instead of starting from Kaggle's GitHub notebook importer:

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

## Verify the running service

### Safe runtime info

```bash
python3 clients/python/translategemma_client.py \
  --api-key-file data/api_key.txt info
```

A healthy TPU worker should report the expected 8 devices, mesh `[1,8]`, BF16, split-compile generation, strict weight loading, and vision support.

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

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service metadata |
| `GET` | `/health/live` | Coordinator liveness |
| `GET` | `/health/ready` | Model/worker readiness |
| `GET` | `/info` | Authenticated safe runtime information |
| `POST` | `/translate` | Text translation |
| `POST` | `/translate/async` | Async text translation |
| `POST` | `/translate/image` | Image OCR/translation |
| `POST` | `/translate/image/async` | Async image OCR/translation |
| `GET` | `/result/<job_id>` | Poll an async/cold-compile job |
| `POST` | `/restart` | Protected worker restart |

Authentication accepts either:

```text
Authorization: Bearer <API_KEY>
```

or:

```text
X-API-Key: <API_KEY>
```

`POST /restart` additionally requires the separate restart secret.

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

JSON/base64 image transport is also supported.

## Clients

### Python

No third-party client package is required:

```bash
python3 clients/python/translategemma_client.py \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  text "Good morning! How are you?"
```

The client automatically polls `/result/<job_id>` when the server returns `202`.

### Node.js

Requires Node.js 18+ and uses built-in `fetch`, `FormData`, and `Blob`:

```bash
node clients/node/translategemma-client.mjs text \
  --base-url http://127.0.0.1:7860 \
  --api-key-file data/api_key.txt \
  --text "Good morning! How are you?"
```

See [clients/README.md](clients/README.md) and [docs/API.md](docs/API.md) for more examples.

## Runtime behavior

### Readiness

`/health/live` only means the coordinator process is alive. `/health/ready` stays unavailable while the TPU worker is loading/compiling and becomes `200` only after the model is genuinely ready.

### Cold compilation

The first text or vision request may take several minutes because JAX compiles the relevant prefill/decode shapes. This is expected. A cold request may return `202`; clients should poll the supplied result endpoint rather than treating it as a failure.

### Kaggle JAX / libtpu policy

The project does **not** blindly replace Kaggle's accelerator stack.

- Existing JAX and JAXLIB are retained.
- Existing `libtpu` is retained, even if its version differs from the known reference.
- If `libtpu` is completely missing, `scripts/ensure_libtpu.py` installs `libtpu==0.0.17` with `--no-deps`.
- The real functional gate is still the TPU preflight requiring exactly 8 TPU devices.

### Known startup noise

A successful TPU run can still contain known low-level startup messages involving TPU `SliceBuilder` port 8471 or CUDA `cuInit 303`. Their presence alone does not indicate an inference failure; use the actual TPU preflight, readiness state, and worker metadata as the source of truth. Unknown stderr should still be investigated.

## Configuration

Runtime defaults are documented in `.env.example`. Important defaults include:

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

## Operations

```bash
bash scripts/status.sh
bash scripts/restart.sh
bash scripts/stop.sh
```

Optional tunnel:

```bash
bash scripts/run_tunnel.sh
python3 scripts/demo_info.py
bash scripts/stop_tunnel.sh
```

Keep API authentication enabled whenever the service is exposed outside localhost.

## Development and validation

CPU-only checks:

```bash
python3 -m pip install -r requirements-ci.txt
bash scripts/test_unit.sh
python3 -m compileall -q src scripts
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
node --check clients/node/translategemma-client.mjs
python3 scripts/secret_scan.py .
```

These checks validate source, contracts, packaging, and security, but they do **not** replace an actual Kaggle TPU run.

### Build a clean source archive

```bash
python3 scripts/package_source.py
```

The packager excludes generated credentials, `.env`, runtime logs, PID/tunnel state, caches, and internal development artifacts.

### Tagged GitHub releases

Pushing a release tag such as `v1.0.0` triggers `.github/workflows/release.yml`, which runs CPU verification, builds public artifacts, writes SHA256 files, uploads them, and creates the GitHub Release.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Kaggle guide](docs/KAGGLE.md)
- [REST API](docs/API.md)
- [Validation notes](docs/BENCHMARKS.md)
- [Client examples](clients/README.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Changelog](CHANGELOG.md)

## Security

Never commit or publish:

- `.env`
- `data/api_key.txt`
- `data/restart_secret.txt`
- bearer/API-key values
- request bodies containing sensitive content
- image base64 payloads
- tunnel credentials
- SSH secret keys

See [SECURITY.md](SECURITY.md).

## License and third-party software

Original source code in this repository is released under the [MIT License](LICENSE). TranslateGemma model weights and third-party libraries/tools retain their own licenses and terms; see [NOTICE.md](NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
