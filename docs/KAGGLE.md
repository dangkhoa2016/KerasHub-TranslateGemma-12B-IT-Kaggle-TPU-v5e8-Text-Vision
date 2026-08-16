# Kaggle guide

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](KAGGLE.vi.md)

## Recommended workflow: import from GitHub

1. Create a new Kaggle Notebook.
2. Open **File → Import Notebook → GitHub**.
3. Search for `dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision`.
4. Select `notebooks/kaggle-tpu-v5e8-text-vision.ipynb`.
5. Click **Import**.
6. Enable **Internet**.
7. Select **TPU v5e-8 / `v5litepod-8`**.
8. Use **Add Input / Models** to attach the Keras TranslateGemma model containing `translategemma_12b_it`.
9. Keep `RUN_TPU_VALIDATION=True`.
10. Use **Restart Session → Run All**.

## Why the notebook clones the repository again

GitHub Import places the notebook cells into Kaggle. The first code cell then clones or hard-refreshes `main` into:

```text
/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
```

This keeps server source, scripts, clients, tests, and documentation anchored to the repository rather than to copied notebook code.

## What Run All validates

The notebook:

1. checks the official Git checkout and prints the exact HEAD;
2. requires a clean working tree;
3. runs dependency/setup checks;
4. runs CPU-friendly unit tests;
5. requires exactly 8 TPU devices when TPU validation is enabled;
6. starts the Flask coordinator and single TPU worker;
7. waits for mesh `[1,8]` readiness;
8. queries authenticated runtime information;
9. runs text translation;
10. runs multipart vision translation;
11. optionally starts a Cloudflare Quick Tunnel;
12. prints final service status.

## CPU/source-only mode

Set:

```python
RUN_TPU_VALIDATION = False
```

only when you intentionally want dependency, unit, and static checks without TPU initialization or server startup. This mode is not evidence of accelerator inference.

## Manual clone for developers

```bash
git clone https://github.com/dangkhoa2016/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision.git \
  /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
cd /kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision
cp .env.example .env
INSTALL_PYTHON_DEPS=auto TPU_PREFLIGHT_MODE=required bash scripts/setup.sh
bash scripts/start.sh
```

## Troubleshooting order

If a run fails, first identify whether the failure happened during Git/source checks, dependency/bootstrap, TPU preflight, model loading, or REST/inference. Avoid changing the TPU engine for failures that occur before model initialization.

## TPU quota

Use CPU/source-only mode for documentation and ordinary unit-test work. Reserve TPU sessions for changes that need real accelerator acceptance testing.
