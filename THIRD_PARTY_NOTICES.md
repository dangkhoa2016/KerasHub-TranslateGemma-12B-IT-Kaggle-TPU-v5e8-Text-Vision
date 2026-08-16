# Third-party notices

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](THIRD_PARTY_NOTICES.vi.md)

This file is an informational inventory, not a replacement for the license text distributed by each dependency or service.

## Runtime libraries

| Component | Role | Source of terms |
|---|---|---|
| Keras | Model/runtime API | Upstream Keras project |
| KerasHub | TranslateGemma model implementation | Upstream KerasHub project |
| JAX / jaxlib | Accelerator execution | Upstream JAX project |
| libtpu | TPU runtime integration | Package/provider terms |
| Flask | REST coordinator | Upstream Flask project |
| Pillow | Image decoding/validation | Upstream Pillow project |
| NumPy | Array operations | Upstream NumPy project |

## External platforms and tools

| Component | Role | Notes |
|---|---|---|
| Kaggle | Notebook compute and model attachment | Governed by Kaggle terms |
| GitHub / GitHub Actions | Source hosting and CI | Governed by GitHub terms |
| Cloudflare Quick Tunnel | Optional temporary tunnel | Governed by Cloudflare terms |

## Model

TranslateGemma 12B IT weights are external to this repository. The project does not redistribute or relicense those weights.

## Generated artifacts

Source ZIP files created by `scripts/package_source.py` contain project source and documentation only. Runtime-generated credentials, logs/state, model weights, and downloaded service binaries are excluded.

## Maintainer note

When adding a new dependency, update this document and its Vietnamese counterpart in the same change and keep their line counts equal.
