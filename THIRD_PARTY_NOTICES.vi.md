# Thông báo phần mềm bên thứ ba

> 🌐 Language / Ngôn ngữ: [English](THIRD_PARTY_NOTICES.md) | **Tiếng Việt**

File này là informational inventory, không thay thế license text do từng dependency hoặc service phân phối.

## Runtime libraries

| Component | Vai trò | Nguồn điều khoản |
|---|---|---|
| Keras | Model/runtime API | Upstream Keras project |
| KerasHub | TranslateGemma model implementation | Upstream KerasHub project |
| JAX / jaxlib | Accelerator execution | Upstream JAX project |
| libtpu | TPU runtime integration | Package/provider terms |
| Flask | REST coordinator | Upstream Flask project |
| Pillow | Image decoding/validation | Upstream Pillow project |
| NumPy | Array operations | Upstream NumPy project |

## External platforms và tools

| Component | Vai trò | Ghi chú |
|---|---|---|
| Kaggle | Notebook compute và model attachment | Theo Kaggle terms |
| GitHub / GitHub Actions | Source hosting và CI | Theo GitHub terms |
| Cloudflare Quick Tunnel | Temporary tunnel tùy chọn | Theo Cloudflare terms |

## Model

TranslateGemma 12B IT weights nằm ngoài repository này. Project không redistribute hoặc relicense các weights đó.

## Generated artifacts

Source ZIP tạo bởi `scripts/package_source.py` chỉ chứa project source và documentation. Runtime-generated credentials, logs/state, model weights và downloaded service binaries bị loại.

## Ghi chú cho maintainer

Khi thêm dependency mới, cập nhật tài liệu này và bản tiếng Việt trong cùng change và giữ số dòng bằng nhau.
