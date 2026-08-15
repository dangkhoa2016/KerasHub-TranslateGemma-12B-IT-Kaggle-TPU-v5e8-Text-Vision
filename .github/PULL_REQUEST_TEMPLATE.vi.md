# Pull request

> 🌐 Language / Ngôn ngữ: [English](PULL_REQUEST_TEMPLATE.md) | **Tiếng Việt**

## Tóm tắt

Mô tả vấn đề và change nhỏ nhất giải quyết được nó.

## Phạm vi

- [ ] Change có phạm vi rõ ràng và không chứa unrelated refactoring.
- [ ] Runtime architecture invariants được giữ trừ khi PR có direct failure evidence bắt buộc thay đổi.
- [ ] Không commit credentials, model weights, runtime logs/state hoặc generated archives.

## Tests

- [ ] `bash scripts/test_unit.sh`
- [ ] `python3 scripts/check_docs.py`
- [ ] `python3 -m compileall -q src scripts tests`
- [ ] Bash syntax checks
- [ ] Node client syntax check
- [ ] `python3 scripts/secret_scan.py .`

## Kaggle TPU validation

- [ ] Không cần cho change này, hoặc
- [ ] Đã chạy real Kaggle TPU v5e-8 validation và exact Git SHA được ghi bên dưới.

Exact tested SHA / notes:

```text
<điền evidence khi áp dụng>
```

## Documentation

- [ ] English và Vietnamese documents được cập nhật cùng nhau.
- [ ] Paired Markdown files có số dòng bằng nhau.
- [ ] Notebook Markdown vẫn song ngữ và user-facing.

## Ghi chú cho reviewer

Nêu compatibility risks, known limitations hoặc follow-up work được chủ ý để ngoài PR này.
