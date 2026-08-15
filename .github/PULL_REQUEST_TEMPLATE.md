# Pull request

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](PULL_REQUEST_TEMPLATE.vi.md)

## Summary

Describe the problem and the smallest change that solves it.

## Scope

- [ ] The change is focused and does not include unrelated refactoring.
- [ ] Runtime architecture invariants are preserved unless the PR contains direct failure evidence requiring a change.
- [ ] No credentials, model weights, runtime logs/state, or generated archives are committed.

## Tests

- [ ] `bash scripts/test_unit.sh`
- [ ] `python3 scripts/check_docs.py`
- [ ] `python3 -m compileall -q src scripts tests`
- [ ] Bash syntax checks
- [ ] Node client syntax check
- [ ] `python3 scripts/secret_scan.py .`

## Kaggle TPU validation

- [ ] Not required for this change, or
- [ ] Real Kaggle TPU v5e-8 validation was run and the exact Git SHA is included below.

Exact tested SHA / notes:

```text
<enter evidence when applicable>
```

## Documentation

- [ ] English and Vietnamese documents were updated together.
- [ ] Paired Markdown files have equal line counts.
- [ ] Notebook Markdown remains bilingual and user-facing.

## Reviewer notes

Call out compatibility risks, known limitations, or follow-up work that is intentionally outside this PR.
