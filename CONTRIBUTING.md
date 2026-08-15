# Contributing

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](CONTRIBUTING.vi.md)

Thank you for helping improve this project.

## Scope

Contributions are welcome for reliability, documentation, tests, API ergonomics, security hardening, Kaggle usability, and clearly reproduced runtime defects.

Please avoid architecture changes that are unrelated to a reproduced problem.

## Runtime invariants

Keep these properties unless a real Kaggle TPU failure demonstrates that a change is necessary:

- one logical TranslateGemma 12B model;
- one TPU inference worker;
- exactly 8 TPU devices;
- ModelParallel mesh `[1,8]`;
- BF16 inference;
- strict checkpoint loading;
- split prefill/decode compilation;
- CPU-side Flask coordinator;
- `202 + /result/<job_id>` for long-running requests.

## Development workflow

1. Create a focused branch.
2. Add or update tests before changing behavior.
3. Keep the notebook checkout at `/kaggle/working/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision`.
4. Run the CPU-friendly verification suite locally.
5. Use Kaggle TPU v5e-8 for accelerator-specific acceptance testing when the change touches runtime behavior.
6. Keep commits small enough to review independently.

## Required checks

```bash
python3 -m pip install -r requirements-ci.txt
bash scripts/test_unit.sh
python3 scripts/check_docs.py
python3 -m compileall -q src scripts tests
find scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
node --check clients/node/translategemma-client.mjs
python3 scripts/secret_scan.py .
```

## Documentation

Every Markdown document must have an English/Vietnamese pair with equal line counts. Keep technical facts, commands, paths, endpoint names, environment variables, and release identity synchronized between both languages.

## Pull requests

Describe the problem, the smallest solution, tests executed, and whether a real Kaggle TPU run was performed. Do not claim accelerator validation when only CPU/static checks were run.

See [.github/PULL_REQUEST_TEMPLATE.md](.github/PULL_REQUEST_TEMPLATE.md) for the review checklist.
