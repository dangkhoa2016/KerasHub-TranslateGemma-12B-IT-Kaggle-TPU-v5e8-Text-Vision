# Security policy

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](SECURITY.vi.md)

## Project and deployment scope

This public community project provides a KerasHub/TranslateGemma service and its supporting scripts.

Cloudflare Quick Tunnel and Kaggle notebooks are demonstration/development environments, not production hosting.

Production deployments are responsible for their own network, operational, and access-control design.

## Supported versions

Security fixes target the current `main` branch and the latest tagged public release.

## Reporting a vulnerability

Do not include credentials, tokens, exploitable details, or sensitive user content in a public issue.

Use GitHub Security Advisories when available to report a vulnerability privately.

## Implemented security boundary

API authentication defaults to enabled; translation, result, and info routes require the API key.

Restart uses a separate restart secret and requires the `X-Restart-Secret` header.

Basic health responses are public, while detailed health responses require the API key.

Configured bounds cover request bytes, text length, image bytes/pixels, output tokens, queue size, stored results, and result TTL.

## Deployment hardening

TLS termination, rate limiting, and tenant isolation must be supplied by a production deployment.

Keep API authentication enabled whenever the service is reachable outside localhost.

Restrict service exposure, protect credentials, and apply monitoring appropriate to the deployment environment.

## Sensitive runtime material

`.env` is optional local configuration and is not present in a clean checkout.

`data/api_key.txt`, `data/restart_secret.txt`, and `data/tunnel_url.txt` are runtime-generated when their related features run.

Generated secret files use restrictive permissions and must not be committed or shared.

- `.env`
- `data/api_key.txt`
- `data/restart_secret.txt`
- `data/tunnel_url.txt`
- bearer/API-key values, tunnel credentials, SSH secret keys, model access tokens, and sensitive request payloads

## Before sharing an artifact

Run:

```bash
python3 scripts/secret_scan.py .
python3 scripts/package_source.py /tmp/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision-v1.0.0.zip
python3 scripts/secret_scan.py /tmp/KerasHub-TranslateGemma-12B-IT-Kaggle-TPU-v5e8-Text-Vision-v1.0.0.zip
```

Clean source packaging excludes optional local configuration, runtime-generated credentials, logs, process state, tunnel state, caches, and generated archives.

## Dependency policy

The Kaggle runtime intentionally preserves the accelerator-compatible JAX/JAXLIB installation.

Do not add blind JAX/JAXLIB upgrades to normal setup; `libtpu` bootstrap is conditional and uses `--no-deps` when the package is absent.
