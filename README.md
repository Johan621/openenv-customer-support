---
title: Openenv Customer Support
emoji: 🔥
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

# OpenEnv Customer Support

A Docker-deployed OpenEnv-compatible customer support ticket triage environment + API server.

This Space/repo is set up to:
- build cleanly with Docker
- expose a FastAPI/uvicorn server
- pass `openenv validate` for multi-mode deployment

## What was updated to make everything work

### 1) Docker build fixes
The Docker image build was stabilized by:
- copying the full project into the image before running `pip install -e .`
- disabling pip’s progress bar inside the container to avoid thread-related build failures on some systems
- simplifying the install step so the build is repeatable

### 2) OpenEnv validation requirements
To satisfy OpenEnv validation checks:
- added the required dependency:
  - `openenv-core>=0.2.0`
- added the required script entry point:
  - `[project.scripts] server = "server.app:main"`
- regenerated `uv.lock` so it matches `pyproject.toml`

## Running locally

### Build
```bash
docker build -t openenv-customer-support .
```

### Run
```bash
docker run --rm -p 7860:7860 openenv-customer-support
```

Open:
- http://localhost:7860

## Validation

Run OpenEnv validation from the repository root:
```bash
openenv validate
```

If your repository includes the validator script, you can also run:
```bash
./scripts/validate-submission.sh https://johan45-openenv-customer-support.hf.space .
```

## Project layout (high level)

- `server/` — FastAPI application and environment entry points
- `Dockerfile` — container build used for deployment
- `pyproject.toml` — Python package metadata + dependencies
- `uv.lock` — locked dependency set used by validation

## Do we need `requirements.txt`?

No. This project is deployed via **Docker** and installs dependencies from `pyproject.toml` (with `uv.lock` kept in sync).  
Add a `requirements.txt` only if some external tool explicitly requires it.