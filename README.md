# 3D-hand-FM

## Setup

Install [uv package and project manager](https://docs.astral.sh/uv/).


Install the dependencies:

```bash
uv sync
```

## Training

```bash
env $(cat .env | xargs) uv run src/train.py
```
