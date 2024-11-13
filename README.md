# 3D-hand-FM

## Setup

Install [uv package and project manager](https://docs.astral.sh/uv/).


Install the dependencies:

```bash
uv pip install pip setuptools ninja
uv sync
```

`pip` and `setuptools` are required to install `chumpy` which is a dependency of `manopth`, see [This issue](https://github.com/astral-sh/uv/issues/7291) for more details.

`ninja` is useful as a faster build system compared to distutils in order to build `pytorch3d`

## Training

```bash
env $(cat .env | xargs) uv run src/train.py
```
