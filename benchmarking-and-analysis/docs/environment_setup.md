# Environment Setup

The reproduction code uses separate Python environments for heavyweight audio
backends. This avoids incompatible `transformers`, CUDA, and model-specific
runtime dependencies.

All commands below are run from `benchmarking-and-analysis/`.

## Our Environment

Our Kimi-Audio environment uses Linux aarch64, Python 3.10.18,
PyTorch 2.8.0+cu129, and Triton 3.4.0. The dependency configuration retains
the CUDA 12.9 PyTorch index and the CPython 3.10 Linux aarch64 Triton wheel
used for this environment. These installation commands target that platform.

Each sync command explicitly selects the environment used by the reproduction
scripts with `UV_PROJECT_ENVIRONMENT`; activating a virtual environment alone
does not select the target of `uv sync`.

## Base Analysis Environment

Use this environment for analysis scripts, tables, plots, and linear-probe
training. The `analysis` extra includes PyTorch for probe training as well as
the plotting and analysis dependencies.

```bash
uv venv --python 3.10 .venv
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --extra analysis
```

## Kimi-Audio

```bash
uv venv --python 3.10 .venv-kimi
UV_PROJECT_ENVIRONMENT=.venv-kimi uv sync --locked --extra kimia
```

Kimi-Audio may require `flash-attn`. This repository does not pin a
machine-local wheel path; if the Kimi-Audio sync command cannot build or resolve
`flash-attn` on your platform, install a compatible wheel or source build in
`.venv-kimi` according to your CUDA, Python, and architecture versions.

Default model id:

```text
moonshotai/Kimi-Audio-7B-Instruct
```

## MiMo-Audio

MiMo-Audio is isolated because its reference stack may require a different
Python and CUDA dependency set.

```bash
uv venv --python 3.12 .venv-mimo
source .venv-mimo/bin/activate
```

Install the MiMo-Audio reference dependencies according to the upstream project
instructions, then make sure the repository `src/` directory is on
`PYTHONPATH`. The reproduction scripts set `PYTHONPATH` automatically.

Default model ids:

```text
XiaomiMiMo/MiMo-Audio-7B-Instruct
XiaomiMiMo/MiMo-Audio-Tokenizer
```

For MiMo-Audio, this repository's runtime calls the upstream
`audio_understanding` generation path without passing local sampling parameters.
MiMo therefore uses the upstream sampling defaults, including temperature 0.3
and top-p 0.95. The local CLI temperature field may still appear in generated
run manifests, but it is not passed into the MiMo-Audio generation call.

## Step-Audio-2 Mini

```bash
uv venv --python 3.10 .venv-stepaudio
UV_PROJECT_ENVIRONMENT=.venv-stepaudio uv sync --locked --extra stepaudio2
```

Default model id:

```text
stepfun-ai/Step-Audio-2-mini
```

## AudioFlamingo3

```bash
uv venv --python 3.10 .venv-af3
UV_PROJECT_ENVIRONMENT=.venv-af3 uv sync --locked --extra audioflamingo3
```

Default model id:

```text
nvidia/audio-flamingo-3-hf
```

## Gemini

Gemini can run from the base environment if `google-genai` is installed.
Provide credentials through `.env` or the shell environment.

```bash
export GOOGLE_API_KEY=...
```

The paper uses Gemini 3 Flash with low thinking level and temperature 1.0.
The runtime code applies the Gemini temperature default for Gemini 3 models.

## Running Reproduction Scripts

Every script under `scripts/reproduce/`:

- infers the `benchmarking-and-analysis/` root from its own location
- sets `PYTHONPATH=<benchmarking-and-analysis>/src`
- loads `.env` when present
- avoids scheduler-specific settings

Examples:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia
bash scripts/reproduce/03_linear_probe_audio.sh MODEL_STEMS="kimi-audio mimo-audio"
```
