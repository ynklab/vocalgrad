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

Our MiMo-Audio environment uses Python 3.12.11 on Linux aarch64, PyTorch
2.8.0+cu129, torchaudio 2.8.0, Triton 3.4.0, Transformers 4.49.0, and
FlashAttention 2.7.4.post1. Use a CUDA-capable GPU and a compatible NVIDIA
driver. Building FlashAttention also requires a CUDA 12.9 toolkit (`nvcc`),
a C++ compiler, and `CUDA_HOME` pointing to the toolkit when it is not
automatically detected.

Run the following from `benchmarking-and-analysis/`. The runtime expects the
[upstream MiMo-Audio checkout](https://github.com/XiaomiMiMo/MiMo-Audio)
at exactly `third_party/MiMo-Audio`:

```bash
mkdir -p third_party
git clone https://github.com/XiaomiMiMo/MiMo-Audio.git third_party/MiMo-Audio
git -C third_party/MiMo-Audio checkout --detach 62d956b4a1a45419bee5e41f477078c3684dbbcc
uv venv --no-project --python 3.12.11 .venv-mimo
uv pip install --python .venv-mimo/bin/python -r requirements/mimo.txt
uv pip install --python .venv-mimo/bin/python packaging setuptools wheel ninja
FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=4 uv pip install \
  --python .venv-mimo/bin/python --no-build-isolation --no-deps \
  flash-attn==2.7.4.post1
uv pip check --python .venv-mimo/bin/python
```

If the upstream checkout already exists, verify that it is clean before
checking out the specified revision. The requirements file pins the runtime
packages to the versions in our environment, including the Python 3.12 aarch64
GPU wheels. Transitive dependencies are resolved at installation time. The
upstream `requirements.txt` pins a different PyTorch/Triton stack; use the
requirements file above for these experiments.

MiMo uses `uv pip` with an explicit Python path because the main project and
its `uv.lock` target Python 3.10. Do not install this project with `uv sync`
or `pip install -e .` into `.venv-mimo`. The reproduction wrappers expose
`src/` through `PYTHONPATH`, and the runtime adds the upstream checkout to its
import path. The base `.venv` analysis environment is still needed for probe
training and result aggregation.

The following check imports the runtime without loading model weights. Run it
on the configured GPU machine after installation:

```bash
PYTHONPATH=src:third_party/MiMo-Audio .venv-mimo/bin/python -c \
  'import torch, torchaudio, flash_attn, peft; from src.mimo_audio.mimo_audio import MimoAudio; from plic.mimo_finetune import load_mimo_runtime; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

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

The runtime downloads the model and audio tokenizer from Hugging Face when
first loaded. For an offline run, populate the same Hugging Face cache first:

```bash
.venv-mimo/bin/hf download XiaomiMiMo/MiMo-Audio-7B-Instruct
.venv-mimo/bin/hf download XiaomiMiMo/MiMo-Audio-Tokenizer
```

With the [dataset layout](data_setup.md) prepared, run a small benchmark in a
separate output directory before the full experiment:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=mimoaudio \
  CATEGORIES=volume MAX_SAMPLES=2 RUN_TABLES=0 \
  RAW_ROOT=outputs/raw/mimo_smoke ANALYSIS_ROOT=outputs/analysis/mimo_smoke
bash scripts/reproduce/01_benchmark.sh BACKEND=mimoaudio
bash scripts/reproduce/03_linear_probe_audio.sh MODEL_STEMS=mimo-audio
bash scripts/reproduce/04_linear_probe_layers.sh MODEL_STEMS=mimo-audio
bash scripts/reproduce/05_finetune_lora.sh MODEL_STEM=mimo-audio \
  TRAIN_CATEGORY=volume SCOPE=all_linear
```

The runtime defaults to seed 1234 (`MIMOAUDIO_SEED`). See
[the experiment map](paper_experiments.md) for the remaining fine-tuning
categories, scopes, and appendix conditions.

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
