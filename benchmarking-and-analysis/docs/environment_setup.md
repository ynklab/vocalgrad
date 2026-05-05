# Environment Setup

The reproduction code uses separate Python environments for heavyweight audio
backends. This avoids incompatible `transformers`, CUDA, and model-specific
runtime dependencies.

All commands below are run from the repository root.

## Base Analysis Environment

Use this environment for analysis scripts, tables, plots, and lightweight
utilities.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate
uv sync --extra analysis
```

## Kimi-Audio

```bash
uv venv --python 3.10 .venv-kimi
source .venv-kimi/bin/activate
uv sync --extra kimia
```

Kimi-Audio may require `flash-attn`. This anonymized repository does not pin a
machine-local wheel path; if `uv sync --extra kimia` cannot build or resolve
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
source .venv-stepaudio/bin/activate
uv sync --extra stepaudio2
```

Default model id:

```text
stepfun-ai/Step-Audio-2-mini
```

## AudioFlamingo3

```bash
uv venv --python 3.10 .venv-af3
source .venv-af3/bin/activate
uv sync --extra audioflamingo3
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

- infers the repository root from its own location
- sets `PYTHONPATH=<repo>/src`
- loads `.env` when present
- avoids scheduler-specific settings

Examples:

```bash
bash scripts/reproduce/01_benchmark.sh BACKEND=kimia
bash scripts/reproduce/03_linear_probe_audio.sh MODEL_STEMS="kimi-audio mimo-audio"
```
