#!/usr/bin/env bash

set -euo pipefail

repro_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$repro_dir/../.." && pwd)"
cd "$repo_root"

export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${LOAD_DOTENV:-1}" == "1" && -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

PAPER_CATEGORIES_DEFAULT="speaking_speed voice_pitch volume audio_distortion audio_roughness background_noise echo voice_clarity voice_vibration"
BEEP_CATEGORIES_DEFAULT="voice_pitch volume background_noise voice_vibration"

resolve_python_for_backend() {
  case "$1" in
    kimia) echo ".venv-kimi/bin/python" ;;
    mimoaudio) echo ".venv-mimo/bin/python" ;;
    stepaudio2) echo ".venv-stepaudio/bin/python" ;;
    audioflamingo3) echo ".venv-af3/bin/python" ;;
    gemini)
      if [[ -x ".venv-gemini/bin/python" ]]; then
        echo ".venv-gemini/bin/python"
      else
        echo ".venv/bin/python"
      fi
      ;;
    *) echo ".venv/bin/python" ;;
  esac
}

resolve_backend_for_model_stem() {
  case "$1" in
    kimi-audio) echo "kimia" ;;
    mimo-audio) echo "mimoaudio" ;;
    step-audio-2-mini) echo "stepaudio2" ;;
    audioflamingo3) echo "audioflamingo3" ;;
    *) echo "unsupported model stem: $1" >&2; return 1 ;;
  esac
}

resolve_model_id_for_model_stem() {
  case "$1" in
    kimi-audio) echo "moonshotai/Kimi-Audio-7B-Instruct" ;;
    mimo-audio) echo "XiaomiMiMo/MiMo-Audio-7B-Instruct" ;;
    step-audio-2-mini) echo "stepfun-ai/Step-Audio-2-mini" ;;
    audioflamingo3) echo "nvidia/audio-flamingo-3-hf" ;;
    *) echo "unsupported model stem: $1" >&2; return 1 ;;
  esac
}

variant_flags() {
  case "$1" in
    default) ;;
    audio-ref) echo "--explicit-audio-clip-reference" ;;
    swap-order) echo "--swap-direction-order" ;;
    swap-order_audio-ref) echo "--swap-direction-order --explicit-audio-clip-reference" ;;
    *) echo "unsupported variant: $1" >&2; return 1 ;;
  esac
}
