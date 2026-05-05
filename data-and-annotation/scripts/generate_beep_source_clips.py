from __future__ import annotations

import csv
import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
OUTPUT_ROOT = Path("data/selected/source_clips/ablation_beep")
OUTPUT_AUDIO_DIR = OUTPUT_ROOT / "audio" / "beep"
MANIFEST_PATH = OUTPUT_ROOT / "manifest.csv"
README_PATH = OUTPUT_ROOT / "README.md"

ONSET_SEC = 0.75
TAIL_SEC = 0.75
ACTIVE_DURATION_SEC = 2.10
BASE_FREQUENCIES_HZ = [700, 850, 1000, 1150, 1300, 1450, 1600, 1750, 1900, 2050]


def synthesize_clip(base_frequency_hz: float) -> tuple[np.ndarray, dict[str, float | int]]:
    onset_samples = int(round(ONSET_SEC * SAMPLE_RATE))
    active_samples = int(round(ACTIVE_DURATION_SEC * SAMPLE_RATE))
    tail_samples = int(round(TAIL_SEC * SAMPLE_RATE))
    total_samples = onset_samples + active_samples + tail_samples

    audio = np.zeros(total_samples, dtype=np.float64)
    t = np.arange(active_samples, dtype=np.float64) / SAMPLE_RATE
    carrier = np.sin(2.0 * math.pi * base_frequency_hz * t)
    active_signal = 0.65 * carrier
    audio[onset_samples : onset_samples + active_samples] = active_signal

    offset_sec = ONSET_SEC + active_samples / SAMPLE_RATE
    peak_rms = float(np.sqrt(np.mean(np.square(active_signal))))

    metadata = {
        "audio_duration_sec": total_samples / SAMPLE_RATE,
        "offset_sec": offset_sec,
        "speech_duration_sec": active_samples / SAMPLE_RATE,
        "onset_num_segments": 1,
        "onset_peak_rms": peak_rms,
    }
    return audio, metadata


def write_wav(path: Path, audio: np.ndarray) -> None:
    pcm16 = np.round(np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm16.tobytes())


def write_readme() -> None:
    README_PATH.write_text(
        "# Ablation Beep Source Clips\n\n"
        "This directory contains 10 synthetic machine-like beep source clips "
        "for ablation studies.\n\n"
        "## Design\n\n"
        "- mono 48 kHz WAV\n"
        "- pure sine-wave carrier only\n"
        "- no pulse-shaped amplitude envelope\n"
        "- no vibrato or frequency modulation\n"
        "- all clips share the same duration and timing\n"
        "- only the base frequency changes across clips\n",
    )


def build_manifest_rows() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for index, base_frequency_hz in enumerate(BASE_FREQUENCIES_HZ, start=1):
        source_item_id = f"beep_{index:02d}"
        audio, metadata = synthesize_clip(base_frequency_hz)
        wav_name = f"{source_item_id}.wav"
        write_wav(OUTPUT_AUDIO_DIR / wav_name, audio)

        rows.append(
            {
                "source_item_id": source_item_id,
                "clip_id": source_item_id,
                "speaker_id": "beep",
                "sentence_id": f"{index:03d}",
                "audio_path": str(OUTPUT_AUDIO_DIR / wav_name),
                "selected_audio_path": f"audio/beep/{wav_name}",
                "original_audio_path": "",
                "text_path": "",
                "text": f"synthetic machine beep clip {index:02d}",
                "audio_duration_sec": f"{metadata['audio_duration_sec']:.6f}",
                "split": "ablation_beep_source",
                "onset_detected": "True",
                "onset_sec": f"{ONSET_SEC:.6f}",
                "offset_sec": f"{metadata['offset_sec']:.6f}",
                "speech_duration_sec": f"{metadata['speech_duration_sec']:.6f}",
                "onset_num_segments": int(metadata["onset_num_segments"]),
                "onset_threshold": "",
                "onset_noise_floor": "",
                "onset_peak_rms": f"{metadata['onset_peak_rms']:.8f}",
                "base_frequency_hz": base_frequency_hz,
            }
        )
    return rows


def main() -> None:
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_manifest_rows()
    with MANIFEST_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_readme()
    print(f"Wrote {len(rows)} beep source clips to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
