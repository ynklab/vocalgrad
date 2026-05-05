from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
from .cli_vocalgrad import build_prompt
from .vocalgrad import (
    DEFAULT_VOCALGRAD_DATASET_ROOT,
    VocalGradSample,
    list_vocalgrad_categories,
    load_local_vocalgrad_samples,
)


LINEAR_PROBE_STAGES = ("mel", "pre_lm_audio", "lm_audio", "lm_text")
DEFAULT_LINEAR_PROBE_FEATURE_ROOT = Path("outputs/features/linear_probe")
DEFAULT_LINEAR_PROBE_ANALYSIS_ROOT = Path("outputs/analysis/linear_probe")


@dataclass(frozen=True)
class CommonMelConfig:
    sample_rate: int = 16_000
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    fmin: float = 0.0
    fmax: float | None = 8_000.0


DEFAULT_COMMON_MEL_CONFIG = CommonMelConfig()


@dataclass(frozen=True)
class ProbeMetrics:
    accuracy: float
    balanced_accuracy: float
    accuracy_up: float | None
    accuracy_down: float | None
    n_train: int
    n_test: int

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "accuracy_up": self.accuracy_up,
            "accuracy_down": self.accuracy_down,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


@dataclass(frozen=True)
class StageFeatureBatch:
    ids: list[str]
    categories: list[str]
    labels: np.ndarray
    features: dict[str, np.ndarray]


@dataclass(frozen=True)
class TorchLinearProbeConfig:
    device: str = "auto"
    epochs: int = 200
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass
class TorchLinearProbe:
    model: object
    mean: object
    std: object
    device: str


def slugify_model_id(model_id: str) -> str:
    lowered = model_id.lower()
    if lowered == "moonshotai/kimi-audio-7b-instruct":
        return "kimi-audio"
    if lowered == "nvidia/audio-flamingo-3-hf":
        return "audioflamingo3"
    if lowered == "stepfun-ai/step-audio-2-mini":
        return "step-audio-2-mini"
    if lowered == "xiaomimimo/mimo-audio-7b-instruct":
        return "mimo-audio"
    return model_id.split("/")[-1].lower().replace("/", "-")


def label_to_int(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "increase":
        return 1
    if normalized == "decrease":
        return 0
    raise ValueError(f"unsupported label: {label}")


def int_to_label(value: int) -> str:
    if int(value) == 1:
        return "increase"
    if int(value) == 0:
        return "decrease"
    raise ValueError(f"unsupported label id: {value}")


def common_log_mel(audio_path: Path, config: CommonMelConfig = DEFAULT_COMMON_MEL_CONFIG) -> np.ndarray:
    waveform, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.T.astype(np.float32, copy=False)


def segment_mean_pool(sequence: np.ndarray, num_segments: int = 16) -> np.ndarray:
    array = np.asarray(sequence, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"expected 2D sequence, got shape={array.shape}")

    length, dim = array.shape
    pooled = np.zeros((num_segments, dim), dtype=np.float32)
    if length == 0:
        return pooled

    boundaries = np.linspace(0, length, num_segments + 1, dtype=np.int64)
    for idx in range(num_segments):
        start = int(boundaries[idx])
        end = int(boundaries[idx + 1])
        if end > start:
            pooled[idx] = array[start:end].mean(axis=0, dtype=np.float32)
        else:
            source_idx = min(start, length - 1)
            pooled[idx] = array[source_idx]
    return pooled


def flatten_segmented_sequence(sequence: np.ndarray, num_segments: int = 16) -> np.ndarray:
    return segment_mean_pool(sequence, num_segments=num_segments).reshape(-1).astype(np.float32, copy=False)


def sample_prompt(sample: VocalGradSample) -> str:
    return build_prompt(sample.category)


def collect_vocalgrad_samples(
    dataset_root: Path = DEFAULT_VOCALGRAD_DATASET_ROOT,
    *,
    categories: Iterable[str] | None = None,
    max_samples_per_category: int | None = None,
) -> list[VocalGradSample]:
    if categories is None:
        category_list = list_vocalgrad_categories(dataset_root)
    else:
        category_list = list(categories)

    samples: list[VocalGradSample] = []
    for category in category_list:
        category_samples = load_local_vocalgrad_samples(dataset_root=dataset_root, category=category)
        if max_samples_per_category is not None:
            category_samples = category_samples[:max_samples_per_category]
        samples.extend(category_samples)
    return samples


def _resolve_torch_device(requested: str) -> str:
    import torch

    normalized = requested.strip().lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
    return normalized


def train_logistic_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    random_state: int = 1234,
    max_iter: int = 200,
    c_value: float = 1.0,
    device: str = "auto",
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    show_progress: bool = False,
    progress_label: str | None = None,
):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.auto import tqdm

    resolved_device = _resolve_torch_device(device)
    torch.manual_seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    x_np = np.asarray(train_x, dtype=np.float32)
    y_np = np.asarray(train_y, dtype=np.float32).reshape(-1, 1)
    mean = x_np.mean(axis=0, keepdims=True)
    std = x_np.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32, copy=False)
    x_scaled = ((x_np - mean) / std).astype(np.float32, copy=False)

    x_tensor = torch.from_numpy(x_scaled)
    y_tensor = torch.from_numpy(y_np)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=True,
        drop_last=False,
        pin_memory=(resolved_device == "cuda"),
    )

    model = nn.Linear(x_scaled.shape[1], 1).to(resolved_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoch_iterator = range(max_iter)
    if show_progress:
        epoch_iterator = tqdm(epoch_iterator, desc=progress_label or "train_probe", leave=False)

    for _ in epoch_iterator:
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            batch_y = batch_y.to(resolved_device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    mean_tensor = torch.from_numpy(mean.astype(np.float32, copy=False)).to(resolved_device)
    std_tensor = torch.from_numpy(std.astype(np.float32, copy=False)).to(resolved_device)
    model.eval()
    return TorchLinearProbe(model=model, mean=mean_tensor, std=std_tensor, device=resolved_device)


def evaluate_probe(
    model: TorchLinearProbe,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> ProbeMetrics:
    import torch

    n_train = int(np.asarray(train_y).shape[0])
    x_np = np.asarray(test_x, dtype=np.float32)
    y_np = np.asarray(test_y, dtype=np.int64)

    with torch.inference_mode():
        x_tensor = torch.from_numpy(x_np).to(model.device)
        x_scaled = (x_tensor - model.mean) / model.std
        logits = model.model(x_scaled)
        pred = (torch.sigmoid(logits) >= 0.5).to(torch.int64).view(-1).cpu().numpy()

    accuracy = float((pred == y_np).mean())

    def _class_accuracy(target: int) -> float | None:
        mask = y_np == target
        if not np.any(mask):
            return None
        return float((pred[mask] == y_np[mask]).mean())

    accuracy_up = _class_accuracy(1)
    accuracy_down = _class_accuracy(0)
    balanced_parts = [value for value in (accuracy_up, accuracy_down) if value is not None]
    balanced_accuracy = float(sum(balanced_parts) / len(balanced_parts)) if balanced_parts else accuracy

    return ProbeMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        accuracy_up=accuracy_up,
        accuracy_down=accuracy_down,
        n_train=n_train,
        n_test=int(y_np.shape[0]),
    )
