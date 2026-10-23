"""Bidirectional RCG/ECG inference used by the Web UI.

The model and diffusion schedule mirror the two training notebooks in this
directory. One checkpoint generates ECG from RCG and the other generates RCG
from ECG. The notebooks use the same architecture and differ mainly in the
training data folder and whether the RCG column is named RCG or RCG_ch1.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SIGNAL_LENGTH = 640
WINDOW_STEP = 30
DEFAULT_T = 100
DEFAULT_BETA_START = 1e-4
DEFAULT_BETA_END = 0.02
DEFAULT_BASE_CHANNELS = 64
DEFAULT_TIME_EMB_DIM = 128


def _normalise_column_names(names: Iterable[str]) -> Dict[str, str]:
    return {str(name).strip().lower(): str(name) for name in names}


def _read_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(
        str(path), delimiter=",", names=True, dtype=float, encoding="utf-8"
    )
    if data.dtype.names is None:
        raise ValueError("CSV must contain a header row")
    columns = _normalise_column_names(data.dtype.names)
    ecg_name = columns.get("ecg")
    rcg_name = columns.get("rcg") or columns.get("rcg_ch1")
    if ecg_name is None or rcg_name is None:
        raise ValueError("CSV must contain ECG and RCG or RCG_ch1 columns")

    ecg = np.asarray(data[ecg_name], dtype=np.float32).reshape(-1)
    rcg = np.asarray(data[rcg_name], dtype=np.float32).reshape(-1)
    if ecg.size != rcg.size:
        raise ValueError("ECG and RCG sample counts do not match")
    if ecg.size < SIGNAL_LENGTH:
        raise ValueError("CSV must contain at least 640 paired samples")
    if not np.isfinite(ecg).all() or not np.isfinite(rcg).all():
        raise ValueError("ECG or RCG contains NaN or infinite values")
    return rcg, ecg


def sliding_window_cut(data: np.ndarray, window_len: int = SIGNAL_LENGTH, step: int = WINDOW_STEP):
    if data.size < window_len:
        return []
    return [data[start : start + window_len] for start in range(0, len(data) - window_len + 1, step)]


def load_dataset_files(file_paths: Sequence[Path], step: int = WINDOW_STEP) -> List[dict]:
    """Load notebook-compatible CSV files and return paired inference windows."""
    windows: List[dict] = []
    for path in sorted((Path(item) for item in file_paths), key=lambda item: item.name):
        if path.suffix.lower() != ".csv":
            continue
        rcg, ecg = _read_pair(path)
        rcg_windows = sliding_window_cut(rcg, SIGNAL_LENGTH, step)
        ecg_windows = sliding_window_cut(ecg, SIGNAL_LENGTH, step)
        if len(rcg_windows) != len(ecg_windows):
            raise ValueError(f"Window count mismatch in {path.name}")
        for index, (rcg_window, ecg_window) in enumerate(zip(rcg_windows, ecg_windows)):
            windows.append(
                {
                    "source": path.name,
                    "index": index,
                    "rcg": rcg_window.astype(np.float32),
                    "ecg": ecg_window.astype(np.float32),
                }
            )
    if not windows:
        raise ValueError("No valid CSV dataset files were uploaded")
    return windows


def sinusoidal_time_embedding(timesteps: torch.Tensor, embed_dim: int) -> torch.Tensor:
    half_dim = embed_dim // 2
    exp_range = torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exp_range = -math.log(10000.0) * exp_range / float(half_dim)
    sinusoid_input = timesteps.float()[:, None] * torch.exp(exp_range)[None, :]
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=1)


def _group_count(channels: int) -> int:
    if channels >= 8 and channels % 8 == 0:
        return 8
    if channels >= 4 and channels % 4 == 0:
        return 4
    if channels >= 2 and channels % 2 == 0:
        return 2
    return 1


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.size()
        weights = self.avg_pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1)
        return x * weights


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        heads = 1 if channels < 8 else (8 if channels % 8 == 0 else max(1, channels // 2))
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.attn = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalised = self.norm(x).permute(0, 2, 1)
        attended, _ = self.attn(normalised, normalised, normalised)
        return x + attended.permute(0, 2, 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: Optional[int]):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_fc = nn.Linear(time_emb_dim, out_channels) if time_emb_dim else None
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.se = SqueezeExcitation(out_channels)
        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.conv1(self.act(self.norm1(x)))
        if self.time_fc is not None and time_emb is not None:
            hidden = hidden + self.time_fc(time_emb)[..., None]
        hidden = self.conv2(self.act(self.norm2(hidden)))
        hidden = self.se(hidden)
        shortcut = self.shortcut(x) if self.shortcut is not None else x
        return shortcut + hidden


class ConditionalDiffusionModel(nn.Module):
    """1-D conditional U-Net used by both RCG-to-ECG notebooks."""

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        time_emb_dim: int = DEFAULT_TIME_EMB_DIM,
        num_timesteps: int = DEFAULT_T,
    ):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.num_timesteps = num_timesteps
        time_mlp_dim = base_channels * 8
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_mlp_dim),
            nn.SiLU(),
            nn.Linear(time_mlp_dim, time_mlp_dim),
        )
        self.init_conv = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)
        down_channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8, base_channels * 8, base_channels * 8]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_channels_now = base_channels
        for index, out_channels in enumerate(down_channels):
            self.down_blocks.append(ResidualBlock(in_channels_now, out_channels, time_mlp_dim))
            if index != len(down_channels) - 1:
                self.downsamples.append(nn.Conv1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
            in_channels_now = out_channels
        self.attn = SelfAttention1D(down_channels[-1])

        up_channels = [base_channels * 8, base_channels * 4, base_channels * 2, base_channels, base_channels]
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        in_channels_now = down_channels[-1]
        for index, out_channels in enumerate(up_channels):
            skip_channels = down_channels[-2 - index]
            self.up_blocks.append(ResidualBlock(in_channels_now + skip_channels, out_channels, time_mlp_dim))
            if index != len(up_channels) - 1:
                self.upsamples.append(nn.ConvTranspose1d(out_channels, out_channels, kernel_size=4, stride=2, padding=1))
            in_channels_now = out_channels
        self.final_conv = nn.Conv1d(up_channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, noisy_ecg: torch.Tensor, rcg: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        hidden = self.init_conv(torch.cat([rcg, noisy_ecg], dim=1))
        time_emb = self.time_mlp(sinusoidal_time_embedding(timestep, self.time_emb_dim))
        skips = []
        for index, block in enumerate(self.down_blocks):
            hidden = block(hidden, time_emb)
            if index != len(self.down_blocks) - 1:
                skips.append(hidden)
                hidden = self.downsamples[index](hidden)
        hidden = self.attn(hidden)
        for index, block in enumerate(self.up_blocks):
            skip = skips[-1 - index]
            if hidden.size(-1) != skip.size(-1):
                hidden = F.interpolate(hidden, size=skip.size(-1), mode="nearest")
            hidden = block(torch.cat([hidden, skip], dim=1), time_emb)
            if index != len(self.up_blocks) - 1:
                hidden = self.upsamples[index](hidden)
        return self.final_conv(hidden)


class DiffusionScheduler:
    def __init__(self, timesteps: int, beta_start: float, beta_end: float, device: torch.device):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)

    @torch.no_grad()
    def sample(self, model: nn.Module, condition: torch.Tensor) -> torch.Tensor:
        shape = condition.shape
        generated = torch.randn(shape, device=self.device)
        model.eval()
        for step in reversed(range(self.timesteps)):
            timestep = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            noise_prediction = model(generated, condition, timestep)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_cum[step]
            mean = (generated - (1.0 - alpha) / torch.sqrt(1.0 - alpha_bar) * noise_prediction) / torch.sqrt(alpha)
            if step == 0:
                generated = mean
            else:
                posterior_variance = self.betas[step] * (1.0 - self.alpha_cum[step - 1]) / (1.0 - alpha_bar)
                generated = mean + torch.sqrt(posterior_variance) * torch.randn_like(generated)
        return generated


def _checkpoint_config(checkpoint: dict, key: str, default):
    model_config = checkpoint.get("model_config", {})
    diffusion_config = checkpoint.get("diffusion_config", {})
    source = model_config if key in model_config else diffusion_config
    return source.get(key, default)


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"RCG-ECG checkpoint not found: {checkpoint_path}. "
            "Set RCG_ECG_CHECKPOINT for RCG->ECG or ECG_RCG_CHECKPOINT for ECG->RCG."
        )
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
    model = ConditionalDiffusionModel(
        in_channels=int(_checkpoint_config(checkpoint, "in_channels", 2)),
        base_channels=int(_checkpoint_config(checkpoint, "base_channels", DEFAULT_BASE_CHANNELS)),
        time_emb_dim=int(_checkpoint_config(checkpoint, "time_emb_dim", DEFAULT_TIME_EMB_DIM)),
        num_timesteps=int(_checkpoint_config(checkpoint, "num_timesteps", DEFAULT_T)),
    ).to(device)
    state = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if state is None:
        raise ValueError("Checkpoint does not contain model_state_dict")
    model.load_state_dict(state, strict=True)
    diffusion = DiffusionScheduler(
        int(_checkpoint_config(checkpoint, "T", DEFAULT_T)),
        float(_checkpoint_config(checkpoint, "beta_start", DEFAULT_BETA_START)),
        float(_checkpoint_config(checkpoint, "beta_end", DEFAULT_BETA_END)),
        device,
    )
    return model, diffusion


def analyse_dataset(
    file_paths: Sequence[Path],
    forward_checkpoint_path: Optional[Path] = None,
    reverse_checkpoint_path: Optional[Path] = None,
    device_name: Optional[str] = None,
    max_windows: int = 8,
) -> dict:
    windows = load_dataset_files(file_paths)
    selected = windows[: max(1, int(max_windows))]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    forward_checkpoint = forward_checkpoint_path or Path(
        os.environ.get("RCG_ECG_CHECKPOINT", "loop/rcg_ecg_model.pt")
    )
    reverse_checkpoint = reverse_checkpoint_path or Path(
        os.environ.get("ECG_RCG_CHECKPOINT", "loop/ecg_rcg_model.pt")
    )
    forward_model, forward_diffusion = load_model(Path(forward_checkpoint), device)
    reverse_model, reverse_diffusion = load_model(Path(reverse_checkpoint), device)
    rcg = torch.from_numpy(np.stack([item["rcg"] for item in selected])[:, None, :]).to(device)
    ecg = torch.from_numpy(np.stack([item["ecg"] for item in selected])[:, None, :]).to(device)
    torch.manual_seed(0)
    predicted_ecg = forward_diffusion.sample(forward_model, rcg).cpu().numpy()[:, 0, :]
    torch.manual_seed(0)
    predicted_rcg = reverse_diffusion.sample(reverse_model, ecg).cpu().numpy()[:, 0, :]
    result_windows = []
    for item, predicted_ecg_window, predicted_rcg_window, ecg_target, rcg_target in zip(
        selected,
        predicted_ecg,
        predicted_rcg,
        ecg.cpu().numpy()[:, 0, :],
        rcg.cpu().numpy()[:, 0, :],
    ):
        rcg_std = float(np.std(rcg_target))
        predicted_rcg_std = float(np.std(predicted_rcg_window))
        if rcg_std > 1e-8 and predicted_rcg_std > 1e-8:
            rcg_similarity = float(np.corrcoef(rcg_target, predicted_rcg_window)[0, 1])
        else:
            rcg_similarity = 0.0
        result_windows.append(
            {
                "source": item["source"],
                "index": item["index"],
                "rcg": item["rcg"].round(6).tolist(),
                "ecg": ecg_target.round(6).tolist(),
                "predicted_ecg": predicted_ecg_window.round(6).tolist(),
                "predicted_rcg": predicted_rcg_window.round(6).tolist(),
                "ecg_mse": float(np.mean((predicted_ecg_window - ecg_target) ** 2)),
                "ecg_mae": float(np.mean(np.abs(predicted_ecg_window - ecg_target))),
                "rcg_similarity": rcg_similarity,
                "rcg_mse": float(np.mean((predicted_rcg_window - rcg_target) ** 2)),
                "rcg_mae": float(np.mean(np.abs(predicted_rcg_window - rcg_target))),
            }
        )
    return {
        "ok": True,
        "device": str(device),
        "forward_checkpoint": str(Path(forward_checkpoint).resolve()),
        "reverse_checkpoint": str(Path(reverse_checkpoint).resolve()),
        "total_windows": len(windows),
        "analysed_windows": len(result_windows),
        "signal_length": SIGNAL_LENGTH,
        "window_step": WINDOW_STEP,
        "windows": result_windows,
    }