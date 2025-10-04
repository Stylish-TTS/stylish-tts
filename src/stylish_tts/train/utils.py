import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from munch import Munch
import os
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nv_init = False


def print_gpu_vram(tag):
    if False:
        global nv_init
        if not nv_init:
            nvmlInit()
            nv_init = True
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        print(f"{tag} - GPU memory occupied: {info.used//1024**2} MB.")


def maximum_path(neg_cent, mask):
    """Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(
        mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    t_s_max = np.ascontiguousarray(
        mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    )
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def get_data_path_list(path):
    result = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            result = f.readlines()
    return result


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def length_to_mask(lengths, max_length) -> torch.Tensor:
    mask = (
        torch.arange(max_length)
        .unsqueeze(0)
        .expand(lengths.shape[0], -1)
        .type_as(lengths)
    )
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask


# for norm consistency loss
def log_norm(x, mean, std, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    # x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    x = (torch.exp(x * std + mean) ** 0.33).sum(dim=dim)
    return x


@torch.no_grad()
def compute_log_mel_stats(
    file_lines,
    wav_root,
    to_mel,
    sample_rate: int,
):
    """Compute dataset-wide mean/std of log-mel values.

    Args:
        file_lines: Iterable[str] of dataset lines `<wav>|<phonemes>|<speaker>|<text>`
        wav_root: Base directory for wav files
        to_mel: A torchaudio MelSpectrogram module configured for the dataset
        sample_rate: Target sample rate

    Returns:
        (mean, std, total_frames)
    """
    import os.path as osp
    import soundfile as sf
    import librosa

    count = 0
    sum_x = torch.zeros((), dtype=torch.float64)
    sum_x2 = torch.zeros((), dtype=torch.float64)
    # Determine device of the mel transform (defaults to CPU if no buffers)
    try:
        buf_iter = to_mel.buffers()
        first_buf = next(buf_iter, None)
        mel_device = first_buf.device if first_buf is not None else torch.device("cpu")
    except Exception:
        mel_device = torch.device("cpu")

    device = torch.device("cpu")
    to_mel = to_mel.to(device)

    for line in file_lines:
        parts = line.strip().split("|")
        if len(parts) < 1:
            continue
        wav_rel = parts[0]
        wav_path = osp.join(wav_root, wav_rel)
        try:
            wave, sr = sf.read(wav_path)
        except Exception:
            continue
        if wave.ndim == 2:
            wave = wave[:, 0]
        if sr != sample_rate:
            wave = librosa.resample(wave, orig_sr=sr, target_sr=sample_rate)
        wave_t = torch.from_numpy(wave).float().to(device)
        mel = to_mel(wave_t)
        log_mel = torch.log(1e-5 + mel)
        # Accumulate on CPU in float64 for numerical stability
        count += int(log_mel.numel())
        sum_x += log_mel.sum(dtype=torch.float64).cpu()
        sum_x2 += (log_mel * log_mel).sum(dtype=torch.float64).cpu()

    if count == 0:
        return -4.0, 4.0, 0
    mean = sum_x / count
    if count > 1:
        var = (sum_x2 - count * mean * mean) / (count - 1)
    else:
        var = torch.tensor(16.0, dtype=torch.float64)
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    to_mel.to(mel_device)
    return float(mean.item()), float(std.item()), int(count)


def plot_spectrogram_to_figure(
    spectrogram,
    title="Spectrogram",
    figsize=(12, 5),  # Increased width for better time resolution view
    dpi=150,  # Increased DPI for higher resolution image
    interpolation="bilinear",  # Smoother interpolation
    cmap="viridis",  # Default colormap, can change to 'magma', 'inferno', etc.
):
    """Converts a spectrogram tensor/numpy array to a matplotlib figure with improved quality."""
    plt.switch_backend("agg")  # Use non-interactive backend

    # Ensure input is a numpy array on CPU
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_np = spectrogram.detach().cpu().numpy()
    elif isinstance(spectrogram, np.ndarray):
        spectrogram_np = spectrogram
    else:
        raise TypeError("Input spectrogram must be a torch.Tensor or numpy.ndarray")

    # Handle potential extra dimensions (e.g., channel dim)
    if spectrogram_np.ndim > 2:
        if spectrogram_np.shape[0] == 1:  # Remove channel dim if it's 1
            spectrogram_np = spectrogram_np.squeeze(0)
        else:
            # If multiple channels, you might want to plot only the first
            # or handle it differently (e.g., separate plots)
            spectrogram_np = spectrogram_np[0, :, :]  # Plot only the first channel
            # Or raise an error/warning:
            # raise ValueError(f"Spectrogram has unexpected shape: {spectrogram_np.shape}")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # Apply figsize and dpi

    # Ensure valid interpolation string
    valid_interpolations = [
        None,
        "none",
        "nearest",
        "bilinear",
        "bicubic",
        "spline16",
        "spline36",
        "hanning",
        "hamming",
        "hermite",
        "kaiser",
        "quadric",
        "catrom",
        "gaussian",
        "bessel",
        "mitchell",
        "sinc",
        "lanczos",
        "blackman",
    ]
    if interpolation not in valid_interpolations:
        print(f"Warning: Invalid interpolation '{interpolation}'. Using 'bilinear'.")
        interpolation = "bilinear"

    im = ax.imshow(
        spectrogram_np,
        aspect="auto",
        origin="lower",
        interpolation=interpolation,
        cmap=cmap,
    )  # Apply interpolation and cmap

    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Mel Channels")  # More specific label
    plt.title(title)
    plt.tight_layout()
    # plt.close(fig) # Don't close here if returning the figure object
    return fig  # Return the figure object directly


def robust_color_limits(
    diff: np.ndarray,
    static_max_abs: float | None = None,
    max_abs_diff_clip: float | None = None,
    mad_multiplier: float = 3.0,
) -> tuple[float, float, float]:
    """Compute symmetric color limits around zero using robust statistics."""

    if static_max_abs is not None:
        max_abs = abs(static_max_abs)
    else:
        median = float(np.median(diff))
        mad = float(np.median(np.abs(diff - median)))
        if mad > 1e-9:
            robust_bound = abs(median) + mad_multiplier * mad
        else:
            robust_bound = float(np.max(np.abs(diff)))

        max_abs = robust_bound if robust_bound > 1e-6 else 1.0

    if max_abs_diff_clip is not None:
        max_abs = min(max_abs, abs(max_abs_diff_clip))

    return -max_abs, max_abs, max_abs


def summarize_residual(diff: np.ndarray) -> dict[str, float]:
    """Return scalar diagnostics for a residual heatmap."""

    abs_diff = np.abs(diff)
    mean_abs = float(np.mean(abs_diff))
    p95 = float(np.percentile(abs_diff, 95)) if abs_diff.size > 0 else 0.0
    rms = float(np.sqrt(np.mean(diff**2))) if abs_diff.size else 0.0
    return {
        "mean_abs": mean_abs,
        "p95_abs": p95,
        "rms": rms,
    }


def collapse_residual(
    diff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate residuals for summary plots."""

    if diff.size == 0:
        empty = np.array([])
        return empty, empty, empty, empty, empty, empty

    abs_diff = np.abs(diff)
    per_mel_abs = abs_diff.mean(axis=1)
    per_frame_abs = abs_diff.mean(axis=0)
    per_frame_pos = np.mean(np.clip(diff, 0.0, None), axis=0)
    per_frame_neg = np.mean(np.clip(diff, None, 0.0), axis=0)
    per_mel_pos = np.mean(np.clip(diff, 0.0, None), axis=1)
    per_mel_neg = np.mean(np.clip(diff, None, 0.0), axis=1)
    return (
        per_mel_abs,
        per_frame_abs,
        per_frame_pos,
        per_frame_neg,
        per_mel_pos,
        per_mel_neg,
    )


def plot_mel_signed_difference_to_figure(
    mel_gt_normalized_np: np.ndarray,
    mel_pred_normalized_np: np.ndarray,
    *,
    title: str = "Signed Mel Log Diff (GT - Pred)",
    figsize: tuple[int, int] = (12, 7),
    dpi: int = 150,
    cmap: str = "seismic",
    max_abs_diff_clip: float | None = None,
    static_max_abs: float | None = None,
    include_aggregates: bool = True,
    contour_zero: bool = True,
    confidence_mask: np.ndarray | None = None,
) -> tuple[plt.Figure, dict[str, float]]:
    """Plot residual heatmap with optional summaries and overlays."""

    plt.switch_backend("agg")

    min_len = min(mel_gt_normalized_np.shape[1], mel_pred_normalized_np.shape[1])
    mel_gt_trimmed = mel_gt_normalized_np[:, :min_len]
    mel_pred_trimmed = mel_pred_normalized_np[:, :min_len]
    diff = mel_pred_trimmed - mel_gt_trimmed

    vmin, vmax, vmax_used = robust_color_limits(
        diff, static_max_abs=static_max_abs, max_abs_diff_clip=max_abs_diff_clip
    )

    if include_aggregates:
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = gridspec.GridSpec(
            2,
            3,
            height_ratios=[3, 1],
            width_ratios=[1, 4, 0.6],
            hspace=0.25,
            wspace=0.3,
        )
        ax_main = fig.add_subplot(gs[0, 1])
        ax_freq = fig.add_subplot(gs[0, 0], sharey=ax_main)
        cax = fig.add_subplot(gs[0, 2])
        ax_time = fig.add_subplot(gs[1, 1], sharex=ax_main)
        ax_freq_stats = fig.add_subplot(gs[1, 0])
        ax_frame_stats = fig.add_subplot(gs[1, 2])
    else:
        fig, ax_main = plt.subplots(figsize=figsize, dpi=dpi)
        ax_time = ax_freq = cax = ax_freq_stats = ax_frame_stats = None

    im = ax_main.imshow(
        diff,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    if contour_zero:
        try:
            ax_main.contour(
                diff,
                levels=[0.0],
                colors="k",
                linewidths=0.4,
                alpha=0.4,
            )
        except Exception:
            pass

    if confidence_mask is not None:
        try:
            mask = confidence_mask[:, :min_len]
            if mask.ndim == 2:
                overlay = 1.0 - np.clip(mask, 0.0, 1.0)
            else:
                overlay = np.broadcast_to(
                    1.0 - np.clip(mask, 0.0, 1.0), diff.shape
                )
            ax_main.imshow(
                overlay,
                aspect="auto",
                origin="lower",
                cmap="gray",
                alpha=0.25,
                vmin=0,
                vmax=1,
            )
        except Exception:
            pass

    ax_main.set_xlabel("Frames")
    ax_main.set_ylabel("Mel Channels")
    if include_aggregates:
        ax_main.set_ylabel("")
        ax_main.tick_params(labelleft=False)
    ax_main.set_title(f"{title} | vmax={vmax_used:.2f}")
    if cax is not None:
        fig.colorbar(im, cax=cax, label="Signed normalized log diff")
    else:
        fig.colorbar(
            im,
            ax=ax_main,
            fraction=0.032,
            pad=0.02,
            label="Signed normalized log diff",
        )

    (
        per_mel_abs,
        per_frame_abs,
        per_frame_pos,
        per_frame_neg,
        per_mel_pos,
        per_mel_neg,
    ) = collapse_residual(diff)
    if include_aggregates:
        frame_axis = np.arange(per_frame_abs.shape[0]) if per_frame_abs.size else np.array([])
        if frame_axis.size:
            ax_time.plot(frame_axis, per_frame_pos, color="#d62728")
            ax_time.plot(frame_axis, per_frame_neg, color="#1f77b4")
            ax_time.fill_between(frame_axis, 0, per_frame_pos, color="#d62728", alpha=0.15)
            ax_time.fill_between(frame_axis, 0, per_frame_neg, color="#1f77b4", alpha=0.15)
        ax_time.axhline(0, color="black", linewidth=0.6, alpha=0.6)
        ax_time.set_ylim(-0.25, 0.25)
        ax_time.set_title("Mean diff per frame")
        ax_time.set_xlabel("Frame")
        ax_time.set_ylabel("Mean diff")
        if frame_axis.size:
            ax_time.legend(loc="upper right", fontsize="small")

        if per_mel_abs.size:
            mel_axis = np.arange(per_mel_abs.shape[0])
            ax_freq.plot(per_mel_pos, mel_axis, color="#d62728")
            ax_freq.plot(per_mel_neg, mel_axis, color="#1f77b4")
            ax_freq.fill_betweenx(mel_axis, 0, per_mel_pos, color="#d62728", alpha=0.15)
            ax_freq.fill_betweenx(mel_axis, 0, per_mel_neg, color="#1f77b4", alpha=0.15)
        ax_freq.axvline(0, color="black", linewidth=0.6, alpha=0.6)
        ax_freq.set_xlim(-0.25, 0.25)
        ax_freq.set_title("Mean diff per mel")
        ax_freq.set_ylabel("Mel bin")
        ax_freq.set_xlabel("Mean diff")
        if per_mel_abs.size:
            ax_freq.legend(loc="lower right", fontsize="small")

        if ax_freq_stats is not None:
            ax_freq_stats.axis("off")
            mel_pos_avg = float(per_mel_pos.mean()) if per_mel_pos.size else 0.0
            mel_neg_avg = float(per_mel_neg.mean()) if per_mel_neg.size else 0.0
            ax_freq_stats.text(
                0.03,
                0.65,
                f"avg: {mel_pos_avg:+.3f}",
                fontsize=10,
                color="#d62728",
                transform=ax_freq_stats.transAxes,
                ha="left",
            )
            ax_freq_stats.text(
                0.03,
                0.25,
                f"avg: {mel_neg_avg:+.3f}",
                fontsize=10,
                color="#1f77b4",
                transform=ax_freq_stats.transAxes,
                ha="left",
            )

        if ax_frame_stats is not None:
            ax_frame_stats.axis("off")
            frame_pos_avg = float(per_frame_pos.mean()) if per_frame_pos.size else 0.0
            frame_neg_avg = float(per_frame_neg.mean()) if per_frame_neg.size else 0.0
            ax_frame_stats.text(
                0.03,
                0.65,
                f"avg: {frame_pos_avg:+.3f}",
                fontsize=10,
                color="#d62728",
                transform=ax_frame_stats.transAxes,
                ha="left",
            )
            ax_frame_stats.text(
                0.03,
                0.25,
                f"avg: {frame_neg_avg:+.3f}",
                fontsize=10,
                color="#1f77b4",
                transform=ax_frame_stats.transAxes,
                ha="left",
            )

    stats = summarize_residual(diff)
    stats["vmax"] = float(vmax_used)
    fig.tight_layout()
    return fig, stats


def plot_residual_temporal_grid(
    diff: np.ndarray,
    *,
    frames_per_window: int = 128,
    max_windows: int = 4,
    cmap: str = "coolwarm",
    vmin: float | None = None,
    vmax: float | None = None,
) -> plt.Figure:
    """Create tiled snapshots of the residual over time windows."""

    plt.switch_backend("agg")
    total_frames = diff.shape[1]
    if total_frames <= frames_per_window:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=150)
        ax.imshow(
            diff,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel Channels")
        ax.set_title("Residual snapshot")
        fig.tight_layout()
        return fig

    window_stride = max(total_frames // (max_windows + 1), frames_per_window)
    windows = []
    start = 0
    while start < total_frames and len(windows) < max_windows:
        end = min(start + frames_per_window, total_frames)
        windows.append(diff[:, start:end])
        start += window_stride

    cols = len(windows)
    fig, axes = plt.subplots(
        1, cols, figsize=(4 * cols, 3), dpi=150, constrained_layout=True
    )
    if cols == 1:
        axes = [axes]

    for idx, (ax, window) in enumerate(zip(axes, windows)):
        im = ax.imshow(
            window,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Frames {idx * window_stride}-{idx * window_stride + window.shape[1]}")
        ax.set_xlabel("Frames")
        if idx == 0:
            ax.set_ylabel("Mel Channels")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.03)
    return fig


def get_image(arrs):
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(arrs)
    plt.colorbar(im, ax=ax)
    return fig


def recursive_munch(d):
    if isinstance(d, dict):
        return Munch((k, recursive_munch(v)) for k, v in d.items())
    elif isinstance(d, list):
        return [recursive_munch(v) for v in d]
    else:
        return d


def get_git_commit_hash():
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        print("Error obtaining git commit hash:", e)
        return "unknown"


def get_git_diff():
    try:
        # Run the git diff command
        diff_output = subprocess.check_output(["git", "diff"]).decode("utf-8")
        return diff_output
    except subprocess.CalledProcessError as e:
        print("Error obtaining git diff:", e)
        return ""


def save_git_diff(out_dir):
    hash = get_git_commit_hash()
    diff = get_git_diff()
    diff_file = os.path.join(out_dir, "git_state.txt")
    with open(diff_file, "w") as f:
        f.write(f"Git commit hash: {hash}\n\n")
        f.write(diff)
    print(f"Git diff saved to {diff_file}")


def clamped_exp(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(-35, 35)
    return torch.exp(x)


def leaky_clamp(
    x_in: torch.Tensor, min_f: float, max_f: float, slope: float = 0.001
) -> torch.Tensor:
    x = x_in
    min_t = torch.full_like(x, min_f, device=x.device)
    max_t = torch.full_like(x, max_f, device=x.device)
    x = torch.maximum(x, min_t + slope * (x - min_t))
    x = torch.minimum(x, max_t + slope * (x - max_t))
    return x


class DecoderPrediction:
    def __init__(
        self,
        *,
        audio,
        magnitude,
        phase,
    ):
        self.audio = audio
        self.magnitude = magnitude
        self.phase = phase


class DurationProcessor(torch.nn.Module):
    def __init__(self, class_count, max_dur):
        super(DurationProcessor, self).__init__()
        self.class_count = class_count
        self.max_dur = max_dur

        class_to_dur_table = torch.Tensor(
            [1, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 38, 46]
        )
        self.register_buffer("class_to_dur_table", class_to_dur_table)
        dur_to_class_table = torch.Tensor(
            [
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                7,
                7,
                8,
                8,
                8,
                9,
                9,
                9,
                10,
                10,
                10,
                11,
                11,
                11,
                11,
                11,
                12,
                12,
                12,
                12,
                12,
                13,
                13,
                13,
                13,
                13,
                14,
                14,
                14,
                14,
                14,
                14,
                14,
                15,
                15,
                15,
                15,
                15,
                15,
                15,
                15,
                15,
            ]
        )
        self.register_buffer("dur_to_class_table", dur_to_class_table)

    # def class_to_dur_soft(self, class_dist):
    #     return class_dist * self.class_to_dur_table

    def class_to_dur_soft(self, softdur):
        result = (softdur * self.class_to_dur_table).sum(dim=-1) / (
            softdur.sum(dim=-1) + 1e-9
        )
        return result

    def class_to_dur_hard(self, classes):
        classes = classes.clamp(min=0, max=self.class_count)
        return self.class_to_dur_table[classes]

    def dur_to_class(self, durs):
        durs = durs.clamp(min=1, max=self.max_dur)
        return self.dur_to_class_table[durs.long()]

    def align_to_class(self, alignment):
        result = alignment.sum(dim=-1).clamp(min=1, max=50)
        result = self.dur_to_class(result)
        return result

    def prediction_to_duration(self, pred, text_length):
        # softdur = self.class_to_dur_soft(torch.softmax(pred, dim=-1))
        # softdur = softdur.sum(dim=-1).round().clamp(min=1)
        # argmax = torch.argmax(pred, dim=-1).long()
        # argdur = self.class_to_dur_hard(argmax)
        confidence = torch.softmax(pred, dim=-1)
        softdur = self.class_to_dur_soft(confidence)
        dur = softdur
        # dur = (argdur * (argdur < 7)) + (softdur * (argdur >= 7))
        # dur = dur[:text_length]
        return dur

    def duration_to_alignment(self, duration: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of duration values to an attention matrix.

        duration -- [t]ext length
        result -- [t]ext length x [a]udio length"""
        indices = torch.repeat_interleave(
            torch.arange(duration.shape[0], device=duration.device),
            duration.to(torch.int),
        )
        result = torch.zeros(
            (duration.shape[0], indices.shape[0]), device=duration.device
        )
        result[indices, torch.arange(indices.shape[0])] = 1
        return result

    def forward(self, pred, text_length):
        duration = self.prediction_to_duration(pred, text_length)
        alignment = self.duration_to_alignment(duration)
        return alignment


def torch_empty_cache(device):
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device == "cpu":
        # torch.cpu.synchronize()
        # torch.cpu.empty_cache()
        pass
    else:
        exit(f"Unknown device {device}. Could not empty cache.")
