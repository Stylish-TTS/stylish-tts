import numpy as np
import torch
import matplotlib.pyplot as plt
from munch import Munch
import os
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import stylish_tts
import importlib

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


def plot_mel_signed_difference_to_figure(
    mel_gt_normalized_np,  # Ground truth (already normalized log mel)
    mel_pred_log_np,  # Predicted (raw log mel)
    mean: float,  # Dataset mean used for normalization
    std: float,  # Dataset std used for normalization
    title="Signed Mel Log Difference (GT - Pred)",  # Updated title
    figsize=(12, 5),
    dpi=150,
    cmap="vanimo",
    max_abs_diff_clip=None,  # Optional: Clip the color range e.g., 3.0
    static_max_abs=None,  # Optional: Static max abs value for consistent color range
):
    """Plots the signed difference between two mel spectrograms using a diverging colormap."""
    plt.switch_backend("agg")

    # Ensure shapes match by trimming to the minimum length
    min_len = min(mel_gt_normalized_np.shape[1], mel_pred_log_np.shape[1])
    mel_gt_trimmed = mel_gt_normalized_np[:, :min_len]
    mel_pred_log_trimmed = mel_pred_log_np[:, :min_len]

    # Normalize the predicted log mel
    mel_pred_normalized_np = (mel_pred_log_trimmed - mean) / std

    # Calculate SIGNED difference in the *normalized* log domain
    diff = mel_gt_trimmed - mel_pred_normalized_np

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if static_max_abs is not None:
        # Use static max abs value for color limits
        vmin = -static_max_abs
        vmax = static_max_abs
    else:
        # Determine symmetric color limits centered at 0
        max_abs_val = np.max(np.abs(diff)) + 1e-9  # Add epsilon for stability
        if max_abs_diff_clip is not None:
            max_abs_val = min(
                max_abs_val, max_abs_diff_clip
            )  # Apply clipping if specified

        vmin = -max_abs_val
        vmax = max_abs_val

    im = ax.imshow(
        diff,
        aspect="auto",
        origin="lower",
        interpolation="none",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )  # Use 'none' for raw diff

    plt.colorbar(
        im, ax=ax, label="Signed Normalized Log Difference (GT - Pred)"
    )  # Updated label
    plt.xlabel("Frames")
    plt.ylabel("Mel Channels")
    plt.title(title)
    plt.tight_layout()
    # plt.close(fig) # Don't close if returning fig
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
    path = os.path.dirname(stylish_tts.__file__)
    try:
        commit_hash = (
            subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except subprocess.CalledProcessError as e:
        version = importlib.metadata.version("stylish_tts")
        return f"Version {version}"


def get_git_diff():
    path = os.path.dirname(stylish_tts.__file__)
    try:
        # Run the git diff command
        diff_output = subprocess.check_output(["git", "-C", path, "diff"]).decode(
            "utf-8"
        )
        return diff_output
    except subprocess.CalledProcessError as e:
        return ""


def save_git_diff(out_dir):
    commit_hash = get_git_commit_hash()
    diff = get_git_diff()
    diff_file = os.path.join(out_dir, "git_state.txt")
    with open(diff_file, "w") as f:
        f.write(f"Git commit hash or version: {commit_hash}\n\n")
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
        confidence = torch.softmax(pred, dim=-1)
        softdur = self.class_to_dur_soft(confidence)
        mask = sequence_mask(text_length).to(softdur.device)
        dur = softdur * mask
        return dur

    def duration_to_alignment(self, duration: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of duration values to an attention matrix.

        duration -- [t]ext length
        result -- [t]ext length x [a]udio length"""
        total_dur = duration.sum(dim=1).round().max().long().item()

        upper_bound = torch.cumsum(duration, dim=1)
        lower_bound = upper_bound - duration
        mean = (lower_bound + upper_bound) / 2
        mean = mean.unsqueeze(2)

        sequence = torch.arange(round(total_dur)).unsqueeze(0).unsqueeze(1)
        sequence = sequence.to(duration.device)
        x = sequence - mean
        alignment = torch.zeros_like(x)

        alignment = 1 - (x * 2 / (duration.unsqueeze(2) + 6)) ** 2
        # for i in range(5):
        #     power = i
        #     coefficient = coefficient_list[:, i, :].unsqueeze(2)
        #     # Coefficients are normalized for a domain of -1, 1.
        #     # We need to scale them to a domain of -duration/2, duration/2
        #     alignment = alignment + coefficient * (x * (2 / (duration.unsqueeze(2) + 6))) ** power

        # duration = torch.nn.functional.pad(duration, (1, 1))
        lower_bound -= 3
        upper_bound += 3
        mask = (sequence > lower_bound.unsqueeze(2)) * (
            sequence < upper_bound.unsqueeze(2)
        )
        alignment = alignment * mask
        alignment = torch.clamp(alignment, min=0.0)

        alignment = torch.softmax(alignment, dim=1)
        return alignment

    # def duration_to_alignment(self, duration: torch.Tensor) -> torch.Tensor:
    #     indices = torch.repeat_interleave(
    #         torch.arange(duration.shape[0], device=duration.device),
    #         duration.to(torch.int),
    #     )
    #     result = torch.zeros(
    #         (duration.shape[0], indices.shape[0]), device=duration.device
    #     )
    #     result[indices, torch.arange(indices.shape[0])] = 1
    #     return result

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
