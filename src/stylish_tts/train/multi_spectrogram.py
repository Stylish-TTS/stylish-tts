import torch
import torchaudio
from einops import rearrange


class Resolution:
    def __init__(self, *, fft, hop, window):
        self.fft = fft
        self.hop = hop
        self.window = window


resolutions = [
    # Resolution(fft=2048, hop=50, window=67),
    # Resolution(fft=2048, hop=100, window=127),
    Resolution(fft=512, hop=128, window=512),  # 257),
    Resolution(fft=1024, hop=256, window=1024),  # 509),
    Resolution(fft=2048, hop=512, window=2048),  # 1021),
    # Resolution(fft=2048, hop=100, window=2048),
]

multi_spectrogram_count = len(resolutions)


class MultiSpectrogram(torch.nn.Module):
    def __init__(
        self, resolutions=resolutions, window=torch.hann_window, *, sample_rate
    ):
        super(MultiSpectrogram, self).__init__()
        self.windows = [torch.hann_window(item.window) for item in resolutions]
        self.mel_scales = [
            torchaudio.transforms.MelScale(
                n_mels=128,
                sample_rate=sample_rate,
                n_stft=item.fft // 2 + 1,
            )
            for item in resolutions
        ]

    def calculate_single(self, audio, index, item):
        window = self.windows[index].to(audio.device)
        stft = torch.stft(
            audio,
            n_fft=item.fft,
            hop_length=item.hop,
            win_length=item.window,
            window=window,
            return_complex=True,
        )
        fft_mag = torch.abs(stft)
        phase = (fft_mag > 1e-3).detach() * torch.angle(stft)
        mag = torch.log1p(self.mel_scales[index].to(audio.device)(fft_mag))
        mag = rearrange(mag, "b f t -> b 1 f t")
        fft_mag = rearrange(fft_mag, "b f t -> b 1 f t")
        return mag, phase, fft_mag

    def forward(self, *, target, pred):
        target_result = []
        pred_result = []
        target_phase_result = []
        pred_phase_result = []
        target_fft_result = []
        pred_fft_result = []
        for index, item in enumerate(resolutions):
            with torch.no_grad():
                t_mag, t_phase, t_fft_mag = self.calculate_single(target, index, item)
                target_fft_result.append(t_fft_mag)
                target_phase_result.append(t_phase)
                target_result.append(t_mag)
            p_mag, p_phase, p_fft_mag = self.calculate_single(pred, index, item)
            pred_fft_result.append(p_fft_mag)
            pred_phase_result.append(p_phase)
            pred_result.append(p_mag)
        return (
            target_result,
            pred_result,
            target_phase_result,
            pred_phase_result,
            target_fft_result,
            pred_fft_result,
        )
