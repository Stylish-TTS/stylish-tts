import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample
from .constants import *
from .model import E2E0, E2E
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0
from safetensors import safe_open


class RMVPE:
    def __init__(self, model_path, device, hop_length=160):
        self.device = device
        self.resample_kernel = {}
        model = E2E0(4, 1, (2, 2))
        checkpoint = {}
        with safe_open(model_path, framework="pt", device=0) as f:
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
        model.load_state_dict(checkpoint)
        model.eval()
        self.model = model.to(device)
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(device)
        self.resample_kernel = {}

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(
                mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"
            )
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def infer_from_audio(
        self, audio, *, sample_rate, device, thred=0.03, use_viterbi=False
    ):
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        if sample_rate == 16000:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(
                    sample_rate, 16000, lowpass_filter_width=128
                ).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)
        mel_extractor = self.mel_extractor
        mel = mel_extractor(audio_res, center=True)
        hidden = self.mel2hidden(mel)
        f0 = self.decode(hidden, thred=thred, use_viterbi=use_viterbi)
        return f0
