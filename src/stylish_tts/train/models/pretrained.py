import torch
import torch.nn as nn
import torchaudio
from transformers import HubertModel
from einops import rearrange
import random
from huggingface_hub import snapshot_download
import os
import yaml
from .samresnet import SimAM_ResNet34_ASP, SimAM_ResNet100_ASP
import logging
import torchaudio.compliance.kaldi as kaldi
import torch.nn.functional as F
from .emotion2vec import Emotion2Vec
from focalcodec import FocalCodec
from pathlib import Path
from kanade_tokenizer import KanadeModel, load_vocoder, vocode


class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


class AdaptiveHubert(nn.Module):
    def __init__(self, hubert_path: str, global_sr: int):
        super().__init__()
        self.model = HubertModelWithFinalProj.from_pretrained(hubert_path)
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        x = self.model(wave)["last_hidden_state"]
        x = rearrange(x, "b t c -> b c t")
        xs = []
        for scale in scales:
            _x = F.interpolate(
                x,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            xs.append(_x)
        return xs


class AdaptiveFocalCodec(nn.Module):
    def __init__(
        self, global_sr: int, codec_path: str = "lucadellalib/focalcodec_50hz"
    ):
        super().__init__()
        self.codec = FocalCodec.from_pretrained(codec_path)
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def remove_encoder(self):
        if hasattr(self.codec, "encoder"):
            del self.codec.encoder
            del self.codec.compressor
            torch.cuda.empty_cache()

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        x = self.codec.sig_to_feats(wave)
        codes = []
        for scale in scales:
            _x = F.interpolate(
                x.mT,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            codes.append(self.codec.feats_to_toks(_x.mT))
        return codes

    def decode(self, codes):
        return self.codec.toks_to_sig(codes)


def load_checkpoint(model: torch.nn.Module, path: str):
    checkpoint = torch.load(path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    for key in missing_keys:
        logging.warning("missing tensor: {}".format(key))
    for key in unexpected_keys:
        logging.warning("unexpected tensor: {}".format(key))


# https://github.com/wenet-e2e/wespeaker/blob/67f0f4a8d472e6e2203d7baca38daba818af17f3/wespeaker/cli/speaker.py#L306
def load_model_pt(model_name_or_path: str):
    """There are the following files in the `model_dir`:
    - config.yaml: the model config file
    - avg_model.pt: the pytorch model file
    """
    model_dir = snapshot_download(model_name_or_path)
    required_files = ["config.yaml", "avg_model.pt"]
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            raise FileNotFoundError(f"{file} not found in {model_dir}")
    # Read config file
    with open(os.path.join(model_dir, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config["model"] == "SimAM_ResNet34_ASP":
        model = SimAM_ResNet34_ASP(**config["model_args"])
    elif config["model"] == "SimAM_ResNet100_ASP":
        model = SimAM_ResNet100_ASP(**config["model_args"])
    else:
        raise NotImplementedError(config["model"])
    load_checkpoint(model, os.path.join(model_dir, "avg_model.pt"))
    model.eval()
    return model


class SpeakerEmbeddingModel(nn.Module):
    def __init__(self, model_sr: int):
        super().__init__()
        self.model = load_model_pt("gaunernst/wespeaker-voxblink2-samresnet34")
        # self.model.pooling = nn.Identity()
        # self.model.bottleneck = nn.Identity()

        self.resample_rate = 16000
        self.window_type = "hamming"
        self.resample = torchaudio.transforms.Resample(model_sr, self.resample_rate)

    def compute_fbank(
        self,
        wavform,
        sample_rate=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        cmn=True,
    ):
        feat = kaldi.fbank(
            wavform,
            num_mel_bins=num_mel_bins,
            frame_length=frame_length,
            frame_shift=frame_shift,
            sample_frequency=sample_rate,
            window_type=self.window_type,
        )
        if cmn:
            feat = feat - torch.mean(feat, 0)
        return feat

    def forward(self, wave):
        device = next(self.parameters()).device
        wave = self.resample(wave)
        num_batch, _ = wave.shape
        feats = []
        for i in range(num_batch):
            _feats = self.compute_fbank(
                wave[i : i + 1, :],
                sample_rate=self.resample_rate,
                cmn=True,
            )
            feats.append(_feats)
        feats = torch.stack(feats, 0).to(device)
        return self.model(feats)


class AdaptiveEmotion2Vec(torch.nn.Module):
    def __init__(self, global_sr):
        super().__init__()
        self.model = Emotion2Vec.from_pretrained()
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        if self.model.cfg.normalize:
            wave = F.layer_norm(wave, wave.shape)
        x = self.model.extract_features(wave)["x"]
        x = rearrange(x, "b t c -> b c t")
        xs = []
        for scale in scales:
            _x = F.interpolate(
                x,
                scale_factor=scale,
                mode="nearest",
            )
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            xs.append(_x)
        return xs


# class AdaptiveS3Codec(torch.nn.Module):
#     def __init__(self, global_sr: int):
#         super().__init__()
#         self.codec = s3tokenizer.S3Tokenizer("speech_tokenizer_v1_25hz")
#         self.codec.init_from_onnx("D:\\TTS\\speech_tokenizer_v1.onnx")
#         self.resample = torchaudio.transforms.Resample(global_sr, 16000)

#     def remove_encoder(self):
#         if hasattr(self, "codec"):
#             del self.codec
#             torch.cuda.empty_cache()

#     def forward(self, wave, *scales, center_pad=True):
#         wave = self.resample(wave)
#         x = s3tokenizer.log_mel_spectrogram(wave)
#         codes = []
#         for scale in scales:
#             _x = F.interpolate(
#                 x,
#                 scale_factor=scale,
#                 mode="nearest",
#             )
#             if center_pad:
#                 # Padding due to center=True??
#                 pad = scale
#                 _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
#             _codes, _ = self.codec(_x, torch.tensor([_x.shape[-1]], device=x.device, dtype=torch.long))
#             codes.append(_codes)
#         return codes


class AdaptiveKanadeCodec(nn.Module):
    def __init__(
        self, global_sr: int, codec_path: str = "frothywater/kanade-25hz-clean"
    ):
        super().__init__()
        self.model = KanadeModel.from_pretrained(codec_path)
        self.vocoder = load_vocoder(self.model.config.vocoder_name)

    def remove_encoder(self):
        pass

    def normalize(self, wave):
        max_val = torch.max(torch.abs(wave)) + 1e-8
        wave = wave / max_val  # Normalize to [-1, 1]
        return wave

    def forward(self, wave, *scales, center_pad=False):
        x = [
            self.model.encode(_wave, True, False).content_token_indices
            for _wave in self.normalize(wave)
        ]
        x = torch.stack(x, 0)
        codes = []
        for scale in scales:
            _x = x.repeat(1, scale)
            if center_pad:
                # Padding due to center=True??
                pad = scale
                _x = F.pad(_x, (pad // 2, pad // 2 + (pad % 2)), "reflect")
            codes.append(_x)
        return codes

    def get_ssl_embeddings(self, waveform: torch.Tensor):
        waveform = self.normalize(waveform)
        audio_length = waveform.size(-1)
        padding = self.model._calculate_waveform_padding(audio_length)
        local_ssl_features, global_ssl_features = self.model.forward_ssl_features(
            waveform, padding=padding
        )
        return local_ssl_features, global_ssl_features

    def get_global_embeddings(self, wave):
        x = [
            self.model.encode(_wave, False, True).global_embedding
            for _wave in self.normalize(wave)
        ]
        x = torch.stack(x, 0)
        return x

    def decode(self, content_tokens, global_embs):
        mel = []
        for content_token, global_emb in zip(content_tokens, global_embs):
            _mel = self.model.decode(
                content_token_indices=content_token,
                global_embedding=global_emb,
            )
            mel.append(_mel)
        return vocode(self.vocoder, torch.stack(mel, 0))


# class AdaptiveOrangeWavLM(nn.Module):
#     def __init__(self, global_sr: int, model_path="Orange/Speaker-wavLM-pro"):
#         super().__init__()
#         self.model = EmbeddingsModel.from_pretrained(model_path)
#         self.resample = torchaudio.transforms.Resample(global_sr, 16000)

#     def forward(self, wave):
#         wave = self.resample(wave)
#         x = self.model(wave)
#         return x
