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
from .vevo_repcodec import VevoRepCodec
from pathlib import Path


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


class AdaptiveVevoCodec(torch.nn.Module):
    def __init__(self, global_sr: int):
        super().__init__()
        self.hubert = torchaudio.pipelines.HUBERT_LARGE.get_model()

        down_dir = snapshot_download(
            repo_id="amphion/Vevo",
            repo_type="model",
            allow_patterns=["tokenizer/vq32/*"],
        )
        down_dir = Path(down_dir, "tokenizer/vq32")
        with open(down_dir / "hubert_large_l18_c32.yaml") as fp:
            conf = yaml.load(fp, Loader=yaml.FullLoader)

        self.vqvae = VevoRepCodec(**conf)
        self.vqvae.quantizer.initial()
        self.vqvae.load_state_dict(
            torch.load(down_dir / "hubert_large_l18_c32.pkl", map_location="cpu")[
                "model"
            ]["repcodec"]
        )
        self.resample = torchaudio.transforms.Resample(global_sr, 16000)

    def remove_encoder(self):
        if hasattr(self, "hubert"):
            del self.hubert
            del self.vqvae
            torch.cuda.empty_cache()

    @torch.no_grad()
    def extract_hubert_feature(self, wavs, wav_lens=None, output_layer=18):
        """
        Args:
            wavs: [B, T]
            wav_lens: [B,]
        Returns:
            feats: [B, T, D]
            feat_lengths: [B]
        """
        if wav_lens is None:
            wav_lens = torch.tensor([wavs.shape[1]] * wavs.shape[0]).to(wavs).int()

        feats, feat_lengths = self.hubert.extract_features(
            wavs, lengths=wav_lens, num_layers=output_layer
        )
        feats = feats[-1]
        return feats, feat_lengths

    def forward(self, wave, *scales, center_pad=True):
        wave = self.resample(wave)
        feats, _ = self.extract_hubert_feature(wave)
        x = self.vqvae.encoder(feats.mT)
        x = self.vqvae.projector(x)
        codes = []
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

            _, idx = self.vqvae.quantizer.codebook.forward_index(_x.mT)
            codes.append(idx[0])
        return codes
