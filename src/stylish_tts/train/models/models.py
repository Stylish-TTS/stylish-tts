import torch
from stylish_tts.lib.config_loader import ModelConfig

from .text_aligner import tdnn_blstm_ctc_model_base

from .discriminator import (
    MultiResolutionDiscriminator,
    SpecDiscriminator,
    ContextFreeDiscriminator,
)

from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor

from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .mel_style_encoder import MelStyleEncoder, PitchStyleEncoder
from .pitch_energy_predictor import PitchEnergyPredictor
from .speech_predictor import SpeechPredictor
from .pitch_discriminator import PitchDiscriminator
from stylish_tts.train.multi_spectrogram import multi_spectrogram_count

from munch import Munch

import logging

logger = logging.getLogger(__name__)


def build_model(model_config: ModelConfig):
    text_aligner = tdnn_blstm_ctc_model_base(
        model_config.text_aligner.n_mels, model_config.text_encoder.tokens
    )

    duration_predictor = DurationPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.inter_dim,
        text_config=model_config.text_encoder,
        duration_config=model_config.duration_predictor,
    )

    pitch_energy_predictor = PitchEnergyPredictor(
        style_dim=model_config.style_dim,
        inter_dim=model_config.pitch_energy_predictor.inter_dim,
        coarse_multiplier=model_config.coarse_multiplier,
        text_config=model_config.text_encoder,
        duration_config=model_config.duration_predictor,
        pitch_energy_config=model_config.pitch_energy_predictor,
    )

    speech_style_encoder = MelStyleEncoder(
        model_config.style_encoder.n_mels,
        model_config.style_dim,
        model_config.style_encoder.max_channels,
        model_config.style_encoder.skip_downsample,
    )
    pe_style_encoder = PitchStyleEncoder(
        model_config.style_encoder.n_mels,
        model_config.style_dim,
        model_config.style_encoder.max_channels,
        model_config.style_encoder.skip_downsample,
        coarse_multiplier=model_config.coarse_multiplier,
    )
    duration_style_encoder = MelStyleEncoder(
        model_config.style_encoder.n_mels,
        model_config.style_dim,
        model_config.style_encoder.max_channels,
        model_config.style_encoder.skip_downsample,
    )

    nets = Munch(
        text_aligner=text_aligner,
        duration_predictor=duration_predictor,
        pitch_energy_predictor=pitch_energy_predictor,
        speech_predictor=SpeechPredictor(model_config),
        disc=ContextFreeDiscriminator(),
        mrd0=SpecDiscriminator(),
        mrd1=SpecDiscriminator(),
        mrd2=SpecDiscriminator(),
        speech_style_encoder=speech_style_encoder,
        pe_style_encoder=pe_style_encoder,
        duration_style_encoder=duration_style_encoder,
        pitch_disc=PitchDiscriminator(dim_in=2, dim_hidden=64, kernel=21),
        dur_disc=PitchDiscriminator(dim_in=1, dim_hidden=64, kernel=5),
    )

    return nets
