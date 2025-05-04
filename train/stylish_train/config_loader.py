from pydantic import BaseModel, Field
from typing import List, Union, Literal
from pathlib import Path
import yaml
import json
import logging

logger = logging.getLogger(__name__)


######## General Configuration ########


class TrainingConfig(BaseModel):
    """
    Training configuration parameters.
    """

    log_interval: int = Field(..., description="Interval (in steps) for logging.")
    save_interval: int = Field(
        ..., description="Interval (in steps) for saving checkpoints."
    )
    val_interval: int = Field(..., description="Interval (in steps) for validation.")
    device: str = Field(..., description="Computational device (e.g., 'cuda').")
    mixed_precision: str = Field(..., description="accelerator use bf16 or fp16 or no")
    probe_batch_max: int = Field(
        ..., description="Maximum batch size to attempt during bin probing."
    )
    probe_alignment_batch_max: int = Field(
        ..., description="Maximum alignment batch size to attempt during bin probing."
    )



class TrainingPlanConfig(BaseModel):
    """
    Training plan configuration that defines the number of epochs for different stages.
    """

    text_encoder: int = Field(
        default=10,
        description="Number of epochs for pretraining the text encoder model.",
    )
    alignment: int = Field(
        default=10,
        description="Number of epochs for pretraining the text alignment model.",
    )
    vocoder: int = Field(
        default=10,
        description="Number of epochs for pretraining the vocoder generator.",
    )
    pre_acoustic: int = Field(
        default=10,
        description="Number of epochs for the pretraining of acoustic models (first stage).",
    )
    acoustic: int = Field(
        default=10,
        description="Number of epochs for joint training of acoustic models (second stage).",
    )
    pre_textual: int = Field(
        default=10,
        description="Number of epochs for the pretraining of textual models stage (third stage).",
    )
    textual: int = Field(
        default=10,
        description="Number of epochs for training of textual models stage (fourth stage).",
    )
    joint: int = Field(
        default=10,
        description="Number of epochs for joint training of textual models stage (fifth stage).",
    )
    sbert: int = Field(
        default=10,
        description="Number of epochs for training of sbert models stage (sixth stage).",
    )


class DatasetConfig(BaseModel):
    """
    Dataset configuration parameters.
    """

    train_data: str = Field(..., description="Path to the training data list.")
    val_data: str = Field(..., description="Path to the validation data list.")
    wav_path: str = Field(..., description="Directory containing WAV files.")
    pitch_path: str = Field(
        ...,
        description="Path to the precomputed pitch safetensor file for your segments.",
    )
    alignment_path: str = Field(
        ...,
        description="Path to the precomputed alignment safetensor file for your segments.",
    )


class LossWeightConfig(BaseModel):
    """
    Loss weight configuration for various loss components.
    """

    mel: float = Field(..., description="Weight for mel spetral convergence loss.")
    generator: float = Field(..., description="Weight for generator loss.")
    slm: float = Field(
        ..., description="Weight for speech-language model feature matching loss."
    )
    mono: float = Field(..., description="Weight for monotonic alignment loss.")
    s2s: float = Field(..., description="Weight for sequence-to-sequence loss.")
    pitch: float = Field(..., description="Weight for F0 pitch reconstruction loss.")
    energy: float = Field(..., description="Weight for energy reconstruction loss.")
    duration: float = Field(..., description="Weight for duration loss.")
    duration_ce: float = Field(
        ..., description="Weight for duration cross-entropy loss."
    )
    style: float = Field(..., description="Weight for style reconstruction loss.")
    magphase: float = Field(..., description="Weight for magnitude/phase loss.")
    amplitude: float = Field(..., description="Weight for log amplitude loss.")
    phase: float = Field(..., description="Weight for phase loss.")
    stft_reconstruction: float = Field(
        ..., description="Weight for STFT reconstruction loss"
    )
    mel_rec: float = Field(..., description="Weight for text mel reconstruction")
    text_gen: float = Field(..., description="Weight for text mel generator model")

class DiscriminatorLossWeightConfig(BaseModel):
    mpd: float = Field(..., description="Weight for Multi-Period discriminator.")
    mrd: float = Field(..., description="Weight for Multi-Resolution discriminator.")
    msbd: float = Field(..., description="Weight for Multi-Scale Sub-Band CQT discriminator.")
    mstftd: float = Field(..., description="Weight for Multi-Scale STFT discriminator.")

class OptimizerConfig(BaseModel):
    """
    Optimizer configuration parameters.
    """

    lr: float = Field(..., description="General learning rate.")
    bert_lr: float = Field(..., description="Learning rate for the PLBERT model.")
    ft_lr: float = Field(..., description="Learning rate for acoustic modules.")
    alignment_lr: float = Field(..., description="Alignment learning rate.")
    text_encoder_lr: float = Field(..., description="Text encoder pretraining learning rate.")


######## Model Configuration ########


class SymbolConfig(BaseModel):
    """
    Configuration for text processing symbols with validation
    """

    pad: str = Field(..., description="Padding symbol for sequence alignment")
    punctuation: str = Field(..., description="Supported punctuation marks")
    letters: str = Field(..., description="Latin alphabet letters")
    letters_ipa: str = Field(
        ..., description="IPA phonetic characters including diacritics"
    )
    voiced: str = Field(..., description="Set of characters which are voiced")
    unvoiced: str = Field(..., description="Set of characters which are unvoiced")


class TextAlignerConfig(BaseModel):
    """
    Configuration for the text aligner component.
    """

    hidden_dim: int = Field(
        ..., description="Dimension of the hidden layers in the text aligner."
    )
    token_embedding_dim: int = Field(
        ..., description="Dimension of the token embeddings in the text aligner."
    )


# class PitchExtractorConfig(BaseModel):
#     """
#     Configuration for the pitch extractor component.
#     """

#     num_class: int = Field(..., description="Number of classes for pitch extraction.")
#     seq_len: int = Field(..., description="Sequence length for pitch extraction.")
#     leaky_relu_slope: float = Field(
#         ..., description="Slope for the leaky ReLU activation function."
#     )


class PLBERTConfig(BaseModel):
    """
    Configuration for the PLBERT model.
    """

    hidden_size: int = Field(..., description="Hidden size of the PLBERT model.")
    num_attention_heads: int = Field(
        ..., description="Number of attention heads in the PLBERT model."
    )
    intermediate_size: int = Field(
        ..., description="Intermediate size of the feed-forward layers in PLBERT."
    )
    max_position_embeddings: int = Field(
        ..., description="Maximum number of position embeddings in PLBERT."
    )
    num_hidden_layers: int = Field(
        ..., description="Number of hidden layers in the PLBERT model."
    )
    dropout: float = Field(..., description="Dropout rate used in the PLBERT model.")


# class HiFiGANDecoderConfig(BaseModel):
#     """
#     Configuration for HiFiGAN decoder.
#     """

#     type: Literal["hifigan"] = "hifigan"
#     hidden_dim: int = Field(..., description="Hidden dimension for HiFiGAN.")
#     resblock_kernel_sizes: List[int] = Field(
#         ..., description="Kernel sizes for residual blocks."
#     )
#     upsample_rates: List[int] = Field(
#         ..., description="Upsample rates for each upsampling layer."
#     )
#     upsample_initial_channel: int = Field(
#         ..., description="Initial channel count for upsampling."
#     )
#     resblock_dilation_sizes: List[List[int]] = Field(
#         ..., description="Dilation sizes for residual blocks."
#     )
#     upsample_kernel_sizes: List[int] = Field(
#         ..., description="Kernel sizes for the upsampling layers."
#     )


class ISTFTNetDecoderConfig(BaseModel):
    """
    Configuration for iSTFTNet decoder.
    """

    type: Literal["istftnet"] = "istftnet"
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")


class RingformerDecoderConfig(BaseModel):
    """
    Configuration for Ringformer decoder.
    """

    type: Literal["ringformer"] = "ringformer"
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    upsample_initial_channel: int = Field(
        ..., description="Initial channel count for upsampling."
    )
    resblock_dilation_sizes: List[List[int]] = Field(
        ..., description="Dilation sizes for residual blocks."
    )
    upsample_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for the upsampling layers."
    )
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_hop_size: int = Field(..., description="Hop size for iSTFT generator.")
    depth: int = Field(..., description="Number of conformer blocks in model")


class VocosDecoderConfig(BaseModel):
    """
    Configuration for Vocos decoder.
    """

    type: Literal["vocos"] = "vocos"
    intermediate_dim: int = Field(
        ..., description="Intermediate dimension size for Vocos."
    )
    num_layers: int = Field(..., description="Number of layers in Vocos model.")
    gen_istft_n_fft: int = Field(..., description="FFT size for iSTFT generator.")
    gen_istft_win_length: int = Field(
        ..., description="Window length for iSTFT generator."
    )
    gen_istft_hop_length: int = Field(
        ..., description="Hop length for iSTFT generator."
    )


class ValidationConfig(BaseModel):
    """
    Setup samples to use during validation
    """

    sample_count: int = Field(
        ..., description="Number of samples to generate during validation."
    )
    force_samples: list = Field(
        default=[], description="List of segments to use during validation."
    )


class FreevDecoderConfig(BaseModel):
    """
    Configuration for FreeV decoder.
    """

    type: Literal["freev"] = "freev"


#     # ASP_channel: int = Field(..., description="Amplitude channel dimension")
#     # ASP_resblock_kernel_sizes: List[int] = Field(
#     #     ..., description="Amplitude residual block kernels"
#     # )
#     # ASP_resblock_dilation_sizes: List[List[int]] = Field(
#     #     ..., description="Amplitude residual block dilation sizes"
#     # )
#     # ASP_input_conv_kernel_size: int = Field(
#     #     ..., description="Amplitude input convolution kerenel size"
#     # )
#     # ASP_output_conv_kernel_size: int = Field(
#     #     ..., description="Amplitude output convolution kernel size"
#     # )
#     # PSP_channel: int = Field(..., description="Phase channel dimension")
#     # PSP_resblock_kernel_sizes: List[int] = Field(
#     #     ..., description="Phase residual block kernels"
#     # )
#     # PSP_resblock_dilation_sizes: List[List[int]] = Field(
#     #     ..., description="Phase residual block dilation sizes"
#     # )
#     # PSP_input_conv_kernel_size: int = Field(
#     #     ..., description="Phase input convolution kerenel size"
#     # )
#     # PSP_output_R_conv_kernel_size: int = Field(
#     #     ..., description="Phase real output convolution kernel size"
#     # )
#     # PSP_output_I_conv_kernel_size: int = Field(
#     #     ..., description="Phase imaginary output convolution kernel size"
#     # )


class TextEncoderConfig(BaseModel):
    """
    Text encoder configuration parameters.
    """

    kernel_size: int = Field(
        ..., description="Kernel size for convolution in the text encoder."
    )
    n_layer: int = Field(..., description="Number of layers in the text encoder.")
    n_token: int = Field(..., description="Number of phoneme tokens.")


class StyleEncoderConfig(BaseModel):
    """
    Style encoder configuration parameters.
    This encoder (which may also act as a prosody encoder) generates a style embedding from audio.
    """

    dim_in: int = Field(..., description="Input dimension for the style encoder.")
    hidden_dim: int = Field(
        ..., description="Hidden dimension for the embedding encoder."
    )
    skip_downsamples: bool = Field(
        ..., description="Flag indicating whether to skip downsampling."
    )


class DurationPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    n_layer: int = Field(..., description="Number of layers in the prosody predictor.")
    max_dur: int = Field(..., description="Maximum duration of a single phoneme.")
    dropout: float = Field(..., description="Dropout rate for the prosody predictor.")


class PitchEnergyPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    dropout: float = Field(..., description="Dropout rate for the prosody predictor.")


class SlmConfig(BaseModel):
    """
    Speech Language Model (SLM) configuration parameters.
    """

    model: str = Field(..., description="Identifier or path for the SLM model.")
    sr: int = Field(..., description="Sampling rate used by the SLM.")
    hidden: int = Field(..., description="Hidden dimension of the SLM.")
    nlayers: int = Field(..., description="Number of layers in the SLM.")
    initial_channel: int = Field(
        ..., description="Initial channel count for the SLM discriminator head."
    )


class Config(BaseModel):
    """
    Top-level configuration model that encompasses all settings.
    """

    training: TrainingConfig = Field(
        ..., description="Training configuration parameters."
    )
    training_plan: TrainingPlanConfig = Field(
        ..., description="Training plan configuration parameters."
    )
    validation: ValidationConfig = Field(
        ..., description="Validation configuration parameters."
    )
    dataset: DatasetConfig = Field(..., description="Dataset configuration parameters.")
    loss_weight: LossWeightConfig = Field(
        ..., description="Loss weight configuration for various loss components."
    )
    discriminator_loss_weight: DiscriminatorLossWeightConfig = Field(
        ..., description="Loss weight configuration for discriminators."
    )
    optimizer: OptimizerConfig = Field(
        ..., description="Optimizer configuration parameters."
    )

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state: dict) -> None:
        self = self.model_copy(update=state)


class ModelConfig(BaseModel):
    multispeaker: bool = Field(
        ..., description="Indicates if the model supports multispeaker input."
    )
    n_mels: int = Field(..., description="Number of mel frequency bins.")
    sample_rate: int = Field(..., description="Sample rate for audio preprocessing.")
    n_fft: int = Field(..., description="FFT size for spectrogram computation.")
    win_length: int = Field(
        ..., description="Window length for spectrogram computation."
    )
    hop_length: int = Field(..., description="Hop length for spectrogram computation.")
    style_dim: int = Field(..., description="Dimension of the style vector.")
    inter_dim: int = Field(
        ..., description="Dimension of the embedding used between models."
    )
    discriminators: List[str] = Field(
        ..., description="List of discriminators to use (can be mpd, mrd, msbd, mstftd)"
    )

    text_aligner: TextAlignerConfig = Field(
        ..., description="Configuration for the text aligner component."
    )
    # pitch_extractor: PitchExtractorConfig = Field(
    #     ..., description="Configuration for the pitch extractor component."
    # )
    use_plbert: bool = Field(..., description="Whether to use PL-BERT or not.")
    plbert: PLBERTConfig = Field(..., description="Configuration for the PLBERT model.")
    decoder: Union[
        # HiFiGANDecoderConfig,
        ISTFTNetDecoderConfig,
        RingformerDecoderConfig,
        VocosDecoderConfig,
        FreevDecoderConfig,
    ] = Field(..., description="Decoder (vocoder) configuration parameters.")
    text_encoder: TextEncoderConfig = Field(
        ..., description="Text encoder configuration parameters."
    )
    style_encoder: StyleEncoderConfig = Field(
        ..., description="Style encoder configuration parameters."
    )
    duration_predictor: DurationPredictorConfig = Field(
        ..., description="Duration predictor configuration parameters."
    )
    pitch_energy_predictor: PitchEnergyPredictorConfig = Field(
        ..., description="Pitch/Energy predictor configuration parameters."
    )
    slm: SlmConfig = Field(
        ..., description="Speech Language Model (SLM) configuration parameters."
    )
    symbol: SymbolConfig = Field(..., description="Text processing symbols")

    def state_dict(self) -> dict:
        return self.model_dump()

    def load_state_dict(self, state: dict) -> None:
        self = self.model_copy(update=state)


def load_config_yaml(config_path: str) -> Config:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(config_path)
    # Load the YAML file into a dictionary
    with path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Parse and validate the configuration dictionary
    return Config.model_validate(config_dict)


def load_model_config_yaml(config_path: str) -> ModelConfig:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
    path = Path(config_path)
    # Load the YAML file into a dictionary
    with path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)

    # Parse and validate the configuration dictionary
    return ModelConfig.model_validate(config_dict)


def dump_to_string(config: Config) -> str:
    """
    Dump the configuration object to a string.

    Args:
        config (Config): Configuration object to dump.

    Returns:
        str: JSON string representation of the configuration object.
    """
    return config.model_dump_json()  # Dumps as a compact single-line JSON string


def load_from_string(config_str: str) -> Config:
    """
    Load a configuration object from a JSON string.

    Args:
        config_str (str): JSON string representation of the configuration object.

    Returns:
        Config: Parsed configuration object.
    """
    # Convert JSON string to a dictionary and validate it with the Config model
    return Config.model_validate(json.loads(config_str))


# Example usage:
if __name__ == "__main__":
    # Path to the YAML configuration file
    config_path = Path("Configs/new.config.yml")

    # Load the YAML file into a dictionary
    with config_path.open("r", encoding="utf-8") as file:
        config_dict = yaml.safe_load(file)
    # Parse and validate the configuration dictionary
    config = Config.model_validate(config_dict)

    # For Pydantic v2, use model_dump_json (or use json.dumps(config.model_dump(), indent=2))
    test = dump_to_string(config)
    logging.debug(test)

    # Load a configuration object from a JSON string
    config_loaded = load_from_string(test)
    logging.debug(config_loaded)
