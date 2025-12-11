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
    vram_reserve: int = Field(
        default=200,
        description="Reserve VRAM buffer during batch probing to avoid OOMs (in MiB).",
    )
    data_workers: int = Field(
        ...,
        description="Number of workers used in the dataloader. Set as high as your CPU allows.",
    )


class TrainingStageConfig(BaseModel):
    """
    Training stage configuration that defines the number of epochs, probe batch max and learning rate for one stage.
    """

    epochs: int = Field(default=10, description="Number of epochs of this stage.")
    probe_batch_max: int = Field(
        default=32, description="Maximum batch size to attempt during bin probing."
    )
    lr: float = Field(default=1e-4, description="General learning rate.")


class TrainingPlanConfig(BaseModel):
    """
    Training plan configuration for every stage.
    """

    # text_encoder: TrainingStageConfig = Field(
    #     default_factory=TrainingStageConfig,
    #     description="Configuration for pretraining the text encoder model."
    # )
    # vocoder: TrainingStageConfig = Field(
    #     default_factory=TrainingStageConfig,
    #     description="Configuration for the vocoder pretraining stage."
    # )
    text_encoder: TrainingStageConfig = None
    vocoder: TrainingStageConfig = None
    alignment: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for pretraining the text alignment model.",
    )
    duration: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training durations solo",
    )
    acoustic: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for joint training of acoustic models stage (second stage).",
    )
    textual: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training of textual models stage (fourth stage).",
    )
    style: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for training of style models stage.",
    )
    joint: TrainingStageConfig = Field(
        default_factory=TrainingStageConfig,
        description="Configuration for joint training of inference models.",
    )

    def get_stage(self, name: str) -> TrainingStageConfig:
        try:
            return getattr(self, name)
        except AttributeError:
            raise ValueError(f"Stage '{name}' not found.")


class DatasetConfig(BaseModel):
    """
    Dataset configuration parameters.
    """

    path: str = Field(
        ...,
        description="Root path for dataset. All other paths in this section are relative to it.",
    )
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
    alignment_model_path: str = Field(
        ...,
        description="Path to the trained alignment model used for aligning your segments.",
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
    pitch: float = Field(..., description="Weight for F0 pitch reconstruction loss.")
    voiced: float = Field(..., description="Weight for F0 voiced reconstruction loss.")
    energy: float = Field(..., description="Weight for energy reconstruction loss.")
    duration: float = Field(..., description="Weight for duration loss.")
    duration_ce: float = Field(
        ..., description="Weight for duration cross-entropy loss."
    )
    style: float = Field(..., description="Weight for style reconstruction loss.")
    mag: float = Field(..., description="Weight for magnitude prediction loss.")
    phase: float = Field(..., description="Weight for phase prediction loss.")
    multi_phase: float = Field(
        ..., description="Weight for multi-resolution phase loss."
    )
    confidence: float = Field(..., description="Weight for alignment confidence")
    align_loss: float = Field(..., description="Weight for alignment loss")
    discriminator: float = Field(..., description="Weight for discriminator loss")


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


class TextAlignerConfig(BaseModel):
    """
    Configuration for the text aligner component.
    """

    n_mels: int = Field(..., description="Number of mel bins for text aligner")
    n_fft: int = Field(..., description="Number of fft bins for text aligner")
    win_length: int = Field(..., description="Window length for text aligner")
    hop_length: int = Field(..., description="Hop length for text aligner")
    hidden_dim: int = Field(
        ..., description="Dimension of the hidden layers in the text aligner."
    )
    token_embedding_dim: int = Field(
        ..., description="Dimension of the token embeddings in the text aligner."
    )


class DecoderConfig(BaseModel):
    """
    Configuration for decoder.
    """

    hidden_dim: int = Field(..., description="Hidden dimension")
    residual_dim: int = Field(..., description="Residual shortcut dimension")


class GeneratorConfig(BaseModel):
    """
    Configuration for generator.
    """

    type: Literal["freegan"] = "freegan"
    input_dim: int = Field(..., description="Input dimension expected from decoder")
    hidden_dim: int = Field(..., description="Hidden dimension, ideally n_fft // 2 + 1")
    conv_intermediate_dim: int = Field(
        ..., description="Internal dimension for convnext blocks"
    )
    io_conv_kernel_size: int = Field(
        ..., description="Kernel size for input/output convolutions"
    )
    conformer_layers: int = Field(
        ..., description="Number of conformer blocks each for phase/amp"
    )
    conv_layers: int = Field(
        ..., description="Number of convnext block each for phase/amp"
    )


class RingformerGeneratorConfig(BaseModel):
    """
    Configuration for Ringformer generator.
    """

    type: Literal["ringformer"] = "ringformer"
    resblock_kernel_sizes: List[int] = Field(
        ..., description="Kernel sizes for residual blocks."
    )
    upsample_rates: List[int] = Field(
        ..., description="Upsample rates for each upsampling layer."
    )
    input_dim: int = Field(..., description="Initial channel count for upsampling.")
    upsample_last_channel: int = Field(
        ..., description="Last channel count before istft"
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


class TextEncoderConfig(BaseModel):
    """
    Text encoder configuration parameters.
    """

    tokens: int = Field(..., description="Number of phoneme tokens.")
    hidden_dim: int = Field(..., description="Hidden dimension")
    filter_channels: int = Field(..., description="Filter Channel for encoder")
    heads: int = Field(..., description="Number of attention heads")
    layers: int = Field(..., description="Number of layers in the text encoder.")
    kernel_size: int = Field(..., description="Kernel size for convolution.")
    dropout: float = Field(..., description="Dropout for internal layers")


class StyleEncoderConfig(BaseModel):
    """
    Style encoder configuration parameters.
    """

    n_mels: int = Field(..., description="Number of mel bins for style encoders")
    n_fft: int = Field(..., description="Number of fft bins for style encoders")
    win_length: int = Field(..., description="Window length for style encoders")
    hop_length: int = Field(..., description="Hop length for style encoders")
    max_channels: int = Field(
        ..., description="Maximum number of channels during downsampling"
    )
    skip_downsample: bool = Field(
        ..., description="Skip one of the downsample layers to allow smaller mel length"
    )


class DurationPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    n_layer: int = Field(..., description="Number of layers in the prosody predictor.")
    duration_classes: int = Field(
        ..., description="Number of classes used to predict duration."
    )
    max_duration: int = Field(..., description="Maximum duration of a single phoneme.")
    dropout: float = Field(..., description="Dropout rate for the prosody predictor.")
    last_dropout: float = Field(
        ..., description="Dropout rate just before final linear layer."
    )


class PitchEnergyPredictorConfig(BaseModel):
    """
    Prosody predictor configuration parameters.
    """

    inter_dim: int = Field(
        ..., description="Interchange dimension for pitch-energy prediction"
    )
    dropout: float = Field(
        ..., description="Dropout rate for the pitch-energy predictor."
    )


class SlmConfig(BaseModel):
    """
    Speech Language Model (SLM) configuration parameters.
    """

    model: str = Field(..., description="Identifier or path for the SLM model.")
    sr: int = Field(..., description="Sampling rate used by the SLM.")


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
    coarse_multiplier: int = Field(
        ...,
        description="Multiplier for hop length for coarse-grained prediction (alignment, p/e prediction)",
    )
    style_dim: int = Field(..., description="Dimension of the style vector.")
    inter_dim: int = Field(
        ..., description="Dimension of the embedding used between models."
    )

    text_aligner: TextAlignerConfig = Field(
        ..., description="Configuration for the text aligner component."
    )
    decoder: DecoderConfig = Field(..., description="Decoder configuration parameters.")
    generator: GeneratorConfig = Field(
        # Union[RingformerGeneratorConfig, GeneratorConfig] = Field(
        ...,
        description="Generator (vocoder) configuration parameters.",
    )
    text_encoder: TextEncoderConfig = Field(
        ..., description="Text encoder configuration parameters."
    )
    style_encoder: StyleEncoderConfig = Field(
        ..., description="Acoustic style encoder configuration parameters."
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
    symbol: SymbolConfig = Field(..., description="Text processing symbols.")

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


def load_model_config_yaml(file) -> ModelConfig:
    """
    Load a configuration file from the specified path.

    Args:
        config_path (Path): Path to the configuration file.

    Returns:
        Config: Parsed configuration object.
    """
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
