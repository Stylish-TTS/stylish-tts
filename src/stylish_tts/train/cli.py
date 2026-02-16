import os.path as osp
import click
import importlib.resources
import multiprocessing
import torch

from stylish_tts.lib.config_loader import load_config_yaml, load_model_config_yaml
import stylish_tts.train.config as config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_config(config_path):
    if osp.exists(config_path):
        config = load_config_yaml(config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)
    return config


def get_model_config(model_config_path):
    if len(model_config_path) == 0:
        path = importlib.resources.files(config) / "model.yml"
        f_model = path.open("r", encoding="utf-8")
    else:
        if osp.exists(model_config_path):
            f_model = open(model_config_path, "r", encoding="utf-8")
        else:
            logger.error(f"Config file not found at {model_config_path}")
            exit(1)
    result = load_model_config_yaml(f_model)
    f_model.close()
    return result


##############################################################################


@click.group("stylish-train")
def cli():
    """Prepare a dataset, train a model, or convert a model to ONNX:

    In order to train, first you `train-align` to create an alignment model, `align` to use that model to generate alignments, `pitch` to generate pitch estimation for the dataset. At this point, as long as your dataset does not change, you do not need to re-run any of these stages again.

    Once you have pre-cached alignments and pitches, you can `train` your model, and finally `convert` your model to ONNX for inference.

    """
    # multiprocessing.set_start_method("forkserver")
    # torch.multiprocessing.set_sharing_strategy("file_system")
    # print("Setting multiprocessing start method to spawn.")


##### train-align #####


@cli.command(
    "train-align",
    short_help="Train an alignment model to use for pre-caching alignments.",
)
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option("--out", type=str, help="Output directory for logs and checkpoints")
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
def train_align(config_path, model_config_path, out, checkpoint, reset_stage):
    """Train alignment model

    <config_path> is your main configuration file and the resulting alignment model will be stored at <path>/<alignment_model_path> as specified in the dataset section.
    """
    print("Train alignment...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.train import train_model

    train_model(
        config,
        model_config,
        out,
        "alignment",
        checkpoint,
        reset_stage,
        config_path,
        model_config_path,
    )


##### align #####


@cli.command(
    short_help="Use a pretrained alignment model to create a cache of alignments for training."
)
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--method",
    type=click.Choice(["k2", "torch"], case_sensitive=False),
    default="k2",
    help="Method for forced alignment. 'k2' (process multiple samples simultaneously), 'torch' (one sample at a time). Default: k2",
)
@click.option(
    "-bs",
    "--batch-size",
    "batch_size",
    default=8,
    type=int,
    help="Number of samples to process simultaneously (only works if method is k2), default to 8.",
)
def align(config_path, model_config_path, method, batch_size):
    """Align dataset

    <config_path> is your main configuration file. Use an alignment model to precache the alignments for your dataset. <config_path> is your main configuration file and the alignment model will be loaded from <path>/<alignment_model_path>. The alignments are saved to <path>/<alignment_path> as specified in the dataset section. 'scores_val.txt' and 'scores_train.txt' containing confidence scores for each segment will be written to the dataset <path>.
    """
    print("Calculate alignment...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.dataprep.align_text import align_text

    align_text(config, model_config, method, batch_size)


##### align-textgrid #####


@cli.command(
    "align-textgrid",
    short_help="Use a pretrained alignment model to create a cache of alignments for training.",
)
@click.argument(
    "audio_path",
    type=str,
)
@click.argument(
    "text",
    type=str,
)
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--method",
    type=click.Choice(["k2", "torch"], case_sensitive=False),
    default="k2",
    help="Method for forced alignment. 'k2' (process multiple samples simultaneously), 'torch' (one sample at a time). Default: k2",
)
def align_textgrid(audio_path, text, config_path, model_config_path, method):
    """Align single sample and save as .textgrid

    <config_path> is your main configuration file. Use an alignment model to precache the alignments for your dataset. <config_path> is your main configuration file and the alignment model will be loaded from <path>/<alignment_model_path>. The alignments are saved to <path>/<alignment_path> as specified in the dataset section. 'scores_val.txt' and 'scores_train.txt' containing confidence scores for each segment will be written to the dataset <path>.
    """
    print("Calculate alignment...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.dataprep.align_textgrid import align_textgrid

    align_textgrid(audio_path, text, config, model_config, method)


##### pitch #####


@cli.command(short_help="Create a cache of pitches to use for training.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "-k",
    "--workers",
    default=8,
    type=int,
    help="Number of worker threads to use for calculation",
)
@click.option(
    "--method",
    default="pyworld",
    type=str,
    help="Method used to calculate. 'pyworld' (CPU based, traditional), 'rmvpe' (GPU based, ML model)",
)
def pitch(config_path, model_config_path, workers, method):
    """Calculate pitch for a dataset

    <config_path> is your main configuration file. Calculates the fundamental frequencies for every segment in your dataset. The pitches are saved to the <path>/<pitch_path> from the dataset section of the config file.
    """
    if method != "pyworld" and method != "rmvpe":
        exit("Pitch calculation must either be pyworld or rmvpe")
    print("Calculate pitch...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.dataprep.pitch_extractor import calculate_pitch

    calculate_pitch(config, model_config, method, workers)


##### train #####


@cli.command(short_help="Train a model using the specified configuration.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--out", type=str, help="Output directory for logs, checkpoints, and models"
)
@click.option(
    "--stage",
    default="acoustic",
    type=str,
    help="Training stage should be one of 'acoustic', 'textual', 'style', 'duration'.",
)
@click.option(
    "--checkpoint",
    default="",
    type=str,
    help="Path to a model checkpoint to load before training.",
)
@click.option(
    "--reset-stage",
    "reset_stage",
    is_flag=True,
    help="If loading a checkpoint, do not skip epochs and data.",
)
def train(config_path, model_config_path, out, stage, checkpoint, reset_stage):
    """Train a model

    <config_path> is your main configuration file. Train a Stylish TTS model. You must have already precached alignment and pitch information for the dataset. Stage should be 'acoustic' to begin with unless you are loading a checkpoint.
    """
    print("Train model...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    from stylish_tts.train.train import train_model

    train_model(
        config,
        model_config,
        out,
        stage,
        checkpoint,
        reset_stage,
        config_path,
        model_config_path,
    )


##### convert #####


@cli.command(short_help="Convert a model to ONNX for use in inference.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option("--speech", required=True, type=str, help="Path to write speech model")
@click.option(
    "--checkpoint",
    required=True,
    type=str,
    help="Path to a model checkpoint to load for conversion",
)
def convert(config_path, model_config_path, speech, checkpoint):
    """Convert a model to ONNX

    The converted model will be saved to path <speech>.
    """
    print("Convert to ONNX...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)

    from pathlib import Path
    from .cli_util import Checkpoint
    from stylish_tts.lib.text_utils import TextCleaner
    from stylish_tts.train.convert_to_onnx import convert_to_onnx
    from stylish_tts.train.dataloader import FilePathDataset
    from stylish_tts.train.utils import get_data_path_list

    state = Checkpoint(checkpoint, config, model_config)

    text_cleaner = TextCleaner(model_config.symbol)
    datalist = get_data_path_list(Path(config.dataset.path) / config.dataset.train_data)
    dataset = FilePathDataset(
        data_list=datalist,
        root_path=Path(config.dataset.path) / config.dataset.wav_path,
        text_cleaner=text_cleaner,
        model_config=model_config,
        pitch_path=Path(config.dataset.path) / config.dataset.pitch_path,
        alignment_path=Path(config.dataset.path) / config.dataset.alignment_path,
        duration_processor=state.duration_processor,
    )

    all_f0 = []
    for f0 in dataset.pitch.values():
        all_f0.append(f0[f0 > 10].flatten())
    all_f0 = torch.cat(all_f0, 0)
    all_f0 = torch.log2(all_f0 + 1e-9)
    f0_log2_mean = all_f0.mean().item()
    f0_log2_std = all_f0.std().item()

    metadata = {
        "pitch_log2_mean": str(f0_log2_mean),
        "pitch_log2_std": str(f0_log2_std),
        "mel_log_mean": str(state.norm.mel_log_mean),
        "mel_log_std": str(state.norm.mel_log_std),
    }

    convert_to_onnx(
        model_config,
        speech,
        state.model,
        config.training.device,
        state.duration_processor,
        metadata,
    )


@cli.command(short_help="Generate a voice pack.")
@click.argument(
    "config_path",
    type=str,
)
@click.option(
    "--dynamic",
    is_flag=True,
    help="Create a dynamic voicepack (default is static)",
)
@click.option(
    "-mc",
    "--model-config",
    "model_config_path",
    default="",
    type=str,
    help="Model configuration (optional), defaults to known-good model parameters.",
)
@click.option(
    "--voicepack",
    "voicepack_path",
    required=True,
    type=str,
    help="Path to write voice pack",
)
@click.option(
    "--checkpoint",
    required=True,
    type=str,
    help="Path to a model checkpoint to load for conversion",
)
def voicepack(config_path, dynamic, model_config_path, voicepack_path, checkpoint):
    from safetensors.torch import save_file
    from stylish_tts.train.voicepack import make_voicepack

    if dynamic:
        print("Generate dynamic voicepack...")
    else:
        print("Generate static voicepack...")
    config = get_config(config_path)
    model_config = get_model_config(model_config_path)
    result = make_voicepack(config, model_config, dynamic, checkpoint)
    key = "voicepack_static"
    if dynamic:
        key = "voicepack_dynamic"
    save_file({key: result}, voicepack_path)
