import torch
from torch.utils.tensorboard.writer import SummaryWriter
import click
import shutil
import logging
import random
from logging import StreamHandler
from stylish_tts.lib.config_loader import load_config_yaml, load_model_config_yaml
from stylish_tts.train.train_context import TrainContext
import hashlib
import numpy as np
from safetensors.torch import save_file

from stylish_tts.train.dataloader import build_dataloader, FilePathDataset
from stylish_tts.train.batch_manager import BatchManager
from stylish_tts.train.stage import Stage, is_valid_stage, valid_stage_list

from stylish_tts.train.models.models import build_model
from stylish_tts.train.losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    WavLMLoss,
    DurationLoss,
)
from stylish_tts.train.utils import get_data_path_list, save_git_diff, torch_empty_cache
from stylish_tts.train.loss_log import combine_logs
from stylish_tts.train.convert_to_onnx import convert_to_onnx
import tqdm

import os.path as osp
import os
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Create a logger for the current module
logger = logging.getLogger(__name__)


class LoggerManager:
    def __init__(self, logger, out_dir):
        logger.setLevel(logging.DEBUG)
        # Prevent messages from being passed to the root logger
        logger.propagate = False

        # Always add a stream handler
        self.err_handler = StreamHandler()
        self.err_handler.setLevel(logging.DEBUG)
        self.err_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(self.err_handler)

        # Always add a file handler
        self.file_handler = self.add_file_handler(logger, out_dir)

    def add_file_handler(self, logger, out_dir):
        file_handler = logging.FileHandler(osp.join(out_dir, "train.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
        return file_handler

    def reset_file_handler(self, logger, out_dir):
        logger.removeHandler(self.file_handler)
        self.file_handler.close()
        self.file_handler = self.add_file_handler(logger, out_dir)


def train_model(
    config,
    model_config,
    out_dir,
    stage,
    checkpoint,
    reset_stage,
    config_path,
    model_config_path,
):
    convert = False
    np.random.seed(1)
    random.seed(1)

    train_logger = logging.getLogger(__name__)

    if (
        stage == "alignment"
        and config.training.device != "cuda"
        and config.training.device != "cpu"
    ):
        logger.info(
            f"Alignment training does not support device {config.training.device}. Falling back on cpu training."
        )
        config.training.device = "cpu"

    train = TrainContext(stage, out_dir, config, model_config, train_logger)

    if not osp.exists(train.out_dir):
        os.makedirs(train.out_dir, exist_ok=True)
    if not osp.exists(train.out_dir):
        exit(f"Failed to create or find log directory at {train.out_dir}.")

    logger_manager = LoggerManager(train_logger, train.out_dir)

    shutil.copy(config_path, osp.join(train.out_dir, osp.basename(config_path)))
    if len(model_config_path) > 0:
        shutil.copy(
            model_config_path, osp.join(train.out_dir, osp.basename(model_config_path))
        )
    save_git_diff(train.out_dir)

    # Set up data loaders and batch manager
    if not osp.exists(train.data_path(train.config.dataset.train_data)):
        exit(
            f"Train data not found at {train.data_path(train.config.dataset.train_data)}"
        )
    if not osp.exists(train.data_path(train.config.dataset.val_data)):
        exit(
            f"Validation data not found at {train.data_path(train.config.dataset.val_data)}"
        )
    if not osp.exists(train.data_path(train.config.dataset.wav_path)):
        exit(
            f"Root wav path not found at {train.data_path(train.config.dataset.wav_path)}"
        )
    if not osp.exists(train.data_path(train.config.dataset.pitch_path)):
        exit(
            f"Pitch path not found at {train.data_path(train.config.dataset.pitch_path)}"
        )
    if (
        not osp.exists(train.data_path(train.config.dataset.alignment_path))
        and stage != "alignment"
    ):
        exit(
            f"Alignment path not found at {train.data_path(train.config.dataset.alignment_path)}"
        )
    val_list = get_data_path_list(train.data_path(train.config.dataset.val_data))
    # force somewhat determanistic selection of validation samples
    hashed_data = []
    for line in val_list:
        fields = line.strip().split("|")
        item_bytes = fields[0].encode("utf-8")
        hash_hex = hashlib.blake2b(item_bytes).hexdigest()
        hashed_data.append((hash_hex, fields[0]))
    hashed_data.sort()
    selected_pairs = hashed_data[: train.config.validation.sample_count]
    selected_files = [file_name for _, file_name in selected_pairs]

    for item in train.config.validation.force_samples:
        if item not in selected_files:
            selected_files.append(item)

    train.config.validation.force_samples = selected_files

    val_dataset = FilePathDataset(
        data_list=val_list,
        root_path=train.data_path(train.config.dataset.wav_path),
        text_cleaner=train.text_cleaner,
        model_config=train.model_config,
        pitch_path=train.data_path(train.config.dataset.pitch_path),
        alignment_path=train.data_path(train.config.dataset.alignment_path),
        duration_processor=train.duration_processor,
    )
    val_time_bins, _ = val_dataset.time_bins()
    train.val_dataloader = build_dataloader(
        val_dataset,
        val_time_bins,
        validation=True,
        num_workers=train.config.training.data_workers,
        device=train.config.training.device,
        multispeaker=train.model_config.multispeaker,
        stage=stage,
        train=train,
        hop_length=train.model_config.hop_length,
    )
    train.val_dataloader = train.accelerator.prepare(train.val_dataloader)
    train.duration_loss = DurationLoss(
        class_count=train.model_config.duration_predictor.duration_classes,
        weight=val_dataset.duration_weights,
    ).to(train.config.training.device)

    train.batch_manager = BatchManager(
        train.config.dataset,
        train.out_dir,
        probe_batch_max=train.config.training_plan.get_stage(stage).probe_batch_max,
        device=train.config.training.device,
        accelerator=train.accelerator,
        multispeaker=train.model_config.multispeaker,
        text_cleaner=train.text_cleaner,
        stage=stage,
        epoch=train.manifest.current_epoch,
        train=train,
    )

    # build model
    train.model = build_model(train.model_config)
    for key in train.model:
        train.model[key] = train.accelerator.prepare(train.model[key])
        train.model[key].to(train.config.training.device)

    train.generator_loss = GeneratorLoss(
        mrd0=train.model.mrd0,
        mrd1=train.model.mrd1,
        mrd2=train.model.mrd2,
        pitch=train.model.pitch_disc,
    ).to(train.config.training.device)
    train.discriminator_loss = DiscriminatorLoss(
        mrd0=train.model.mrd0,
        mrd1=train.model.mrd1,
        mrd2=train.model.mrd2,
        pitch=train.model.pitch_disc,
    ).to(train.config.training.device)
    train.wavlm_loss = WavLMLoss(
        train.model_config.slm.model,
        train.model_config.sample_rate,
        train.model_config.slm.sr,
    ).to(train.config.training.device)

    if not is_valid_stage(stage):
        exit(f"{stage} is not a valid stage. Must be one of {valid_stage_list()}")
    train.stage = Stage(stage, train, train.batch_manager.time_bins, val_time_bins)

    train.manifest.current_epoch = 1
    train.manifest.current_total_step = 0
    should_fast_forward = False

    assert train.stage is not None
    if checkpoint:
        train.accelerator.load_state(checkpoint)
        train.config = config
        # if we are not loading on a epoch boundary we need to resume the loader and skip to the correct step
        if train.manifest.stage == stage:
            if train.manifest.current_step != 0 and not reset_stage:
                should_fast_forward = True
            if reset_stage:
                train.manifest.current_epoch = 1
                train.manifest.current_step = 0
                # TODO: Do we need some create a different function to reset the optimizer and scheduler?
                train.stage.begin_stage(stage, train)
        else:
            train.manifest.current_epoch = 1
            train.manifest.current_step = 0
            train.stage.begin_stage(stage, train)
        logger.info(f"Loaded last checkpoint at {checkpoint} ...")

    # Compute or load dataset normalization stats (after checkpoint load so we can reuse)
    train.init_normalization()

    train.manifest.stage = stage

    if convert:
        filename = convert_to_onnx(
            train.model_config,
            train.base_output_dir,
            train.model,
            train.config.training.device,
            train.duration_processor,
        )
        logger.info(f"Export to ONNX file {filename} complete")
        exit(0)

    done = False
    while not done:
        train.logger.info(f"Training stage {train.manifest.stage}")
        train.manifest.best_loss = float("inf")  # best test loss
        torch_empty_cache(train.config.training.device)
        # save_checkpoint(train, prefix="checkpoint_test", long=False)
        # from models.stft import STFT
        # stft = STFT(
        #     filter_length=train.model_config.generator.gen_istft_n_fft,
        #     hop_length=train.model_config.generator.gen_istft_hop_size,
        #     win_length=train.model_config.generator.gen_istft_n_fft,
        # )
        # train.model.speech_predictor.generator.stft = stft.to(train.config.training.device).eval()
        # train.stage.validate(train)
        # exit(0)
        if not train.stage.batch_sizes_exist():
            train.batch_manager.probe_loop(train)
            should_fast_forward = False
        train_val_loop(train, should_fast_forward=should_fast_forward)
        train.logger.info(f"Training complete for stage {train.manifest.stage}")
        should_fast_forward = False
        next_stage = train.stage.get_next_stage()
        if next_stage is not None:
            train.manifest.current_epoch = 1
            train.manifest.current_step = 0
            train.manifest.stage = next_stage
            train.stage.begin_stage(next_stage, train)
            if not osp.exists(train.out_dir):
                os.makedirs(train.out_dir, exist_ok=True)
            if not osp.exists(train.out_dir):
                exit(f"Failed to create or find log directory at {train.out_dir}.")
            shutil.copy(config_path, osp.join(train.out_dir, osp.basename(config_path)))
            if len(model_config_path) > 0:
                shutil.copy(
                    model_config_path,
                    osp.join(train.out_dir, osp.basename(model_config_path)),
                )
            save_git_diff(train.out_dir)
            # Copy normalization stats into the new stage directory
            try:
                with open(
                    osp.join(train.out_dir, "normalization.json"), "w", encoding="utf-8"
                ) as f:
                    json.dump(
                        {
                            "mel_log_mean": train.normalization.mel_log_mean,
                            "mel_log_std": train.normalization.mel_log_std,
                            "frames": train.normalization.frames,
                            "sample_rate": train.model_config.sample_rate,
                            "n_mels": train.model_config.n_mels,
                            "n_fft": train.model_config.n_fft,
                            "hop_length": train.model_config.hop_length,
                            "win_length": train.model_config.win_length,
                        },
                        f,
                    )
            except Exception:
                pass
            if train.accelerator.is_main_process:
                assert train.writer is not None
                train.writer.close()
                train.writer = SummaryWriter(train.out_dir + "/tensorboard")
                logger_manager.reset_file_handler(train_logger, train.out_dir)
        else:
            done = True
    if train.manifest.stage == "alignment":
        save_alignment(train)
    train.accelerator.end_training()


def train_val_loop(train: TrainContext, should_fast_forward=False):
    assert (
        train.stage is not None
        and train.batch_manager is not None
        and train.model is not None
    )
    logs = []
    while train.manifest.current_epoch <= train.stage.max_epoch:
        train.batch_manager.init_epoch(train, should_fast_forward=should_fast_forward)

        _ = [train.model[key].train() for key in train.model]
        progress_bar = None
        if train.accelerator.is_main_process:
            iterator = tqdm.tqdm(
                iterable=enumerate(train.batch_manager.loader),
                desc=f"Train {train.manifest.stage} [{train.manifest.current_epoch}/{train.stage.max_epoch}]",
                total=train.manifest.steps_per_epoch,
                unit="steps",
                initial=train.manifest.current_step,
                bar_format="{desc}{bar}| {n_fmt}/{total_fmt} {remaining}{postfix} ",
                colour="GREEN",
                delay=5,
                leave=False,
                dynamic_ncols=True,
            )
            progress_bar = iterator
        else:
            iterator = enumerate(train.batch_manager.loader)
        loss = None
        for _, batch in iterator:
            postfix = {}
            next_log = train.batch_manager.train_iterate(
                batch, train, progress_bar=progress_bar
            )
            train.manifest.current_total_step += 1
            train.manifest.current_step += 1
            train.manifest.total_trained_audio_seconds += (
                float(len(batch[0][0]) * len(batch[0])) / train.model_config.sample_rate
            )
            if train.accelerator.is_main_process:
                if next_log is not None:
                    logs.append(next_log)
                    if loss is None:
                        if "mel" in next_log.metrics:
                            loss = next_log.metrics["mel"]
                        else:
                            loss = next_log.total()
                    else:
                        if "mel" in next_log.metrics:
                            loss = loss * 0.9 + next_log.metrics["mel"] * 0.1
                        else:
                            loss = loss * 0.9 + next_log.total() * 0.1
                    postfix = {"mel_loss": f"{loss:.3f}"}
                if len(logs) >= train.config.training.log_interval:
                    progress_bar.clear() if progress_bar is not None else None
                    combine_logs(logs).broadcast(train.manifest, train.stage)
                    progress_bar.display() if progress_bar is not None else None
                    logs = []
            num = (
                train.manifest.current_step
                + (train.manifest.current_epoch - 1) * train.manifest.steps_per_epoch
            )
            val_step = train.config.training.val_interval
            save_step = train.config.training.save_interval
            do_val = num % val_step == 0
            do_save = num % save_step == 0
            next_val = val_step - num % val_step - 1
            next_save = save_step - num % save_step - 1
            if next_val < next_save:
                postfix["val"] = str(next_val)
            else:
                postfix["save"] = str(next_save)
            progress_bar.set_postfix(postfix) if progress_bar is not None else None
            if do_val or do_save:
                progress_bar.clear() if progress_bar is not None else None

                # EXPERIMENTAL: DO THE UNTHINKABLE
                if train.manifest.stage == "alignment":
                    print("Training on the validation data like a boss.")
                    for _, batch in enumerate(train.val_dataloader):
                        train.batch_manager.train_iterate(
                            batch, train, progress_bar=None
                        )

                train.stage.validate(train)
                progress_bar.display() if progress_bar is not None else None
            if do_save:
                progress_bar.clear() if progress_bar is not None else None
                save_checkpoint(train, prefix="checkpoint")
                progress_bar.display() if progress_bar is not None else None
        if len(logs) > 0:
            combine_logs(logs).broadcast(train.manifest, train.stage)
            logs = []
        train.align_loss.on_train_epoch_end(train)
        train.manifest.current_epoch += 1
        train.manifest.current_step = 0
        train.manifest.training_log.append(
            f"Completed 1 epoch of {train.manifest.stage} training"
        )
        progress_bar.close() if progress_bar is not None else None
    train.stage.validate(train)
    save_checkpoint(train, prefix="checkpoint_final", long=False)


def save_alignment(train: TrainContext) -> None:
    path = osp.join(
        train.config.dataset.path, train.config.dataset.alignment_model_path
    )
    logger.info(f"Saving alignment to {path}")
    save_file(train.model.text_aligner.state_dict(), path)


def save_checkpoint(
    train: TrainContext,
    prefix: str = "checkpoint",
    long: bool = True,
) -> None:
    """
    Saves checkpoint using a checkpoint.
    """
    logger.info("Saving...")
    checkpoint_dir = osp.join(train.out_dir, f"{prefix}")
    if long:
        checkpoint_dir += f"_{train.manifest.current_epoch:05d}_step_{train.manifest.current_total_step:09d}"

    # Let the accelerator save all model/optimizer/LR scheduler/rng states
    train.accelerator.save_state(checkpoint_dir, safe_serialization=False)

    logger.info(f"Saved checkpoint to {checkpoint_dir}")


# if __name__ == "__main__":
#     main()
