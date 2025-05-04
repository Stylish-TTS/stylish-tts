import torch
from torch.utils.tensorboard.writer import SummaryWriter
import click
import shutil
import logging
import random
from logging import StreamHandler
from config_loader import load_config_yaml, load_model_config_yaml
from train_context import TrainContext
import hashlib
import numpy as np
from safetensors.torch import save_file

from meldataset import build_dataloader, FilePathDataset
from batch_manager import BatchManager
from stage_context import StageContext, is_valid_stage, valid_stage_list

from models.models import build_model, load_defaults
from losses import GeneratorLoss, DiscriminatorLoss, WavLMLoss
from utils import get_data_path_list, save_git_diff
from loss_log import combine_logs
import tqdm

import os.path as osp
import os

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


@click.command()
@click.option("-p", "--config_path", default="configs/new.config.yml", type=str)
@click.option("-cp", "--model_config_path", default="config/model.config.yml", type=str)
@click.option("--out_dir", type=str)
@click.option("--stage", default="first_tma", type=str)
@click.option("--checkpoint", default="", type=str)
@click.option("--reset_stage", default=False, type=bool)
def main(config_path, model_config_path, out_dir, stage, checkpoint, reset_stage):
    np.random.seed(1)
    random.seed(1)
    if osp.exists(config_path):
        config = load_config_yaml(config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {config_path}")
        exit(1)

    if osp.exists(model_config_path):
        model_config = load_model_config_yaml(model_config_path)
    else:
        # TODO: we may be able to pull it out of the model if a model is passed in instead
        logger.error(f"Config file not found at {model_config_path}")
        exit(1)

    train_logger = logging.getLogger(__name__)

    train = TrainContext(stage, out_dir, config, model_config, train_logger)

    if not osp.exists(train.out_dir):
        os.makedirs(train.out_dir, exist_ok=True)
    if not osp.exists(train.out_dir):
        exit(f"Failed to create or find log directory at {train.out_dir}.")

    logger_manager = LoggerManager(train_logger, train.out_dir)

    shutil.copy(config_path, osp.join(train.out_dir, osp.basename(config_path)))
    shutil.copy(
        model_config_path, osp.join(train.out_dir, osp.basename(model_config_path))
    )
    save_git_diff(train.out_dir)

    # Set up data loaders and batch manager
    if not osp.exists(train.config.dataset.train_data):
        exit(f"Train data not found at {train.config.dataset.train_data}")
    if not osp.exists(train.config.dataset.val_data):
        exit(f"Validation data not found at {train.config.dataset.val_data}")
    if not osp.exists(train.config.dataset.wav_path):
        exit(f"Root wav path not found at {train.config.dataset.wav_path}")
    if not osp.exists(train.config.dataset.pitch_path):
        exit(f"Pitch path not found at {train.config.dataset.pitch_path}")
    if not osp.exists(train.config.dataset.alignment_path) and stage != "alignment":
        exit(f"Alignment path not found at {train.config.dataset.alignment_path}")
    val_list = get_data_path_list(train.config.dataset.val_data)
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
        root_path=train.config.dataset.wav_path,
        text_cleaner=train.text_cleaner,
        model_config=train.model_config,
        pitch_path=train.config.dataset.pitch_path,
        alignment_path=train.config.dataset.alignment_path,
    )
    val_time_bins = val_dataset.time_bins()
    train.val_dataloader = build_dataloader(
        val_dataset,
        val_time_bins,
        validation=True,
        num_workers=4,
        device=train.config.training.device,
        multispeaker=train.model_config.multispeaker,
        train=train,
    )
    train.val_dataloader = train.accelerator.prepare(train.val_dataloader)

    train.batch_manager = BatchManager(
        train.config.dataset,
        train.out_dir,
        probe_alignment_batch_max=train.config.training.probe_alignment_batch_max,
        probe_batch_max=train.config.training.probe_batch_max,
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
        mpd=train.model.mpd,
        mrd=train.model.mrd,
        msbd=train.model.msbd,
        mstftd=train.model.mstftd,
        discriminators=train.model_config.discriminators,
        loss_weights=train.config.discriminator_loss_weight,
    ).to(train.config.training.device)
    train.discriminator_loss = DiscriminatorLoss(
        mpd=train.model.mpd,
        mrd=train.model.mrd,
        msbd=train.model.msbd,
        mstftd=train.model.mstftd,
        discriminators=train.model_config.discriminators,
        loss_weights=train.config.discriminator_loss_weight,
    ).to(train.config.training.device)
    train.wavlm_loss = WavLMLoss(
        train.model_config.slm.model,
        train.model_config.sample_rate,
        train.model_config.slm.sr,
    ).to(train.config.training.device)

    if not is_valid_stage(stage):
        exit(f"{stage} is not a valid stage. Must be one of {valid_stage_list()}")
    train.stage = StageContext(
        stage, train, train.batch_manager.time_bins, val_time_bins
    )

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
        # train.stage.optimizer.reset_discriminator_schedulers()
        # if train.manifest.stage == "acoustic" and stage == "textual":
        #     logger.info("Cloning style encoder into prosodic style encoder...")
        #     train.model.prosodic_style_encoder.load_state_dict(
        #         train.model.style_encoder.state_dict()
        #     )
        logger.info(f"Loaded last checkpoint at {checkpoint} ...")
    else:
        load_defaults(train, train.model)

    train.manifest.stage = stage

    done = False
    while not done:
        train.logger.info(f"Training stage {train.manifest.stage}")
        train.manifest.best_loss = float("inf")  # best test loss
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
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
            shutil.copy(
                model_config_path,
                osp.join(train.out_dir, osp.basename(model_config_path)),
            )
            save_git_diff(train.out_dir)
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
    # train.stage.validate(train)
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
    path = osp.join(train.base_output_dir, "alignment_model.safetensors")
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


if __name__ == "__main__":
    main()
