import json
import math
import os
import os.path as osp
import traceback
from typing import Callable, List, Any, Dict, Optional
import torch
from munch import Munch
import tqdm
import matplotlib.pyplot as plt

from loss_log import combine_logs
from stage_train import (
    train_alignment,
    train_acoustic,
    train_textual,
)

from stage_validate import (
    validate_alignment,
    validate_acoustic,
    validate_textual,
)

from optimizers import build_optimizer
from utils import (
    get_image,
    plot_spectrogram_to_figure,
    plot_mel_signed_difference_to_figure,
)


class StageConfig:
    def __init__(
        self,
        next_stage: Optional[str],
        train_fn: Callable,
        validate_fn: Callable,
        train_models: List[str],
        eval_models: List[str],
        discriminators: List[str],
        inputs: List[str],
    ):
        self.next_stage: Optional[str] = next_stage
        self.train_fn: Callable = train_fn
        self.validate_fn: Callable = validate_fn
        self.train_models: List[str] = train_models
        self.eval_models: List[str] = eval_models
        self.discriminators = discriminators
        self.inputs: List[str] = inputs


stages = {
    "alignment": StageConfig(
        next_stage=None,
        train_fn=train_alignment,
        validate_fn=validate_alignment,
        train_models=["text_aligner"],
        eval_models=[],
        discriminators=[],
        inputs=[
            "text",
            "text_length",
            "align_mel",
            "mel_length",
        ],
    ),
    "acoustic": StageConfig(
        next_stage="textual",
        train_fn=train_acoustic,
        validate_fn=validate_acoustic,
        train_models=[
            "text_encoder",
            "textual_style_encoder",
            "decoder",
            "generator",
        ],
        eval_models=[],
        discriminators=["mpd", "mrd"],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "alignment",
        ],
    ),
    "textual": StageConfig(
        next_stage=None,
        train_fn=train_textual,
        validate_fn=validate_textual,
        train_models=[
            "text_pe_encoder",
            "textual_pe_encoder",
            "pe_duration_encoder",
            "text_duration_encoder",
            "textual_prosody_encoder",
            "duration_predictor",
            "pitch_energy_predictor",
        ],
        eval_models=[
            "text_encoder",
            "textual_style_encoder",
            "decoder",
            "generator",
        ],
        discriminators=[],
        inputs=[
            "text",
            "text_length",
            "mel",
            "mel_length",
            "audio_gt",
            "pitch",
            "alignment",
        ],
    ),
}


def is_valid_stage(name):
    return name in stages


def valid_stage_list():
    return list(stages.keys())


class Stage:
    def __init__(
        self, name: str, train, train_time_bins: dict, val_time_bins: dict
    ) -> None:
        self.name: str = name
        self.max_epoch: int = train.config.training_plan.get_stage(name).epochs
        self.train_time_bins: dict = train_time_bins
        self.val_time_bins: dict = val_time_bins

        self.batch_sizes: Dict[str, int] = {}
        self.last_batch_load = None
        self.out_dir = train.out_dir
        self.load_batch_sizes()
        train.manifest.steps_per_epoch = self.get_steps_per_epoch()

        self.train_fn: Callable = stages[name].train_fn
        self.validate_fn: Callable = stages[name].validate_fn
        self.optimizer = build_optimizer(self.name, train=train)
        self.optimizer.prepare(train.accelerator)

    def begin_stage(self, name, train):
        self.name = name
        self.max_epoch = train.config.training_plan.get_stage(name).epochs
        self.train_fn = stages[name].train_fn
        self.validate_fn = stages[name].validate_fn
        self.optimizer.reset_lr(name, train)
        train.reset_out_dir(name)
        self.last_batch_load = None
        self.out_dir = train.out_dir
        self.load_batch_sizes()

    def get_next_stage(self) -> Optional[str]:
        return stages[self.name].next_stage

    def set_batch_size(self, i: int, batch_size: int) -> None:
        self.batch_sizes[str(i)] = batch_size

    def get_batch_size(self, key: int) -> int:
        if str(key) in self.batch_sizes:
            return self.batch_sizes[str(key)]
        else:
            return 1

    def reset_batch_sizes(self) -> None:
        self.batch_sizes = {}

    def batch_sizes_exist(self):
        return self.last_batch_load is not None

    def load_batch_sizes(self) -> None:
        batch_file = osp.join(self.out_dir, f"{self.name}_batch_sizes.json")
        if osp.isfile(batch_file):
            modified = os.stat(batch_file).st_mtime
            if self.last_batch_load is None or modified > self.last_batch_load:
                with open(batch_file, "r") as batch_input:
                    self.batch_sizes = json.load(batch_input)
                    self.last_batch_load = modified

    def save_batch_sizes(self) -> None:
        batch_file = osp.join(self.out_dir, f"{self.name}_batch_sizes.json")
        with open(batch_file, "w") as o:
            json.dump(self.batch_sizes, o)

    def get_steps_per_val(self) -> int:
        return self.get_steps(self.val_time_bins)

    def get_steps_per_epoch(self) -> int:
        return self.get_steps(self.train_time_bins)

    def get_steps(self, time_bins):
        total = 0
        for key in time_bins.keys():
            val = time_bins[key]
            total_batch = self.get_batch_size(key)
            if total_batch > 0:
                total += len(val) // total_batch + 1
        return total

    def train_batch(self, inputs, train, probing=False):
        config = stages[self.name]
        batch = prepare_batch(inputs, train.config.training.device, config.inputs)
        model = prepare_model(
            train.model,
            train.config.training.device,
            config.train_models,
            config.eval_models,
            config.discriminators,
            train,
        )
        result, audio = self.train_fn(batch, model, train, probing)
        optimizer_step(self.optimizer, config.train_models)
        if len(config.discriminators) > 0:
            audio_gt = batch.audio_gt.unsqueeze(1)
            audio = audio.detach()
            train.stage.optimizer.zero_grad()
            d_loss = train.discriminator_loss(audio_gt, audio, config.discriminators)
            train.accelerator.backward(d_loss * math.sqrt(batch.text.shape[0]))
            optimizer_step(self.optimizer, config.discriminators)
            train.stage.optimizer.zero_grad()
            result.add_loss("discriminator", d_loss)
        return result.detach()

    def validate(self, train):
        sample_count = train.config.validation.sample_count
        for key in train.model:
            train.model[key].eval()
        logs = []
        progress_bar = None
        if train.accelerator.is_main_process:
            iterator = tqdm.tqdm(
                iterable=enumerate(train.val_dataloader),
                desc=f"Validating {self.name}",
                total=self.get_steps_per_val(),
                unit="steps",
                bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} {remaining}{postfix} ",
                colour="BLUE",
                delay=2,
                leave=False,
                dynamic_ncols=True,
            )
            progress_bar = iterator
        else:
            iterator = enumerate(train.val_dataloader)

        sample_map = {
            item: j for j, item in enumerate(train.config.validation.force_samples)
        }
        for index, inputs in iterator:
            try:
                batch = prepare_batch(
                    inputs, train.config.training.device, stages[self.name].inputs
                )
                next_log, attention, audio_out, audio_gt = self.validate_fn(
                    batch, train
                )
                logs.append(next_log)
                samples = [
                    (i, sample_map[item])
                    for i, item in enumerate(inputs[8])
                    if item in sample_map
                ]

                if (
                    len(train.config.validation.force_samples) == 0
                    and index < sample_count
                ):
                    samples = [(0, index)]
                if train.accelerator.is_main_process and len(samples) > 0:
                    steps = train.manifest.current_total_step
                    sample_rate = train.model_config.sample_rate
                    if attention is not None:
                        attention = attention.cpu().numpy().squeeze()
                        train.writer.add_figure(
                            f"eval/attention_{index}", get_image(attention), steps
                        )
                    for inputs_index, samples_index in samples:
                        mel_gt_np = None
                        mel_pred_log_np = None
                        fig_mel_diff = None
                        if (
                            audio_out is not None
                            and audio_out[inputs_index] is not None
                        ):
                            audio_out_data = (
                                audio_out[inputs_index].cpu().numpy().squeeze()
                            )
                            # write audio
                            train.writer.add_audio(
                                f"eval/sample_{samples_index}",
                                audio_out_data,
                                steps,
                                sample_rate=sample_rate,
                            )
                            # write mel
                            audio_pred_cpu = (
                                audio_out[inputs_index].squeeze(0).cpu().float()
                            )
                            to_mel_cpu = train.to_mel.to("cpu")
                            mel_pred_tensor = to_mel_cpu(audio_pred_cpu)
                            mel_pred_log = torch.log(
                                torch.clamp(mel_pred_tensor, min=1e-5)
                            )
                            mel_pred_log_np = mel_pred_log.cpu().numpy()
                            fig_mel_pred = plot_spectrogram_to_figure(
                                mel_pred_log_np, title=f"Predicted Mel (Step {steps})"
                            )
                            train.writer.add_figure(
                                f"eval/sample_{samples_index}/mel",
                                fig_mel_pred,
                                global_step=steps,
                            )
                            plt.close(fig_mel_pred)
                        if audio_gt is not None and audio_gt[inputs_index] is not None:
                            audio_gt_data = (
                                audio_gt[inputs_index].cpu().numpy().squeeze()
                            )
                            # write audio
                            train.writer.add_audio(
                                f"eval/sample_{samples_index}_gt",
                                audio_gt_data,
                                0,
                                sample_rate=sample_rate,
                            )
                            try:
                                mel_gt_np = batch.mel[inputs_index].cpu().numpy()
                                fig_mel_gt = plot_spectrogram_to_figure(
                                    mel_gt_np, title="GT Mel"
                                )
                                train.writer.add_figure(
                                    f"eval/sample_{samples_index}/mel_gt",
                                    fig_mel_gt,
                                    global_step=0,
                                )
                                plt.close(fig_mel_gt)
                            except Exception as e:
                                train.logger.warning(
                                    f"Could not plot GT mel for sample index {samples_index}: {e}"
                                )
                        # --- NEW: Plot Mel Difference ---
                        if mel_gt_np is not None and mel_pred_log_np is not None:
                            try:
                                # Define or retrieve mean and std
                                dataset_mean = -4.0
                                dataset_std = 4.0

                                fig_mel_signed_diff = plot_mel_signed_difference_to_figure(
                                    mel_gt_np,  # Already normalized log mel
                                    mel_pred_log_np,  # Raw log mel
                                    dataset_mean,  # Pass normalization mean
                                    dataset_std,  # Pass normalization std
                                    title=f"Signed Mel Log Diff (GT - Pred) (Step {steps})",
                                    static_max_abs=2.5,
                                    # Optionally add clipping: max_abs_diff_clip=3.0
                                )
                                train.writer.add_figure(
                                    f"eval/sample_{samples_index}/mel_difference_normalized",
                                    fig_mel_signed_diff,
                                    global_step=steps,
                                )
                                plt.close(fig_mel_diff)  # Explicitly close figure
                            except Exception as e:
                                train.logger.warning(
                                    f"Could not plot mel difference for sample index {samples_index}: {e}"
                                )
                                if fig_mel_diff is not None and plt.fignum_exists(
                                    fig_mel_diff.number
                                ):
                                    plt.close(fig_mel_diff)
                if train.accelerator.is_main_process:
                    interim = combine_logs(logs)
                    if progress_bar is not None and interim is not None:
                        progress_bar.set_postfix({"loss": f"{interim.total():.3f}"})

            except Exception as e:
                path = inputs[8]
                progress_bar.clear() if progress_bar is not None else None
                train.logger.error(f"Validation failed {path}: {e}")
                traceback.print_exc()
                progress_bar.display() if progress_bar is not None else None
                continue
        progress_bar.close() if progress_bar is not None else None
        validation = combine_logs(logs)
        if validation is not None:
            validation.broadcast(train.manifest, train.stage, validation=True)
            total = validation.total()
            if total < train.manifest.best_loss:
                train.manifest.best_loss = total
        for key in train.model:
            train.model[key].train()


batch_names = [
    "audio_gt",
    "text",
    "text_length",
    "ref_text",
    "ref_length",
    "mel",
    "mel_length",
    "ref_mel",
    "path",
    "pitch",
    "align_mel",
    "alignment",
]


def prepare_batch(
    inputs: List[Any], device: torch.device, keys_to_transfer: List[str]
) -> Munch:
    """
    Transfers selected batch elements to the specified device.
    """
    prepared = {}
    for i, key in enumerate(batch_names):
        if key in keys_to_transfer:
            if key != "paths":
                prepared[key] = inputs[i].to(device)
            else:
                prepared[key] = inputs[i]
    return Munch(**prepared)


def prepare_model(
    model, device, training_set, eval_set, discriminators, train
) -> Munch:
    """
    Prepares models for training or evaluation, attaches them to the cpu memory if unused, returns an object which contains only the models that will be used.
    """
    result = {}
    for key in model:
        if key in training_set or key in eval_set or key in discriminators:
            result[key] = model[key]
            result[key].to(device)
        else:
            model[key].to("cpu")
        # if key in training_set or key in disc_set:
        #    result[key].train()
        # elif key in eval_set:
        #    result[key].eval()
    return Munch(**result)


def optimizer_step(optimizer, keys: List[str]) -> None:
    """
    Steps the optimizer for each module key in keys.
    """
    for key in keys:
        optimizer.step(key)
