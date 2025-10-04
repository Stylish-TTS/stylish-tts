import json
import math
import random
import os
import os.path as osp
import traceback
from typing import List, Any, Dict, Optional
import numpy as np
import torch
from munch import Munch
import tqdm
import matplotlib.pyplot as plt

from stylish_tts.train.loss_log import combine_logs
from stylish_tts.train.stage_type import stages, is_valid_stage, valid_stage_list
from stylish_tts.train.optimizers import build_optimizer
from stylish_tts.train.utils import (
    get_image,
    plot_spectrogram_to_figure,
    plot_mel_signed_difference_to_figure,
    plot_residual_temporal_grid,
)


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
        if probing:
            disc_index = 0
        else:
            # TODO: Fix hardcoded value
            disc_index = random.randrange(3)
        result, target_spec, pred_spec = self.train_fn(
            batch, model, train, probing, disc_index
        )
        optimizer_step(self.optimizer, config.train_models)
        if len(config.discriminators) > 0:
            # audio_gt = batch.audio_gt.unsqueeze(1)
            # audio = audio.detach()
            train.stage.optimizer.zero_grad()
            d_loss = train.discriminator_loss(
                target_list=target_spec,
                pred_list=pred_spec,
                used=config.discriminators,
                index=disc_index,
            )
            train.accelerator.backward(d_loss * math.sqrt(batch.text.shape[0]))
            disc_list = [f"mrd{disc_index}"]
            optimizer_step(self.optimizer, disc_list)  # config.discriminators)
            train.stage.optimizer.zero_grad()
            result.add_loss("discriminator", d_loss)
        return result.detach()

    def validate(self, train):
        sample_count = train.config.validation.sample_count
        for key in train.model:
            train.model[key].eval()
            # train.model[key].train()
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
                    for i, item in enumerate(inputs[3])
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
                        mel_gt_log_np = None
                        mel_gt_normalized_np = None
                        mel_pred_log_np = None
                        mel_pred_normalized_np = None
                        confidence_mask = None
                        fig_mel_signed_diff = None
                        fig_temporal = None
                        dataset_mean = 0.0
                        dataset_std = 1.0
                        if (
                            hasattr(train, "normalization")
                            and train.normalization is not None
                        ):
                            dataset_mean = train.normalization.mel_log_mean
                            dataset_std = train.normalization.mel_log_std
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
                            if self.name != "duration":
                                # write mel
                                audio_pred = audio_out[inputs_index].squeeze(0).float()
                                mel_pred_tensor = train.to_mel(audio_pred).cpu()
                                mel_pred_log = torch.log(
                                    torch.clamp(mel_pred_tensor, min=1e-5)
                                )
                                mel_pred_log_np = mel_pred_log.cpu().numpy()
                                if abs(dataset_std) > 1e-9:
                                    mel_pred_normalized_np = (
                                        mel_pred_log_np - dataset_mean
                                    ) / dataset_std
                                else:
                                    mel_pred_normalized_np = mel_pred_log_np - dataset_mean
                                fig_mel_pred = plot_spectrogram_to_figure(
                                    mel_pred_log_np,
                                    title=f"Predicted Mel (Step {steps})",
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
                            if self.name != "duration":
                                try:
                                    audio_gt_tensor = audio_gt[inputs_index]
                                    if audio_gt_tensor.dim() > 1:
                                        audio_gt_tensor = audio_gt_tensor.squeeze(0)
                                    audio_gt_tensor = audio_gt_tensor.float()
                                    mel_gt_tensor = train.to_mel(audio_gt_tensor).cpu()
                                    mel_gt_log = torch.log(
                                        torch.clamp(mel_gt_tensor, min=1e-5)
                                    )
                                    mel_gt_log_np = mel_gt_log.numpy()
                                    fig_mel_gt = plot_spectrogram_to_figure(
                                        mel_gt_log_np, title="GT Mel (log)"
                                    )
                                    train.writer.add_figure(
                                        f"eval/sample_{samples_index}/mel_gt",
                                        fig_mel_gt,
                                        global_step=0,
                                    )
                                    plt.close(fig_mel_gt)
                                    if abs(dataset_std) > 1e-9:
                                        mel_gt_normalized_np = (
                                            mel_gt_log_np - dataset_mean
                                        ) / dataset_std
                                    else:
                                        mel_gt_normalized_np = mel_gt_log_np - dataset_mean
                                    frame_energy = mel_gt_tensor.sum(dim=0).numpy()
                                    if frame_energy.size > 0:
                                        max_energy = float(np.max(frame_energy))
                                        if max_energy > 0:
                                            energy_norm = frame_energy / max_energy
                                        else:
                                            energy_norm = frame_energy
                                        confidence_mask = np.clip(energy_norm, 0.0, 1.0)
                                        confidence_mask = np.tile(
                                            confidence_mask[np.newaxis, :],
                                            (mel_gt_tensor.shape[0], 1),
                                        )
                                except Exception as e:
                                    train.logger.warning(
                                        f"Could not plot GT mel for sample index {samples_index}: {e}"
                                    )
                        # --- NEW: Plot Mel Difference ---
                        if (
                            mel_gt_normalized_np is not None
                            and mel_pred_normalized_np is not None
                        ):
                            if self.name != "duration":
                                try:
                                    voiced_mask = None
                                    if (
                                        hasattr(batch, "pitch")
                                        and batch.pitch is not None
                                    ):
                                        try:
                                            pitch_tensor = (
                                                batch.pitch[inputs_index]
                                                .detach()
                                                .cpu()
                                                .numpy()
                                            )
                                            if pitch_tensor.ndim == 1:
                                                pitch_mask = (pitch_tensor > 10).astype(
                                                    np.float32
                                                )
                                                pitch_mask = pitch_mask[np.newaxis, :]
                                            else:
                                                pitch_mask = (pitch_tensor > 10).astype(
                                                    np.float32
                                                )
                                            voiced_mask = np.tile(
                                                pitch_mask[:1],
                                                (mel_gt_normalized_np.shape[0], 1),
                                            )
                                        except Exception:
                                            voiced_mask = None

                                    combined_confidence = confidence_mask
                                    if voiced_mask is not None:
                                        if combined_confidence is not None:
                                            min_len_mask = min(
                                                combined_confidence.shape[1],
                                                voiced_mask.shape[1],
                                            )
                                            combined_confidence = combined_confidence[
                                                :, :min_len_mask
                                            ].copy()
                                            voiced_trim = voiced_mask[:, :min_len_mask]
                                            combined_confidence = np.minimum(
                                                combined_confidence, voiced_trim
                                            )
                                        else:
                                            combined_confidence = voiced_mask

                                    min_len_for_plot = min(
                                        mel_gt_normalized_np.shape[1],
                                        mel_pred_normalized_np.shape[1],
                                    )
                                    if (
                                        combined_confidence is not None
                                        and combined_confidence.shape[1]
                                        > min_len_for_plot
                                    ):
                                        combined_confidence = combined_confidence[
                                            :, :min_len_for_plot
                                        ]

                                    fig_mel_signed_diff, diff_stats = plot_mel_signed_difference_to_figure(
                                        mel_gt_normalized_np,
                                        mel_pred_normalized_np,
                                        title=f"Signed Mel Log Diff (GT - Pred) (Step {steps})",
                                        static_max_abs=2.5,
                                        confidence_mask=combined_confidence,
                                    )
                                    train.writer.add_figure(
                                        f"eval/sample_{samples_index}/mel_difference_normalized",
                                        fig_mel_signed_diff,
                                        global_step=steps,
                                    )
                                    min_len = min(
                                        mel_gt_normalized_np.shape[1],
                                        mel_pred_normalized_np.shape[1],
                                    )
                                    diff_matrix = (
                                        mel_gt_normalized_np[:, :min_len]
                                        - mel_pred_normalized_np[:, :min_len]
                                    )
                                    plt.close(fig_mel_signed_diff)  # Explicitly close figure
                                except Exception as e:
                                    train.logger.warning(
                                        f"Could not plot mel difference for sample index {samples_index}: {e}"
                                    )
                                    if (
                                        fig_mel_signed_diff is not None
                                        and plt.fignum_exists(
                                            fig_mel_signed_diff.number
                                        )
                                    ):
                                        plt.close(fig_mel_signed_diff)
                if train.accelerator.is_main_process:
                    interim = combine_logs(logs)
                    if progress_bar is not None and interim is not None:
                        progress_bar.set_postfix({"loss": f"{interim.total():.3f}"})

            except Exception as e:
                path = inputs[3]
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
    "path",
    "pitch",
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
