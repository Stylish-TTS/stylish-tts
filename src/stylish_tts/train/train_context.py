from stylish_tts.lib.config_loader import Config, ModelConfig
from stylish_tts.train.batch_manager import BatchManager
from typing import Optional, Any
import os.path as osp
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import logging
from torch.utils.data import DataLoader
from stylish_tts.train.losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    WavLMLoss,
    MultiResolutionSTFTLoss,
    CTCLossWithLabelPriors,
    MagPhaseLoss,
    DurationLoss,
)
from torch.utils.tensorboard.writer import SummaryWriter
from stylish_tts.lib.text_utils import TextCleaner
import torchaudio
from stylish_tts.train.utils import DurationProcessor
from stylish_tts.train.multi_spectrogram import MultiSpectrogram
from pathlib import Path
import traceback


class Manifest:
    def __init__(self) -> None:
        self.current_epoch: int = 0
        self.current_step: int = 0
        self.steps_per_epoch: int = 0
        self.current_total_step: int = 0
        self.total_trained_audio_seconds: float = 0.0
        self.stage: str = "first"
        self.best_loss: float = float("inf")
        self.training_log: list = []

    def state_dict(self) -> dict:
        return self.__dict__.copy()

    def load_state_dict(self, state: dict) -> None:
        for key, value in state.items():
            if hasattr(self, key):
                setattr(self, key, value)


class NormalizationStats:
    def __init__(self) -> None:
        self.mel_log_mean: float = -4.0
        self.mel_log_std: float = 4.0
        self.energy_log2_mean: float = 0.0
        self.energy_log2_std: float = 1.0
        self.frames: int = 0

    def state_dict(self) -> dict:
        return {
            "mel_log_mean": float(self.mel_log_mean),
            "mel_log_std": float(self.mel_log_std),
            "energy_log2_mean": float(self.energy_log2_mean),
            "energy_log2_std": float(self.energy_log2_std),
            "frames": int(self.frames),
        }

    def load_state_dict(self, state: dict) -> None:
        self.mel_log_mean = float(state.get("mel_log_mean", -4.0))
        self.mel_log_std = float(state.get("mel_log_std", 4.0))
        self.energy_log2_mean = float(state.get("energy_log2_mean", 0.0))
        self.energy_log2_std = float(state.get("energy_log2_std", 0.0))
        self.frames = int(state.get("frames", 0))


class TrainContext:
    def __init__(
        self,
        stage_name: str,
        base_out_dir: str,
        config: Config,
        model_config: ModelConfig,
        logger: logging.Logger,
    ) -> None:
        import stylish_tts.train.stage

        self.base_output_dir: str = base_out_dir
        self.out_dir: str = ""
        self.reset_out_dir(stage_name)
        self.config: Config = config
        self.model_config: ModelConfig = model_config
        self.batch_manager: Optional[BatchManager] = None
        self.stage: Optional[stage.Stage] = None
        self.manifest: Manifest = Manifest()
        self.normalization: NormalizationStats = NormalizationStats()
        self.writer: Optional[SummaryWriter] = None

        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False, find_unused_parameters=True
        )
        self.accelerator = Accelerator(
            project_dir=self.base_output_dir,
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=self.config.training.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self.accelerator.even_batches = False

        if self.accelerator.is_main_process:
            self.writer = SummaryWriter(self.out_dir + "/tensorboard")

        # TODO Replace these with json files, pickling is bad
        self.accelerator.register_for_checkpointing(self.config)
        self.accelerator.register_for_checkpointing(self.model_config)
        self.accelerator.register_for_checkpointing(self.manifest)
        self.accelerator.register_for_checkpointing(self.normalization)

        self.val_dataloader: Optional[DataLoader] = None

        self.model: Optional[Any] = None

        self.logger: logging.Logger = logger

        # Losses
        self.multi_spectrogram = MultiSpectrogram(
            sample_rate=self.model_config.sample_rate
        ).to(self.config.training.device)
        self.generator_loss: Optional[GeneratorLoss] = None  # Generator Loss
        self.discriminator_loss: Optional[DiscriminatorLoss] = (
            None  # Discriminator Loss
        )
        self.wavlm_loss: Optional[WavLMLoss] = None  # WavLM Loss
        self.stft_loss: MultiResolutionSTFTLoss = MultiResolutionSTFTLoss(
            sample_rate=self.model_config.sample_rate
        ).to(self.config.training.device)
        self.align_loss: CTCLossWithLabelPriors = CTCLossWithLabelPriors(
            prior_scaling_factor=0.3, blank=model_config.text_encoder.tokens
        )
        # self.magphase_loss: MagPhaseLoss = MagPhaseLoss(
        #     n_fft=self.model_config.generator.gen_istft_n_fft,
        #     hop_length=self.model_config.generator.gen_istft_hop_size,
        #     win_length=self.model_config.generator.gen_istft_n_fft,
        # ).to(self.config.training.device)
        # TODO: Remove hardcoded valude
        self.magphase_loss: MagPhaseLoss = MagPhaseLoss(
            n_fft=self.model_config.n_fft // 2,
            hop_length=self.model_config.hop_length // 3,
            win_length=self.model_config.win_length // 2,
        ).to(self.config.training.device)
        self.duration_loss: DurationLoss = None

        self.text_cleaner = TextCleaner(self.model_config.symbol)
        self.duration_processor = DurationProcessor(
            class_count=self.model_config.duration_predictor.duration_classes,
            max_dur=self.model_config.duration_predictor.max_duration,
        ).to(self.config.training.device)

        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_config.n_mels,
            n_fft=self.model_config.n_fft,
            win_length=self.model_config.win_length,
            hop_length=self.model_config.hop_length,
            sample_rate=self.model_config.sample_rate,
        ).to(self.config.training.device)

        self.to_style_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_config.style_encoder.n_mels,
            n_fft=self.model_config.style_encoder.n_fft,
            win_length=self.model_config.style_encoder.win_length,
            hop_length=self.model_config.style_encoder.hop_length,
            sample_rate=self.model_config.sample_rate,
        ).to(self.config.training.device)

        self.to_align_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=self.model_config.text_aligner.n_mels,
            n_fft=self.model_config.text_aligner.n_fft,
            win_length=self.model_config.text_aligner.win_length,
            hop_length=self.model_config.hop_length
            * self.model_config.coarse_multiplier,
            sample_rate=self.model_config.sample_rate,
        ).to(self.config.training.device)

        self.pe_style_dict = {}
        self.pe_style_sequence = None
        self.pe_knn = None

    def reset_out_dir(self, stage_name):
        self.out_dir = osp.join(self.base_output_dir, stage_name)

    def data_path(self, path: str) -> Path:
        return Path(self.config.dataset.path) / path

    def init_normalization(self) -> None:
        """Initialize normalization stats for this run.

        Priority order:
          1) Use stats loaded from checkpoint (if available),
          2) Load from <stage_out>/normalization.json,
          3) Compute from train split and save.
        """
        import json
        import os.path as osp
        from stylish_tts.train.utils import compute_log_mel_stats, get_data_path_list

        try:
            import tqdm as _tqdm
        except Exception:
            _tqdm = None

        import torch

        all_f0 = []
        for f0 in self.batch_manager.dataset.pitch.values():
            all_f0.append(f0[f0 > 10].flatten())
            # all_f0.append(f0.flatten())
        all_f0 = torch.cat(all_f0, 0)
        all_f0 = torch.clamp(all_f0, min=1, max=600)
        all_f0 = torch.log(all_f0 + 1e-9)
        self.f0_log2_mean = all_f0.mean()
        self.f0_log2_std = all_f0.std()

        # 1) Already from checkpoint
        if self.normalization is not None and self.normalization.frames > 0:
            out_path = osp.join(self.out_dir, "normalization.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "mel_log_mean": self.normalization.mel_log_mean,
                            "mel_log_std": self.normalization.mel_log_std,
                            "energy_log2_mean": self.normalization.energy_log2_mean,
                            "energy_log2_std": self.normalization.energy_log2_std,
                            "frames": self.normalization.frames,
                            "sample_rate": self.model_config.sample_rate,
                            "n_mels": self.model_config.n_mels,
                            "n_fft": self.model_config.n_fft,
                            "hop_length": self.model_config.hop_length,
                            "win_length": self.model_config.win_length,
                        },
                        f,
                    )
            except Exception:
                pass
            self.logger.info(
                f"Using normalization stats from checkpoint: mean={self.normalization.mel_log_mean:.4f}, std={self.normalization.mel_log_std:.4f}"
            )
            return

        # 2) Load from file if present
        out_path = osp.join(self.out_dir, "normalization.json")
        if osp.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.normalization.mel_log_mean = float(data.get("mel_log_mean", -4.0))
                self.normalization.mel_log_std = float(data.get("mel_log_std", 4.0))
                self.normalization.energy_log2_mean = float(
                    data.get("energy_log2_mean", 0.0)
                )
                self.normalization.energy_log2_std = float(
                    data.get("energy_log2_std", 1.0)
                )
                self.normalization.frames = int(data.get("frames", 0))
                self.logger.info(
                    f"Loaded normalization stats: mean={self.normalization.mel_log_mean:.4f}, std={self.normalization.mel_log_std:.4f}, frames={self.normalization.frames}"
                )
                self.logger.info(
                    f"Loaded energy normalization: mean={self.normalization.energy_log2_mean:.4f}, std={self.normalization.energy_log2_std:.4f}"
                )
                if self.normalization.frames == 0 or (
                    abs(self.normalization.mel_log_mean - (-4.0)) < 1e-6
                    and abs(self.normalization.mel_log_std - 4.0) < 1e-6
                ):
                    self.logger.warning(
                        "Normalization stats appear to be defaults (-4, 4) or empty; delete normalization.json to trigger recompute."
                    )
                return
            except Exception as e:
                self.logger.warning(
                    f"Failed to load normalization.json, will recompute: {e}"
                )

        # 3) Compute from dataset
        train_list = get_data_path_list(self.data_path(self.config.dataset.train_data))
        iterator = train_list
        if _tqdm is not None and self.accelerator.is_main_process:
            iterator = _tqdm.tqdm(
                train_list,
                desc="Computing normalization stats",
                unit="segments",
                total=len(train_list),
                colour="MAGENTA",
                dynamic_ncols=True,
            )

        mean, std, energy_log2_mean, energy_log2_std, frames = compute_log_mel_stats(
            iterator,
            str(self.data_path(self.config.dataset.wav_path)),
            self.to_mel,
            self.model_config.sample_rate,
        )
        self.normalization.mel_log_mean = mean
        self.normalization.mel_log_std = std
        self.normalization.energy_log2_mean = energy_log2_mean
        self.normalization.energy_log2_std = energy_log2_std
        self.normalization.frames = frames
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mel_log_mean": mean,
                        "mel_log_std": std,
                        "energy_log2_mean": energy_log2_mean,
                        "energy_log2_std": energy_log2_std,
                        "frames": frames,
                        "sample_rate": self.model_config.sample_rate,
                        "n_mels": self.model_config.n_mels,
                        "n_fft": self.model_config.n_fft,
                        "hop_length": self.model_config.hop_length,
                        "win_length": self.model_config.win_length,
                    },
                    f,
                )
            if frames == 0 or (abs(mean - (-4.0)) < 1e-6 and abs(std - 4.0) < 1e-6):
                self.logger.warning(
                    "Computed normalization stats are defaults (-4, 4) or zero frames; check dataset lists and audio."
                )
            else:
                self.logger.info(
                    f"Computed normalization stats: mean={mean:.4f}, std={std:.4f}, frames={frames}"
                )
                self.logger.info(
                    f"Computed energy normalization: mean={self.normalization.energy_log2_mean:.4f}, std={self.normalization.energy_log2_std:.4f}"
                )
            # Also write a copy under dataset root for reuse across runs
            ds_copy = osp.join(self.config.dataset.path, "normalization.json")
            try:
                with open(ds_copy, "w", encoding="utf-8") as f2:
                    json.dump(
                        {
                            "mel_log_mean": mean,
                            "mel_log_std": std,
                            "energy_log2_mean": energy_log2_mean,
                            "energy_log2_std": energy_log2_std,
                            "frames": frames,
                            "sample_rate": self.model_config.sample_rate,
                            "n_mels": self.model_config.n_mels,
                            "n_fft": self.model_config.n_fft,
                            "hop_length": self.model_config.hop_length,
                            "win_length": self.model_config.win_length,
                        },
                        f2,
                    )
            except Exception as e:
                self.logger.warning(f"Failed to write dataset normalization.json: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to write normalization.json: {e}")
