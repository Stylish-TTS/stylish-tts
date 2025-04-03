from config_loader import Config, ModelConfig
from batch_manager import BatchManager
from typing import Optional, Any
import os.path as osp
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
import logging
from torch.utils.data import DataLoader
from losses import (
    GeneratorLoss,
    DiscriminatorLoss,
    WavLMLoss,
    MultiResolutionSTFTLoss,
    PitchReconstructionLoss,
)
from torch.utils.tensorboard.writer import SummaryWriter
from text_utils import TextCleaner


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


class TrainContext:
    def __init__(
        self,
        stage_name: str,
        base_out_dir: str,
        config: Config,
        model_config: ModelConfig,
        logger: logging.Logger,
    ) -> None:
        import stage_context

        self.base_output_dir: str = base_out_dir
        self.out_dir: str = ""
        self.reset_out_dir(stage_name)
        self.config: Config = config
        self.model_config: ModelConfig = model_config
        self.batch_manager: Optional[BatchManager] = None
        self.stage: Optional[stage_context.StageContext] = None
        self.manifest: Manifest = Manifest()
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

        self.val_dataloader: Optional[DataLoader] = None

        self.model: Optional[Any] = None

        self.logger: logging.Logger = logger

        # Losses
        self.generator_loss: Optional[GeneratorLoss] = None  # Generator Loss
        self.discriminator_loss: Optional[DiscriminatorLoss] = (
            None  # Discriminator Loss
        )
        self.wavlm_loss: Optional[WavLMLoss] = None  # WavLM Loss
        self.stft_loss: MultiResolutionSTFTLoss = MultiResolutionSTFTLoss(
            sample_rate=self.model_config.sample_rate
        ).to(self.config.training.device)
        self.pitch_r_loss: PitchReconstructionLoss = PitchReconstructionLoss(
            hop_length=self.model_config.hop_length,
            win_length=self.model_config.win_length,
        ).to(self.config.training.device)

        # Run parameters
        self.n_down: int = 1  # TODO: Use train.model.text_aligner.n_down
        self.text_cleaner = TextCleaner(self.model_config.symbol)

    def reset_out_dir(self, stage_name):
        self.out_dir = osp.join(self.base_output_dir, stage_name)
