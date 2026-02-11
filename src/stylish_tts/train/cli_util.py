# from models.export_model import ExportModel
# count_parameters(ExportModel(**train.model, device=train.config.training.device))
# exit(0)


def count_parameters(model):
    table = PrettyTable(["Module", "Parameters (M)"])
    summary = defaultdict(float)
    total_params = 0

    for name, parameter in model.named_parameters():
        module = ".".join(name.split(".")[:2])
        summary[module] += parameter.numel() / 1_000_000
        total_params += parameter.numel() / 1_000_000

    for module, params in summary.items():
        table.add_row([module, f"{params:.3}M"])

    print(table)
    print(f"Total Trainable Params: {total_params:,.2f}M")
    return total_params


class Checkpoint:
    def __init__(self, checkpoint, config, model_config):
        from stylish_tts.train.train_context import Manifest
        from stylish_tts.train.models.models import build_model
        from stylish_tts.train.utils import DurationProcessor
        from accelerate import Accelerator
        from accelerate import DistributedDataParallelKwargs
        from stylish_tts.train.losses import DiscriminatorLoss

        self.config = config
        self.model_config = model_config
        self.duration_processor = DurationProcessor(
            class_count=model_config.duration_predictor.duration_classes,
            max_dur=model_config.duration_predictor.max_duration,
        ).to(config.training.device)
        self.manifest = Manifest()

        ddp_kwargs = DistributedDataParallelKwargs(
            broadcast_buffers=False, find_unused_parameters=True
        )
        self.accelerator = Accelerator(
            project_dir=".",
            split_batches=True,
            kwargs_handlers=[ddp_kwargs],
            mixed_precision=config.training.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        self.model = build_model(model_config)
        for key in self.model:
            self.model[key] = self.accelerator.prepare(self.model[key])
            self.model[key].to(config.training.device)

        self.disc_loss = DiscriminatorLoss(
            disc=self.model.disc,
            mrd0=self.model.mrd0,
            mrd1=self.model.mrd1,
            mrd2=self.model.mrd2,
            pitch=self.model.pitch_disc,
            duration=self.model.dur_disc,
        )

        from stylish_tts.train.train_context import NormalizationStats

        self.norm = NormalizationStats()
        self.accelerator.register_for_checkpointing(config)
        self.accelerator.register_for_checkpointing(model_config)
        self.accelerator.register_for_checkpointing(self.manifest)
        self.accelerator.register_for_checkpointing(self.norm)
        self.accelerator.register_for_checkpointing(self.disc_loss)

        self.accelerator.load_state(checkpoint)
