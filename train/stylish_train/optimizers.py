# coding:utf-8
import torch
from torch.optim import AdamW
import logging
import transformers

logger = logging.getLogger(__name__)
logical_step_limit = 10000
# logical_step_warmup = 250
logical_step_warmup = 0

discriminators = {"mpd", "mrd", "msbd", "mstftd"}


class MultiOptimizer:
    def __init__(self, *, optimizers, schedulers, discriminator_loss):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.discriminator_loss = discriminator_loss
        self.keys = list(optimizers.keys())

    def prepare(self, accelerator):
        for key in self.optimizers.keys():
            self.optimizers[key] = accelerator.prepare(self.optimizers[key])
            if key not in discriminators:
                self.schedulers[key] = accelerator.prepare(self.schedulers[key])

    def reset_lr(self, stage_name, train):
        for key in train.model.keys():
            if key not in discriminators:
                lr, _, _ = calculate_lr(key, stage_name, train=train)
                for param_group in self.optimizers[key].param_groups:
                    if isinstance(param_group["lr"], torch.Tensor):
                        param_group["lr"].fill_(lr)
                        param_group["initial_lr"].fill_(lr)
                    else:
                        param_group["lr"] = lr
                        param_group["initial_lr"] = lr
                self.schedulers[key].scheduler.last_epoch = -1
                self.schedulers[key].scheduler.base_lrs = [
                    group["initial_lr"] for group in self.optimizers[key].param_groups
                ]
                self.schedulers[key].step()
        self.reset_discriminator_schedulers()

    def step_discriminator_schedulers(self):
        gen_lr = self.optimizers["decoder"].param_groups[0]["lr"]
        if isinstance(gen_lr, torch.Tensor):
            gen_lr = gen_lr.item()
        for key in discriminators:
            multiplier = self.discriminator_loss.get_disc_lr_multiplier(key)
            lr = gen_lr * multiplier
            for param_group in self.optimizers[key].param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(lr)
                else:
                    param_group["lr"] = lr

    def reset_discriminator_schedulers(self):
        lr = self.optimizers["decoder"].param_groups[0]["lr"]
        if isinstance(lr, torch.Tensor):
            lr = lr.item()
        for key in discriminators:
            for param_group in self.optimizers[key].param_groups:
                if isinstance(param_group["lr"], torch.Tensor):
                    param_group["lr"].fill_(lr)
                else:
                    param_group["lr"] = lr

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, step: int, step_limit: int, stage: str):
        logical_step = step * logical_step_limit // step_limit
        plateau = 0.9
        if stage == "pre_acoustic":
            plateau = 0.5
        logical_step = min(logical_step, logical_step_limit * plateau)
        for key in self.keys:
            if key not in discriminators:
                self.schedulers[key].scheduler.last_epoch = logical_step
                self.schedulers[key].step()


def build_optimizer(stage_name: str, *, train):
    optim = {}
    schedulers = {}
    for key in train.model.keys():
        lr, weight_decay, betas = calculate_lr(key, stage_name, train=train)
        optim[key] = AdamW(
            train.model[key].parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=1e-9,
        )
        if key not in discriminators:
            schedulers[key] = transformers.get_cosine_schedule_with_warmup(
                optim[key],
                num_warmup_steps=logical_step_warmup,
                num_training_steps=logical_step_limit,
            )
    multi_optim = MultiOptimizer(
        optimizers=optim,
        schedulers=schedulers,
        discriminator_loss=train.discriminator_loss,
    )
    return multi_optim


def calculate_lr(key, stage_name, *, train):
    # is_second = (
    #     stage_name == "second"
    #     or stage_name == "second_style"
    #     or stage_name == "second_joint"
    #     or stage_name == "textual"
    #     or stage_name == "joint"
    # )
    lr = train.config.optimizer.lr
    if stage_name == "alignment" or stage_name == "text_encoder":
        lr /= 10
    # elif stage_name == "pre_acoustic":
    #     lr /= 5
    # weight_decay = 1e-4
    # betas = (0.85, 0.99)
    weight_decay = 1e-4
    betas = (0.85, 0.99)
    # if is_second:
    #     if key == "bert":
    #         lr = train.config.optimizer.bert_lr
    #         weight_decay = 1e-2
    #         betas = (0.9, 0.99)
    #     elif key in {"decoder", "style_encoder"}:
    #         lr = train.config.optimizer.ft_lr
    return lr, weight_decay, betas
