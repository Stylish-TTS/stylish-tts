from __future__ import annotations
import logging
import torch
from typing import Optional
import stylish_tts.train.train_context

logger = logging.getLogger(__name__)


def build_loss_log(train: train_context.TrainContext):
    return LossLog(train.logger, train.writer, train.config.loss_weight)


class LossLog:
    def __init__(self, logger, writer, loss_weight):
        self.logger = logger
        self.writer = writer
        self.weights = loss_weight
        self.weight_dict = loss_weight.model_dump()
        self.metrics = {}
        self.total_loss = None

    def total(self):
        if self.total_loss is None:
            self.calculate_metrics()
        return self.total_loss

    def broadcast(self, manifest, stage, validation=False):
        if self.total_loss is None:
            self.calculate_metrics()
        loss_list = [f"{k}: {v:.3f}" for k, v in self.metrics.items()]
        loss_string = f"loss: {self.total_loss:.3f}, " + ", ".join(loss_list)
        if validation:
            writer_type = "eval"
            best_string = ""
            if manifest.best_loss != float("inf"):
                best_string = f", (best was {manifest.best_loss})"
            self.logger.info(
                f"Validation step {manifest.current_total_step}: "
                + loss_string
                + best_string
            )
        else:
            writer_type = "train"
            lr = stage.optimizer.optimizers["speech_predictor"].param_groups[0]["lr"]
            lr_string = f", lr: {lr:.7f}"
            disc_lr = stage.optimizer.optimizers["mrd0"].param_groups[0]["lr"]
            disc_lr_string = f", disc_lr: {disc_lr:.7f}"
            self.logger.info(
                f"Epoch [{manifest.current_epoch}/{stage.max_epoch}], "
                + f"Step [{manifest.current_step}/{manifest.steps_per_epoch}], "
                + loss_string
                + lr_string
                + disc_lr_string
            )
        self.writer.add_scalar(
            f"{writer_type}/loss", self.total_loss, manifest.current_total_step
        )
        for key, value in self.metrics.items():
            self.writer.add_scalar(
                f"{writer_type}/{key}", value, manifest.current_total_step
            )

    def weight(self, key: str):
        if key in self.weight_dict:
            return self.weight_dict[key]
        else:
            # logging.error(f"WARNING: Unknown weight for key {key}, defaulting to 1")
            logging.debug(f"self.weights: {self.weights}")
            return 1

    def calculate_metrics(self):
        total = 0
        total_weight = 0
        for key, value in self.metrics.items():
            weight = self.weight(key)
            loss = value * weight
            total += loss
            total_weight += weight
        self.total_loss = total  # / total_weight

    def backwards_loss(self):
        total = 0
        for key, value in self.metrics.items():
            if key == "generator" or key == "align_loss":
                loss = value
            else:
                if torch.is_tensor(value):
                    loss = value / (value.detach() + 1e-9)
                else:
                    loss = value
            weight = self.weight(key)
            total += loss * weight
        return total

    def detach(self):
        for key, value in self.metrics.items():
            if torch.is_tensor(value):
                self.metrics[key] = value.item()
        if torch.is_tensor(self.total_loss):
            self.total_loss = self.total_loss.item()
        return self

    def add_loss(self, key, value):
        self.metrics[key] = value
        self.total_loss = None


def combine_logs(loglist) -> Optional[LossLog]:
    result = None
    if len(loglist) > 0:
        result = LossLog(loglist[0].logger, loglist[0].writer, loglist[0].weights)
        totals = {}
        counts = {}
        for log in loglist:
            for key in log.metrics.keys():
                if key not in totals:
                    totals[key] = 0
                    counts[key] = 0
                totals[key] += log.metrics[key]
                counts[key] += 1
        for key in totals.keys():
            result.metrics[key] = totals[key] / counts[key]
    return result
