import math
from typing import List, Optional, Union, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoModel
import numpy as np
import k2
from einops import rearrange
from stylish_tts.train.multi_spectrogram import multi_spectrogram_count
from stylish_tts.train.models.discriminator import run_discriminator_model


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        *,
        sample_rate,
    ):
        super(MultiResolutionSTFTLoss, self).__init__()

    def spectral_convergence_loss(self, target, pred):
        return torch.norm(target - pred, p=1) / (torch.norm(target, p=1) + 1e-6)

    def forward(self, *, target_list, pred_list, log):
        loss = 0.0
        for target, pred in zip(target_list, pred_list):
            loss += self.spectral_convergence_loss(target, pred)
        loss /= len(target_list)

        log.add_loss("mel", loss)

        return loss


def anti_wrapping_loss(phase_diff, weights):
    loss = torch.abs(phase_diff - 2 * math.pi * torch.round(phase_diff / (2 * math.pi)))
    return loss * weights


def differential_phase_loss(pred, target, n_fft):
    freq_size = target.shape[1]
    weights = torch.arange(freq_size).to(pred.device)
    base = math.exp(math.log(2.5) / (freq_size // 2))
    weights = torch.pow(base, weights)
    weights = rearrange(weights, "w -> 1 w 1")

    phase_loss = anti_wrapping_loss(pred - target, weights).mean()

    freq_matrix = (
        torch.triu(torch.ones(freq_size, freq_size), diagonal=1)
        - torch.triu(torch.ones(freq_size, freq_size), diagonal=2)
        - torch.eye(freq_size)
    )
    freq_matrix = freq_matrix.to(pred.device)
    pred_freq = torch.matmul(pred.transpose(1, 2), freq_matrix)
    target_freq = torch.matmul(target.transpose(1, 2), freq_matrix)
    phase_loss += anti_wrapping_loss(
        (pred_freq - target_freq).transpose(1, 2), weights
    ).mean()

    frames = target.shape[2]
    time_matrix = (
        torch.triu(torch.ones(frames, frames), diagonal=1)
        - torch.triu(torch.ones(frames, frames), diagonal=2)
        - torch.eye(frames)
    )
    time_matrix = time_matrix.to(pred.device)
    pred_time = torch.matmul(pred, time_matrix)
    target_time = torch.matmul(target, time_matrix)
    phase_loss += anti_wrapping_loss(pred_time - target_time, weights).mean()

    return phase_loss


def multi_phase_loss(pred_list, target_list, n_fft):
    loss = 0
    for pred, target in zip(pred_list, target_list):
        loss += differential_phase_loss(pred, target, n_fft)
    return loss / len(pred_list)


class MagPhaseLoss(torch.nn.Module):
    """Magnitude/Phase Loss for Ringformer"""

    def __init__(self, *, n_fft, hop_length, win_length):
        super(MagPhaseLoss, self).__init__()
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # weights = torch.arange(n_fft // 2 + 1)
        # base = math.exp(math.log(2.5) / (n_fft // 2))
        # weights = torch.pow(base, weights)
        # weights = rearrange(weights, "w -> 1 w 1")
        # self.register_buffer("weights", weights)

    # def phase_loss(self, phase_diff):
    #     loss = torch.abs(
    #         phase_diff - 2 * math.pi * torch.round(phase_diff / (2 * math.pi))
    #     )
    #     return loss * self.weights

    def forward(self, pred, gt, log):
        y_stft = torch.stft(
            gt,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            window=self.window,
        )
        target_mag = torch.abs(y_stft) + 1e-14
        target_phase = (target_mag > 1e-3).detach() * torch.angle(y_stft)
        pred.phase = (target_mag > 1e-3).detach() * pred.phase
        # target_phase = torch.angle(y_stft)
        log.add_loss(
            "mag",
            torch.nn.functional.l1_loss(
                pred.magnitude,
                torch.log(target_mag + 1e-9),
            ),
        )
        # phase_loss = self.phase_loss(pred.phase - target_phase).mean()

        # n_fft = self.n_fft
        # freq_matrix = (
        #     torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=1)
        #     - torch.triu(torch.ones(n_fft // 2 + 1, n_fft // 2 + 1), diagonal=2)
        #     - torch.eye(n_fft // 2 + 1)
        # )
        # freq_matrix = freq_matrix.to(pred.phase.device)
        # pred_freq = torch.matmul(pred.phase.transpose(1, 2), freq_matrix)
        # target_freq = torch.matmul(target_phase.transpose(1, 2), freq_matrix)
        # phase_loss += self.phase_loss((pred_freq - target_freq).transpose(1, 2)).mean()

        # frames = target_phase.shape[2]
        # time_matrix = (
        #     torch.triu(torch.ones(frames, frames), diagonal=1)
        #     - torch.triu(torch.ones(frames, frames), diagonal=2)
        #     - torch.eye(frames)
        # )
        # time_matrix = time_matrix.to(pred.phase.device)
        # pred_time = torch.matmul(pred.phase, time_matrix)
        # target_time = torch.matmul(target_phase, time_matrix)
        # phase_loss += self.phase_loss(pred_time - target_time).mean()

        phase_loss = differential_phase_loss(pred.phase, target_phase, self.n_fft)

        log.add_loss("phase", phase_loss)


class DiscriminatorLoss(torch.nn.Module):
    def __init__(self, *, mrd0, mrd1, mrd2, pitch, duration):
        super(DiscriminatorLoss, self).__init__()
        self.discriminators = torch.nn.ModuleDict(
            {
                "mrd0": DiscriminatorLossHelper(mrd0, 1),  # multi_spectrogram_count),
                "mrd1": DiscriminatorLossHelper(mrd1, 1),  # multi_spectrogram_count),
                "mrd2": DiscriminatorLossHelper(mrd2, 1),  # multi_spectrogram_count),
                "pitch_disc": DiscriminatorLossHelper(pitch, 1),
                "dur_disc": DiscriminatorLossHelper(duration, 1),
            }
        )
        self.disc_list = [
            self.discriminators["mrd0"],
            self.discriminators["mrd1"],
            self.discriminators["mrd2"],
            self.discriminators["pitch_disc"],
            self.discriminators["dur_disc"],
        ]

    def get_disc_lr_multiplier(self, key):
        return self.discriminators[key].get_disc_lr_multiplier()

    def forward(self, *, target_list, pred_list, used, index):
        if "pitch_disc" in used:
            loss = self.discriminators["pitch_disc"](
                target=target_list[0], pred=pred_list[0]
            )
        elif "dur_disc" in used:
            loss = self.discriminators["dur_disc"](
                target=target_list[0], pred=pred_list[0]
            )
        else:
            loss = self.disc_list[index](
                target=target_list[index], pred=pred_list[index]
            )
        # loss = 0
        # for key in used:
        #     loss += self.discriminators[key](
        #         target_list=target_list, pred_list=pred_list
        #     )
        return loss.mean()

    def state_dict(self, *args, **kwargs):
        state = {}
        for key, helper in self.discriminators.items():
            state[f"discriminators.{key}.last_loss"] = helper.last_loss
            state[f"discriminators.{key}.weight"] = 1
        return state

    def load_state_dict(self, state_dict, strict=True):
        for key, helper in self.discriminators.items():
            if f"discriminators.{key}.last_loss" in state_dict:
                helper.last_loss = state_dict[f"discriminators.{key}.last_loss"]
        return state_dict


class DiscriminatorLossHelper(torch.nn.Module):
    """
    Discriminator Loss Helper: Returns discriminator loss for a single discriminator
    """

    def __init__(self, model, sub_count):
        super(DiscriminatorLossHelper, self).__init__()
        self.model = model
        self.last_loss = 0.5 * sub_count
        self.ideal_loss = 0.5 * sub_count
        self.f_max = 4.0
        self.h_min = 0.01
        self.x_max = 0.05 * sub_count
        self.x_min = 0.05 * sub_count

    def get_disc_lr_multiplier(self):
        x = abs(self.last_loss - self.ideal_loss)
        result = 1.0
        if self.last_loss > self.ideal_loss + self.ideal_loss * self.x_max:
            result = self.f_max
        elif self.last_loss < self.ideal_loss - self.ideal_loss * self.x_min:
            result = self.h_min
        elif self.last_loss > self.ideal_loss:
            result = min(math.pow(self.f_max, x / self.x_max), self.f_max)
        else:
            result = max(math.pow(self.h_min, x / self.x_min), self.h_min)
        return result

    def discriminator_loss(
        self,
        real_score: List[torch.Tensor],
        gen_score: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_score, gen_score):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss

        return loss

    def tprls_loss(
        self,
        real_score: List[torch.Tensor],
        gen_score: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_score, gen_score):
            tau = 0.04
            m_dg = torch.median((dr - dg))
            l_rel_unreduced = (((dr - dg) - m_dg) ** 2)[dr < dg + m_dg]
            l_rel = torch.sum(l_rel_unreduced) / (
                math.prod(l_rel_unreduced.shape) + 1e-9
            )
            loss += tau - F.relu(tau - l_rel)
        return loss

    def forward(self, *, target, pred):
        # real_score, gen_score, _, _ = self.model(
        #     target_list=target_list, pred_list=pred_list
        # )
        real_score, gen_score, _, _ = run_discriminator_model(self.model, target, pred)
        disc = self.discriminator_loss(real_score, gen_score)
        tprls = self.tprls_loss(real_score, gen_score)
        self.last_loss = self.last_loss * 0.95 + disc.item() * 0.05
        return disc + tprls


class GeneratorLoss(torch.nn.Module):
    def __init__(self, *, mrd0, mrd1, mrd2, pitch, duration):
        super(GeneratorLoss, self).__init__()
        self.generators = torch.nn.ModuleDict(
            {
                "mrd0": GeneratorLossHelper(mrd0),
                "mrd1": GeneratorLossHelper(mrd1),
                "mrd2": GeneratorLossHelper(mrd2),
                "pitch_disc": GeneratorLossHelper(pitch),
                "dur_disc": GeneratorLossHelper(duration),
            }
        )
        self.gen_list = [
            self.generators["mrd0"],
            self.generators["mrd1"],
            self.generators["mrd2"],
            self.generators["pitch_disc"],
            self.generators["dur_disc"],
        ]

    def forward(self, *, target_list, pred_list, used, index):
        if "pitch_disc" in used:
            loss = self.generators["pitch_disc"](
                target=target_list[0], pred=pred_list[0]
            )
        elif "dur_disc" in used:
            loss = self.generators["dur_disc"](target=target_list[0], pred=pred_list[0])
        else:
            loss = self.gen_list[index](
                target=target_list[index], pred=pred_list[index]
            )
        # loss = 0
        # for key in used:
        #     loss += self.generators[key](target_list=target_list, pred_list=pred_list)
        return loss.mean()


class GeneratorLossHelper(torch.nn.Module):
    """
    Generator Loss Helper: Returns generator loss for a single discriminator
    """

    def __init__(self, model):
        super(GeneratorLossHelper, self).__init__()
        self.model = model

    def generator_loss(self, gen_score: List[torch.Tensor]) -> torch.Tensor:
        loss = 0
        for dg in gen_score:
            loss += torch.mean((1 - dg) ** 2)
        return loss

    def feature_loss(
        self, real_features: List[torch.Tensor], gen_features: List[torch.Tensor]
    ) -> torch.Tensor:
        loss = 0
        for dr, dg in zip(real_features, gen_features):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    def tprls_loss(self, real_score, gen_score):
        loss = 0
        for dg, dr in zip(real_score, gen_score):
            tau = 0.04
            m_DG = torch.median((dr - dg))
            L_rel = torch.mean((((dr - dg) - m_DG) ** 2)[dr < dg + m_DG])
            loss += tau - F.relu(tau - L_rel)
        return loss

    def forward(self, *, target, pred):
        real_score, gen_score, real_features, gen_features = run_discriminator_model(
            self.model, target, pred
        )
        # real_score, gen_score, real_features, gen_features = self.model(
        #     target_list=target_list, pred_list=pred_list
        # )
        feature = self.feature_loss(real_features, gen_features)
        gen = self.generator_loss(gen_score)
        tprls = self.tprls_loss(real_score, gen_score)
        return feature + gen + tprls


class WavLMLoss(torch.nn.Module):
    def __init__(self, model, model_sr, slm_sr=16000):
        super(WavLMLoss, self).__init__()
        self.wavlm = AutoModel.from_pretrained(model)
        self.resample = torchaudio.transforms.Resample(model_sr, slm_sr)

    def forward(self, wav, y_rec):
        with torch.no_grad():
            wav_16 = self.resample(wav)
            wav_embeddings = self.wavlm(
                input_values=wav_16, output_hidden_states=True
            ).hidden_states
            wav_tensor = torch.stack(wav_embeddings)
        y_rec_16 = self.resample(y_rec)
        y_rec_embeddings = self.wavlm(
            input_values=y_rec_16.squeeze(1), output_hidden_states=True
        ).hidden_states
        y_rec_tensor = torch.stack(y_rec_embeddings)
        return torch.nn.functional.l1_loss(wav_tensor, y_rec_tensor)


class CDW_CCELoss(nn.Module):
    def __init__(self, classes, *, alpha=1.0, weight=None):
        super(CDW_CCELoss, self).__init__()
        indices = torch.arange(classes)
        distance_list = []
        for i in range(classes):
            distance_list.append(torch.abs(i - indices).clamp(max=7) ** alpha)
        distance_weight = torch.stack(distance_list)
        self.register_buffer("distance_weight", distance_weight)
        if weight is not None:
            self.register_buffer("custom_weight", weight)
        else:
            self.weight = None

    def forward(self, pred, target):
        """pred is of shape NxC, target is of shape N"""
        index = torch.arange(pred.shape[0], device=pred.device)
        if self.custom_weight is not None:
            weight = self.custom_weight[target]
            ce = torch.log_softmax(pred, dim=1)[index, target] * (
                weight / (weight.sum() + 1e-9)
            )
        else:
            ce = torch.log_softmax(pred, dim=1)[index, target] / pred.shape[0]
            exit(1)
            pass
        distance = self.distance_weight[target]
        cdw = torch.log((1 - torch.softmax(pred, dim=1)) + 1e-9)  # * distance
        cdw = cdw * (distance / distance.sum(dim=1, keepdim=True))
        cdw = cdw / pred.shape[0]
        return -ce.sum(), -cdw.sum() * 100


class DurationLoss(torch.nn.Module):
    def __init__(self, *, class_count, weight):
        super(DurationLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.sqrt(weight))
        # self.loss = CDW_CCELoss(class_count, alpha=2, weight=weight)

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, text_length: torch.Tensor):
        loss_ce = 0
        loss_cdw = 0
        for i in range(text_length.shape[0]):
            # ce, cdw = self.loss(
            #     pred[i, : text_length[i]], gt[i, : text_length[i]].long()
            # )
            ce = self.loss(pred[i, : text_length[i]], gt[i, : text_length[i]].long())
            loss_ce += ce
            # loss_cdw += cdw
        return loss_ce / text_length.shape[0], loss_cdw / text_length.shape[0]


# The following code was adapated from: https://github.com/huangruizhe/audio/blob/aligner_label_priors/examples/asr/librispeech_alignment/loss.py

# BSD 2-Clause License
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


class CTCLossWithLabelPriors(nn.Module):
    def __init__(self, prior_scaling_factor=0.0, blank=0, reduction="mean"):
        super().__init__()

        self.blank = blank
        self.reduction = reduction

        self.log_priors = None
        self.log_priors_sum = None
        self.num_samples = 0
        self.prior_scaling_factor = prior_scaling_factor  # This corresponds to the `alpha` hyper parameter in the paper
        self.k2_device = "cpu"
    
    def to(self, device):
        super().to(device)
        self.k2_device = device if k2.with_cuda else "cpu"
        return self

    def encode_supervisions(
        self, targets, target_lengths, input_lengths
    ) -> Tuple[torch.Tensor, Union[List[str], List[List[int]]]]:
        # https://github.com/k2-fsa/icefall/blob/master/icefall/utils.py#L181

        batch_size = targets.size(0)
        supervision_segments = torch.stack(
            (
                torch.arange(batch_size),
                torch.zeros(batch_size),
                input_lengths.cpu(),
            ),
            1,
        ).to(torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)
        supervision_segments = supervision_segments[indices]

        # Be careful: the targets here are already padded! We need to remove paddings from it
        res = targets[indices].tolist()
        res_lengths = target_lengths[indices].tolist()
        res = [l[:l_len] for l, l_len in zip(res, res_lengths)]

        return supervision_segments, res, indices

    def forward(
        self,
        log_probs: Tensor,
        targets: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        step_type="train",
    ) -> Tensor:
        supervision_segments, token_ids, indices = self.encode_supervisions(
            targets, target_lengths, input_lengths
        )

        decoding_graph = k2.ctc_graph(
            token_ids, modified=False, device=self.k2_device
        )

        # Accumulate label priors for this epoch
        log_probs = log_probs.permute(1, 0, 2)  # (T, N, C) -> (N, T, C)
        if step_type == "train":
            log_probs_flattened = []
            for lp, le in zip(log_probs, input_lengths):
                log_probs_flattened.append(lp[: int(le.item())])
            log_probs_flattened = torch.cat(log_probs_flattened, 0)

            # Note, the log_probs here is already log_softmax'ed.
            T = log_probs_flattened.size(0)
            self.num_samples += T
            log_batch_priors_sum = torch.logsumexp(
                log_probs_flattened, dim=0, keepdim=True
            )
            log_batch_priors_sum = log_batch_priors_sum.detach()
            if self.log_priors_sum is None:
                self.log_priors_sum = log_batch_priors_sum
            else:
                _temp = torch.stack([self.log_priors_sum, log_batch_priors_sum], dim=-1)
                self.log_priors_sum = torch.logsumexp(_temp, dim=-1)

            # Apply the label priors
            if self.log_priors is not None and self.prior_scaling_factor > 0:
                log_probs = log_probs - self.log_priors * self.prior_scaling_factor

        # Compute CTC loss
        dense_fsa_vec = k2.DenseFsaVec(
            log_probs.to(self.k2_device),  # (N, T, C)
            supervision_segments.to(self.k2_device),
        )

        loss = k2.ctc_loss(
            decoding_graph=decoding_graph,
            dense_fsa_vec=dense_fsa_vec,
            output_beam=10,
            reduction=self.reduction,
            use_double_scores=True,
            target_lengths=target_lengths.to(self.k2_device),
        )

        return loss.to(log_probs.device)

    def on_train_epoch_end(self, train):
        if self.log_priors_sum is not None:
            log_priors_sums = train.accelerator.gather(self.log_priors_sum.unsqueeze(0))
            log_priors_sums = torch.logsumexp(log_priors_sums, dim=0, keepdim=True)
            num_samples = train.accelerator.gather(
                torch.Tensor([self.num_samples]).to(log_priors_sums.device)
            )
            num_samples = torch.log(num_samples.sum() + 1e-9).to(log_priors_sums.device)
            new_log_prior = log_priors_sums - num_samples
            if False:
                print(
                    "new_priors: ",
                    ["{0:0.2f}".format(i) for i in new_log_prior[0][0].exp().tolist()],
                )
                print(
                    "new_log_prior: ",
                    ["{0:0.2f}".format(i) for i in new_log_prior[0][0].tolist()],
                )
                if self.log_priors is not None:
                    _a1 = new_log_prior.exp()
                    _b1 = self.log_priors.exp()
                    print(
                        "diff%: ",
                        [
                            "{0:0.2f}".format(i)
                            for i in ((_a1 - _b1) / _b1 * 100)[0][0].tolist()
                        ],
                    )

            prior_threshold = -12.0
            new_log_prior = torch.where(
                new_log_prior < prior_threshold, prior_threshold, new_log_prior
            )

            self.log_priors = new_log_prior
            self.log_priors_sum = None
            self.num_samples = 0
