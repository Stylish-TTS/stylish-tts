import torch
from .text_encoder import TextEncoder
from .text_style_encoder import TextStyleEncoder
from .prosody_encoder import ProsodyEncoder
from .duration_predictor import DurationPredictor
from .pitch_energy_predictor import PitchEnergyPredictor
from .decoder import Decoder
from .generator import UpsampleGenerator, Generator, MultiGenerator


class SpeechPredictor(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # self.text_encoder = TextEncoder(
        #     inter_dim=model_config.inter_dim, config=model_config.text_encoder
        # )

        self.teacher_mel_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(model_config.n_mels, model_config.generator.input_dim, 1),
            BasicConvNeXtBlock(model_config.generator.input_dim, 768),
        )
        self.quantizer = FSQ(
            dim=model_config.generator.input_dim,
            levels=[5] * 6,  # 15625 codes
            num_codebooks=1,
        )

        self.student_text_encoder = torch.nn.Conv1d(768, model_config.inter_dim, 1)
        self.student_decoder = Decoder(
            dim_in=model_config.inter_dim,
            style_dim=model_config.style_dim,
            dim_out=model_config.generator.input_dim,
            hidden_dim=model_config.decoder.hidden_dim,
            residual_dim=model_config.decoder.residual_dim,
        )

        # self.generator = UpsampleGenerator(
        #     style_dim=model_config.style_dim,
        #     resblock_kernel_sizes=model_config.generator.resblock_kernel_sizes,
        #     upsample_rates=model_config.generator.upsample_rates,
        #     upsample_initial_channel=model_config.generator.input_dim,
        #     upsample_last_channel=model_config.generator.upsample_last_channel,
        #     resblock_dilation_sizes=model_config.generator.resblock_dilation_sizes,
        #     upsample_kernel_sizes=model_config.generator.upsample_kernel_sizes,
        #     gen_istft_n_fft=model_config.generator.gen_istft_n_fft,
        #     gen_istft_hop_size=model_config.generator.gen_istft_hop_size,
        #     sample_rate=model_config.sample_rate,
        # )
        self.generator = MultiGenerator(
            style_dim=model_config.style_dim,
            n_fft=model_config.n_fft,
            win_length=model_config.win_length,
            hop_length=model_config.hop_length,
            sample_rate=model_config.sample_rate,
            config=model_config.generator,
        )

    def _freeze(self, model):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    def forward(
        self,
        texts,
        # text_lengths,
        # alignment,
        pitch,
        energy,
        voiced,
        style,
        denormal_pitch,
        prior_mel=None,
        train_student=False,
    ):
        # text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        mel, distil_loss = None, None
        if self.training:
            assert prior_mel is not None
            if train_student:
                self._freeze(self.teacher_mel_encoder)
                self._freeze(self.quantizer)
                self._freeze(self.generator)
            teacher_mel = self.teacher_mel_encoder(prior_mel)
            teacher_mel, _ = self.quantizer(teacher_mel.mT)
            mel = teacher_mel.mT
        if train_student or not self.training:
            text_encoding = self.student_text_encoder(texts)
            student_mel, _ = self.student_decoder(
                text_encoding,  # text_encoding @ alignment,
                pitch,
                energy,
                style,
                voiced,
            )
            student_mel, _ = self.quantizer(student_mel.mT)
            mel = student_mel.mT
            if self.training:
                distil_loss = torch.nn.functional.mse_loss(student_mel, teacher_mel)

        assert mel is not None

        prediction = self.generator(
            mel=mel,
            style=style,
            pitch=denormal_pitch,
            energy=energy,
            voiced=voiced,
        )
        return prediction, distil_loss
