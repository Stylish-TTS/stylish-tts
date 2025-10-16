import torch
from einops import rearrange
from stylish_tts.train.utils import DurationProcessor


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        speech_predictor,
        pitch_energy_predictor,
        duration_predictor,
        device,
        class_count,
        max_dur,
        **kwargs,
    ):
        super(ExportModel, self).__init__()

        for model in [
            speech_predictor,
            duration_predictor,
            pitch_energy_predictor,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.duration_predictor = duration_predictor
        self.duration_processor = DurationProcessor(class_count, max_dur).to(device)
        self.speech_predictor = speech_predictor
        self.pitch_energy_predictor = pitch_energy_predictor

    def forward(
        self, texts, text_lengths, speech_style, pe_style, duration_style
    ):  # , alignment):
        dur_pred = self.duration_predictor(texts, text_lengths, duration_style)
        alignment = self.duration_processor(dur_pred, text_lengths)
        pitch, energy, voiced = self.pitch_energy_predictor(
            texts, text_lengths, alignment, pe_style
        )
        prediction = self.speech_predictor(
            texts, text_lengths, alignment, pitch, energy, voiced.round(), speech_style
        )
        audio = rearrange(prediction.audio, "1 1 l -> l")
        return audio
