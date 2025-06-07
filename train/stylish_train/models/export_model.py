import torch


class ExportModel(torch.nn.Module):
    def __init__(
        self,
        *,
        text_encoder,
        text_duration_encoder,
        text_pe_encoder,
        textual_style_encoder,
        textual_prosody_encoder,
        textual_pe_encoder,
        duration_predictor,
        pitch_energy_predictor,
        pe_duration_encoder,
        decoder,
        generator,
        device="cuda",
        **kwargs
    ):
        super(ExportModel, self).__init__()

        for model in [
            text_encoder,
            text_duration_encoder,
            text_pe_encoder,
            textual_style_encoder,
            textual_prosody_encoder,
            textual_pe_encoder,
            duration_predictor,
            pitch_energy_predictor,
            pe_duration_encoder,
            decoder,
            generator,
        ]:
            model.to(device).eval()
            for p in model.parameters():
                p.requires_grad = False

        self.device = device
        self.text_encoder = text_encoder
        self.text_duration_encoder = text_duration_encoder
        self.text_pe_encoder = text_pe_encoder
        self.textual_style_encoder = textual_style_encoder
        self.textual_prosody_encoder = textual_prosody_encoder
        self.textual_pe_encoder = textual_pe_encoder
        self.duration_predictor = duration_predictor
        self.pitch_energy_predictor = pitch_energy_predictor
        self.pe_duration_encoder = pe_duration_encoder
        self.decoder = decoder
        self.generator = generator

    def decoding_single(
        self,
        text_encoding,
        duration,
        pitch,
        energy,
        style,
    ):
        style = style @ duration
        mel, _ = self.decoder(
            text_encoding @ duration, pitch, energy, style, probing=False
        )
        prediction = self.generator(mel=mel, style=style, pitch=pitch, energy=energy)
        return prediction

    def duration_predict(self, duration_encoding, prosody_embedding, text_lengths):
        duration = self.duration_predictor(
            duration_encoding,
            prosody_embedding,
            text_lengths,
        )
        duration = torch.sigmoid(duration).sum(axis=-1)

        pred_dur = torch.round(duration).clamp(min=1).long().squeeze()
        indices = torch.repeat_interleave(
            torch.arange(duration_encoding.shape[2], device=self.device), pred_dur
        )
        pred_aln_trg = torch.zeros(
            (duration_encoding.shape[2], indices.shape[0]), device=self.device
        )
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0).to(self.device)

        return pred_aln_trg

    def pe_predict(self, pe_encoding, pe_embedding, pred_aln_trg):
        d = self.pe_duration_encoder.text_encoder.infer(pe_encoding, pe_embedding)
        pe = d.permute(0, 2, 1) @ pred_aln_trg
        return pe

    def forward(self, texts, text_lengths):
        text_encoding, _, _ = self.text_encoder(texts, text_lengths)
        duration_encoding, _, _ = self.text_duration_encoder(texts, text_lengths)
        pe_encoding, _, _ = self.text_pe_encoder(texts, text_lengths)

        style_embedding = self.textual_style_encoder(text_encoding)
        prosody_embedding = self.textual_prosody_encoder(duration_encoding)
        pe_embedding = self.textual_pe_encoder(pe_encoding)

        duration_prediction = self.duration_predict(
            duration_encoding,
            prosody_embedding,
            text_lengths,
        )
        pe = self.pe_duration_encoder(
            pe_encoding,
            pe_embedding,
            text_lengths,
        )
        pitch_prediction, energy_prediction = self.pitch_energy_predictor(
            pe.transpose(-1, -2), pe_embedding @ duration_prediction
        )
        prediction = self.decoding_single(
            text_encoding,
            duration_prediction,
            pitch_prediction,
            energy_prediction,
            style_embedding,
        )
        return prediction.audio.squeeze()
