from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch
import torchaudio
import tqdm
from stylish_tts.train.cli_util import Checkpoint
from stylish_tts.train.dataloader import build_dataloader, FilePathDataset
from stylish_tts.lib.text_utils import TextCleaner
from stylish_tts.train.utils import get_data_path_list, calculate_mel, log_norm


def make_voicepack(config, model_config, dynamic, checkpoint):
    device = config.training.device
    state = Checkpoint(checkpoint, config, model_config)
    if state.norm.frames <= 0:
        exit("No normalization state found. Cannot generate voicepack.")

    to_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=model_config.n_mels,
        n_fft=model_config.n_fft,
        win_length=model_config.win_length,
        hop_length=model_config.hop_length,
        sample_rate=model_config.sample_rate,
    ).to(config.training.device)

    to_style_mel = torchaudio.transforms.MelSpectrogram(
        n_mels=model_config.style_encoder.n_mels,
        n_fft=model_config.style_encoder.n_fft,
        win_length=model_config.style_encoder.win_length,
        hop_length=model_config.style_encoder.hop_length,
        sample_rate=model_config.sample_rate,
    ).to(config.training.device)
    text_cleaner = TextCleaner(model_config.symbol)

    datalist = get_data_path_list(Path(config.dataset.path) / config.dataset.train_data)

    if dynamic:
        sbert = SentenceTransformer("stsb-mpnet-base-v2")
        paths = []
        plaintexts = []
        for line in datalist:
            fields = line.strip().split("|")
            paths.append(fields[0])
            plaintexts.append(fields[3])
        sbert_embeddings = sbert.encode(plaintexts)

        path_to_embedding = {}
        for i in range(len(paths)):
            path_to_embedding[paths[i]] = sbert_embeddings[i]

    dataset = FilePathDataset(
        data_list=datalist,
        root_path=Path(config.dataset.path) / config.dataset.wav_path,
        text_cleaner=text_cleaner,
        model_config=model_config,
        pitch_path=Path(config.dataset.path) / config.dataset.pitch_path,
        alignment_path=Path(config.dataset.path) / config.dataset.alignment_path,
        duration_processor=state.duration_processor,
    )

    time_bins, _ = dataset.time_bins()
    dataloader = build_dataloader(
        dataset,
        time_bins,
        validation=True,
        num_workers=0,
        device=config.training.device,
        multispeaker=model_config.multispeaker,
        stage="voicepack",
        train=None,
        hop_length=model_config.hop_length,
    )

    iterator = tqdm.tqdm(
        iterable=enumerate(dataloader),
        desc="Generating styles",
        total=len(datalist),
        unit="steps",
        initial=0,
        colour="GREEN",
        leave=False,
        dynamic_ncols=True,
    )
    state.model.speech_style_encoder.eval()
    state.model.pe_style_encoder.eval()
    state.model.duration_style_encoder.eval()

    if dynamic:
        result = make_dynamic(
            iterator, state, to_mel, to_style_mel, device, path_to_embedding
        )
    else:
        result = make_static(iterator, state, to_mel, to_style_mel, device)
    return result


def make_dynamic(iterator, state, to_mel, to_style_mel, device, path_to_embedding):
    styles = []
    for _, batch in iterator:
        path = batch[3][0]
        combined = calculate_style(batch, state, to_mel, to_style_mel, device)
        embedding = torch.from_numpy(path_to_embedding[path]).to(device)
        combined = torch.cat(
            [
                combined,
                embedding,
            ],
            dim=0,
        )
        styles.append(combined)

    result = torch.stack(styles, dim=0)
    return result


def make_static(iterator, state, to_mel, to_style_mel, device):
    styles = [[] for _ in range(512)]
    for _, batch in iterator:
        length = batch[2][0].to(device)
        combined = calculate_style(batch, state, to_mel, to_style_mel, device)
        styles[length.item() - 1].append(combined)

    result = []
    for i in range(512):
        lower = i
        upper = i + 1
        while total_len(styles[lower:upper]) < 100:
            lower -= 1
            upper += 1
            if lower < 0 and upper > 512:
                exit("Need at least 100 styles to make a voicepack")
        flattened = sum(styles[lower:upper], [])
        average = torch.stack(flattened, dim=0).mean(dim=0)
        result.append(average)
    result = torch.stack(result, dim=0)
    return result


def calculate_style(batch, state, to_mel, to_style_mel, device):
    with torch.no_grad():
        wave = batch[0].to(device)
        pitch = batch[4].to(device)
        energy_mel, _ = calculate_mel(
            wave,
            to_mel,
            state.norm.mel_log_mean,
            state.norm.mel_log_std,
        )
        energy = log_norm(
            energy_mel.unsqueeze(1),
            state.norm.mel_log_mean,
            state.norm.mel_log_std,
        ).squeeze(1)
        energy = torch.log(energy + 1e-9)
        style_mel, _ = calculate_mel(
            wave, to_style_mel, state.norm.mel_log_mean, state.norm.mel_log_std
        )
        speech_style = state.model.speech_style_encoder(style_mel.unsqueeze(1))
        pe_style = state.model.pe_style_encoder(style_mel, pitch, energy)
        duration_style = state.model.duration_style_encoder(style_mel.unsqueeze(1))

        combined = torch.cat(
            [
                speech_style.squeeze(0),
                pe_style.squeeze(0),
                duration_style.squeeze(0),
            ],
            dim=0,
        )
    return combined


def total_len(listlist):
    result = 0
    for item in listlist:
        result += len(item)
    return result
