import pathlib, sys
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

import click
import numpy
import torch
from torch.nn import functional as F
import librosa

from safetensors.torch import save_file
from stylish_tts.train.dataprep.align_text import audio_list, tqdm_wrapper
import pyworld
import tqdm
from stylish_tts.train.dataloader import get_frame_count, get_time_bin
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download


def calculate_pitch(config, model_config, method, workers):
    root = pathlib.Path(config.dataset.path)
    out = root / config.dataset.pitch_path
    wavdir = root / config.dataset.wav_path
    vals = calculate_pitch_set(
        "Val Set",
        method,
        root / config.dataset.val_data,
        wavdir,
        model_config,
        workers,
        config.training.device,
    )
    trains = calculate_pitch_set(
        "Train set",
        method,
        root / config.dataset.train_data,
        wavdir,
        model_config,
        workers,
        config.training.device,
    )
    result = vals | trains
    save_file(result, out)


def calculate_pitch_set(label, method, path, wavdir, model_config, workers, device):
    model = None
    if method == "rmvpe":
        calculate_single = calculate_pitch_rmvpe
        from .rmvpe import RMVPE, SAMPLE_RATE

        model = RMVPE(
            hf_hub_download("stylish-tts/pitch_extractor", "rmvpe.safetensors"),
            device=device,
            hop_length=SAMPLE_RATE
            // (model_config.sample_rate // model_config.hop_length),
        )
    elif method == "pyworld":
        calculate_single = calculate_pitch_pyworld
    else:
        exit("Invalid pitch calculation method passed")

    import concurrent.futures

    with path.open("r", encoding="utf-8") as f:
        total_segments = sum(1 for _ in f)

    max_queue_size = workers * 2
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {}
        iterator = tqdm_wrapper(
            audio_list(path, wavdir, model_config),
            total=total_segments,
            desc="Pitch " + label,
            color="GREEN",
        )
        
        result = {}
        
        for name, text_raw, wave in iterator:
            while len(future_map) >= max_queue_size:
                done, _ = concurrent.futures.wait(
                    future_map.keys(), 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    if future in future_map:
                        name_done = future_map.pop(future)
                        try:
                            current = future.result()
                            result[name_done] = current
                        except Exception as e:
                            print(f"{name_done} generated an exception: {str(e)}")

            future = executor.submit(
                calculate_single,
                name,
                text_raw,
                wave,
                model_config.sample_rate,
                model_config.hop_length,
                model,
                device,
            )
            future_map[future] = name

        for future in as_completed(future_map):
            name = future_map[future]
            try:
                current = future.result()
                result[name] = current
            except Exception as e:
                print(f"{name} generated an exception: {str(e)}")
    return result


def calculate_pitch_pyworld(
    name, text_raw, wave, sample_rate, hop_length, model, device
):
    bad_f0 = 5
    zero_value = -10
    frame_period = hop_length / sample_rate * 1000
    f0, t = pyworld.harvest(wave, sample_rate, frame_period=frame_period)
    # if harvest fails, try dio
    if sum(f0 != 0) < bad_f0:
        f0, t = pyworld.dio(wave, sample_rate, frame_period=frame_period)
    pitch = pyworld.stonemask(wave, f0, t, sample_rate)
    pitch = torch.from_numpy(pitch).float().unsqueeze(0)
    if torch.any(torch.isnan(pitch)):
        pitch[torch.isnan(pitch)] = zero_value
    pitch = pitch[:, :-1]
    return pitch


def calculate_pitch_rmvpe(name, text_raw, wave, sample_rate, hop_length, model, device):
    zero_value = -10
    pitch = (
        torch.from_numpy(
            model.infer_from_audio(wave, sample_rate=sample_rate, device=device)
        )
        .float()
        .unsqueeze(0)
    )
    pitch = pitch[:, :-1]
    if torch.any(torch.isnan(pitch)):
        pitch[torch.isnan(pitch)] = zero_value
    return pitch
