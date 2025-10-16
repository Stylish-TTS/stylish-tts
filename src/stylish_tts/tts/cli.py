import sys
import click
from scipy.io.wavfile import write
import numpy as np
from .stylish_model import StylishModel
import pyloudnorm as pyln

# TODO: Remove torch/safetensors dependency
import torch
from safetensors import safe_open


@click.group("stylish-tts")
def cli():
    pass


@cli.command(
    "speak",
    short_help="Use a Stylish TTS model to convert text from stdin to audio, one utterance per line.",
)
@click.argument("model", type=str)
@click.argument("voicepack", type=str)
@click.argument("infile", type=str)
@click.argument("outfile", type=str)
@click.option(
    "--lang",
    type=str,
    default="phonemes",
    help="ISO 639 language code to use for G2P or 'phonemes' for no G2P",
)
def speak_document(model, voicepack, infile, outfile, lang):
    if lang != "phonemes":
        exit("Only phoneme input supported for now")
    with safe_open(voicepack, framework="pt", device="cpu") as f:
        if "basic_voicepack" in f.keys():
            voicepack = f.get_tensor("basic_voicepack")
        else:
            exit(f"Could not find basic voicepack key in {voicepack}")

    model = StylishModel(model)
    meter = pyln.Meter(model.sample_rate())  # create BS.1770 meter
    results = []
    with open(infile, "r") as f:
        for line in f:
            tokens = model.tokenize(line.strip())
            voice_index = max(511, min(2, len(tokens)))
            s_style, pe_style, d_style = torch.chunk(
                voicepack[voice_index], chunks=3, dim=0
            )
            s_style = s_style.unsqueeze(0).detach().cpu().numpy()
            pe_style = pe_style.unsqueeze(0).detach().cpu().numpy()
            d_style = d_style.unsqueeze(0).detach().cpu().numpy()
            audio = model.generate_speech(tokens, [s_style, pe_style, d_style])
            loudness = meter.integrated_loudness(audio)
            audio = pyln.normalize.loudness(audio, loudness, -20.0)
            audio = np.multiply(audio, 32768).astype(np.int16)
            results.append(audio)
            sys.stderr.write(".")
            sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.flush()
    combined = np.concatenate(results)
    print("Saving to:", outfile)
    write(outfile, model.sample_rate(), combined)
