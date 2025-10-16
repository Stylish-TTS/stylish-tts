import sys
import click
from scipy.io.wavfile import write
import numpy as np
from .stylish_model import StylishModel
import pyloudnorm as pyln

# TODO: Remove torch/safetensors dependency
import torch
from safetensors import safe_open
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors


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
        if "voicepack" in f.keys():
            voicepack = f.get_tensor("voicepack").detach().cpu().numpy()
        else:
            exit(f"Could not find voicepack key in {voicepack}")

    speech_pack = voicepack[:, :64]
    pe_pack = voicepack[:, 64:128]
    duration_pack = voicepack[:, 128:192]
    sbert_pack = voicepack[:, 192:]

    sbert = SentenceTransformer("stsb-mpnet-base-v2")
    matcher = NearestNeighbors(n_neighbors=20, algorithm="ball_tree")
    matcher.fit(sbert_pack)

    model = StylishModel(model)
    meter = pyln.Meter(model.sample_rate())  # create BS.1770 meter
    results = []
    with open(infile, "r") as f:
        for line in f:
            fields = line.strip().split("|")
            tokens = model.tokenize(fields[0])
            plaintext = fields[1]
            embedding = sbert.encode([plaintext])

            indices = matcher.kneighbors(embedding, return_distance=False)
            speech_style = speech_pack[indices].mean(axis=1)
            pe_style = pe_pack[indices].mean(axis=1)
            duration_style = duration_pack[indices].mean(axis=1)

            # voice_index = max(511, min(2, len(tokens)))
            # s_style, pe_style, d_style = torch.chunk(
            #     voicepack[voice_index], chunks=3, dim=0
            # )
            # s_style = s_style.unsqueeze(0).detach().cpu().numpy()
            # pe_style = pe_style.unsqueeze(0).detach().cpu().numpy()
            # d_style = d_style.unsqueeze(0).detach().cpu().numpy()
            audio = model.generate_speech(
                tokens, [speech_style, pe_style, duration_style]
            )
            # loudness = meter.integrated_loudness(audio)
            # audio = pyln.normalize.loudness(audio, loudness, -20.0)
            audio = np.multiply(audio, 32768).astype(np.int16)
            results.append(audio)
            sys.stderr.write(".")
            sys.stderr.flush()

    sys.stderr.write("\n")
    sys.stderr.flush()
    combined = np.concatenate(results)
    print("Saving to:", outfile)
    write(outfile, model.sample_rate(), combined)
