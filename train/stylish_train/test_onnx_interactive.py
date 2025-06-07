import torch
import sys

sys.path.insert(0, "D:\\stylish-tts\\lib")
from stylish_lib.config_loader import ModelConfig
from stylish_lib.text_utils import TextCleaner
import numpy as np
import onnxruntime as ort
import click
from scipy.io.wavfile import write
import onnx
from time import perf_counter
import sounddevice as sd
from misaki.vi import VIG2P
from underthesea import sent_tokenize
import threading
import queue
from collections import deque
import time

g2p = VIG2P()


def read_meta_data_onnx(filename, key):
    model = onnx.load(filename)
    for prop in model.metadata_props:
        if prop.key == key:
            return prop.value
    return None


def g2p_preprocess(inp):
    re = []
    for sent in sent_tokenize(inp):
        text = g2p(sent)[0]
        text = text.replace("\u0306", "").replace("\u0361", "").replace("͡", "")
        text = (
            text.replace("'", "")
            .replace("1", "")
            .replace("5", "↗")
            .replace("2", "↘")
            .replace("4", "ʌ")
            .replace("3", "→")
            .replace("6", "↓")
        )
        re.append(text)
    return re


class SeamlessAudioPlayer:
    def __init__(self, sample_rate=24000, buffer_size=1024):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.stream = None
        self.audio_buffer = deque()
        self.current_audio = np.array([], dtype=np.int16)
        self.position = 0

    def start_stream(self):
        """Start the audio stream"""
        if self.stream is None:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.int16,
                callback=self._audio_callback,
                blocksize=self.buffer_size,
            )
            self.stream.start()
            self.is_playing = True

    def stop_stream(self):
        """Stop the audio stream"""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.is_playing = False

    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback function for continuous playback"""
        if status:
            print(f"Audio callback status: {status}")

        # Fill output buffer
        output_frames = 0
        outdata.fill(0)  # Initialize with silence

        while output_frames < frames:
            # If current audio is exhausted, get next from buffer
            if self.position >= len(self.current_audio):
                if self.audio_buffer:
                    self.current_audio = self.audio_buffer.popleft()
                    self.position = 0
                else:
                    # No more audio, fill with silence
                    break

            # Calculate how many frames to copy
            available_frames = len(self.current_audio) - self.position
            frames_to_copy = min(frames - output_frames, available_frames)

            # Copy audio data
            end_pos = self.position + frames_to_copy
            outdata[output_frames : output_frames + frames_to_copy, 0] = (
                self.current_audio[self.position : end_pos]
            )

            self.position += frames_to_copy
            output_frames += frames_to_copy

    def add_audio(self, audio_data):
        """Add audio data to the playback buffer"""
        if isinstance(audio_data, np.ndarray):
            self.audio_buffer.append(audio_data.flatten())

    def is_buffer_empty(self):
        """Check if audio buffer is empty"""
        return len(self.audio_buffer) == 0 and self.position >= len(self.current_audio)

    def wait_for_completion(self):
        """Wait until all audio has been played"""
        while not self.is_buffer_empty():
            time.sleep(0.01)


class TTSProcessor:
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.model_config = None
        self.text_cleaner = None
        self.session = None
        self.audio_player = SeamlessAudioPlayer()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ONNX model and related components"""
        model_config = read_meta_data_onnx(self.onnx_path, "model_config")
        assert (
            model_config
        ), "model_config metadata not found. Please rerun ONNX conversion."

        self.model_config = ModelConfig.model_validate_json(model_config)
        self.text_cleaner = TextCleaner(self.model_config.symbol)
        self.session = ort.InferenceSession(
            self.onnx_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def generate_audio(self, text):
        """Generate audio from text"""
        tokens = torch.tensor(self.text_cleaner(text)).unsqueeze(0)
        texts = torch.zeros([1, tokens.shape[1] + 2], dtype=int)
        texts[0][1 : tokens.shape[1] + 1] = tokens
        text_lengths = torch.zeros([1], dtype=int)
        text_lengths[0] = tokens.shape[1] + 2

        outputs = self.session.run(
            None,
            {
                "texts": texts.cpu().numpy(),
                "text_lengths": text_lengths.cpu().numpy(),
            },
        )

        sample = np.multiply(outputs[0], 32768).astype(np.int16)
        return sample.flatten()[5000:-5000]

    def process_and_queue_audio(self, inp):
        """Process text and queue audio for seamless playback"""
        processed_texts = g2p_preprocess(inp)

        for i, text in enumerate(processed_texts):
            start_time = perf_counter()
            audio_data = self.generate_audio(text)
            generation_time = perf_counter() - start_time

            # Add audio to player buffer
            self.audio_player.add_audio(audio_data)

            # Start playback on first chunk
            if i == 0 and not self.audio_player.is_playing:
                self.audio_player.start_stream()

            # Calculate RTF (Real Time Factor)
            audio_duration = len(audio_data) / 24000
            rtf = generation_time / audio_duration
            print(f"Chunk {i+1}: {generation_time:.3f}s, RTF: {rtf:.3f}")

    def start(self):
        """Start the interactive TTS session"""
        print("Seamless Vietnamese TTS Ready!")
        print("Type 'q' to quit, 'stop' to stop current playback")

        try:
            while True:
                inp = input("\nVietnamese text: ").strip()

                if inp.lower() == "q":
                    break
                elif inp.lower() == "stop":
                    self.audio_player.stop_stream()
                    self.audio_player = SeamlessAudioPlayer()  # Reset player
                    print("Playback stopped.")
                    continue
                elif not inp:
                    continue

                # Process input in a separate thread for better responsiveness
                threading.Thread(
                    target=self.process_and_queue_audio, args=(inp,), daemon=True
                ).start()

        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        if self.audio_player:
            self.audio_player.stop_stream()


@click.command()
@click.option(
    "--onnx_path", type=str, required=True, help="Path to the ONNX model file"
)
@click.option("--buffer_size", type=int, default=1024, help="Audio buffer size")
def main(onnx_path, buffer_size):
    """Seamless Vietnamese Text-to-Speech with continuous audio playback"""
    try:
        processor = TTSProcessor(onnx_path)
        processor.audio_player.buffer_size = buffer_size
        processor.start()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
