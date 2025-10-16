from stylish_tts.lib.config_loader import ModelConfig
from stylish_tts.lib.text_utils import TextCleaner
import onnxruntime as ort
import onnx
import numpy as np


class StylishModel:
    def __init__(self, model_path):
        model_config = read_meta_data_onnx(model_path, "model_config")
        if not model_config:
            exit("Could not read model metadata")
        self.model_config = ModelConfig.model_validate_json(model_config)
        self.text_cleaner = TextCleaner(self.model_config.symbol)
        self.model = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def tokenize(self, phonemes):
        tokens = np.array(self.text_cleaner(phonemes))
        tokens = np.expand_dims(tokens, 0)
        return tokens

    def generate_speech(self, tokens, styles):
        texts = np.zeros([1, tokens.shape[1] + 2], dtype=int)
        texts[0][1 : tokens.shape[1] + 1] = tokens
        text_lengths = np.zeros([1], dtype=int)
        text_lengths[0] = tokens.shape[1] + 2
        outputs = self.model.run(
            None,
            {
                "texts": texts,
                "text_lengths": text_lengths,
                "speech_style": styles[0],
                "pe_style": styles[1],
                "duration_style": styles[2],
            },
        )
        return outputs[0]

    def sample_rate(self):
        return self.model_config.sample_rate


def read_meta_data_onnx(filename, key):
    model = onnx.load(filename)
    for prop in model.metadata_props:
        if prop.key == key:
            return prop.value
    return None
