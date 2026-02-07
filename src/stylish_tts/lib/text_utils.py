# IPA Phonemizer: https://github.com/bootphon/phonemizer
# from config_loader import SymbolConfig
import logging

logger = logging.getLogger(__name__)


class TextCleaner:
    def __init__(self, symbols):
        self._pad = symbols.pad  # "$"
        self._punctuation = symbols.punctuation  # ';:,.!?¡¿—…"()“” '
        self._letters = (
            symbols.letters
        )  # 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        self._letters_ipa = (
            symbols.letters_ipa
        )  # "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        self.word_index_dictionary = self.build_text_cleaner()
        self.index_word_dictionary = {
            v: k for k, v in self.word_index_dictionary.items()
        }
        # logger.debug(len(dicts))

    def __call__(self, text):
        text = self._pad + text + self._pad
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                logger.error("Meld " + char + ": " + text)
        return indexes

    def build_text_cleaner(self):
        # Export all symbols:
        symbols = (
            [self._pad]
            + list(self._punctuation)
            + list(self._letters)
            + list(self._letters_ipa)
        )
        symbol_dict = {}
        for i in range(len((symbols))):
            symbol_dict[symbols[i]] = i
        return symbol_dict
