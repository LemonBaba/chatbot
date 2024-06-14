from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class DataPreprocessor:
    def __init__(self, max_seq_length=10):
        self.max_seq_length = max_seq_length
        self.tokenizers = {}

    def fit_tokenizer(self, texts, lang):
        tokenizer = Tokenizer(char_level=False, filters='')
        tokenizer.fit_on_texts(texts)
        # Add special tokens
        tokenizer.word_index['\t'] = len(tokenizer.word_index) + 1
        tokenizer.word_index['\n'] = len(tokenizer.word_index) + 2
        self.tokenizers[lang] = tokenizer

    def texts_to_sequences(self, texts, lang):
        tokenizer = self.tokenizers[lang]
        sequences = tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')

    def create_decoder_target_data(self, texts, lang):
        tokenizer = self.tokenizers[lang]
        sequences = tokenizer.texts_to_sequences(texts)
        sequences = pad_sequences(sequences, maxlen=self.max_seq_length, padding='post')
        decoder_target_data = np.zeros((len(texts), self.max_seq_length, len(tokenizer.word_index) + 1), dtype='float32')
        for i, seq in enumerate(sequences):
            for t, word in enumerate(seq):
                if t > 0:
                    decoder_target_data[i, t - 1, word] = 1.0
        return decoder_target_data
