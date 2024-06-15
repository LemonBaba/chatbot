import os
import re
import tensorflow as tf
import unicodedata


class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.max_length_input = 20
        self.max_length_output = 20

    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, sentence):
        sentence = self.unicode_to_ascii(sentence.lower().strip())
        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()
        return sentence

    def load_conversations(self):
        conversations = []
        with open(os.path.join(self.data_path, 'movie_conversations.txt'), 'r', encoding='utf-8',
                  errors='ignore') as file:
            for line in file.readlines():
                line = line.strip().split(" +++$+++ ")
                conversations.append(line[-1][1:-1].replace("'", "").split(", "))
        return conversations

    def load_lines(self):
        lines = {}
        with open(os.path.join(self.data_path, 'movie_lines.txt'), 'r', encoding='utf-8', errors='ignore') as file:
            for line in file.readlines():
                parts = line.strip().split(" +++$+++ ")
                lines[parts[0]] = parts[-1]
        return lines

    def create_dataset(self):
        lines = self.load_lines()
        conversations = self.load_conversations()
        input_lang = []
        target_lang = []

        for conv in conversations:
            for i in range(len(conv) - 1):
                input_lang.append(self.preprocess_sentence(lines[conv[i]]))
                target_lang.append(self.preprocess_sentence(lines[conv[i + 1]]))

        return input_lang, target_lang

    def tokenize(self, lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
        return tensor, lang_tokenizer

    def preprocess_data(self):
        input_lang, target_lang = self.create_dataset()
        input_tensor, input_tokenizer = self.tokenize(input_lang)
        target_tensor, target_tokenizer = self.tokenize(target_lang)
        return input_tensor, target_tensor, input_tokenizer, target_tokenizer
