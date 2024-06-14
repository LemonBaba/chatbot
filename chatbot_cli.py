from langdetect import detect
import numpy as np

class ChatbotCLI:
    def __init__(self, model_paths, data_preprocessor, seq2seq_model):
        self.model_paths = model_paths
        self.data_preprocessor = data_preprocessor
        self.seq2seq_model = seq2seq_model
        self.models = {}
        self.load_models()

    def load_models(self):
        for lang, path in self.model_paths.items():
            model = self.seq2seq_model.load_model(path)
            _, encoder_model, decoder_model = self.seq2seq_model.build_model(
                num_encoder_tokens=len(self.data_preprocessor.tokenizers[lang].word_index) + 1,
                num_decoder_tokens=len(self.data_preprocessor.tokenizers[lang].word_index) + 1
            )
            self.models[lang] = (encoder_model, decoder_model)

    def decode_sequence(self, input_seq, lang):
        if lang not in self.models:
            print(f"Detected language '{lang}' is not supported. Falling back to English.")
            lang = 'en'
        encoder_model, decoder_model = self.models[lang]

        states_value = encoder_model.predict(input_seq)

        target_seq = np.zeros((1, 1))
        tokenizer = self.data_preprocessor.tokenizers[lang]
        target_seq[0, 0] = tokenizer.word_index['\t']

        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = tokenizer.index_word[sampled_token_index]
            decoded_sentence += sampled_char

            if sampled_char == '\n' or len(decoded_sentence) > self.data_preprocessor.max_seq_length:
                stop_condition = True

            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]

        return decoded_sentence

    def start(self):
        print("Chatbot is ready. Type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            detected_lang = detect(user_input)
            if detected_lang not in self.models:
                print(f"Detected language '{detected_lang}' is not supported. Falling back to English.")
                detected_lang = 'en'
            input_seq = self.data_preprocessor.texts_to_sequences([user_input], detected_lang)
            response = self.decode_sequence(input_seq, detected_lang)
            print(f"Bot: {response}")
