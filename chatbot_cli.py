import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ChatbotCLI:
    def __init__(self, model, input_tokenizer, target_tokenizer):
        self.model = model
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length_input = 20
        self.max_length_output = 20

    def start_chat(self):
        print("Start chatting with the bot (type 'quit' to stop)!")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'quit':
                break
            response = self.generate_response(user_input)
            print(f"Bot: {response}")

    def preprocess_input(self, sentence):
        sentence_seq = self.input_tokenizer.texts_to_sequences([sentence])
        sentence_seq = pad_sequences(sentence_seq, maxlen=self.max_length_input, padding='post')
        return sentence_seq

    def generate_response(self, sentence):
        sentence_seq = self.preprocess_input(sentence)
        response = self.model.predict(sentence_seq)
        return response
