import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dot, Activation, Concatenate


class Seq2SeqModel:
    def __init__(self, vocab_inp_size, vocab_tar_size, input_tokenizer, target_tokenizer, embedding_dim=256, units=512):
        self.vocab_tar_size = vocab_tar_size
        self.embedding_dim = embedding_dim
        self.units = units
        self.encoder = self.build_encoder(vocab_inp_size, embedding_dim, units)
        self.decoder = self.build_decoder(vocab_tar_size, embedding_dim, units)
        self.model = self.build_seq2seq_model(vocab_inp_size, vocab_tar_size, embedding_dim, units)
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer

    def build_encoder(self, vocab_size, embedding_dim, units):
        inputs = Input(shape=(None,))
        embedding = Embedding(vocab_size, embedding_dim)(inputs)
        outputs, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(embedding)
        return Model(inputs, [outputs, state_h, state_c])

    def build_decoder(self, vocab_size, embedding_dim, units):
        inputs = Input(shape=(None,))
        enc_outputs = Input(shape=(None, units))
        enc_state_h = Input(shape=(units,))
        enc_state_c = Input(shape=(units,))
        embedding = Embedding(vocab_size, embedding_dim)(inputs)
        lstm_out, state_h, state_c = LSTM(units, return_sequences=True, return_state=True)(
            embedding, initial_state=[enc_state_h, enc_state_c]
        )
        attention = Dot(axes=[2, 2])([lstm_out, enc_outputs])
        attention = Activation('softmax')(attention)
        context = Dot(axes=[2, 1])([attention, enc_outputs])
        concat = Concatenate()([lstm_out, context])
        outputs = Dense(vocab_size, activation='softmax')(concat)
        return Model([inputs, enc_outputs, enc_state_h, enc_state_c], outputs)

    def build_seq2seq_model(self, vocab_inp_size, vocab_tar_size, embedding_dim, units):
        encoder_inputs = Input(shape=(None,))
        encoder_outputs, state_h, state_c = self.encoder(encoder_inputs)

        decoder_inputs = Input(shape=(None,))
        decoder_outputs = self.decoder([decoder_inputs, encoder_outputs, state_h, state_c])

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, input_tensor, target_tensor, epochs=10, batch_size=64):
        decoder_input_data = np.zeros(target_tensor.shape)
        decoder_input_data[:, 1:] = target_tensor[:, :-1]
        self.model.fit([input_tensor, decoder_input_data], target_tensor, epochs=epochs, batch_size=batch_size)

    def predict(self, input_seq):
        encoder_output, state_h, state_c = self.encoder.predict(input_seq)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = self.input_tokenizer.word_index['startseq']

        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
            output_tokens = self.decoder.predict([target_seq, encoder_output, state_h, state_c])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.target_tokenizer.index_word.get(sampled_token_index, '')

            if sampled_word == 'endseq' or len(decoded_sentence) > self.max_length_output:
                stop_condition = True
            else:
                decoded_sentence.append(sampled_word)
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = sampled_token_index
                state_h, state_c = state_h, state_c

        return ' '.join(decoded_sentence)
