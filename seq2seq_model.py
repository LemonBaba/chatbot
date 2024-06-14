from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding

class Seq2SeqModel:
    def __init__(self, latent_dim=256, max_seq_length=10):
        self.latent_dim = latent_dim
        self.max_seq_length = max_seq_length

    def build_model(self, num_encoder_tokens, num_decoder_tokens):
        # Encoder
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(num_encoder_tokens, self.latent_dim)(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        # Decoder
        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(num_decoder_tokens, self.latent_dim)
        decoder_embedding2 = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding2, initial_state=encoder_states)
        decoder_dense = Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn encoder_input_data & decoder_input_data into decoder_target_data
        training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        # Encoder model for inference
        encoder_model = Model(encoder_inputs, encoder_states)

        # Decoder model for inference
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_embedding_inf = decoder_embedding(decoder_inputs)
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_embedding_inf, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return training_model, encoder_model, decoder_model

    def compile_and_train(self, model, encoder_input_data, decoder_input_data, decoder_target_data, epochs=100):
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=64,
            epochs=epochs,
            validation_split=0.2
        )

    def save_model(self, model, path):
        model.save(path)

    def load_model(self, path):
        from tensorflow.keras.models import load_model
        return load_model(path)
