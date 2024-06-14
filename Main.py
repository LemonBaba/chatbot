from data_preprocessor import DataPreprocessor
from seq2seq_model import Seq2SeqModel
from chatbot_cli import ChatbotCLI

if __name__ == '__main__':
    data_preprocessor = DataPreprocessor(max_seq_length=10)

    input_texts_en = ["Hi", "How are you?", "What is your name?", "Goodbye"]
    target_texts_en = ["\tHello\n", "\tI'm fine, thank you.\n", "\tI am a chatbot.\n", "\tSee you later!\n"]

    input_texts_es = ["Hola", "¿Cómo estás?", "¿Cuál es tu nombre?", "Adiós"]
    target_texts_es = ["\tHola\n", "\tEstoy bien, gracias.\n", "\tSoy un chatbot.\n", "\t¡Hasta luego!\n"]

    data_preprocessor.fit_tokenizer(input_texts_en + target_texts_en, 'en')
    data_preprocessor.fit_tokenizer(input_texts_es + target_texts_es, 'es')

    encoder_input_data_en = data_preprocessor.texts_to_sequences(input_texts_en, 'en')
    decoder_input_data_en = data_preprocessor.texts_to_sequences(target_texts_en, 'en')
    decoder_target_data_en = data_preprocessor.create_decoder_target_data(target_texts_en, 'en')

    encoder_input_data_es = data_preprocessor.texts_to_sequences(input_texts_es, 'es')
    decoder_input_data_es = data_preprocessor.texts_to_sequences(target_texts_es, 'es')
    decoder_target_data_es = data_preprocessor.create_decoder_target_data(target_texts_es, 'es')

    seq2seq_model = Seq2SeqModel(latent_dim=256, max_seq_length=10)
    model_en, encoder_model_en, decoder_model_en = seq2seq_model.build_model(
        len(data_preprocessor.tokenizers['en'].word_index) + 1,
        len(data_preprocessor.tokenizers['en'].word_index) + 1
    )
    model_es, encoder_model_es, decoder_model_es = seq2seq_model.build_model(
        len(data_preprocessor.tokenizers['es'].word_index) + 1,
        len(data_preprocessor.tokenizers['es'].word_index) + 1
    )

    seq2seq_model.compile_and_train(model_en, encoder_input_data_en, decoder_input_data_en, decoder_target_data_en, epochs=100)
    seq2seq_model.compile_and_train(model_es, encoder_input_data_es, decoder_input_data_es, decoder_target_data_es, epochs=100)

    seq2seq_model.save_model(model_en, 'model_en.h5')
    seq2seq_model.save_model(model_es, 'model_es.h5')

    model_paths = {'en': 'model_en.h5', 'es': 'model_es.h5'}
    chatbot_cli = ChatbotCLI(model_paths, data_preprocessor, seq2seq_model)
    chatbot_cli.start()
