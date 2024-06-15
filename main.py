from data_preprocessor import DataPreprocessor
from seq2seq_model import Seq2SeqModel
from chatbot_cli import ChatbotCLI

def main():
    # Data preprocessing
    data_preprocessor = DataPreprocessor('data/')
    input_tensor, target_tensor, input_tokenizer, target_tokenizer = data_preprocessor.preprocess_data()

    # Model creation and training
    vocab_inp_size = len(input_tokenizer.word_index) + 1
    vocab_tar_size = len(target_tokenizer.word_index) + 1
    seq2seq_model = Seq2SeqModel(vocab_inp_size, vocab_tar_size, input_tokenizer, target_tokenizer)
    seq2seq_model.train(input_tensor, target_tensor, epochs=10, batch_size=64)

    # Start chatbot CLI
    chatbot_cli = ChatbotCLI(seq2seq_model, input_tokenizer, target_tokenizer)
    chatbot_cli.start_chat()

if __name__ == "__main__":
    main()
