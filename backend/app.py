from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Sample data for demonstration
corpus = [
    'What is your name?',
    'How are you doing today?',
    'What are the weather conditions today?',
    'Tell me a joke.',
    'What is the capital of France?',
    'How do I reset my password?',
    'Where can I find the nearest coffee shop?',
    'Thanks for your help.'
]

# Tokenization and preprocessing
word_tokenizer = nltk.tokenize.WordPunctTokenizer()
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform TF-IDF vectors
tfidf_vectors = tfidf_vectorizer.fit_transform(corpus)


def get_intent(query):
    # Vectorize the input query
    query_vector = tfidf_vectorizer.transform([query])

    # Calculate similarities with each document
    similarities = cosine_similarity(query_vector, tfidf_vectors)

    # Get the index of the most similar document
    most_similar_index = np.argmax(similarities)

    return most_similar_index


def extract_entities(query):
    # Placeholder function - implement your entity recognition logic here
    # This could involve named entity recognition (NER) or regex matching
    # Example:
    # entities = your_entity_extraction_function(query)
    entities = []
    return entities


def generate_response(intent_index, entities):
    # Responses corresponding to intents
    responses = [
        "My name is Chatbot.",
        "I'm doing fine, thank you.",
        "The weather is sunny with a chance of rain.",
        "Why don't skeletons fight each other? They don't have the guts.",
        "The capital of France is Paris.",
        "Please visit our support page for password reset instructions.",
        "There is a coffee shop at 123 Main Street.",
        "You're welcome!"
    ]

    # Adjust responses based on entities or additional logic if needed
    response = responses[intent_index]
    return response


@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    user_input = data['user_input']

    # Get intent
    intent_index = get_intent(user_input)

    # Extract entities (optional)
    entities = extract_entities(user_input)

    # Generate response
    response = generate_response(intent_index, entities)

    return jsonify({'response': response})


if __name__ == '__main__':
    app.run(debug=True)
