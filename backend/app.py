from flask import Flask, request, jsonify
from flask_cors import CORS
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)
CORS(app)

# Download NLTK data (only needed once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Sample data for demonstration
corpus = [
    'What is your name?',
    'How are you doing today?',
    'How do I reset my password?',
    'Thanks for your help.',
    "What are your operating hours?",
    "How can I update my billing information?",
    "Where can I find your pricing?",
    "I'm having trouble logging in. What should I do?",
    "How do I cancel my subscription?",
    "Do you offer a free trial?",
    "How can I contact your support team?",
    "What payment methods do you accept?",
    "Can I change my account email address?",
    "How do I update my profile picture?",
    "Is there a mobile app available?",
    "How do I unsubscribe from marketing emails?",
    "How do I download my invoice?",
    "Where can I find your Terms of Service?",
    "What happens if I forget my security question answer?",
    "How do I delete my account?",
    "Is my data secure with your service?",
    "How do I upgrade my plan?",
    "What should I do if I encounter a bug?"
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
    # Tokenize the query
    words = word_tokenize(query)

    # Part-of-speech tagging
    pos_tags = pos_tag(words)

    # Named Entity Recognition (NER) using NLTK's ne_chunk
    ner_tags = ne_chunk(pos_tags)

    # Extract entities from NER tags
    entities = []
    for chunk in ner_tags:
        if hasattr(chunk, 'label'):
            entities.append(' '.join(c[0] for c in chunk.leaves()))

    return entities

def generate_response(intent_index, entities):
    # Responses corresponding to intents
    responses = [
        "My name is Chatbot.",
        "I'm doing fine, thank you.",
        "Please visit our support page for password reset instructions.",
        "You're welcome!",
        "Our support team is available from 9 AM to 5 PM, Monday to Friday.",
        "You can update your billing information in your account settings under the billing section.",
        "You can find our pricing information on our website [link].",
        "Please try resetting your password using the 'Forgot Password' link on the login page.",
        "You can cancel your subscription from your account settings under the subscription section.",
        "Yes, we offer a free trial. You can sign up on our website [link].",
        "You can contact our support team via email at [email] or through our live chat.",
        "We accept Visa, MasterCard, American Express, and PayPal.",
        "To change your account email address, go to your account settings and update the email field.",
        "To update your profile picture, go to your profile settings and upload a new image.",
        "Yes, we have a mobile app available for iOS and Android. You can download it from the App Store or Google Play.",
        "You can unsubscribe from marketing emails by clicking the 'Unsubscribe' link at the bottom of any email.",
        "You can download your invoice from your account settings under the billing section.",
        "You can find our Terms of Service on our website [link].",
        "If you forget your security question answer, please contact our support team for assistance.",
        "To delete your account, go to your account settings and choose the delete account option.",
        "Yes, your data is secured with encryption and strict access controls.",
        "To upgrade your plan, go to your account settings and choose the upgrade option.",
        "If you encounter a bug, please report it to our support team with details about the issue."
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

    # Extract entities
    entities = extract_entities(user_input)

    # Generate response
    response = generate_response(intent_index, entities)

    return jsonify({'response': response, 'entities': entities})

if __name__ == '__main__':
    app.run(debug=True)
