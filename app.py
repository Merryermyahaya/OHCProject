from flask import Flask, render_template, request, jsonify, url_for
import joblib
import os
import requests
from sentence_transformers.util import cos_sim
import torch

# Download model file if not present
MODEL_PATH = 'chatbot_model.pkl'
MODEL_URL = 'https://drive.google.com/uc?export=download&id=YOUR_FILE_ID'  # replace with actual file ID

if not os.path.exists(MODEL_PATH):
    print("Downloading chatbot model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, 'wb') as f:
        f.write(response.content)
    print("Model downloaded!")

# Define the chatbot class
class SemanticChatbot:
    def __init__(self, df, index, model):  # fixed typo: init → __init__
        self.df = df
        self.index = index
        self.model = model

    def get_response(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        question_embeddings = self.model.encode(self.df["Preprocessed_Question"].tolist(), convert_to_tensor=True)
        scores = cos_sim(query_embedding, question_embeddings)
        top_idx = scores.argmax()
        best_match = self.df.iloc[top_idx.item()]
        return best_match["Answer"]

# Initialize Flask app
app = Flask(__name__)

# Load the chatbot
chatbot = joblib.load(MODEL_PATH)

# Greetings
greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "good night"]

def is_greeting(text):
    return any(word in text for word in greetings)

# Routes
@app.route('/')
def home():
    return render_template('web1.html')

@app.route('/get', methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data.get("message", "").strip().lower()

    if is_greeting(user_input):
        return jsonify({"response": "Hello! How can I assist you with admissions today?"})

    try:
        query_embedding = chatbot.model.encode(user_input, convert_to_tensor=True)
        question_embeddings = chatbot.model.encode(chatbot.df["Preprocessed_Question"].tolist(), convert_to_tensor=True)
        scores = cos_sim(query_embedding, question_embeddings)
        max_score = torch.max(scores).item()

        if max_score < 0.65:
            reply = "I only answer questions related to admissions!!! Ask something related 😊"
        else:
            top_idx = scores.argmax()
            reply = chatbot.df.iloc[top_idx.item()]["Answer"]

    except Exception as e:
        reply = "Something went wrong while processing your request. Please try again."

    return jsonify({"response": reply})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)