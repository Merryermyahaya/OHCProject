# build_model.py

from semantic_chatbot import SemanticChatbot
import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

# Sample data — replace this with your actual admissions Q&A
df = pd.DataFrame({
    "Preprocessed_Question": [
        "what are the admission requirements?",
        "how do I apply?",
        "what is the deadline?"
    ],
    "Answer": [
        "You must submit your transcripts and test scores.",
        "Apply online through our admissions portal.",
        "The deadline is June 30th."
    ]
})

# Load the model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create chatbot instance
chatbot = SemanticChatbot(df=df, index=None, model=model)

# Save the chatbot model
joblib.dump(chatbot, "chatbot_model.pkl")
print("✅ chatbot_model.pkl saved successfully.")
