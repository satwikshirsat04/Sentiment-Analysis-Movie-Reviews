from flask import Flask, render_template, request, jsonify
import pickle
import re

app = Flask(__name__)

with open("sentiment.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(review):
    review = preprocess_text(review)
    review_vector = vectorizer.transform([review]) 
    prediction = model.predict(review_vector)      
    if prediction[0] == 1:
        return "Positive"
    elif prediction[0] == 0:
        return "Negative"
    else:
        return "Neutral"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data.get('review', '')
    sentiment = predict_sentiment(review)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
