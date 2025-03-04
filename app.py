from flask import Flask, request, jsonify
import nltk
import heapq
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from transformers import pipeline

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

app = Flask(__name__)

# Initialize the abstractive summarizer model
abstractive_summarizer = pipeline("summarization", model="t5-small")

def extractive_summary(text, num_sentences=3):
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)

    # Calculate word frequencies
    word_frequencies = {}
    for word in words:
        if word.lower() not in stop_words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1

    # Normalize word frequencies
    max_freq = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    # Score sentences based on word frequencies
    sentence_list = sent_tokenize(text)
    sentence_scores = {}
    for sent in sentence_list:
        for word in word_tokenize(sent.lower()):
            if word in word_frequencies:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + word_frequencies[word]

    # Get the top-ranked sentences
    summary_sentences = heapq.nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
    return ' '.join(summary_sentences)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running!"})    

@app.route("/extractive", methods=["POST"])
def extractive():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = extractive_summary(text)
    return jsonify({"summary": summary})

@app.route("/abstractive", methods=["POST"])
def abstractive():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    summary = abstractive_summarizer(text, max_length=150, min_length=30, do_sample=False)
    return jsonify({"summary": summary[0]["summary_text"]})

if __name__ == "__main__":
    app.run(debug=True)
