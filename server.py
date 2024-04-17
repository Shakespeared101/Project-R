# server.py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import Flask, render_template, request, jsonify
from scraper import scrape_website
from transformers import pipeline
from model_loader import load_and_predict_model
from textblob import TextBlob
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import spacy
import re
import nltk
import pandas as pd
import numpy as np
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model
nltk.download('stopwords')
nltk.download('punkt')
with open('C:/Users/ASUS/Desktop/Projects/help/3.0/model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
available_policies = ['CCPA.txt', 'GDPR.txt', 'DPDP.txt']
model = load_model('C:/Users/ASUS/Desktop/Projects/help/3.0/model/your_model.h5')
stop_words = set(stopwords.words('english'))
df = pd.read_csv('C:/Users/ASUS/Desktop/Projects/help/3.0/Datasets/fake_reviews_dataset.csv', nrows=35000)
max_len_text = 504
def preprocess_text(text):
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text
df['clean_text'] = df['text_'].apply(preprocess_text)

# Apply preprocessing to the 'text_' column

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_vectorizer.fit(df['clean_text'])
# Function to preprocess and predict label with probability
def predict_label(review_text):
    clean_review = preprocess_text(review_text)
    review_tfidf_features = tfidf_vectorizer.transform([clean_review])
    review_tfidf_features.sort_indices()
    review_tfidf_features_dense = review_tfidf_features.toarray()
    predicted_prob = model.predict(review_tfidf_features_dense)[0][0]
    predicted_label = "CG" if predicted_prob >= 0.8 else "OR"
    return predicted_label, "{:.2f}".format(predicted_prob * 100)  # Convert probability to percentage and format to two decimal places
app = Flask(__name__)
summarizer = pipeline("summarization")
nlp = spacy.load("en_core_web_md")
def calculate_compatibility(government_policy, company_policy):
    # Use spaCy for text similarity
    gov_doc = nlp(government_policy)
    comp_doc = nlp(company_policy)
    similarity_score = gov_doc.similarity(comp_doc)

    # Use TextBlob for sentiment analysis
    gov_sentiment = TextBlob(government_policy).sentiment.polarity
    comp_sentiment = TextBlob(company_policy).sentiment.polarity

    # Combine scores for a compatibility score
    compatibility_score = (similarity_score + (gov_sentiment + comp_sentiment) / 2) / 2

    # Determine color based on the compatibility score
    if compatibility_score >= 0.7:
        color = "#4caf50"  # Green
    elif compatibility_score >= 0.4:
        color = "#ffc107"  # Yellow
    else:
        color = "#e53935"  # Red

    return {
        "score": round(compatibility_score, 2),
        "color": color
    }

def save_predictions_to_txt(predictions, output_file):
    with open(output_file, 'w') as file:
        for entry in predictions:
            if entry['prediction'] == 1:
                file.write(f"{entry['text']}\n\n")
    
from flask import send_file

@app.route('/scrape', methods=['POST'])
def scrape_and_predict():
    try:
        data = request.json
        url = data.get('url')
        if not url:
            return jsonify({'error': 'Missing URL in the request'})

        scraped_data_file = scrape_website(url)

        if scraped_data_file:
            predictions = load_and_predict_model(scraped_data_file)
            formatted_predictions = [{'text': entry['text'], 'prediction': entry['prediction']} for entry in predictions]
            save_predictions_to_txt(formatted_predictions, 'final.txt')
            with open('final.txt', 'r') as file:
                data = file.read()
            formatted_data = '\n'.join(line.strip() for line in data.split('\n') if line.strip())
            with open('final.txt', 'w') as file:
                file.write(formatted_data)
            return send_file('final.txt', as_attachment=True)
            return jsonify({'status': 'Data scraped and predicted successfully', 'predictions': formatted_predictions})
        else:
            return jsonify({'error': 'Error during data scraping'})
    except Exception as e:
        error_message = f'Error during scraping and prediction: {str(e)}'
        return jsonify({'error': error_message})
    

@app.route('/')
def index():
    return render_template('index.html')
def index1():
    return render_template('index1.html')
def home():
    return render_template('index2.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        article = request.form['article']
        max_chunk_length = 1000
        chunks = [article[i:i + max_chunk_length] for i in range(0, len(article), max_chunk_length)]

        chunk_summaries = []
        for chunk in chunks:
            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            chunk_summaries.append(summary)

        total_summary = ' '.join(chunk_summaries)

        return render_template('index.html', article=article, summary=total_summary)

@app.route('/compare', methods=['POST', 'GET'])
def compare_policies():
    if request.method == 'POST':
        # Inside compare_policies function
        selected_policy = request.form['government_policy']
        company_policy = request.form['company_policy']

        # Load the selected government policy text
        policy_file_path = f"C:/Users/ASUS/Desktop/Projects/help/3.0/static/{selected_policy}"
        with open(policy_file_path, "r", encoding="utf-8") as file:
            government_policy_text = file.read()

        # Perform text comparison and sentiment analysis using spaCy and vaderSentiment
        similarity_score = calculate_similarity(government_policy_text, company_policy)
        sentiment_score = calculate_sentiment(company_policy)

        # Combine scores to get the overall compatibility score
        compatibility_score = (similarity_score + sentiment_score) / 2
        compatibility_score = round(compatibility_score, 2)
        if compatibility_score < 0:
            compatibility_score = 0.1

        # Determine color based on the compatibility score
        color = get_color(compatibility_score)

        result = {
            'score': compatibility_score,
            'color': color
        }

        # Display the content of the selected policy text file
        with open(policy_file_path, "r", encoding="utf-8") as file:
            selected_policy_content = file.read()

        return render_template('index1.html', result=result, available_policies=available_policies, selected_policy_content=selected_policy_content)

    return render_template('index1.html', result=None, available_policies=available_policies, selected_policy_content=None)

def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    similarity_score = doc1.similarity(doc2)
    return similarity_score

def calculate_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']
    return sentiment_score

def get_color(score):
    if score >= 0.8:
        return 'green'
    elif 0.6 <= score < 0.8:
        return 'yellow'
    else:
        return 'red'
@app.route('/feedback')
def index3():
    return render_template('index3.html')

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form['feedback']
    feedback_data.append(feedback)
    return jsonify({"message": "Feedback submitted successfully!"})

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    return jsonify({"feedback":feedback_data})
                    
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        # Render the initial page with the form
        return render_template('index2.html', prediction_text=None, text_input=None)
    elif request.method == 'POST':
        # Get the text and rating from the form
        text = request.form['text']
        rating = float(request.form['rating'])  # Convert rating to float
    
        # Tokenize and pad the text (make sure you have defined tokenizer and max_len_text)
        text_sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(text_sequence, maxlen=max_len_text)
        
        # Preprocess rating input
        rating_input = np.array(rating).reshape(-1, 1)
        
        # Make prediction
        # Make prediction
        prediction = model.predict([np.array(padded_sequence), np.array(rating_input)])

        print(prediction)
        
        # Decode the prediction
        predicted_label = np.argmax(prediction)
        prediction_text = 'Predicted label: Original Review' if predicted_label == 1 else 'Predicted label: Computer Generated Fake review'
        
        # Return the prediction along with the input text
        if (predicted_label==1):
            return render_template('index2.html', prediction_text='Predicted label: Orginal Review', text_input=text)
        else:
            return render_template('index2.html', prediction_text='Predicted label: Computer Generated Fake review', text_input=text)
if __name__ == '__main__':
    app.run(port=5000, debug=True)
