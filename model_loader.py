import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def load_and_predict_model(file_path):
    # Load your machine learning model
    model = load_model('model/dark_pattern_text_detection_model.h5')
    df = pd.read_csv('C:/Users/ASUS/Desktop/Projects/help/3.0/output.csv')
    df = df.drop(" page_id", axis=1)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    
    # Preprocess the text if needed
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['label'])
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df['text'])
    
    # Load the text from the file and split each line
    with open(file_path, 'r', encoding='utf-8') as file:
        new_data = [line.strip() for line in file]
    
    # Tokenize and pad sequences for the new data
    sequences = tokenizer.texts_to_sequences(new_data)
    padded_sequences = pad_sequences(sequences, maxlen=100)
    
    # Perform prediction
    prediction = model.predict(padded_sequences)
    decoded_predictions = label_encoder.inverse_transform(np.round(prediction).astype(int).flatten())
    
    # Convert the NumPy array to a regular Python list
    decoded_predictions_list = decoded_predictions.tolist()
    
    # Combine the text and predictions into a list of dictionaries
    result_data = [{'text': text, 'prediction': pred} for text, pred in zip(new_data, decoded_predictions_list)]
    
    return result_data
