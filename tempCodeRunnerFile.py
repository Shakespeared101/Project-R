_sequence = pad_sequences(text_sequence, maxlen=max_len_text)
    
    # Preprocess rating input
    rating_input = np.array(rating).reshape(-1, 1)
    
    # Make prediction
    prediction = model.predict([padded_sequence, rating_input])
    print(prediction)
    # Decode the prediction
    # predicted_class = 1 if prediction[0][1] > 0.5 else 0
    predicted_label = np.argmax(prediction)
    # Return the prediction
    if (predicted_label==1):
        return render_template('index2.html', prediction_text='Predicted label: Orginal Review', text_input=text)
    else:
        return render_template('index2.html', prediction_text='Predicted label: Computer Generated Fake review', text_input=text)