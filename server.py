# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import string

app = Flask(__name__)

stop_words = set(stopwords.words('english'))
delchars = ''.join(c for c in map(chr, range(256)) if not c.isalpha() and not c.isspace())

tf = pickle.load(open("tfidf.pkl", "rb"))

# Load the model
model = pickle.load(open('logistic_model.sav','rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    review = ''
    result = ''

    if request.method == "POST":
        data = request.form['review-text']
    
        data = data.lower().translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', delchars))
        data = ' '.join([word for word in data.split() if word not in stop_words])

        data = [data]
        test_data = tf.transform(data)
        #print(str(test_data))

        # Make prediction using model loaded from disk as per the data.
        prediction = model.predict(test_data)

        result = 'Sentiment : '
        if prediction[0] == 0:
            result = result + 'Negative'
        elif prediction[0] == 1:
            result = result + 'Positive'

        review = 'Review Text : ' + request.form['review-text']

    return render_template('index.html', review=review, result=result)


if __name__ == '__main__':
    try:
        app.run(port=5000)
    except:
        print("Server is exited unexpectedly. Please contact server admin.")