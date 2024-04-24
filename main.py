from flask import Flask
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def home():
    return 'My API'

@app.route('/sentiment/<text>')
def sentiment(text):
    tb = TextBlob(text)
    
    polarity = tb.sentiment.polarity
    
    return f'Polarity: {polarity}'

if __name__ == "__main__":
    app.run(debug=True)
