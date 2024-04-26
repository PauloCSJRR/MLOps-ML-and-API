from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle

model = pickle.load(open('artifacts/model.sav','rb'))
list = ['LotArea','YearBuilt','GarageCars']

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'Paulo'
app.config['BASIC_AUTH_PASSWORD'] = 'admin'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return 'My API'

@app.route('/sentiment/<text>')
@basic_auth.required
def sentiment(text):
    tb = TextBlob(text)
    
    polarity = tb.sentiment.polarity
    
    return f'Polarity: {polarity}'

@app.route('/housepricing/', methods=['POST'])
@basic_auth.required
def house_pricing():
    data = request.get_json()
    data_input = [data[col] for col in list]
    predicted_price = model.predict([data_input])
    
    return jsonify(predicted_price=predicted_price[0])

if __name__ == "__main__":
    app.run(debug=True)
