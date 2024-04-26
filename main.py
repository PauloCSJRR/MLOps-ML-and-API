from flask import Flask, request, jsonify
from textblob import TextBlob
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

train = pd.read_csv('house-pricing/data/house-prices-advanced-regression-techniques/train.csv')
list = ['LotArea','YearBuilt','GarageCars']

train_filtered = train[['LotArea','YearBuilt','GarageCars','SalePrice']]

train_filtered.rename(columns={'LotArea':'Area'}, inplace=True)

X = train_filtered.drop('SalePrice', axis=1)
y = train_filtered['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)




app = Flask(__name__)

@app.route('/')
def home():
    return 'My API'

@app.route('/sentiment/<text>')
def sentiment(text):
    tb = TextBlob(text)
    
    polarity = tb.sentiment.polarity
    
    return f'Polarity: {polarity}'

@app.route('/housepricing/', methods=['POST'])
def house_pricing():
    data = request.get_json()
    data_input = [data[col] for col in list]
    predicted_price = lr.predict([data_input])
    
    return jsonify(predicted_price=predicted_price[0])

if __name__ == "__main__":
    app.run(debug=True)
