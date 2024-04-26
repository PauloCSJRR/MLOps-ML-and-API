import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

train = pd.read_csv('house-pricing/data/house-prices-advanced-regression-techniques/train.csv')
list = ['LotArea','YearBuilt','GarageCars']

train_filtered = train[['LotArea','YearBuilt','GarageCars','SalePrice']]

train_filtered.rename(columns={'LotArea':'Area'}, inplace=True)

X = train_filtered.drop('SalePrice', axis=1)
y = train_filtered['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()

lr.fit(X_train, y_train)

pickle.dump(lr, open('artifacts/model.sav', 'wb'))