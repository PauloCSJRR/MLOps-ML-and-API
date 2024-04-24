from flask import Flask

app = Flask('my_app')

@app.route('/')

def home():
    return 'My API'

app.run()