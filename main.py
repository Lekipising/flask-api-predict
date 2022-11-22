import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import boto3
import json

from flask import (
    Flask,
    request
)

import s3fs
fs = s3fs.S3FileSystem(anon=True)


app = Flask(__name__)
app.config["DEBUG"] = True

# model = load_model('ml-objs/nextword1.h5')
client = boto3.client('s3', aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                      aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                      region_name="us-west-2")
client.download_file("ml-objs",
                     'model.h5',
                     'model.h5')
client.download_file("ml-objs",
                     'tokenizer1.pkl',
                     'tokenizer1.pkl')
model = load_model('model.h5')
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))


def predictor(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])[0]
    sequence = np.array(sequence)

    preds = model.predict(sequence)
    counter = 0
    index = 0
    highest_prob = preds[0][0]
    for k in preds:
        for j in k:
            if j > highest_prob:
                highest_prob = j
                index = counter
            else:
                counter += 1

    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == index:
            predicted_word = key

    return predicted_word


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(data["5words"])

    text = data["5words"].split(" ")
    text = text[-1]

    text = ''.join(text)

    prediction = predictor(model, tokenizer, text)

    return json.dumps(prediction)

# defalut route


@app.route('/', methods=['GET'])
def home():
    return "<h1>Next Word Predictor</h1>"


app.run()
