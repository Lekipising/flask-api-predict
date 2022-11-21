import pickle
import numpy as np
from tensorflow.keras.models import load_model
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
with fs.open('ml-objs/nextword1.h5', 'rb') as handle:
    model = load_model(handle)
with fs.open('ml-objs/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


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


app.run()
