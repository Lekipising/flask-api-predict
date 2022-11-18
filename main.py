import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json

from flask import (
    Flask,
    request
)

app = Flask(__name__)
app.config["DEBUG"] = True

model = load_model('nextword1.h5')
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))


def Predict_Next_Words(model, tokenizer, text):
    for i in range(3):
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = np.array(sequence)

        preds = model.predict_classes(sequence)
#         print(preds)
        predicted_word = ""

        for key, value in tokenizer.word_index.items():
            if value == preds:
                predicted_word = key
                break

        print(predicted_word)
        return predicted_word


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    print(data["5words"])

    text = data["5words"].split(" ")
    text = text[-1]

    text = ''.join(text)

    prediction = Predict_Next_Words(model, tokenizer, text)

    return json.dumps(prediction)


app.run()
