import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)
model = pickle.load(open('venv/model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame(
        {'age': [int_features[0]/80], 'bp': [int_features[1]/10], 'al':[ int_features[2]/5], 'pcc': [int_features[3]],
         'bgr': [int_features[4]/146], 'bu': [int_features[5]/418], 'sod': [int_features[6]/164], 'pot': [int_features[7]/130.1],
         'hemo': [int_features[8]/21.7], 'rbcc': [int_features[9]/12.8], 'dm': [int_features[10]], 'appet': [int_features[11]]})
    prediction = model.predict(final_features)
    print(prediction[0][0])
    pred=[1 if y>=0.5 else 0 for y in prediction]

    return render_template('index.html', prediction_text='you have this disease ${}'.format(pred))


if __name__ == "__main__":
    app.run(debug=True)
