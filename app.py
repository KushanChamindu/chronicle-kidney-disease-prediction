# import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
import json
import pickle
from flask import make_response

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json(silent=True, force=True)
    print('Request: ')
    print(json.dumps(req['queryResult']['parameters'], indent=4))
    int_features = json.dumps(req['queryResult']['parameters'], indent=4)
    int_features = json.loads(int_features)

    final_features = pd.DataFrame(
        {'age': [float(int_features['age']['amount']) / 80], 'bp': [float(int_features['bp']) / 10],
         'al': [float(int_features['al']) / 5], 'pcc': [float(int_features['pcc'])],
         'bgr': [float(int_features['bgr']) / 146], 'bu': [float(int_features['bu']) / 418],
         'sod': [float(int_features['sod']) / 164], 'pot': [float(int_features['pot']) / 130.1],
         'hemo': [float(int_features['hemo']) / 21.7], 'rbcc': [float(int_features['rbcc']) / 12.8],
         'dm': [float(int_features['dm'])], 'appet': [float(int_features['appet'])]})
    prediction = model.predict(final_features)
    print(prediction[0][0])
    pred = (prediction[0][0] * 0.96)

    if pred > 0.75:
        result_prediction = 'Don\'t worry you have ' + str((1 - pred) * 100) + '% precentage to have chronicle kidney disease. We have 99.06% confident for say that !! '
    elif (pred > 0.5):
        result_prediction = 'You are in some denger zone. You have ' + str((1 - pred) * 100) + '% percentae to have chronicle kidney disease. We have 99.06% confident for say that !! Care full'
    else:
        result_prediction = 'You are in very danger zone.You have ' + str((1 - pred) * 100) + '% percentae to have chronicle kidney disease. We have 99.06% confident for say that !! Meet your doctor as soon as possible'
    res = {
        "fulfillmentText": result_prediction
    }
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r

if __name__ == "__main__":
    app.run(debug=True)
